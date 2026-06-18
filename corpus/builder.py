"""
真实研报语料构建器
从研报向量库中提取、清洗、分类、去重真实研报文本，构建高质量金融训练语料。
不生成合成数据——所有语料内容均来自真实研报原文。
"""
import json
import os
import re
import hashlib
from datetime import datetime
from collections import Counter
from typing import Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

from config import FAISS_DIR, CORPUS_DIR, EMBEDDING_MODEL, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from corpus.scenarios import SCENARIOS

# ─── 研报常见 boilerplate 模式 ───
BOILERPLATE_PATTERNS = [
    r"请阅读最后一页免责声明.*",
    r"免责声明.*",
    r"本报告由.*证券.*制作",
    r"投资评级说明.*",
    r"分析师声明.*",
    r"重要声明.*",
    r"风险提示.*投资需谨慎",
    r"证券投资咨询.*资格证书号.*",
    r"本公司不承担.*责任",
    r"未经.*书面许可.*不得.*转载",
    r"版权所有.*",
    r"扫码关注.*",
    r"更多研报.*关注.*",
    r"评级标准.*|评级说明.*",
]
BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE)

# ─── 金融实体提取正则 ───
STOCK_CODE_RE = re.compile(r"[036]\d{5}\.SZ|[036]\d{5}\.SH|[68]\d{4}\.SH")
MONEY_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(万亿|亿|万元|万元|亿元)")
PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
DATE_RE = re.compile(r"(20\d{2})\s*[年/-]\s*(\d{1,2})\s*[月/-]?")
RATING_RE = re.compile(r"(买入|增持|中性|减持|卖出|强烈推荐|推荐|优于大市|同步大市|弱于大市)")


class CorpusBuilder:
    """
    真实研报语料构建器
    流程：抽样 → 清洗 → 分类 → 去重 → 质量过滤 → 导出
    """

    def __init__(self):
        embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.db = FAISS.load_local(FAISS_DIR, embedding, allow_dangerous_deserialization=True)
        self.client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        self.chunks: list[dict] = []
        self._content_hashes = set()

    # ─── 1. 抽样 ───
    def sample(self, queries: list[str], k_per_query: int = 30) -> list[dict]:
        """用多个检索词从向量库抽样，确保覆盖面"""
        seen = set()
        all_chunks = []
        for q in queries:
            results = self.db.similarity_search(q, k=k_per_query)
            for r in results:
                content = r.page_content.strip()
                h = hashlib.md5(content.encode()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                all_chunks.append({
                    "content": content,
                    "org": r.metadata.get("org", ""),
                    "title": r.metadata.get("title", r.metadata.get("report_title", "")),
                    "stock": r.metadata.get("stock", ""),
                    "chunk_hash": h,
                })
        print(f"抽样: {len(all_chunks)} 条（去重后）")
        return all_chunks

    # ─── 2. 清洗 ───
    def clean(self, chunks: list[dict]) -> list[dict]:
        """清洗研报文本：去boilerplate、去页眉页脚、规范化空白"""
        cleaned = []
        removed = 0
        for c in chunks:
            text = c["content"]
            # 去 boilerplate
            text = BOILERPLATE_RE.sub("", text)
            # 去纯符号行
            lines = text.split("\n")
            lines = [l.strip() for l in lines if l.strip()]
            # 过滤明显不是正文的行
            lines = [
                l for l in lines
                if not l.startswith("数据来源：")
                and not l.startswith("图")
                and not l.startswith("表")
                and not l.startswith("证券研究报告")
                and len(l) > 3
            ]
            text = "\n".join(lines)
            # 规范化空白
            text = re.sub(r"\s{3,}", "  ", text)
            text = text.strip()

            if len(text) < 30:  # 太短的弃掉
                removed += 1
                continue

            c["content"] = text
            c["char_count"] = len(text)
            cleaned.append(c)

        print(f"清洗: {len(chunks)} → {len(cleaned)} 条（去除 {removed} 条过短）")
        return cleaned

    # ─── 3. 场景分类 ───
    def classify(self, chunks: list[dict], sample_size: int = 100) -> list[dict]:
        """
        用LLM对文本进行场景分类（仅分类，不生成内容）
        为节省API调用，对大批量数据：LLM分类小样本 → 关键词匹配扩展到全部
        """
        if len(chunks) <= sample_size:
            # 全部用LLM分类
            return self._llm_classify(chunks)

        # 小样本LLM分类 → 提取关键词 → 规则扩展
        print(f"  小样本LLM分类 {sample_size} 条，其余用规则扩展...")
        import random
        random.seed(42)
        sample = random.sample(chunks, sample_size)
        labeled = self._llm_classify(sample)

        # 从已标注样本中提取各场景的关键词
        scenario_keywords = self._extract_keywords(labeled)

        # 规则扩展到全部
        for c in chunks:
            if c.get("scenario"):
                continue
            c["scenario"] = self._rule_classify(c["content"], scenario_keywords)
            c["scenario_name"] = SCENARIOS.get(c["scenario"], {}).get("name", "未分类")

        return chunks

    def _llm_classify(self, chunks: list[dict]) -> list[dict]:
        """使用LLM批量分类"""
        scenario_desc = "\n".join(
            f"- {k}: {v['name']}（{v['description']}）" for k, v in SCENARIOS.items()
        )

        batch_texts = []
        for i, c in enumerate(chunks):
            batch_texts.append(f"[{i}] {c['content'][:300]}")

        prompt = f"""将以下金融研报片段分类到对应场景。每行输出序号和场景代码。

场景代码：
{scenario_desc}

输出格式（每行一个）：
[序号] scenario_code

以下是待分类文本：
{chr(10).join(batch_texts[:50])}

请输出分类结果："""

        try:
            resp = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            text = resp.choices[0].message.content
            # 解析分类结果
            for line in text.split("\n"):
                match = re.match(r"\[(\d+)\]\s*(\w+)", line.strip())
                if match:
                    idx = int(match.group(1))
                    code = match.group(2).strip()
                    if idx < len(chunks) and code in SCENARIOS:
                        chunks[idx]["scenario"] = code
                        chunks[idx]["scenario_name"] = SCENARIOS[code]["name"]
        except Exception as e:
            print(f"  ⚠ LLM分类失败: {e}")

        # 未分类的归为 financial_qa
        for c in chunks:
            if "scenario" not in c:
                c["scenario"] = "financial_qa"
                c["scenario_name"] = "金融问答"

        return chunks

    def _extract_keywords(self, labeled: list[dict]) -> dict[str, list[str]]:
        """从已标注数据中提取各场景高频关键词"""
        from collections import Counter
        keywords = {}
        for sk in SCENARIOS:
            texts = [c["content"] for c in labeled if c.get("scenario") == sk]
            if not texts:
                keywords[sk] = []
                continue
            combined = " ".join(texts)
            # 简单提取2-4字高频词
            words = re.findall(r"[一-鿿]{2,4}", combined)
            counter = Counter(words)
            # 过滤通用词
            stopwords = {"证券", "报告", "公司", "行业", "数据", "同比", "增长", "发展", "市场"}
            top = [w for w, _ in counter.most_common(30) if w not in stopwords][:15]
            keywords[sk] = top
        return keywords

    def _rule_classify(self, text: str, scenario_keywords: dict) -> str:
        """基于关键词匹配做规则分类"""
        scores = {}
        for sk, kws in scenario_keywords.items():
            score = sum(1 for kw in kws if kw in text)
            scores[sk] = score
        if max(scores.values()) == 0:
            return "financial_qa"
        return max(scores, key=scores.get)

    # ─── 4. 实体提取 ───
    def extract_entities(self, chunks: list[dict]) -> list[dict]:
        """提取每段文本中的金融实体"""
        for c in chunks:
            text = c["content"]
            c["extracted"] = {
                "stock_codes": list(set(STOCK_CODE_RE.findall(text))),
                "money_values": [m[0] + m[1] for m in MONEY_RE.findall(text)][:5],
                "percentages": PCT_RE.findall(text)[:5],
                "ratings": list(set(RATING_RE.findall(text))),
            }
        return chunks

    # ─── 5. 去重 ───
    def deduplicate(self, chunks: list[dict], similarity_threshold: float = 0.85) -> list[dict]:
        """基于内容相似度去重（简单Jaccard）"""
        def tokenize(text):
            return set(re.findall(r"[一-鿿]+", text))

        deduped = []
        seen_tokens = []
        for c in chunks:
            tokens = tokenize(c["content"][:200])
            if not tokens:
                continue
            is_dup = False
            for st in seen_tokens[-50:]:  # 只检查最近50条，避免O(n²)
                overlap = len(tokens & st) / max(1, len(tokens | st))
                if overlap > similarity_threshold:
                    is_dup = True
                    break
            if not is_dup:
                seen_tokens.append(tokens)
                deduped.append(c)

        print(f"去重: {len(chunks)} → {len(deduped)} 条")
        return deduped

    # ─── 6. 质量评分 ───
    def quality_filter(self, chunks: list[dict], min_chars: int = 50, max_chars: int = 3000) -> list[dict]:
        """质量过滤"""
        filtered = []
        for c in chunks:
            text = c["content"]
            chars = len(text)
            if chars < min_chars or chars > max_chars:
                continue
            # 纯数字/符号占比过高
            alpha_chars = len(re.findall(r"[一-鿿]", text))
            if alpha_chars / max(1, chars) < 0.3:
                continue
            # 太多连续重复字符
            if re.search(r"(.)\1{10,}", text):
                continue
            filtered.append(c)

        print(f"质量过滤: {len(chunks)} → {len(filtered)} 条")
        return filtered

    # ─── 7. 导出 ───
    def export(self, chunks: list[dict] | None = None, filename: str = "corpus.jsonl") -> str:
        """导出为JSONL格式训练语料"""
        if chunks is None or isinstance(chunks, str):
            if isinstance(chunks, str):
                filename = chunks
            chunks = self.chunks
        path = os.path.join(CORPUS_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            for c in chunks:
                record = {
                    "text": c["content"],
                    "scenario": c.get("scenario", ""),
                    "scenario_name": c.get("scenario_name", ""),
                    "source_org": c.get("org", ""),
                    "source_title": c.get("title", ""),
                    "source_stock": c.get("stock", ""),
                    "char_count": c.get("char_count", len(c["content"])),
                    "chunk_hash": c.get("chunk_hash", ""),
                    "extracted_entities": c.get("extracted", {}),
                    "processed_at": datetime.now().isoformat(),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"✓ 导出 {len(chunks)} 条真实语料到 {path}")
        return path

    # ─── 全流程 ───
    def build(
        self,
        scenario_keys: Optional[list[str]] = None,
        total_chunks: int = 500,
    ) -> list[dict]:
        """完整语料构建流程"""
        if scenario_keys is None:
            scenario_keys = list(SCENARIOS.keys())

        search_queries = {
            "investment_research": [
                "行业发展趋势分析", "市场竞争格局", "产业链上下游", "公司核心竞争力",
                "业绩增长驱动因素", "行业政策影响", "技术路线对比", "市场份额变化",
            ],
            "financial_qa": [
                "估值方法分析", "PE PB ROE指标", "财务指标解读", "投资策略分析",
                "行业景气度判断", "盈利预测调整", "目标价分析", "分红政策",
            ],
            "report_interpretation": [
                "营业收入增长分析", "净利润变动原因", "毛利率变化趋势",
                "资产负债结构", "现金流状况分析", "费用率变化", "应收账款周转",
            ],
            "compliance": [
                "监管政策解读", "信息披露要求", "风险管理措施", "合规审查标准",
                "行业监管动态", "内部控制制度", "关联交易披露", "重大事项公告",
            ],
        }

        all_queries = []
        for sk in scenario_keys:
            all_queries.extend(search_queries.get(sk, [])[:10])

        print("=" * 60)
        print("  真实金融研报语料构建")
        print(f"  目标场景: {', '.join(SCENARIOS[k]['name'] for k in scenario_keys)}")
        print(f"  目标条数: {total_chunks}")
        print("=" * 60)

        # Step 1: 抽样
        chunks = self.sample(all_queries, k_per_query=max(10, total_chunks // len(all_queries)))
        # Step 2: 清洗
        chunks = self.clean(chunks)
        # Step 3: 分类
        chunks = self.classify(chunks)
        # Step 4: 实体提取
        chunks = self.extract_entities(chunks)
        # Step 5: 去重
        chunks = self.deduplicate(chunks)
        # Step 6: 质量过滤
        chunks = self.quality_filter(chunks)
        # 截取目标数量
        chunks = chunks[:total_chunks]

        self.chunks = chunks
        return chunks

    def stats(self) -> dict:
        """语料统计"""
        if not self.chunks:
            return {}
        scenarios = Counter(c.get("scenario_name", "未分类") for c in self.chunks)
        orgs = Counter(c.get("org", "") for c in self.chunks if c.get("org"))
        stocks = Counter(c.get("stock", "") for c in self.chunks if c.get("stock"))
        total_chars = sum(c.get("char_count", 0) for c in self.chunks)
        return {
            "total_chunks": len(self.chunks),
            "total_characters": total_chars,
            "avg_chars_per_chunk": total_chars // max(1, len(self.chunks)),
            "by_scenario": dict(scenarios),
            "top_orgs": dict(orgs.most_common(10)),
            "top_stocks": dict(stocks.most_common(10)),
        }
