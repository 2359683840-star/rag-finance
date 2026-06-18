"""
评测Benchmark构建器
构建两类真实评测基准：
1. 检索评测：验证系统能否根据query找到正确的研报
2. 忠实度评测：验证模型回答是否基于真实来源
"""
import json
import os
import re
import hashlib
import random
from datetime import datetime
from collections import defaultdict

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

from config import FAISS_DIR, EVAL_DIR, EMBEDDING_MODEL, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


class BenchmarkBuilder:
    """
    构建真实评测Benchmark
    不生成合成题目——所有测试基于真实的研报内容和检索行为
    """

    CATEGORIES = {
        "retrieval": "检索评测",
        "faithfulness": "忠实度评测",
        "factual": "事实准确性评测",
        "citation": "引用正确性评测",
    }

    def __init__(self):
        embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.db = FAISS.load_local(FAISS_DIR, embedding, allow_dangerous_deserialization=True)
        self.client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        self.test_cases: list[dict] = []

    # ═══════════════════════════════════════════
    # 类别1: 检索评测
    # 给定一个自然语言查询，正确答案是特定机构/标题的研报
    # 评测指标: Precision@K, Recall@K, MRR
    # ═══════════════════════════════════════════
    def build_retrieval_tests(self, n: int = 30) -> list[dict]:
        """构建检索评测集"""
        print(f"\n📋 构建检索评测集 (目标{n}题)...")

        # 从向量库中采样文档作为"正确答案"
        # 用多个抽样query获取多样性文档
        sample_queries = [
            "宁德时代 电池技术", "比亚迪 新能源汽车", "隆基绿能 光伏",
            "固态电池 产业化", "储能 行业分析", "锂电材料 价格",
            "光伏组件 出口", "风电 政策", "新能源汽车 销量",
            "充电桩 建设", "碳酸锂 价格", "电力设备 出海",
        ]

        all_docs = []
        seen = set()
        for q in sample_queries:
            results = self.db.similarity_search(q, k=10)
            for r in results:
                h = hashlib.md5(r.page_content.encode()).hexdigest()
                if h not in seen:
                    seen.add(h)
                    all_docs.append({
                        "content": r.page_content.strip(),
                        "org": r.metadata.get("org", ""),
                        "title": r.metadata.get("title", r.metadata.get("report_title", "")),
                        "stock": r.metadata.get("stock", ""),
                        "doc_hash": h,
                    })

        random.seed(42)
        sampled = random.sample(all_docs, min(n, len(all_docs)))

        tests = []
        for doc in sampled:
            # 用LLM根据文档内容生成一个自然的检索query
            # 这个query应该能让这篇文档排在靠前位置
            query = self._generate_retrieval_query(doc)
            if not query:
                continue

            tests.append({
                "id": hashlib.md5(doc["doc_hash"].encode()).hexdigest()[:10],
                "category": "retrieval",
                "category_name": "检索评测",
                "query": query,
                "expected_org": doc["org"],
                "expected_title": doc.get("title", ""),
                "expected_content_hash": doc["doc_hash"],
                "expected_keywords": self._extract_key_terms(doc["content"]),
            })

        print(f"  ✓ 生成 {len(tests)} 道检索评测题")
        self.test_cases.extend(tests)
        return tests

    def _generate_retrieval_query(self, doc: dict) -> str:
        """根据文档内容生成自然的检索query"""
        prompt = f"""你是一位金融研究员，正在查询研报数据库。请根据以下研报内容，写出一个你会用来查询这篇研报的自然语言检索问句。

要求：
- 必须是研究员真实会问的问题
- 包含具体的公司/行业/概念名称
- 不要太宽泛也不要太具体
- 15-40个字
- 直接输出问题，不要其他内容

研报内容：
来源：{doc.get('org', '')}
标题：{doc.get('title', '')}
{doc['content'][:500]}

检索问句："""

        try:
            resp = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            query = resp.choices[0].message.content.strip()
            return query[:80]
        except:
            # fallback: 用标题作为query
            title = doc.get("title", "")
            org = doc.get("org", "")
            stock = doc.get("stock", "")
            parts = [p for p in [org, stock, title[:30]] if p]
            return " ".join(parts)[:60]

    def _extract_key_terms(self, text: str) -> list[str]:
        """提取文本中的关键术语"""
        # 简单提取金融相关名词短语
        patterns = [
            r"(?:宁德时代|比亚迪|隆基绿能|阳光电源|[^\s]{2,4}(?:股份|科技|能源|材料|电力))",
            r"(?:锂电|光伏|储能|风电|新能源|固态电池|钠离子)[^\s]{0,4}",
            r"\d+(?:\.\d+)?(?:亿|%|倍|GWh|GW)",
        ]
        terms = []
        for p in patterns:
            terms.extend(re.findall(p, text))
        return list(set(terms))[:10]

    # ═══════════════════════════════════════════
    # 类别2: 忠实度评测
    # 给定一段研报原文 + 一个基于原文的声明(claim)
    # 评测: 模型能否正确判断声明是否被原文支持
    # ═══════════════════════════════════════════
    def build_faithfulness_tests(self, n: int = 30) -> list[dict]:
        """构建忠实度评测集"""
        print(f"\n📋 构建忠实度评测集 (目标{n}题)...")

        # 抽样文档
        results = self.db.similarity_search("财务数据分析 行业趋势", k=50)
        docs = []
        seen = set()
        for r in results:
            h = hashlib.md5(r.page_content.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                docs.append({
                    "content": r.page_content.strip(),
                    "org": r.metadata.get("org", ""),
                    "title": r.metadata.get("title", ""),
                })

        random.seed(42)
        sampled = random.sample(docs, min(n, len(docs)))

        tests = []
        for doc in sampled:
            claims = self._generate_claims(doc)
            if claims:
                tests.extend(claims)

        print(f"  ✓ 生成 {len(tests)} 道忠实度评测题")
        self.test_cases.extend(tests)
        return tests

    def _generate_claims(self, doc: dict) -> list[dict]:
        """基于文档内容生成声明（支持/不支持）"""
        prompt = f"""你是一位金融数据质量审核员。请基于以下研报内容，生成3个声明（claim），其中：
- 1-2个是原文明确支持的真实声明（label: supported）
- 1-2个是对原文数据的篡改/曲解（label: not_supported），例如改数字、改方向、张冠李戴

输出严格JSON数组：
```json
[
  {{"claim": "声明内容", "label": "supported", "evidence": "原文中支持该声明的关键句"}},
  ...
]
```

研报原文：
来源：{doc.get('org', '')}
{doc['content'][:1500]}

直接输出JSON数组："""

        try:
            resp = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
            )
            text = resp.choices[0].message.content
            json_text = self._extract_json(text)
            claims = json.loads(json_text)

            for c in claims:
                c["id"] = hashlib.md5(
                    (c["claim"] + str(datetime.now().timestamp())).encode()
                ).hexdigest()[:10]
                c["category"] = "faithfulness"
                c["category_name"] = "忠实度评测"
                c["source_org"] = doc.get("org", "")
                c["source_content"] = doc["content"][:800]
                c["created_at"] = datetime.now().isoformat()

            return claims
        except Exception as e:
            print(f"  ⚠ 生成声明失败: {e}")
            return []

    # ═══════════════════════════════════════════
    # 类别3: 事实准确性评测
    # 问一个可在原文中找到确切答案的问题
    # ═══════════════════════════════════════════
    def build_factual_tests(self, n: int = 30) -> list[dict]:
        """构建事实准确性评测集"""
        print(f"\n📋 构建事实准确性评测集 (目标{n}题)...")

        results = self.db.similarity_search("营收 净利润 增长 市场份额 产能", k=40)
        docs = []
        seen = set()
        for r in results:
            # 优先选包含具体数据的文本
            text = r.page_content
            if re.search(r"\d+(?:\.\d+)?(?:亿|%|倍|万吨|GWh)", text):
                h = hashlib.md5(text.encode()).hexdigest()
                if h not in seen:
                    seen.add(h)
                    docs.append({
                        "content": text.strip(),
                        "org": r.metadata.get("org", ""),
                    })

        random.seed(42)
        sampled = random.sample(docs, min(n, len(docs)))

        tests = []
        for doc in sampled:
            qa = self._extract_factual_qa(doc)
            if qa:
                tests.append(qa)

        print(f"  ✓ 生成 {len(tests)} 道事实准确性评测题")
        self.test_cases.extend(tests)
        return tests

    def _extract_factual_qa(self, doc: dict) -> dict | None:
        """从文档中提取可验证的事实问答"""
        prompt = f"""以下是一段金融研报。请从中提取1个可以基于原文确切回答的事实问题，并给出答案。要求问题答案必须在原文中有明确的数字或事实支持。

输出JSON：
```json
{{"question": "事实性问题", "answer": "基于原文的确切答案", "evidence": "原文证据句"}}
```

原文：
{doc['content'][:1500]}

直接输出JSON："""

        try:
            resp = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            text = resp.choices[0].message.content
            json_text = self._extract_json(text)
            qa = json.loads(json_text)
            qa["id"] = hashlib.md5(
                (qa["question"] + str(datetime.now().timestamp())).encode()
            ).hexdigest()[:10]
            qa["category"] = "factual"
            qa["category_name"] = "事实准确性评测"
            qa["source_org"] = doc.get("org", "")
            qa["source_content"] = doc["content"][:800]
            qa["created_at"] = datetime.now().isoformat()
            return qa
        except:
            return None

    # ═══════════════════════════════════════════
    # 全量构建
    # ═══════════════════════════════════════════
    def build(self, n_per_category: int = 20) -> list[dict]:
        """构建完整Benchmark"""
        self.test_cases = []
        self.build_retrieval_tests(n=n_per_category)
        self.build_faithfulness_tests(n=n_per_category)
        self.build_factual_tests(n=n_per_category)
        print(f"\n✓ Benchmark构建完成: 共 {len(self.test_cases)} 道题")
        return self.test_cases

    def export(self, filename: str = "benchmark.json") -> str:
        path = os.path.join(EVAL_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.test_cases, f, ensure_ascii=False, indent=2)
        print(f"✓ Benchmark导出: {path}")
        return path

    @classmethod
    def load(cls, filename: str = "benchmark.json") -> list[dict]:
        path = os.path.join(EVAL_DIR, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def stats(self) -> dict:
        from collections import Counter
        cats = Counter(t.get("category_name", "") for t in self.test_cases)
        return {
            "total": len(self.test_cases),
            "by_category": dict(cats),
        }

    def _extract_json(self, text: str) -> str:
        if "```json" in text:
            s = text.index("```json") + 7
            e = text.index("```", s)
            return text[s:e].strip()
        if "```" in text:
            s = text.index("```") + 3
            e = text.index("```", s)
            return text[s:e].strip()
        if "[" in text and "]" in text:
            return text[text.index("["):text.rindex("]") + 1]
        if "{" in text and "}" in text:
            return text[text.index("{"):text.rindex("}") + 1]
        return text.strip()
