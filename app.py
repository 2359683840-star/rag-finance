"""
金融LLM数据建设与评测平台
核心原则：所有数据来自真实研报，所有评测可追溯到原文来源。
— 对标 月之暗面 数据专家实习生-金融
"""
import os
import json
import time
import hashlib
from datetime import datetime
from collections import Counter

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import streamlit as st
import pandas as pd

st.set_page_config(page_title="金融LLM数据与评测平台", page_icon="📊", layout="wide")

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

from config import FAISS_DIR, CORPUS_DIR, EVAL_DIR, EMBEDDING_MODEL
from corpus.scenarios import SCENARIOS

# ─── API ───
try:
    api_key = os.getenv("DASHSCOPE_API_KEY") or st.secrets.get("DASHSCOPE_API_KEY", "")
except:
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
try:
    base_url = os.getenv("API_BASE_URL") or st.secrets.get("API_BASE_URL", "https://api.deepseek.com")
except:
    base_url = os.getenv("API_BASE_URL", "https://api.deepseek.com")
try:
    model_name = os.getenv("LLM_MODEL") or st.secrets.get("LLM_MODEL", "deepseek-chat")
except:
    model_name = os.getenv("LLM_MODEL", "deepseek-chat")

client = OpenAI(api_key=api_key, base_url=base_url)


# ─── Vector DB ───
@st.cache_resource
def load_embedding():
    """加载Embedding模型，支持通过HF_ENDPOINT环境变量指定镜像"""
    import os as _os
    # 优先使用环境变量/Secrets中配置的镜像端点
    _hf_endpoint = _os.getenv("HF_ENDPOINT", "")
    if _hf_endpoint:
        _os.environ["HF_ENDPOINT"] = _hf_endpoint
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource
def load_db():
    idx_path = os.path.join(FAISS_DIR, "index.faiss")
    if not os.path.exists(idx_path):
        return None
    try:
        embedding = load_embedding()
        return FAISS.load_local(FAISS_DIR, embedding, allow_dangerous_deserialization=True)
    except OSError as e:
        st.error(f"""
        **Embedding模型加载失败**

        可能原因：
        1. HuggingFace 连接超时 → 在 Secrets 中添加 `HF_ENDPOINT = "https://hf-mirror.com"`
        2. 模型未缓存 → 首次部署需要下载约400MB模型文件

        请在 App settings → Secrets 中检查配置后重启应用。
        """)
        return None
    except Exception as e:
        st.error(f"向量库加载失败: {e}")
        return None


vectordb = load_db()


def search_reports(query: str, k: int = 10) -> list[dict]:
    if vectordb is None:
        return []
    results = vectordb.similarity_search(query, k=k)
    return [
        {
            "content": r.page_content.strip(),
            "org": r.metadata.get("org", ""),
            "title": r.metadata.get("title", r.metadata.get("report_title", "")),
            "stock": r.metadata.get("stock", ""),
        }
        for r in results
    ]


# ─── Helpers ───
def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except:
                pass
    return data


def load_json(path: str):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════
st.sidebar.title("📊 金融LLM数据与评测平台")
st.sidebar.markdown("*基于真实研报来源*")

page = st.sidebar.radio(
    "导航",
    ["🏗️ 语料工坊", "📋 Benchmark管理", "🔬 模型评测", "📈 评测报告", "🔍 研报检索"],
)

st.sidebar.markdown("---")

if vectordb is None:
    st.sidebar.warning("⚠️ 向量库未初始化\n\n运行 `python build_knowledge_base.py`")
else:
    st.sidebar.success("✅ 向量库已就绪")

corpus_files = [f for f in os.listdir(CORPUS_DIR) if f.endswith(".jsonl")] if os.path.exists(CORPUS_DIR) else []
benchmark_path = os.path.join(EVAL_DIR, "benchmark.json")
eval_results_path = os.path.join(EVAL_DIR, "eval_results.json")

if corpus_files:
    st.sidebar.info(f"📝 已缓存语料: {len(corpus_files)} 个文件")
if os.path.exists(benchmark_path):
    st.sidebar.info("📋 Benchmark已构建")
if os.path.exists(eval_results_path):
    st.sidebar.info("📊 评测结果已缓存")

st.sidebar.markdown("---")
st.sidebar.caption("Moonshot.AI · 数据专家实习生")

# ═══════════════════════════════════════
# Page 1: 语料工坊
# ═══════════════════════════════════════
if page == "🏗️ 语料工坊":
    st.title("🏗️ 语料工坊 — 真实研报训练语料")

    st.markdown("""
    > **核心原则：所有语料来自真实研报原文。** 流程：抽样 → 清洗 → 场景分类 → 实体提取 → 去重 → 质量过滤 → 导出。
    > 不做LLM合成——只对真实文本做加工和结构化。
    """)

    tab1, tab2, tab3 = st.tabs(["⚙️ 构建语料", "📋 语料浏览", "📊 统计"])

    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            selected = st.multiselect(
                "场景覆盖",
                [s["name"] for s in SCENARIOS.values()],
                default=[s["name"] for s in SCENARIOS.values()],
            )
            nk = {v["name"]: k for k, v in SCENARIOS.items()}
            scenario_keys = [nk[n] for n in selected]
        with col2:
            total = st.slider("目标语料条数", 50, 1000, 300, 50)

        if st.button("🚀 构建真实语料", type="primary", disabled=vectordb is None):
            from corpus.builder import CorpusBuilder

            builder = CorpusBuilder()
            progress = st.progress(0)
            status = st.empty()

            status.text("Step 1/6: 抽样...")
            chunks = builder.sample(
                [q for qs in [
                    ["行业趋势", "市场竞争", "产业链", "核心竞争力", "业绩驱动"],
                    ["估值方法", "财务指标", "投资策略", "盈利预测"],
                    ["营收增长", "净利润", "毛利率", "资产负债", "现金流"],
                    ["监管政策", "信息披露", "风险管理", "合规审查"],
                ] for q in qs], k_per_query=max(5, total // 20)
            )
            progress.progress(15)

            status.text("Step 2/6: 清洗（去boilerplate、规范化）...")
            chunks = builder.clean(chunks)
            progress.progress(30)

            status.text("Step 3/6: 场景分类...")
            chunks = builder.classify(chunks)
            progress.progress(50)

            status.text("Step 4/6: 实体提取...")
            chunks = builder.extract_entities(chunks)
            progress.progress(65)

            status.text("Step 5/6: 去重...")
            chunks = builder.deduplicate(chunks)
            progress.progress(80)

            status.text("Step 6/6: 质量过滤 + 导出...")
            chunks = builder.quality_filter(chunks)
            builder.chunks = chunks[:total]
            path = builder.export(chunks[:total])
            progress.progress(100)
            status.text("✓ 完成！")

            stats = builder.stats()
            st.success(f"**{stats['total_chunks']}** 条真实语料 → `{path}`")
            c1, c2, c3 = st.columns(3)
            c1.metric("总字符数", f"{stats.get('total_characters', 0):,}")
            c2.metric("平均每条", f"{stats.get('avg_chars_per_chunk', 0)} 字符")
            c3.metric("覆盖机构", len(stats.get("top_orgs", {})))
            with st.expander("场景分布"):
                st.json(stats.get("by_scenario", {}))

    with tab2:
        corpus_path = os.path.join(CORPUS_DIR, corpus_files[0]) if corpus_files else None
        if corpus_path:
            data = load_jsonl(corpus_path)
            scenario_filter = st.selectbox("场景筛选", ["全部"] + [s["name"] for s in SCENARIOS.values()])
            filtered = data if scenario_filter == "全部" else [d for d in data if d.get("scenario_name") == scenario_filter]
            st.info(f"{len(filtered)} 条语料")

            rows = [{
                "场景": d.get("scenario_name", ""),
                "来源机构": d.get("source_org", ""),
                "相关股票": d.get("source_stock", ""),
                "长度": d.get("char_count", 0),
                "文本预览": d.get("text", "")[:100],
            } for d in filtered[:100]]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            if filtered:
                idx = st.number_input("查看详情（序号）", 0, len(filtered) - 1, 0)
                d = filtered[idx]
                st.markdown("---")
                st.markdown(f"**场景**: {d.get('scenario_name','')} | **来源**: {d.get('source_org','')} | **股票**: {d.get('source_stock','')}")
                st.markdown(f"**标题**: {d.get('source_title','')}")
                with st.expander("全文"):
                    st.text(d.get("text", ""))
                with st.expander("提取的实体"):
                    st.json(d.get("extracted_entities", {}))
        else:
            st.info("暂无语料，请先构建")

    with tab3:
        corpus_path = os.path.join(CORPUS_DIR, corpus_files[0]) if corpus_files else None
        if corpus_path:
            data = load_jsonl(corpus_path)
            c1, c2, c3 = st.columns(3)
            c1.metric("总条数", len(data))
            c2.metric("总字符", f"{sum(d.get('char_count', 0) for d in data):,}")
            c3.metric("覆盖机构", len(set(d.get("source_org", "") for d in data)))

            sc = Counter(d.get("scenario_name", "") for d in data)
            st.bar_chart(pd.DataFrame({"数量": sc}))

            orgs = Counter(d.get("source_org", "") for d in data if d.get("source_org"))
            if orgs:
                st.caption(f"来源机构 TOP10: {dict(orgs.most_common(10))}")
        else:
            st.info("暂无数据")


# ═══════════════════════════════════════
# Page 2: Benchmark管理
# ═══════════════════════════════════════
elif page == "📋 Benchmark管理":
    st.title("📋 Benchmark管理 — 来源锚定评测集")

    st.markdown("""
    > 三类评测基准，全部基于真实研报内容：
    > - **检索评测**：给定查询，验证能否找到正确机构/文档
    > - **忠实度评测**：判断声明是否被原文支持
    > - **事实准确性评测**：验证回答中的事实是否与原文一致
    """)

    tab1, tab2 = st.tabs(["⚙️ 构建", "📋 浏览"])

    with tab1:
        n = st.slider("每类题目数", 5, 30, 12, 5)
        if st.button("🚀 构建Benchmark", type="primary", disabled=vectordb is None):
            from evaluation.benchmark import BenchmarkBuilder

            bb = BenchmarkBuilder()
            progress = st.progress(0)
            bb.build(n_per_category=n)
            progress.progress(80)
            bb.export()
            progress.progress(100)
            stats = bb.stats()
            st.success(f"**{stats['total']}** 道评测题已保存")
            st.json(stats)

    with tab2:
        if os.path.exists(benchmark_path):
            benchmark = load_json(benchmark_path)
            st.info(f"共 {len(benchmark)} 道题")

            cat_filter = st.selectbox("类别", ["全部", "检索评测", "忠实度评测", "事实准确性评测"])
            filtered = benchmark if cat_filter == "全部" else [b for b in benchmark if b.get("category_name") == cat_filter]

            rows = []
            for b in filtered:
                if b["category"] == "retrieval":
                    rows.append({"ID": b["id"][:8], "类别": "检索评测", "查询": b.get("query", "")[:80], "目标机构": b.get("expected_org", "")})
                elif b["category"] == "faithfulness":
                    rows.append({"ID": b["id"][:8], "类别": "忠实度评测", "声明": b.get("claim", "")[:80], "标签": b.get("label", "")})
                else:
                    rows.append({"ID": b["id"][:8], "类别": "事实准确性", "问题": b.get("question", "")[:80], "答案": b.get("answer", "")[:60]})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=350)

            if filtered:
                idx = st.number_input("查看详情", 0, len(filtered) - 1, 0)
                b = filtered[idx]
                st.markdown("---")
                if b["category"] == "retrieval":
                    st.subheader(f"🔍 {b.get('query', '')}")
                    st.markdown(f"目标机构: **{b.get('expected_org', '')}**")
                    st.caption(f"关键词: {b.get('expected_keywords', [])}")
                elif b["category"] == "faithfulness":
                    st.subheader(f"📝 声明: {b.get('claim', '')}")
                    st.markdown(f"标签: `{b.get('label', '')}`")
                    with st.expander("原文"):
                        st.text(b.get("source_content", "")[:1500])
                else:
                    st.subheader(f"❓ {b.get('question', '')}")
                    st.markdown(f"标准答案: **{b.get('answer', '')}**")
                    with st.expander("原文依据"):
                        st.text(b.get("source_content", "")[:1500])
        else:
            st.info("暂无Benchmark，请先构建")


# ═══════════════════════════════════════
# Page 3: 模型评测
# ═══════════════════════════════════════
elif page == "🔬 模型评测":
    st.title("🔬 模型评测 — 来源锚定评测")

    st.markdown("""
    > **评测指标全部可追溯到真实研报**，不依赖LLM主观打分。
    > - 检索命中率 / 关键词覆盖率
    > - 忠实度判断准确率
    > - 事实准确性 / 幻觉检测
    """)

    tab1, tab2 = st.tabs(["🔬 批量评测", "📝 单题评测"])

    with tab1:
        if not os.path.exists(benchmark_path):
            st.warning("请先在「Benchmark管理」中构建测试集")
        else:
            benchmark = load_json(benchmark_path)
            st.info(f"当前Benchmark: {len(benchmark)} 道题")

            col1, col2 = st.columns(2)
            with col1:
                sample_size = st.slider("评测样本数", 5, min(40, len(benchmark)), 15, 5)
            with col2:
                model_label = st.text_input("模型标签", "RAG-Baseline")

            if st.button("🚀 开始评测", type="primary"):
                from evaluation.judge import LLMJudge
                from evaluation.reporter import EvalReporter
                import random

                random.seed(42)
                sampled = random.sample(benchmark, min(sample_size, len(benchmark)))

                # Step 1: 生成RAG回答
                st.info("Step 1/2: RAG生成回答 + 检索结果...")
                progress = st.progress(0)
                retrieved_docs_map = {}
                model_answers = {}

                for i, case in enumerate(sampled):
                    cat = case.get("category", "")
                    cid = case.get("id", str(i))

                    if cat == "retrieval":
                        docs = search_reports(case.get("query", ""), k=10)
                        retrieved_docs_map[case.get("query", "")] = [
                            {"content": d["content"][:500], "org": d["org"], "title": d["title"]}
                            for d in docs
                        ]
                    elif cat == "faithfulness":
                        prompt = f"""判断声明是否被原文支持。仅输出 supported 或 not_supported。

原文：{case.get('source_content', '')[:1000]}

声明：{case.get('claim', '')}

判断："""
                        resp = client.chat.completions.create(
                            model=model_name, messages=[{"role": "user", "content": prompt}], temperature=0.0
                        )
                        model_answers[cid] = resp.choices[0].message.content.strip()

                    elif cat == "factual":
                        docs = search_reports(case.get("question", ""), k=5)
                        ctx = "\n\n".join(f"[{d['org']}] {d['content'][:400]}" for d in docs)
                        prompt = f"""基于资料回答问题，资料不足请说明。

资料：{ctx[:2500]}

问题：{case.get('question', '')}

回答："""
                        resp = client.chat.completions.create(
                            model=model_name, messages=[{"role": "user", "content": prompt}], temperature=0.2
                        )
                        model_answers[cid] = resp.choices[0].message.content

                    progress.progress((i + 1) / len(sampled))
                    time.sleep(0.2)

                # Step 2: 评测
                st.info("Step 2/2: 来源锚定评测中...")
                judge = LLMJudge()
                eval_results = judge.batch_evaluate(sampled, retrieved_docs_map, model_answers)
                progress.progress(100)

                reporter = EvalReporter(eval_results, model_label=model_label)
                reporter.save_results_json()
                reporter.save_report()

                st.success(f"✓ 评测完成: {len(eval_results)} 条")
                scores = reporter.scores
                by_cat = scores.get("by_category", {})

                # 摘要
                st.markdown("### 评测摘要")
                cols = st.columns(4)
                cols[0].metric("综合评分", f"{scores.get('overall_score', 'N/A')}")
                if "retrieval" in by_cat:
                    cols[1].metric("检索命中率", f"{by_cat['retrieval']['org_hit_rate']:.0%}")
                if "faithfulness" in by_cat:
                    cols[2].metric("忠实度准确率", f"{by_cat['faithfulness']['accuracy']:.0%}")
                if "factual" in by_cat:
                    cols[3].metric("无幻觉率", f"{by_cat['factual']['no_hallucination_rate']:.0%}")

                st.info("详细报告 → 「📈 评测报告」页面")

    with tab2:
        st.subheader("单题评测")
        test_question = st.text_area("题目", "分析宁德时代的核心竞争力与主要风险")
        test_response = st.text_area("模型回答", "")

        if st.button("🔍 评测", disabled=not (test_question and test_response)):
            # 检索原文作为锚定
            docs = search_reports(test_question, k=5)
            source_text = "\n\n".join(f"[{d['org']}] {d['content'][:500]}" for d in docs)

            from evaluation.judge import LLMJudge
            judge = LLMJudge()
            result = judge.evaluate_factual(
                question=test_question,
                expected_answer="",  # 无预设标准答案，纯来源锚定
                source_text=source_text,
                model_answer=test_response,
            )

            scores = result.get("scores", {})
            c1, c2, c3 = st.columns(3)
            c1.metric("事实准确性", f"{scores.get('factual_accuracy', 0):.0%}")
            c2.metric("是否准确", "✅" if scores.get("is_correct") else "❌")
            c3.metric("幻觉检测", "⚠️ 发现" if scores.get("hallucination", 1.0) == 0 else "✅ 未发现")

            with st.expander("详情"):
                st.json(result.get("details", {}))
            with st.expander("来源文档"):
                st.text(source_text[:3000])


# ═══════════════════════════════════════
# Page 4: 评测报告
# ═══════════════════════════════════════
elif page == "📈 评测报告":
    st.title("📈 评测报告 — 来源锚定分析")

    if not os.path.exists(eval_results_path):
        st.info("暂无评测数据，请先在「🔬 模型评测」中运行评测")
    else:
        eval_results = load_json(eval_results_path)
        from evaluation.reporter import EvalReporter
        from evaluation.metrics import compute_scores, dimension_breakdown, bad_case_analysis

        model_label = eval_results[0].get("test_case_id", "Unknown")[:10] if eval_results else ""
        reporter = EvalReporter(eval_results, model_label=model_label)

        scores = reporter.scores
        by_cat = scores.get("by_category", {})

        # 综合
        st.markdown("## 📊 综合评分")
        col1, col2, col3 = st.columns(3)
        col1.metric("综合评分", f"{scores.get('overall_score', 'N/A')}", help="检索30% + 忠实度30% + 事实40%")
        col2.metric("评测样本", scores.get("total_cases", 0))
        col3.metric("评测类别", len(by_cat))

        # 检索
        if "retrieval" in by_cat:
            st.markdown("## 🔍 检索评测")
            r = by_cat["retrieval"]
            cols = st.columns(3)
            cols[0].metric("机构命中率", f"{r['org_hit_rate']:.0%}", help="检索结果中包含目标机构的比例")
            cols[1].metric("关键词覆盖", f"{r['avg_keyword_coverage']:.0%}", help="检索结果覆盖预期关键词的比例")
            cols[2].metric("综合检索分", f"{r['avg_overall']:.2f}")
            if r["org_hit_rate"] < 0.5:
                st.warning("⚠️ 机构命中率偏低 → 建议优化Embedding或增加元数据索引")

        # 忠实度
        if "faithfulness" in by_cat:
            st.markdown("## 📝 忠实度评测")
            f = by_cat["faithfulness"]
            st.metric("判断准确率", f"{f['accuracy']:.0%}", help="模型判断'声明是否被原文支持'的准确率")
            if f["accuracy"] < 0.8:
                st.warning("⚠️ 忠实度偏低 → 建议增加声明验证类训练数据")

        # 事实准确性
        if "factual" in by_cat:
            st.markdown("## ✅ 事实准确性评测")
            fa = by_cat["factual"]
            cols = st.columns(3)
            cols[0].metric("事实准确率", f"{fa['avg_accuracy']:.0%}")
            cols[1].metric("完全正确率", f"{fa['correct_rate']:.0%}")
            cols[2].metric("无幻觉率", f"{fa['no_hallucination_rate']:.0%}")
            if fa["no_hallucination_rate"] < 0.9:
                st.warning("⚠️ 存在幻觉 → 建议加强检索约束和数据验证")

        # Bad cases
        bad = bad_case_analysis(eval_results, threshold=0.5)
        if bad:
            st.markdown("## 🔍 低分样本")
            for i, case in enumerate(bad[:5]):
                with st.expander(f"Case {i+1}: 得分 {case.get('_overall', 'N/A')}"):
                    st.json({k: v for k, v in case.items() if k != "_overall"})

        # 优化建议
        st.markdown("## 💡 优化建议")
        for i, sug in enumerate(reporter._generate_suggestions()):
            st.markdown(f"**{i+1}. {sug['title']}**")
            st.info(sug["detail"])

        # 下载
        if st.button("📥 生成完整报告"):
            report_path = reporter.save_report()
            with open(report_path, "r", encoding="utf-8") as f:
                report_md = f.read()
            st.download_button("下载报告", report_md, "eval_report.md", "text/markdown")


# ═══════════════════════════════════════
# Page 5: 研报检索
# ═══════════════════════════════════════
elif page == "🔍 研报检索":
    st.title("🔍 研报检索 — RAG问答")

    if vectordb is None:
        st.warning("向量库未初始化。运行 `python build_knowledge_base.py`")
    else:
        query = st.text_input("检索关键词或问题", placeholder="固态电池产业化进展、比亚迪2024业绩分析...")
        k = st.slider("返回数量", 5, 30, 12)

        if query:
            results = search_reports(query, k=k)
            st.info(f"检索到 {len(results)} 条")

            orgs = Counter(r["org"] for r in results if r["org"])
            if orgs:
                st.bar_chart(pd.DataFrame({"机构": list(orgs.keys()), "数量": list(orgs.values())}).set_index("机构"))

            rows = [{"机构": r["org"], "股票": r["stock"], "标题": r["title"][:50], "摘要": r["content"][:150]} for r in results]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            if st.button("🤖 RAG分析"):
                if not api_key:
                    st.error("请配置 API Key")
                else:
                    ctx = "\n\n".join(f"[{r['org']} - {r['title']}]\n{r['content'][:400]}" for r in results[:8])
                    prompt = f"""你是资深行业研究员。基于以下资料分析问题。分点阐述，引用来源，资料不足明示。

资料：{ctx[:4000]}

问题：{query}

分析："""
                    with st.spinner("分析中..."):
                        resp = client.chat.completions.create(
                            model=model_name, messages=[{"role": "user", "content": prompt}]
                        )
                    st.markdown("---")
                    st.markdown(resp.choices[0].message.content)
