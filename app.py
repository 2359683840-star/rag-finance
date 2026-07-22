"""
金融LLM数据建设与评测平台
— 对标 月之暗面 数据专家实习生-金融
"""
import os, json, time, hashlib
from datetime import datetime
from collections import Counter

import streamlit as st
import pandas as pd
from openai import OpenAI

st.set_page_config(page_title="金融LLM数据与评测平台", page_icon="📊", layout="wide")

from config import FAISS_DIR, CORPUS_DIR, EVAL_DIR
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

client = OpenAI(api_key=api_key, base_url=base_url) if api_key else None

# ─── Vector DB (lazy load, graceful failure) ───
@st.cache_resource
def load_db():
    idx_path = os.path.join(FAISS_DIR, "index.faiss")
    if not os.path.exists(idx_path):
        return None
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        embedding = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
        return FAISS.load_local(FAISS_DIR, embedding, allow_dangerous_deserialization=True)
    except Exception:
        return None

# 延迟加载，不在启动时阻塞
if "db_loaded" not in st.session_state:
    st.session_state.db = load_db()
    st.session_state.db_loaded = True

vectordb = st.session_state.db

def search_reports(query: str, k: int = 10) -> list[dict]:
    if vectordb is not None:
        results = vectordb.similarity_search(query, k=k)
        return [{"content": r.page_content.strip(), "org": r.metadata.get("org",""),
                 "title": r.metadata.get("title", r.metadata.get("report_title","")),
                 "stock": r.metadata.get("stock","")} for r in results]
    # Fallback: keyword search using faiss_docs.json
    return _keyword_search(query, k)

@st.cache_data
def _load_fallback_docs():
    path = os.path.join(os.path.dirname(__file__), "faiss_docs.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _keyword_search(query: str, k: int = 10) -> list[dict]:
    docs = _load_fallback_docs()
    if not docs:
        return []
    keywords = query.lower().split()
    scored = []
    for d in docs:
        text = (d.get("content","") + d.get("org","") + d.get("title","") + d.get("stock","")).lower()
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in scored[:k]]

# ─── Data helpers ───
def load_jsonl(path: str) -> list:
    if not os.path.exists(path): return []
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try: data.append(json.loads(line.strip()))
            except: pass
    return data

def load_json(path: str):
    if not os.path.exists(path): return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ─── Sidebar ───
st.sidebar.title("📊 金融LLM数据与评测平台")
st.sidebar.markdown("*金融LLM数据建设 · Benchmark评测 · 报告分析*")

page = st.sidebar.radio("导航", ["🏗️ 语料工坊", "📋 Benchmark管理", "🔬 模型评测", "📈 评测报告", "🔍 研报检索"])

st.sidebar.markdown("---")
if vectordb is not None:
    st.sidebar.success("✅ 向量语义检索已就绪")
elif os.path.exists(os.path.join(os.path.dirname(__file__), "faiss_docs.json")):
    st.sidebar.info("🔍 关键词检索模式（语义模型云端暂不可用）")
else:
    st.sidebar.warning("⚠️ 检索不可用\n\n语料/评测/报告页面可正常使用")

corpus_files = [f for f in os.listdir(CORPUS_DIR) if f.endswith(".jsonl")] if os.path.exists(CORPUS_DIR) else []
benchmark_path = os.path.join(EVAL_DIR, "benchmark.json")
eval_results_path = os.path.join(EVAL_DIR, "eval_results.json")

if corpus_files: st.sidebar.info(f"📝 语料: {len(corpus_files)} 个文件")
if os.path.exists(benchmark_path): st.sidebar.info("📋 Benchmark已缓存")
if os.path.exists(eval_results_path): st.sidebar.info("📊 评测结果已缓存")
st.sidebar.caption("Moonshot.AI · 数据专家实习生")

# ═══════════════════════════════════════
# 语料工坊
# ═══════════════════════════════════════
if page == "🏗️ 语料工坊":
    st.title("🏗️ 语料工坊 — 真实研报训练语料")
    st.markdown("> 所有语料来自真实研报原文，经过 抽样→清洗→场景分类→实体提取→去重→质量过滤。不做LLM合成。")

    corpus_path = os.path.join(CORPUS_DIR, corpus_files[0]) if corpus_files else None
    if not corpus_path:
        st.warning("暂无缓存语料。本地运行 `python scripts/build_corpus.py --max 200` 生成。")
    else:
        data = load_jsonl(corpus_path)
        scenario_filter = st.selectbox("场景筛选", ["全部"] + [s["name"] for s in SCENARIOS.values()])
        filtered = data if scenario_filter == "全部" else [d for d in data if d.get("scenario_name") == scenario_filter]
        st.info(f"{len(filtered)} 条语料")

        # 统计
        c1, c2, c3 = st.columns(3)
        c1.metric("总条数", len(data))
        c2.metric("总字符", f"{sum(d.get('char_count',0) for d in data):,}")
        c3.metric("覆盖机构", len(set(d.get("source_org","") for d in data)))

        sc = Counter(d.get("scenario_name","") for d in data)
        st.bar_chart(pd.DataFrame({"数量": sc}))

        # 列表
        rows = [{"场景": d.get("scenario_name",""), "来源": d.get("source_org",""),
                 "股票": d.get("source_stock",""), "长度": d.get("char_count",0),
                 "文本": d.get("text","")[:100]} for d in filtered[:100]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if filtered:
            idx = st.number_input("查看详情", 0, len(filtered)-1, 0)
            d = filtered[idx]
            st.markdown(f"**场景**: {d.get('scenario_name','')} | **来源**: {d.get('source_org','')} | **股票**: {d.get('source_stock','')}")
            st.text(d.get("text", ""))

# ═══════════════════════════════════════
# Benchmark管理
# ═══════════════════════════════════════
elif page == "📋 Benchmark管理":
    st.title("📋 Benchmark管理 — 来源锚定评测集")
    st.markdown("> 三类评测基准：检索评测·忠实度评测·事实准确性评测。全部基于真实研报内容。")

    if not os.path.exists(benchmark_path):
        st.warning("暂无Benchmark。本地运行 `python scripts/run_eval.py --build-benchmark` 生成。")
    else:
        benchmark = load_json(benchmark_path)
        st.info(f"共 {len(benchmark)} 道评测题")

        cat_filter = st.selectbox("类别", ["全部", "检索评测", "忠实度评测", "事实准确性评测"])
        filtered = benchmark if cat_filter == "全部" else [b for b in benchmark if b.get("category_name") == cat_filter]

        rows = []
        for b in filtered:
            if b["category"] == "retrieval":
                rows.append({"ID": b["id"][:8], "类别": "检索评测", "内容": b.get("query","")[:80], "目标": b.get("expected_org","")})
            elif b["category"] == "faithfulness":
                rows.append({"ID": b["id"][:8], "类别": "忠实度评测", "内容": b.get("claim","")[:80], "标签": b.get("label","")})
            else:
                rows.append({"ID": b["id"][:8], "类别": "事实准确性", "内容": b.get("question","")[:80]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=400)

        if filtered:
            idx = st.number_input("查看详情", 0, len(filtered)-1, 0)
            b = filtered[idx]
            st.json(b)

# ═══════════════════════════════════════
# 模型评测
# ═══════════════════════════════════════
elif page == "🔬 模型评测":
    st.title("🔬 模型评测 — 来源锚定评测")
    st.markdown("> 评测指标全部可追溯到真实研报，不依赖LLM主观打分。")

    tab1, tab2 = st.tabs(["📝 单题评测", "🔬 批量评测（本地）"])

    with tab1:
        test_question = st.text_area("题目", "分析宁德时代在动力电池领域的核心竞争力与主要风险因素")
        test_response = st.text_area("模型回答", "")
        if st.button("🔍 评测", disabled=not (test_question and test_response and client)):
            if not client: st.error("API Key 未配置")
            else:
                docs = search_reports(test_question, k=5)
                source_text = "\n\n".join(f"[{d['org']}] {d['content'][:400]}" for d in docs) if docs else "（向量库未加载，无法检索原文）"
                prompt = f"""对比模型回答与原文，评估事实准确性。

原文：{source_text[:2000]}

问题：{test_question}

模型回答：{test_response}

评估维度：1)事实是否与原文一致 2)有无编造 3)关键数字是否正确
输出JSON：{{"factual_score": 1-5, "is_correct": true/false, "hallucination": true/false, "summary": "一句话总结"}}"""
                with st.spinner("评测中..."):
                    resp = client.chat.completions.create(model=model_name, messages=[{"role":"user","content":prompt}], temperature=0.0)
                try:
                    result = json.loads(resp.choices[0].message.content.split("```json")[-1].split("```")[0] if "```" in resp.choices[0].message.content else resp.choices[0].message.content)
                    c1,c2,c3 = st.columns(3)
                    c1.metric("事实准确性", f"{result.get('factual_score',0)}/5")
                    c2.metric("是否正确", "✅" if result.get("is_correct") else "❌")
                    c3.metric("幻觉", "⚠️有" if result.get("hallucination") else "✅无")
                    st.info(result.get("summary",""))
                except:
                    st.markdown(resp.choices[0].message.content)

    with tab2:
        st.info("批量评测请本地运行: `python scripts/run_eval.py --full`")

# ═══════════════════════════════════════
# 评测报告
# ═══════════════════════════════════════
elif page == "📈 评测报告":
    st.title("📈 评测报告")

    if not os.path.exists(eval_results_path):
        st.warning("暂无评测结果。本地运行 `python scripts/run_eval.py --full` 生成。")
    else:
        results = load_json(eval_results_path)
        st.info(f"共 {len(results)} 条评测记录")

        # 按类别统计
        by_cat = Counter(r.get("category","") for r in results)
        st.markdown("### 评测概览")
        for cat, count in by_cat.items():
            cat_results = [r for r in results if r.get("category") == cat]
            if cat == "retrieval":
                scores = [r["scores"].get("overall",0) for r in cat_results]
                org_hits = sum(1 for r in cat_results if r["scores"].get("org_hit",0) > 0.5)
                st.metric(f"检索评测 ({count}题)", f"机构命中率 {org_hits/count:.0%}" if count else "N/A")
            elif cat == "faithfulness":
                correct = sum(1 for r in cat_results if r["scores"].get("faithfulness_correct",0) > 0.5)
                st.metric(f"忠实度评测 ({count}题)", f"准确率 {correct/count:.0%}" if count else "N/A")

        # Bad cases
        bad = [r for r in results if r.get("scores",{}).get("overall", r.get("scores",{}).get("faithfulness_correct",1)) < 0.5]
        if bad:
            st.markdown(f"### ⚠️ Bad Cases ({len(bad)}个)")
            for case in bad[:5]:
                with st.expander(f"得分 {case.get('scores',{})}"):
                    st.json(case)

        # 优化建议
        st.markdown("### 💡 模型优化建议")
        suggestions = [
            "提升检索精度: 优化Embedding模型，混合元数据过滤+语义检索",
            "增强忠实度: 增加'声明验证'类训练数据，引入多证据交叉验证",
            "减少幻觉: 强化检索约束，对数字类回答增加后验证",
            "扩展评测场景: 增加更多金融子领域，定期更新Benchmark",
        ]
        for i, s in enumerate(suggestions):
            st.markdown(f"{i+1}. {s}")

# ═══════════════════════════════════════
# 研报检索
# ═══════════════════════════════════════
elif page == "🔍 研报检索":
    st.title("🔍 研报检索 — RAG问答")

    has_search = vectordb is not None or os.path.exists(os.path.join(os.path.dirname(__file__), "faiss_docs.json"))
    if not has_search:
        st.warning("""
        **检索功能不可用**

        原因：向量库与文档索引均未加载。这不影响语料工坊、Benchmark、评测报告等页面。

        如需完整RAG功能，请本地运行：
        ```
        streamlit run app.py
        ```
        """)
    else:
        if vectordb is None:
            st.info("🔍 当前使用关键词检索模式（云端无法下载语义模型），检索精度略低于本地语义检索，LLM 分析与来源溯源功能正常")
        query = st.text_input("检索关键词", placeholder="固态电池产业化进展...")
        k = st.slider("返回数量", 5, 30, 12)
        if query:
            results = search_reports(query, k=k)
            st.info(f"检索到 {len(results)} 条")
            orgs = Counter(r["org"] for r in results if r["org"])
            if orgs:
                st.bar_chart(pd.DataFrame({"机构": list(orgs.keys()), "数量": list(orgs.values())}).set_index("机构"))
            rows = [{"机构": r["org"], "股票": r["stock"], "摘要": r["content"][:150]} for r in results]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            if st.button("🤖 RAG分析") and client:
                ctx = "\n\n".join(f"[{r['org']}] {r['content'][:400]}" for r in results[:8])
                prompt = f"""你是资深行业研究员。基于以下资料分析问题。分点阐述，引用来源，资料不足明示。
资料：{ctx[:4000]}
问题：{query}
分析："""
                with st.spinner("分析中..."):
                    resp = client.chat.completions.create(model=model_name, messages=[{"role":"user","content":prompt}])
                st.markdown(resp.choices[0].message.content)
