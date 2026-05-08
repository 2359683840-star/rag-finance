"""
Streamlit 可视化界面（兼容 Streamlit Cloud 部署）
"""
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

FAISS_DIR = "./faiss_db"

# API 配置（Streamlit Cloud Secrets 优先）
api_key = st.secrets.get("DASHSCOPE_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
base_url = st.secrets.get("API_BASE_URL") or os.getenv("API_BASE_URL", "https://api.deepseek.com")
model_name = st.secrets.get("LLM_MODEL") or os.getenv("LLM_MODEL", "deepseek-chat")


@st.cache_resource
def load_db():
    embedding = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
    return FAISS.load_local(FAISS_DIR, embedding, allow_dangerous_deserialization=True)


vectordb = load_db()
client = OpenAI(api_key=api_key, base_url=base_url)


def ask(question):
    results = vectordb.similarity_search(question, k=5)
    context_parts = []
    for i, r in enumerate(results):
        source = r.metadata.get("source", "未知来源")
        context_parts.append(f"[来源: {os.path.basename(source)}]\n{r.page_content}")
    context = "\n\n".join(context_parts)

    prompt = f"""你是一位资深行业研究员。请基于以下资料，对问题进行全面、深入的分析。

要求：
- 分点阐述，每个观点附上数据支撑
- 引用来源时用 [来源 X] 标注（资料开头已经标注了来源编号）
- 如果资料中有多个机构的观点，要综合对比
- 指出资料中提到的具体数据（增长率、市场份额等）
- 如果资料信息不足以完整回答，说明"从现有资料看，还缺少XX方面的信息"

资料：
{context}

问题：{question}

分析："""

    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content


st.set_page_config(page_title="金融研报问答助手", page_icon="📊")
st.title("📊 金融研报问答助手")
st.caption("基于RAG的金融知识库——上传研报后即可提问")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("输入你的问题"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("检索中..."):
            reply = ask(prompt)
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
