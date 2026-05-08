"""
Streamlit 可视化界面
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_NO_TF"] = "1"

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI
import streamlit as st

CHROMA_DIR = "./chroma_db"


@st.cache_resource
def load_db():
    embedding = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)


vectordb = load_db()

api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = os.getenv("API_BASE_URL", "https://api.deepseek.com")
model_name = os.getenv("LLM_MODEL", "deepseek-chat")
client = OpenAI(api_key=api_key, base_url=base_url)


def ask(question):
    results = vectordb.similarity_search(question, k=3)
    context = "\n\n".join([r.page_content for r in results])
    prompt = f"""你是一个金融研究助手。请基于以下资料回答问题。
如果资料不足以回答，就说"资料中没有相关信息"。

资料：
{context}

问题：{question}

回答："""
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
