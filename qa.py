"""
检索 + 大模型回答（命令行交互）
"""
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

FAISS_DIR = "./faiss_db"

# 1. 加载向量库
embedding = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
vectordb = FAISS.load_local(FAISS_DIR, embedding, allow_dangerous_deserialization=True)

# 2. 配置大模型 API
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    api_key = input("请输入你的 API Key（DeepSeek/通义千问 等）: ").strip()

base_url = os.getenv("API_BASE_URL", "https://api.deepseek.com")
model_name = os.getenv("LLM_MODEL", "deepseek-chat")

client = OpenAI(api_key=api_key, base_url=base_url)


def ask(question):
    results = vectordb.similarity_search(question, k=5)
    context = "\n\n".join([r.page_content for r in results])

    prompt = f"""你是一位资深行业研究员。请基于以下资料，对问题进行全面、深入的分析。

要求：
- 分点阐述，每个观点附上数据支撑
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


if __name__ == "__main__":
    print("金融研报问答助手（输入 exit 退出）")
    while True:
        q = input("\n问题：")
        if q.lower() in ["exit", "quit"]:
            break
        try:
            answer = ask(q)
            print(f"\n回答：{answer}")
        except Exception as e:
            print(f"\n错误：{e}")
