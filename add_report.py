"""
手动添加研报：把一份或多份PDF添加到已有向量库
用法：
  python add_report.py 报告1.pdf 报告2.pdf
  python add_report.py reports\新报告.pdf
"""
import os
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_NO_TF"] = "1"

from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_DIR = "./chroma_db"


def add_pdfs(file_paths):
    if not file_paths:
        print("用法: python add_report.py <PDF文件路径> [多个文件]")
        return

    docs = []
    for fpath in file_paths:
        if not os.path.exists(fpath):
            print(f"  ✗ 文件不存在: {fpath}")
            continue
        try:
            loader = PDFPlumberLoader(fpath)
            pages = loader.load()
            docs.extend(pages)
            print(f"  ✓ {os.path.basename(fpath)} ({len(pages)}页)")
        except Exception as e:
            print(f"  ✗ 解析失败: {e}")

    if not docs:
        print("没有成功解析的文件")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\n\n", "\n", "。", "，", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"切分成 {len(chunks)} 个片段")

    print("追加到向量库...")
    embedding = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
    vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
    vectordb.add_documents(chunks)
    vectordb.persist()
    print("✓ 完成")


if __name__ == "__main__":
    add_pdfs(sys.argv[1:])
