"""
自动更新：从 akshare 获取研报元数据并入库
"""
import os, json, time
from datetime import datetime

os.environ["TRANSFORMERS_NO_TF"] = "1"

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

FAISS_DIR = "./faiss_db"
RECORD_FILE = "./metadata_record.json"
META_FILE = "./reports_meta.json"

STOCKS = [
    {"code": "300750", "name": "宁德时代"}, {"code": "002594", "name": "比亚迪"},
    {"code": "300014", "name": "亿纬锂能"}, {"code": "002074", "name": "国轩高科"},
    {"code": "002709", "name": "天赐材料"}, {"code": "300568", "name": "星源材质"},
    {"code": "002460", "name": "赣锋锂业"}, {"code": "002466", "name": "天齐锂业"},
    {"code": "300073", "name": "当升科技"}, {"code": "603659", "name": "璞泰来"},
    {"code": "601012", "name": "隆基绿能"}, {"code": "688599", "name": "天合光能"},
    {"code": "600438", "name": "通威股份"}, {"code": "300274", "name": "阳光电源"},
    {"code": "600406", "name": "国电南瑞"}, {"code": "601877", "name": "正泰电器"},
]

def load_record():
    if os.path.exists(RECORD_FILE):
        with open(RECORD_FILE, "r") as f:
            return json.load(f)
    return {"seen": []}

def save_record(r):
    with open(RECORD_FILE, "w") as f:
        json.dump(r, f)

def fetch_metadata():
    import akshare as ak
    record = load_record()
    seen = set(record["seen"])
    new_docs = []

    for stock in STOCKS:
        print(f"  {stock['name']}({stock['code']})...", end=" ")
        try:
            df = ak.stock_research_report_em(symbol=stock["code"])
            count = 0
            for _, row in df.iterrows():
                url = str(row.iloc[-1])
                if url in seen:
                    continue
                title = str(row.iloc[2])
                org = str(row.iloc[4])
                rating = str(row.iloc[3])
                industry = str(row.iloc[13])
                date = str(row.iloc[-3])
                text = f"机构：{org} | 股票：{stock['name']} | 评级：{rating}\n"
                text += f"标题：{title}\n行业：{industry} | 日期：{date}\n"
                # 盈利预测（第7-12列）
                for i in range(7, 13):
                    val = row.iloc[i]
                    if val and str(val) != "nan":
                        text += f"{df.columns[i]}：{val} "
                doc = Document(
                    page_content=text,
                    metadata={"org": org, "stock": stock["name"], "title": title[:50], "date": date}
                )
                new_docs.append(doc)
                seen.add(url)
                count += 1
            print(f"{count} 条新")
            time.sleep(1)
        except Exception as e:
            print(f"失败: {e}")

    record["seen"] = list(seen)
    save_record(record)
    return new_docs

def add_to_vector_db(docs):
    if not docs:
        print("没有新数据")
        return
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"切分成 {len(chunks)} 个片段")
    embedding = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
    if os.path.exists(os.path.join(FAISS_DIR, "index.faiss")):
        db = FAISS.load_local(FAISS_DIR, embedding, allow_dangerous_deserialization=True)
        db.add_documents(chunks)
    else:
        db = FAISS.from_documents(documents=chunks, embedding=embedding)
    db.save_local(FAISS_DIR)
    print(f"入库完成，新增 {len(docs)} 条")

def run():
    print(f"研报元数据自动更新 ｜ {datetime.now()}")
    docs = fetch_metadata()
    add_to_vector_db(docs)

if __name__ == "__main__":
    run()
