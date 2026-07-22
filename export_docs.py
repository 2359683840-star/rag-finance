import pickle, json

with open("./faiss_db/index.pkl", "rb") as f:
    a, b = pickle.load(f)

docstore = a if hasattr(a, '_dict') else b
items = docstore._dict

docs = []
for doc_id, doc in items.items():
    docs.append({
        "content": doc.page_content.strip(),
        "org": doc.metadata.get("org", ""),
        "title": doc.metadata.get("title", ""),
        "stock": doc.metadata.get("stock", ""),
        "date": doc.metadata.get("date", ""),
    })

with open("./faiss_docs.json", "w", encoding="utf-8") as f:
    json.dump(docs, f, ensure_ascii=False)
print(f"Exported {len(docs)} documents")
