# import chromadb

# client = chromadb.PersistentClient(path="chroma_persistent_storage")
# collection = client.get_collection("document_qa_collection")

# # print(f"Total stored chunks: {collection.count()}")
# # print(collection.get(limit=5))  # peek at first 5

# result = collection.get(limit=10)
# # embs = result["embeddings"][0]
# # print(result.keys())
# print(result["embeddings"])
# # print(result["ids"])
# print(result["documents"])
# print(result["uris"])
# print(result["included"])
# print(result["data"])
# print(result["metadatas"])
# print("Returned IDs:", result["ids"])
# print("First text snippet:", result["documents"][0][:200])
# print("embedding length:", len(embs))
# print("first 5 values:", embs[:5])
# print("metadata:", result["metadatas"][0])


import chromadb
import pandas as pd

client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection = client.get_collection("document_qa_collection")

# Get some rows from Chroma
result = collection.get(
    include=["documents", "embeddings", "metadatas"],
    limit=100,     # or however many you want to inspect
)

rows = []
for doc, emb, meta in zip(
    result["documents"],
    result["embeddings"],
    result["metadatas"],
):
    rows.append({
        "text": doc,
        "embedding_len": len(emb),
        "embedding_first5": emb[:5],
        "metadata": meta,
    })

df = pd.DataFrame(rows)