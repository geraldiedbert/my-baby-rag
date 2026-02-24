import chromadb

client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection = client.get_collection("document_qa_collection")

print(f"Total stored chunks: {collection.count()}")
print(collection.get(limit=5))  # peek at first 5