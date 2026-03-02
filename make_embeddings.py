import chromadb
import google.genai
import time

chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection = chroma_client.get_or_create_collection(
    name="document_qa_collection"
)


BATCH_SIZE = 40

all_texts = [c["text"] for c in chunked_documents]
all_ids   = [c["id"]   for c in chunked_documents]

for i in range(0, len(all_texts), BATCH_SIZE):
    batch_texts = all_texts[i : i + BATCH_SIZE]
    batch_ids   = all_ids[i : i + BATCH_SIZE]

    response = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=batch_texts,
    )

    batch_embeddings = [e.values for e in response.embeddings]

    collection.upsert(
        ids=[f"{bid}_chunk_{i + j}" for j, bid in enumerate(batch_ids)],
        documents=batch_texts,
        embeddings=batch_embeddings,
        metadatas=[{"source": bid} for bid in batch_ids],
    )

    print(f"Embedded and stored batch {i // BATCH_SIZE + 1} / {-(-len(all_texts) // BATCH_SIZE)}")
    time.sleep(20)