import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
# from qdrant_client import QdrantClient
import chromadb
from chromadb.utils import embedding_functions
import math
import time


# load env var from .env
load_dotenv()
# setting up Gemini client
google_api_key = os.getenv("GOOGLE_API_KEY")
# set up embedding function
gemini_ef = embedding_functions.GoogleGenaiEmbeddingFunction(
    model_name="gemini-embedding-001"
)
gemini_client = genai.Client(api_key=google_api_key)


# processing documents
def load_documents_from_directory(directory_path):
    documents = [] # going to be an array of arrays
    for filename in os.listdir(directory_path):
        if filename.endswith((".txt")):
            with open(os.path.join(directory_path, filename)
            ) as file:
                documents.append({"id": filename, "text": file.read()})
    return documents

# to split text
def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 20):
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

# do embeddings
def get_googlegenai_embeddings(text):
    response = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    print("generating embeddings ...")
    return response.embeddings[0].values





directory_path = "./news_articles"
documents = load_documents_from_directory(directory_path)
print(f"Loaded {len(documents)} documents from {directory_path}")

for doc in documents:
    print(f"id: {doc['id']} with text length {len(doc['text'])}")
    print(f"chunking will result in {1 + math.ceil((len(doc['text']) - 1000) / 980)} chunks\n")

# chunk documents
chunked_documents = []
for doc in documents:
    chunks = split_text(doc['text'])
    for i, chunk in enumerate(chunks):
        chunked_documents.append({'id': doc['id'], 'text': chunk})

# create chroma collection
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection = chroma_client.get_or_create_collection(
    name="news_articles_collection"
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



"""
LOGIC BEHIND EMBEDDINGS
every pass, we wait 20s
every pass, we embed the content, and then we upsert to the collection
every pass, we will do the process for a batch size of BATCH_SIZE
that's why we have range(0, len(all_texts), BATCH_SIZE)
for i in range(0, len(all_texts), BATCH_SIZE):
    response = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents = 
    )

    collection.upsert(
        ids = something[i]['id']
        documents =
        embeddings =
        metadatas = 

    )
    time.sleep(20)

"""

"""
DATA TYPES USED
documents = list of dict
total_chunks = list
chunked_documents = list of dict
incoming chroma storage collection = dict
"""
