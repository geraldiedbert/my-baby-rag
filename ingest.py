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
    documents = [] # array or dict?
    for filename in os.listdir(directory_path):
        if filename.endswith((".txt", ".pdf")):
            with open(os.path.join(directory_path, filename)
            ) as file:
                documents.append({"id": filename, "text": file.read()})
    return documents

# so now we have an understanding of how the document is ingested and separated into chunks using fixed chunking

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




# to run
directory_path = "./news_articles3"
documents = load_documents_from_directory(directory_path)
print(f"Loaded {len(documents)} documents from {directory_path}")

for doc in documents:
    print(f"id: {doc['id']} with text length {len(doc['text'])}")
    print(f"chunking will result in {1 + math.ceil((len(doc['text']) - 1000) / 980)} chunks\n")

# chunked_documents = []
# for doc in documents:
#     chunks = split_text(doc['text'])
#     for i, chunk in enumerate(chunks):
#         chunked_documents.append({'id': doc['id'], 'text': chunk})

