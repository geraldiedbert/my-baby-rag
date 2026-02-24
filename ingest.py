import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
import chromadb
from chromadb.utils import embedding_functions
import math

# model = genai.GenerativeModel("gemini-2.5-flash")

# load env var from .env
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

# set up embedding function
gemini_ef = embedding_functions.GoogleGenaiEmbeddingFunction(
    model_name="gemini-embedding-001"
)

# initializing chroma client
chroma_Client = chromadb.PersistentClient(path="chroma_persistent_storage")

# create actual collection
collection_name = "document_qa_collection"
collection = chroma_Client.get_or_create_collection(
    name=collection_name, embedding_function=gemini_ef
)

gemini_client = genai.Client(api_key=google_api_key)

def load_documents_from_directory(directory_path):
    documents = [] # array or dict?
    for filename in os.listdir(directory_path):
        if filename.endswith((".txt", ".pdf")):
            with open(os.path.join(directory_path, filename)
            ) as file:
                documents.append({"id": filename, "text": file.read()})
    return documents



# load docs from directory
directory_path = "./news_articles2"
documents = load_documents_from_directory(directory_path)
print(f"Loaded {len(documents)} documents from {directory_path}")

for doc in documents:
    print(f"id: {doc['id']} with text length {len(doc['text'])}")
    print(f"chunking results in {1 + math.ceil((len(doc['text']) - 1000) / 980)} chunks\n")

# so now we have an understanding of how the document is ingested and separated into chunks using fixed chunking

def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 20):
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks