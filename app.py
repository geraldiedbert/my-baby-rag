import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
import chromadb
from chromadb.utils import embedding_functions

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


def split_text(text, chunk_size: int = 1000, chunk_overlap: int = 20):
    """
    Split a text into fixed-size character chunks with optional overlap.

    - chunk_size: maximum number of characters per chunk
    - chunk_overlap: number of characters each chunk shares with the previous one
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    text_length = len(text)
    start = 0

    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap

    return chunks

# load docs from directory
directory_path = "./news_articles2"
documents = load_documents_from_directory(directory_path)
print(f"Loaded {len(documents)} documents from {directory_path}")

# split document into chunks
chunked_documents = []
doc_length = 0
for doc in documents:
    chunks = split_text(doc['text'])
    doc_length += len(doc['text'])
    print(f"splitting chunks for {doc['id'][0:20]} ...")
    for i, chunk in enumerate(chunks): 
        chunked_documents.append({"id": f"{doc['id']}", "text": chunk})

print(f"document is split into {len(chunked_documents)} chunks")
print(f"total length of documents: {doc_length}")


# do embeddings
def get_googlegenai_embeddings(text):
    response = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    print("generating embeddings ...")
    return response.embeddings[0].values

def estimate_tokens(text: str) -> int:
    # Rough token count for Gemini (≈4 chars per token for English)
    return (len(text) + 3) // 4

def estimate_tokens_batch(texts: list[str]) -> int:
    return sum(estimate_tokens(t) for t in texts)

len_text = 0
len_id = 0
for chunk in chunked_documents:
    len_text += len(chunk)
    print(len(chunk))

print(len_text)
print(len_id)

# total_tokens = 0
# for chunk in chunked_documents:
#     print(f"estimated tokens for chunk {chunk['id'][0:10]} is {estimate_tokens(chunk)}")
#     total_tokens += estimate_tokens(chunk)

# print(f"total tokens: {total_tokens}")





# for doc in documents:
#     doc["embedding"] = get_googlegenai_embeddings(doc['text'])

# print(documents["embedding"])
# def main():
    
#     # Prompts user to input API key if non existent inside .env file
#     if "GOOGLE_API_KEY" not in os.environ:
#         gapikey = input("Please enter your Google API Key: ")
#         genai.configure(api_key=gapikey)
#         google_api_key = gapikey
#     else:
#         google_api_key = os.environ["GOOGLE_API_KEY"]



    
    







