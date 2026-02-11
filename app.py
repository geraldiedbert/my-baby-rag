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

resp = gemini_client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        types.Content(
            role="user",
            parts=[types.Part(text="what is human life expectancy in Mars?")]
        )
    ],
    config=types.GenerateContentConfig(
              system_instruction=['You are a helpful assistant.']
    )
)


# def main():
    
#     # Prompts user to input API key if non existent inside .env file
#     if "GOOGLE_API_KEY" not in os.environ:
#         gapikey = input("Please enter your Google API Key: ")
#         genai.configure(api_key=gapikey)
#         google_api_key = gapikey
#     else:
#         google_api_key = os.environ["GOOGLE_API_KEY"]



    
    







