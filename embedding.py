from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os

def create_vector_store(documents, index_path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(index_path)

def load_vector_store(index_path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(index_path, embeddings)
