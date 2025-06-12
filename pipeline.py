from app.document_loader import load_and_split_documents
from app.embedding_store import create_vector_store, load_vector_store
from app.llm_interface import create_qa_chain
import os

def initialize_rag_pipeline(doc_dir="data/sample_docs", rebuild=False):
    if rebuild or not os.path.exists("faiss_index"):
        print("Loading and processing local ")
        documents = load_and_split_documents(doc_dir)
        print("Creating vector store document 1")
        create_vector_store(documents)
    print("Loading vector store")
    vectorstore = load_vector_store()
    print("Creating QA chain")
    return create_qa_chain(vectorstore)
