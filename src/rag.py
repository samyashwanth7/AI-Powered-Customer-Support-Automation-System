import os
import sys

# Compatibility shim for langchain_core 0.3+/1.0+ pydantic_v1 deprecation
try:
    import pydantic.v1 as pydantic_v1
    sys.modules['langchain_core.pydantic_v1'] = pydantic_v1
except ImportError:
    pass

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def setup_rag():
    """Initializes the RAG retriever with in-memory Chroma vectorstore for maximum speed and reliability."""
    docs_dir = "docs"
    documents = []
    
    if os.path.exists(docs_dir):
        for file in os.listdir(docs_dir):
            if file.endswith(".txt"):
                loader = TextLoader(os.path.join(docs_dir, file), encoding='utf-8')
                documents.extend(loader.load())
            
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(documents) if documents else []
    
    # In-memory vectorstore eliminates SQLite schema migration conflicts and disk lock errors
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

retriever = setup_rag()
