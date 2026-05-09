import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def embed_jd(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    db = Chroma.from_texts(chunks, embeddings, persist_directory="data/chroma")
    return db

def retrieve_context(query: str, db) -> str:
    results = db.similarity_search(query, k=3)
    return "\n".join([r.page_content for r in results])