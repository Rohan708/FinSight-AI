# src/embeddings.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize embedding model
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Create ChromaDB collection from text chunks
def create_vectorstore(text_chunks, persist_directory="chroma_store"):
    embeddings = get_embedding_model()
    vectorstore = Chroma.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectorstore.persist()
    return vectorstore

# Query ChromaDB for similar chunks
def query_vectorstore(vectorstore, query, k=3):
    results = vectorstore.similarity_search(query, k=k)
    return results
