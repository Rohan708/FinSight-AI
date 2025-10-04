# src/document_loader.py
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdf(file) -> str:
    """
    Extract raw text from uploaded PDF file.
    """
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""  # handle pages with no text
    return text

def split_text(text: str, chunk_size=1000, overlap=200):
    """
    Split text into chunks for embeddings.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    return chunks
