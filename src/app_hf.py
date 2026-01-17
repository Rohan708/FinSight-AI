import os
import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv
import google.generativeai as genai

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from streamlit_chat import message  # pip install streamlit-chat

# -----------------------------
# Setup
# -----------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

CHROMA_DIR = "chroma_store"

# -----------------------------
# Helpers
# -----------------------------
def extract_text_from_pdf(file):
    """Extract text from uploaded PDF using PyMuPDF."""
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text


def split_text(text, chunk_size=1000, overlap=100):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def create_vectorstore(chunks):
    """Create Chroma vectorstore with Hugging Face embeddings."""
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma.from_texts(
        texts=chunks,
        embedding=embedding,
        persist_directory=CHROMA_DIR
    )


def get_answer(vectorstore, query, model_name="models/gemini-2.5-flash"):
    """Retrieve context and query Gemini."""
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    model = genai.GenerativeModel(model_name)
    response = model.generate_content(
        f"""
        Answer the question strictly based on the context below.

        Context:
        {context}

        Question:
        {query}
        """
    )
    return response.text


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="📊 Financial Document AI Assistant (HF)",
    page_icon="💹",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #f9fafb;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image(
    "https://img.icons8.com/?size=100&id=38180&format=png",
    width=70
)
st.sidebar.title("💼 Financial AI Assistant (HF)")
st.sidebar.markdown(
    "Upload financial PDFs and ask questions interactively "
    "(Hugging Face embeddings + Gemini)."
)

uploaded_files = st.sidebar.file_uploader(
    "📂 Upload PDFs",
    accept_multiple_files=True,
    type=["pdf"]
)

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.session_state.vectorstore = None

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Process PDFs
if uploaded_files:
    with st.spinner("Processing documents..."):
        all_text = ""
        for f in uploaded_files:
            all_text += extract_text_from_pdf(f) + "\n"

        chunks = split_text(all_text)
        st.session_state.vectorstore = create_vectorstore(chunks)

    st.success("✅ Documents processed successfully!")

# Main UI
st.title("📊 Financial Document Q&A (Hugging Face)")
st.markdown("### Ask questions about your financial documents in real time.")

# Display chat
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        message(msg["content"], is_user=True, key=f"user_{i}")
    else:
        message(msg["content"], key=f"bot_{i}")

# Input box
if prompt := st.chat_input("Type your financial question here..."):
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    if st.session_state.vectorstore is None:
        response = "⚠️ Please upload a financial document first."
    else:
        response = get_answer(
            st.session_state.vectorstore,
            prompt
        )

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
    st.rerun()
