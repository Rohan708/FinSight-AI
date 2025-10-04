import os
import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from streamlit_chat import message  # pip install streamlit-chat

# -----------------------------
# Setup
# -----------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# -----------------------------
# Gemini Embedding Wrapper
# -----------------------------
class GeminiEmbeddings:
    def __init__(self, model="models/embedding-001"):
        self.model = model

    def embed_documents(self, texts):
        embeddings = []
        for t in texts:
            emb = genai.embed_content(model=self.model, content=t)["embedding"]
            embeddings.append(emb)
        return embeddings

    def embed_query(self, text):
        return genai.embed_content(model=self.model, content=text)["embedding"]

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
    """Split text into chunks for embeddings."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def create_vectorstore(chunks, embedding_choice):
    """Create FAISS vectorstore with chosen embedding model."""
    if embedding_choice == "Gemini Embeddings":
        embedding = GeminiEmbeddings()
    else:
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    return FAISS.from_texts(texts=chunks, embedding=embedding)

def get_answer(vectorstore, query, mode="Normal", model_name="models/gemini-2.5-flash"):
    """Retrieve context and query Gemini for an answer."""
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    model = genai.GenerativeModel(model_name)

    if mode == "Tips Mode":
        prompt = f"""
        You are a financial analysis assistant.
        Answer the question based on the provided context from financial documents.

        Then, provide additional *general financial insights* related to the question.
        Always include the disclaimer at the start:

        ⚠️ This is not financial advice. For educational purposes only.

        Context:
        {context}

        Question: {query}
        """
    else:
        prompt = f"""
        Answer the question strictly based on the context below.
        Context:
        {context}

        Question: {query}
        """

    response = model.generate_content(prompt)
    return response.text

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="📊 Financial Document AI Assistant",
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
    .stChatMessage {
        border-radius: 12px;
        padding: 10px 15px;
        margin: 5px 0;
    }
    .user-msg {
        background-color: #DCF8C6;
        text-align: right;
    }
    .bot-msg {
        background-color: #FFFFFF;
        border: 1px solid #ddd;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/?size=100&id=38180&format=png", width=70)
st.sidebar.title("💼 Financial AI Assistant")

embedding_choice = st.sidebar.radio(
    "🔎 Choose Embedding Model",
    ["Gemini Embeddings", "Hugging Face Embeddings"],
    index=1
)

mode_choice = st.sidebar.radio(
    "💡 Choose Mode",
    ["Normal", "Tips Mode"],
    index=0
)

uploaded_files = st.sidebar.file_uploader("📂 Upload PDFs", accept_multiple_files=True, type=["pdf"])

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []

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
        st.session_state.vectorstore = create_vectorstore(chunks, embedding_choice)
    st.success(f"✅ Documents processed with **{embedding_choice}** in **{mode_choice}**!")

# Main header
st.title("📊 Financial Document Q&A")
st.markdown("### Ask questions about your financial documents in real time.")

if mode_choice == "Tips Mode":
    st.markdown("⚠️ **Tips Mode is ON** — answers will include general financial insights (educational only).")

# Display chat
for msg in st.session_state.messages:
    if msg["role"] == "user":
        message(msg["content"], is_user=True, key=msg["content"])
    else:
        message(msg["content"], key=msg["content"])

# Input box
if prompt := st.chat_input("Type your financial question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    if st.session_state.vectorstore is None:
        response = "⚠️ Please upload a financial document first."
    else:
        response = get_answer(st.session_state.vectorstore, prompt, mode_choice)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
