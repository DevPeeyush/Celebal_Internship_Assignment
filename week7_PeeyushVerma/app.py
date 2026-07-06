import streamlit as st
import os
import time

# Set Streamlit page config (must be the first Streamlit command)
st.set_page_config(
    page_title="RAG Document QA",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    /* Main layout */
    .main {
        background-color: #f8f9fa;
        color: #212529;
    }
    /* Chat containers */
    .stChatMessage {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    /* Source highlight */
    .source-box {
        background-color: #e9ecef;
        border-left: 4px solid #0d6efd;
        padding: 10px;
        border-radius: 4px;
        margin-top: 10px;
        font-size: 0.9em;
    }
    /* Headers */
    h1, h2, h3 {
        color: #0d6efd;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Imports after page config
from config import Config
from ingestion.loader import DocumentLoader
from ingestion.chunker import TextChunker
from retrieval.retriever import VectorStoreManager
from llm.generator import LLMGenerator
from utils.helpers import save_uploaded_file
from utils.metrics import metrics_instance

# Initialize core components in session state to persist them
if 'vector_manager' not in st.session_state:
    st.session_state.vector_manager = VectorStoreManager()
if 'llm_generator' not in st.session_state:
    st.session_state.llm_generator = LLMGenerator()
if 'doc_loader' not in st.session_state:
    st.session_state.doc_loader = DocumentLoader()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def process_uploaded_files(uploaded_files, chunk_size, chunk_overlap):
    """Handle document ingestion pipeline."""
    if not uploaded_files:
        return
        
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file in enumerate(uploaded_files):
        status_text.text(f"Processing: {file.name}")
        file_path = save_uploaded_file(file, Config.DATA_DIR)
        
        if file_path:
            docs = st.session_state.doc_loader.load_document(file_path)
            chunks = chunker.split_documents(docs)
            all_chunks.extend(chunks)
            
        progress_bar.progress((i + 1) / len(uploaded_files))
        
    if all_chunks:
        status_text.text(f"Building vector database with {len(all_chunks)} chunks...")
        success = st.session_state.vector_manager.build_vectorstore(all_chunks)
        if success:
            st.success(f"Successfully processed {len(uploaded_files)} files into {len(all_chunks)} chunks!")
            status_text.empty()
            progress_bar.empty()
        else:
            st.error("Failed to build vector database.")
    else:
        st.warning("No readable content found in the uploaded files.")

# --- Sidebar UI ---
with st.sidebar:
    st.title("⚙️ System Configuration")
    
    st.header("1. Data Ingestion")
    uploaded_files = st.file_uploader("Upload PDF / TXT files", type=["pdf", "txt"], accept_multiple_files=True)
    
    hf_dataset = st.text_input("Or load HuggingFace Dataset (e.g. databricks/databricks-dolly-15k)", placeholder="username/dataset_name")
    
    st.header("2. Chunking Settings")
    chunk_size = st.slider("Chunk Size", 200, 2000, Config.CHUNK_SIZE, 100)
    chunk_overlap = st.slider("Chunk Overlap", 0, 500, Config.CHUNK_OVERLAP, 50)
    
    if st.button("Build Knowledge Base", type="primary", use_container_width=True):
        if uploaded_files:
            process_uploaded_files(uploaded_files, chunk_size, chunk_overlap)
        elif hf_dataset:
            with st.spinner("Loading dataset from HuggingFace..."):
                docs = st.session_state.doc_loader.load_huggingface_dataset(hf_dataset)
                if docs:
                    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    chunks = chunker.split_documents(docs)
                    st.session_state.vector_manager.build_vectorstore(chunks)
                    st.success(f"Loaded dataset into {len(chunks)} chunks!")
                else:
                    st.error("Failed to load dataset.")
        else:
            st.warning("Please upload files or specify a dataset.")
            
    if st.button("Clear Vector Database", use_container_width=True):
        if st.session_state.vector_manager.clear_vectorstore():
            st.success("Vector database cleared.")
            st.session_state.chat_history = []
            st.rerun()

    st.divider()
    
    st.header("📊 System Metrics")
    m_col1, m_col2 = st.columns(2)
    m_col1.metric("Vector DB Size", f"{metrics_instance.vector_db_size}")
    m_col2.metric("Total Chunks", f"{metrics_instance.chunk_count}")
    
    st.metric("LLM Provider", "Groq (Llama-3.1)")
    st.metric("Vector DB", "FAISS (bge-small-en)")

# --- Main UI ---
st.title("📚 RAG Document QA System")
st.markdown("Ask questions about your uploaded PDFs, Text files, or Datasets.")

# Display Chat History
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display sources if available
        if "sources" in msg and msg["sources"]:
            with st.expander("View Source Documents"):
                for idx, (doc, score) in enumerate(msg["sources"]):
                    source = doc.metadata.get('source', 'Unknown')
                    page = doc.metadata.get('page', 'N/A')
                    st.markdown(f"""
                    <div class="source-box">
                        <strong>Source {idx+1}:</strong> {os.path.basename(source)} (Page {page})<br>
                        <strong>Similarity Score:</strong> {score:.4f}<br>
                        <em>"{doc.page_content[:300]}..."</em>
                    </div>
                    """, unsafe_allow_html=True)
                    
        # Display metrics if available
        if "metrics" in msg:
            cols = st.columns(2)
            cols[0].caption(f"⏱️ Retrieval: {msg['metrics'].get('retrieval', 0):.3f}s")
            cols[1].caption(f"⚡ Generation: {msg['metrics'].get('generation', 0):.3f}s")

# User Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Check if vector DB is populated
    if metrics_instance.vector_db_size == 0:
        st.warning("Please upload documents and build the knowledge base first!")
    elif not Config.GROQ_API_KEY:
        st.error("GROQ_API_KEY is missing. Please add it to your .env file.")
    else:
        # Display user message
        st.chat_message("user").markdown(prompt)
        
        # Add to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Process and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant context..."):
                retrieved_chunks = st.session_state.vector_manager.search(prompt, k=Config.TOP_K)
                
            with st.spinner("Generating answer..."):
                answer = st.session_state.llm_generator.generate_answer(prompt, retrieved_chunks)
                
            st.markdown(answer)
            
            # Store metrics
            current_metrics = {
                "retrieval": metrics_instance.last_retrieval_latency,
                "generation": metrics_instance.last_generation_latency
            }
            
            # Display sources in UI
            if retrieved_chunks:
                with st.expander("View Source Documents"):
                    for idx, (doc, score) in enumerate(retrieved_chunks):
                        source = doc.metadata.get('source', 'Unknown')
                        page = doc.metadata.get('page', 'N/A')
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>Source {idx+1}:</strong> {os.path.basename(source)} (Page {page})<br>
                            <strong>Similarity Score:</strong> {score:.4f}<br>
                            <em>"{doc.page_content[:300]}..."</em>
                        </div>
                        """, unsafe_allow_html=True)
                        
            # Display latency metrics in UI
            cols = st.columns(2)
            cols[0].caption(f"⏱️ Retrieval: {current_metrics['retrieval']:.3f}s")
            cols[1].caption(f"⚡ Generation: {current_metrics['generation']:.3f}s")
            
            # Save to history
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": answer,
                "sources": retrieved_chunks,
                "metrics": current_metrics
            })
