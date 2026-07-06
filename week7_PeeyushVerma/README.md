# RAG Document Question Answering System

A production-ready Retrieval-Augmented Generation (RAG) system built with Streamlit, LangChain, FAISS, and Groq.
This system allows you to upload documents (PDFs, TXT files) or load HuggingFace datasets, and ask questions about them.

## Features
- **Document Ingestion**: Supports PDFs, TXTs, and HuggingFace datasets.
- **Advanced Chunking**: RecursiveCharacterTextSplitter with configurable size and overlap.
- **Fast Retrieval**: Powered by FAISS and BAAI/bge-small-en-v1.5 embeddings.
- **LLM Generation**: Uses Llama 3.1 8B via the Groq API for blazing-fast responses.
- **Metrics & Logging**: Built-in latency tracking and extensive logging.

## Installation

This project uses `uv` for lightning-fast dependency management.

1. Ensure `uv` is installed on your system.
2. Install dependencies:
   ```bash
   uv sync
   ```
3. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your `GROQ_API_KEY` (and optionally `HUGGINGFACE_API_KEY`)

## Running the Application

```bash
uv run streamlit run app.py
```

## Architecture

- **`ingestion/`**: Document loaders and text chunking logic.
- **`embeddings/`**: HuggingFace embeddings model initialization.
- **`retrieval/`**: FAISS vector store management and search logic.
- **`llm/`**: Groq LLM integration and prompt construction.
- **`utils/`**: Shared helpers, logging, and metrics tracking.
- **`app.py`**: Streamlit frontend.
