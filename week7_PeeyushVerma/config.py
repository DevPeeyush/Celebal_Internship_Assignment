import os
from dotenv import load_dotenv  # pyrefly: ignore [missing-import]

# Load environment variables from .env file
load_dotenv()

class Config:
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    
    # Document Processing Settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Embedding Model Settings
    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
    
    # Reranker Settings
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Retrieval Settings
    TOP_K = 5
    
    # LLM Settings
    LLM_MODEL = "llama-3.1-8b-instant"

# Ensure directories exist
os.makedirs(Config.DATA_DIR, exist_ok=True)
os.makedirs(Config.VECTORSTORE_DIR, exist_ok=True)
os.makedirs(Config.LOG_DIR, exist_ok=True)
