from langchain_huggingface import HuggingFaceEmbeddings
from utils.logger import get_logger
from config import Config

logger = get_logger(__name__)

class EmbeddingModel:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            logger.info(f"Initializing embedding model: {Config.EMBEDDING_MODEL}")
            try:
                cls._instance = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
                logger.info("Embedding model initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize embedding model: {e}")
                raise e
        return cls._instance
