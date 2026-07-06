import os
from langchain_community.vectorstores import FAISS
from embeddings.embedding_model import EmbeddingModel
from utils.logger import get_logger
from utils.metrics import track_latency, metrics_instance
from config import Config

logger = get_logger(__name__)

class VectorStoreManager:
    def __init__(self):
        self.embeddings = EmbeddingModel.get_instance()
        self.vector_store_path = Config.VECTORSTORE_DIR
        self.index_name = "faiss_index"
        self.vectorstore = None
        self.load_vectorstore()

    def load_vectorstore(self):
        """Load existing vector store if available."""
        index_path = os.path.join(self.vector_store_path, f"{self.index_name}.faiss")
        if os.path.exists(index_path):
            try:
                self.vectorstore = FAISS.load_local(
                    folder_path=self.vector_store_path,
                    embeddings=self.embeddings,
                    index_name=self.index_name,
                    allow_dangerous_deserialization=True
                )
                metrics_instance.vector_db_size = self.vectorstore.index.ntotal
                logger.info(f"Loaded existing FAISS index with {metrics_instance.vector_db_size} vectors.")
            except Exception as e:
                logger.error(f"Failed to load existing FAISS index: {e}")
                self.vectorstore = None
        else:
            logger.info("No existing FAISS index found. Ready to build a new one.")

    def build_vectorstore(self, chunks):
        """Build or add to a FAISS vector store."""
        if not chunks:
            logger.warning("No chunks provided to build vector store.")
            return False

        logger.info(f"Building vector store with {len(chunks)} chunks...")
        try:
            if self.vectorstore is None:
                self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
            else:
                self.vectorstore.add_documents(chunks)
            
            self.save_vectorstore()
            return True
        except Exception as e:
            logger.error(f"Failed to build vector store: {e}")
            return False

    def save_vectorstore(self):
        """Save the vector store to disk."""
        if self.vectorstore is not None:
            self.vectorstore.save_local(
                folder_path=self.vector_store_path,
                index_name=self.index_name
            )
            metrics_instance.vector_db_size = self.vectorstore.index.ntotal
            logger.info(f"Saved FAISS index. Total vectors: {metrics_instance.vector_db_size}")

    def clear_vectorstore(self):
        """Clear the existing vector store."""
        try:
            index_path = os.path.join(self.vector_store_path, f"{self.index_name}.faiss")
            pkl_path = os.path.join(self.vector_store_path, f"{self.index_name}.pkl")
            if os.path.exists(index_path):
                os.remove(index_path)
            if os.path.exists(pkl_path):
                os.remove(pkl_path)
                
            self.vectorstore = None
            metrics_instance.vector_db_size = 0
            metrics_instance.chunk_count = 0
            metrics_instance.doc_count = 0
            logger.info("Cleared vector store.")
            return True
        except Exception as e:
            logger.error(f"Failed to clear vector store: {e}")
            return False

    @track_latency("retrieval")
    def search(self, query, k=Config.TOP_K):
        """Perform similarity search on the vector store."""
        if self.vectorstore is None:
            logger.warning("Vector store is empty. Cannot search.")
            return []
            
        logger.info(f"Searching for: '{query}' (top_k={k})")
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        # FAISS returns L2 distance (lower is better), we might want to log it
        for i, (doc, score) in enumerate(results):
            logger.debug(f"Result {i+1}: score={score:.4f}, doc={doc.metadata.get('source', 'unknown')}")
            
        return results
