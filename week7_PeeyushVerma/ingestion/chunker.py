from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.logger import get_logger
from utils.metrics import metrics_instance
from config import Config

logger = get_logger(__name__)

class TextChunker:
    def __init__(self, chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def split_documents(self, documents):
        """Split a list of Document objects into chunks."""
        if not documents:
            logger.warning("No documents provided to chunker.")
            return []
            
        logger.info(f"Chunking {len(documents)} documents (size={self.chunk_size}, overlap={self.chunk_overlap})...")
        chunks = self.splitter.split_documents(documents)
        
        metrics_instance.chunk_count += len(chunks)
        metrics_instance.doc_count += len(documents)
        
        logger.info(f"Created {len(chunks)} chunks.")
        
        # Calculate avg length
        if chunks:
            avg_len = sum(len(c.page_content) for c in chunks) / len(chunks)
            logger.info(f"Average chunk length: {avg_len:.2f} characters.")
            
        return chunks
