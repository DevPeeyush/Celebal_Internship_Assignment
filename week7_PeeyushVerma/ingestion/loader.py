import os
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from datasets import load_dataset
from langchain_core.documents import Document
from utils.logger import get_logger

logger = get_logger(__name__)

class DocumentLoader:
    def __init__(self):
        pass

    def load_pdf(self, file_path):
        """Load a PDF document."""
        try:
            logger.info(f"Loading PDF from {file_path}")
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages from PDF.")
            return documents
        except Exception as e:
            logger.error(f"Failed to load PDF {file_path}: {e}")
            return []

    def load_text(self, file_path):
        """Load a TXT document."""
        try:
            logger.info(f"Loading TXT from {file_path}")
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} text documents.")
            return documents
        except Exception as e:
            logger.error(f"Failed to load TXT {file_path}: {e}")
            return []

    def load_huggingface_dataset(self, dataset_name, split="train", text_column="text"):
        """Load a text dataset from HuggingFace."""
        try:
            logger.info(f"Loading HuggingFace dataset: {dataset_name} (split: {split})")
            dataset = load_dataset(dataset_name, split=split)
            
            documents = []
            for item in dataset:
                if text_column in item:
                    doc = Document(
                        page_content=item[text_column],
                        metadata={"source": dataset_name, "type": "hf_dataset"}
                    )
                    documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} documents from HF dataset.")
            return documents
        except Exception as e:
            logger.error(f"Failed to load HF dataset {dataset_name}: {e}")
            return []

    def load_document(self, file_path):
        """Automatically route to correct loader based on extension."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            return self.load_pdf(file_path)
        elif ext == '.txt':
            return self.load_text(file_path)
        else:
            logger.warning(f"Unsupported file extension: {ext}")
            return []
