import os
from utils.logger import get_logger

logger = get_logger(__name__)

def save_uploaded_file(uploaded_file, directory):
    """Save a Streamlit uploaded file to the specified directory."""
    try:
        file_path = os.path.join(directory, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logger.info(f"Saved uploaded file to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving file {uploaded_file.name}: {e}")
        return None
