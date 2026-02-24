import os
import tempfile
from contextlib import contextmanager
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(file_path):
    """Loads a PDF file and returns the documents."""
    loader = PyPDFLoader(file_path)
    return loader.load()

def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    """Splits documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

@contextmanager
def secure_temp_file(file_content, suffix=".pdf"):
    """
    Securely creates a temporary file with the given content and yields its path.
    Ensures the file is deleted after use.
    """
    # Create a temporary file. delete=False to allow re-opening on Windows if needed.
    # The file is created with 0600 permissions by default on Unix.
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(file_content)
        tmp_path = tmp_file.name

    try:
        yield tmp_path
    finally:
        # Ensure the file is deleted
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass  # Best effort cleanup
