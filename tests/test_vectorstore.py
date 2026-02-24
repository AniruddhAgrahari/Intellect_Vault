import unittest
from unittest.mock import patch, MagicMock
import sys

# Mock dependencies before they are imported by src.vectorstore
mock_langchain_google_genai = MagicMock()
sys.modules["langchain_google_genai"] = mock_langchain_google_genai
sys.modules["langchain_community.embeddings"] = MagicMock()
sys.modules["langchain_community.vectorstores"] = MagicMock()
sys.modules["langchain_core.documents"] = MagicMock()
sys.modules["tenacity"] = MagicMock()
sys.modules["langchain_google_genai._common"] = MagicMock()

from src.vectorstore import get_vectorstore, load_existing_vectorstore, get_embeddings

class TestVectorStore(unittest.TestCase):

    def setUp(self):
        get_embeddings.cache_clear()

    @patch('src.vectorstore.HuggingFaceEmbeddings')
    @patch('src.vectorstore.Chroma')
    def test_get_vectorstore_local(self, mock_chroma, mock_huggingface):
        documents = [MagicMock()]
        get_vectorstore(documents, use_local=True)

        mock_huggingface.assert_called_once_with(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        mock_chroma.from_documents.assert_called_once()

    @patch('src.vectorstore.BatchedGoogleEmbeddings')
    @patch('src.vectorstore.Chroma')
    def test_get_vectorstore_google(self, mock_chroma, mock_google):
        documents = [MagicMock()]
        get_vectorstore(documents, use_local=False)

        mock_google.assert_called_once_with(
            model="models/embedding-001",
            batch_size=5,
            delay=2
        )
        mock_chroma.from_documents.assert_called_once()

    @patch('src.vectorstore.HuggingFaceEmbeddings')
    @patch('src.vectorstore.Chroma')
    def test_load_existing_vectorstore_local(self, mock_chroma, mock_huggingface):
        load_existing_vectorstore(use_local=True)

        mock_huggingface.assert_called_once_with(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        mock_chroma.assert_called_once()

    @patch('src.vectorstore.BatchedGoogleEmbeddings')
    @patch('src.vectorstore.Chroma')
    def test_load_existing_vectorstore_google(self, mock_chroma, mock_google):
        load_existing_vectorstore(use_local=False)

        mock_google.assert_called_once_with(
            model="models/embedding-001",
            batch_size=5,
            delay=2
        )
        mock_chroma.assert_called_once()

if __name__ == '__main__':
    unittest.main()
