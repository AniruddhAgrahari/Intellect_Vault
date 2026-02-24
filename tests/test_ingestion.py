import unittest
from unittest.mock import patch, MagicMock
import sys

# Mock dependencies before they are imported by src.ingestion
mock_langchain_document_loaders = MagicMock()
sys.modules["langchain_community.document_loaders"] = mock_langchain_document_loaders

mock_langchain_text_splitters = MagicMock()
sys.modules["langchain_text_splitters"] = mock_langchain_text_splitters

from src.ingestion import load_pdf, split_documents

class TestIngestion(unittest.TestCase):

    @patch('src.ingestion.PyPDFLoader')
    def test_load_pdf(self, mock_loader):
        file_path = "dummy.pdf"
        documents = [MagicMock()]
        mock_instance = mock_loader.return_value
        mock_instance.load.return_value = documents

        result = load_pdf(file_path)

        mock_loader.assert_called_once_with(file_path)
        mock_instance.load.assert_called_once()
        self.assertEqual(result, documents)

    @patch('src.ingestion.RecursiveCharacterTextSplitter')
    def test_split_documents(self, mock_splitter_cls):
        mock_splitter_instance = mock_splitter_cls.return_value
        documents = [MagicMock()]
        expected_chunks = [MagicMock(), MagicMock()]
        mock_splitter_instance.split_documents.return_value = expected_chunks

        result = split_documents(documents, chunk_size=500, chunk_overlap=50)

        mock_splitter_cls.assert_called_once_with(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )
        mock_splitter_instance.split_documents.assert_called_once_with(documents)
        self.assertEqual(result, expected_chunks)

if __name__ == '__main__':
    unittest.main()
