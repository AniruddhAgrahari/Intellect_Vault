import unittest
from unittest.mock import patch, MagicMock
import sys

# Mock dependencies before they are imported by src.ingestion
mock_langchain_community_document_loaders = MagicMock()
sys.modules["langchain_community.document_loaders"] = mock_langchain_community_document_loaders

mock_langchain_text_splitters = MagicMock()
sys.modules["langchain_text_splitters"] = mock_langchain_text_splitters

from src.ingestion import split_documents

class TestIngestion(unittest.TestCase):

    @patch('src.ingestion.RecursiveCharacterTextSplitter')
    def test_split_documents_default_args(self, mock_splitter_cls):
        # Setup mock instance
        mock_splitter_instance = MagicMock()
        mock_splitter_cls.return_value = mock_splitter_instance

        # Setup mock documents and return value
        mock_documents = [MagicMock(), MagicMock()]
        mock_split_docs = [MagicMock(), MagicMock(), MagicMock()]
        mock_splitter_instance.split_documents.return_value = mock_split_docs

        # Call function
        result = split_documents(mock_documents)

        # Assertions
        mock_splitter_cls.assert_called_once_with(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )
        mock_splitter_instance.split_documents.assert_called_once_with(mock_documents)
        self.assertEqual(result, mock_split_docs)

    @patch('src.ingestion.RecursiveCharacterTextSplitter')
    def test_split_documents_custom_args(self, mock_splitter_cls):
        # Setup mock instance
        mock_splitter_instance = MagicMock()
        mock_splitter_cls.return_value = mock_splitter_instance

        mock_documents = [MagicMock()]

        # Call function with custom args
        split_documents(mock_documents, chunk_size=500, chunk_overlap=50)

        # Assertions
        mock_splitter_cls.assert_called_once_with(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )
        mock_splitter_instance.split_documents.assert_called_once_with(mock_documents)

    @patch('src.ingestion.RecursiveCharacterTextSplitter')
    def test_split_documents_empty_list(self, mock_splitter_cls):
        # Setup mock instance
        mock_splitter_instance = MagicMock()
        mock_splitter_cls.return_value = mock_splitter_instance
        mock_splitter_instance.split_documents.return_value = []

        # Call function with empty list
        result = split_documents([])

        # Assertions
        mock_splitter_instance.split_documents.assert_called_once_with([])
        self.assertEqual(result, [])

if __name__ == '__main__':
    unittest.main()
