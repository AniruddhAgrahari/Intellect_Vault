import unittest
from unittest.mock import patch, MagicMock
import sys

# Mock dependencies before they are imported by src.ingestion
sys.modules["langchain_community.document_loaders"] = MagicMock()
sys.modules["langchain_text_splitters"] = MagicMock()

# Now import the module to be tested
from src.ingestion import load_pdf, split_documents

class TestIngestion(unittest.TestCase):

    @patch('src.ingestion.PyPDFLoader')
    def test_load_pdf(self, mock_loader_cls):
        # Setup mock
        mock_loader_instance = mock_loader_cls.return_value
        expected_docs = [MagicMock(), MagicMock()]
        mock_loader_instance.load.return_value = expected_docs

        # Execute
        file_path = "dummy.pdf"
        result = load_pdf(file_path)

        # Verify
        mock_loader_cls.assert_called_once_with(file_path)
        mock_loader_instance.load.assert_called_once()
        self.assertEqual(result, expected_docs)

    @patch('src.ingestion.RecursiveCharacterTextSplitter')
    def test_split_documents(self, mock_splitter_cls):
        # Setup mock
        mock_splitter_instance = mock_splitter_cls.return_value
        expected_chunks = [MagicMock(), MagicMock(), MagicMock()]
        mock_splitter_instance.split_documents.return_value = expected_chunks

        documents = [MagicMock()]

        # Execute
        result = split_documents(documents)

        # Verify
        mock_splitter_cls.assert_called_once_with(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )
        mock_splitter_instance.split_documents.assert_called_once_with(documents)
        self.assertEqual(result, expected_chunks)

if __name__ == '__main__':
    unittest.main()
