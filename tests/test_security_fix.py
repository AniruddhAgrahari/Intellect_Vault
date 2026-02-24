import unittest
from unittest.mock import MagicMock, patch, mock_open
import sys
import builtins
import os

class TestSecurityFix(unittest.TestCase):
    def setUp(self):
        self.mock_st = MagicMock()
        self.mock_os = MagicMock()
        self.mock_dotenv = MagicMock()
        self.mock_ingestion = MagicMock()
        self.mock_vectorstore = MagicMock()
        self.mock_retrieval = MagicMock()
        self.mock_uuid = MagicMock()

        # Setup st.file_uploader
        mock_file = MagicMock()
        mock_file.size = 100
        mock_file.name = "test.pdf"
        mock_file.getbuffer.return_value = b"test content"
        self.mock_st.file_uploader.return_value = mock_file

        # Setup st.button to trigger processing
        self.mock_st.button.return_value = True

        # Setup st.text_input for API Key
        self.mock_st.text_input.return_value = "fake-api-key"

        # Setup os.environ
        self.mock_os.environ = {}

        # Setup st.session_state
        self.mock_st.session_state = {}

        # Setup st.status context manager
        self.mock_st.status.return_value.__enter__.return_value = MagicMock()

        # Ensure os.path.exists returns True so os.remove is called
        self.mock_os.path.exists.return_value = True

    def test_temp_file_cleanup_on_error(self):
        # Read app.py content
        with open('app.py', 'r') as f:
            code = f.read()

        # Simulate exception in load_pdf
        self.mock_ingestion.load_pdf.side_effect = Exception("Simulated processing failure")

        # Mock uuid to predict filename
        self.mock_uuid.uuid4.return_value = "test-uuid"

        with patch.dict(sys.modules, {
            'streamlit': self.mock_st,
            'os': self.mock_os,
            'dotenv': self.mock_dotenv,
            'src.ingestion': self.mock_ingestion,
            'src.vectorstore': self.mock_vectorstore,
            'src.retrieval': self.mock_retrieval,
            'uuid': self.mock_uuid
        }):
            # Mock built-in open for the temp file write
            with patch('builtins.open', mock_open()) as mocked_file:
                # Execute app.py
                try:
                    exec(code, {'__name__': '__main__'})
                except Exception as e:
                    # Expect the simulated exception
                    if str(e) != "Simulated processing failure":
                        raise e

        # Assert os.remove was called
        expected_filename = "temp_test-uuid.pdf"

        self.mock_os.remove.assert_called_with(expected_filename)

if __name__ == '__main__':
    unittest.main()
