import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import runpy

# Create mocks for modules
mock_st = MagicMock()
mock_ingestion = MagicMock()
mock_vectorstore = MagicMock()
mock_retrieval = MagicMock()
mock_dotenv = MagicMock()

# Patch sys.modules to prevent actual imports
sys.modules["streamlit"] = mock_st
sys.modules["src.ingestion"] = mock_ingestion
sys.modules["src.vectorstore"] = mock_vectorstore
sys.modules["src.retrieval"] = mock_retrieval
sys.modules["dotenv"] = mock_dotenv

class SessionStateMock(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"'SessionStateMock' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

class TestApp(unittest.TestCase):
    def setUp(self):
        # Reset mocks before each test
        mock_st.reset_mock()
        mock_retrieval.reset_mock()

    def test_app_uses_invoke(self):
        # Setup session state
        # We need to initialize it with values that app.py expects or sets
        session_state = SessionStateMock({
            "vectorstore": MagicMock(),
            "chat_history": [],
            "messages": [],
            "processed": True
        })
        mock_st.session_state = session_state

        # Setup user input to trigger the chat logic
        mock_st.chat_input.return_value = "What is the summary?"

        # Setup sidebar inputs
        mock_st.text_input.return_value = "fake_api_key"

        # Setup file uploader to return None so we don't trigger processing
        mock_st.file_uploader.return_value = None

        # Setup QA chain mock
        mock_qa_chain = MagicMock()
        mock_response = {
            "answer": "This is a summary.",
            "source_documents": [
                MagicMock(page_content="Content", metadata={"source": "doc.pdf", "page": 1})
            ]
        }
        mock_qa_chain.return_value = mock_response
        mock_qa_chain.invoke.return_value = mock_response

        mock_retrieval.get_qa_chain.return_value = mock_qa_chain

        # Mock context managers
        mock_st.sidebar.__enter__.return_value = None
        mock_st.sidebar.__exit__.return_value = None

        mock_st.chat_message.return_value.__enter__.return_value = None
        mock_st.chat_message.return_value.__exit__.return_value = None

        mock_st.spinner.return_value.__enter__.return_value = None
        mock_st.spinner.return_value.__exit__.return_value = None

        mock_st.expander.return_value.__enter__.return_value = None
        mock_st.expander.return_value.__exit__.return_value = None

        mock_st.status.return_value.__enter__.return_value = None
        mock_st.status.return_value.__exit__.return_value = None

        # Run app.py
        with patch.dict(os.environ, {}, clear=False):
             runpy.run_path("app.py")

        # Assertions
        # Check that get_qa_chain was called
        mock_retrieval.get_qa_chain.assert_called()

        # Check that invoke was called on the chain
        mock_qa_chain.invoke.assert_called_once()

        # Ensure deprecated __call__ is NOT used
        mock_qa_chain.assert_not_called()

if __name__ == "__main__":
    unittest.main()
