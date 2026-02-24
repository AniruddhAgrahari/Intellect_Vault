import unittest
from unittest.mock import MagicMock, patch
import sys
import os

class MockSessionState(dict):
    """Mock for streamlit session state that supports both dict and attribute access"""
    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"'MockSessionState' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

class TestAppLogic(unittest.TestCase):

    def test_app_uses_invoke(self):
        # Create mocks
        mock_st = MagicMock()
        mock_ingestion = MagicMock()
        mock_vectorstore = MagicMock()
        mock_retrieval = MagicMock()
        mock_dotenv = MagicMock()

        # Setup session state
        mock_st.session_state = MockSessionState({
            "vectorstore": MagicMock(),
            "chat_history": [],
            "messages": [],
            "processed": True
        })

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

        mock_st.status.return_value.__enter__.return_value = MagicMock()
        mock_st.status.return_value.__exit__.return_value = None

        # Patch sys.modules to prevent actual imports and isolate the test
        with patch.dict(sys.modules, {
            "streamlit": mock_st,
            "src.ingestion": mock_ingestion,
            "src.vectorstore": mock_vectorstore,
            "src.retrieval": mock_retrieval,
            "dotenv": mock_dotenv
        }):
            # Run app.py
            import runpy

            # Use patch.dict to set environment variable if needed
            with patch.dict(os.environ, {}, clear=False):
                runpy.run_path("app.py")

        # Assertions
        # Check that get_qa_chain was called
        mock_retrieval.get_qa_chain.assert_called()

        # Check that invoke was called on the chain
        mock_qa_chain.invoke.assert_called_once()

if __name__ == "__main__":
    unittest.main()
