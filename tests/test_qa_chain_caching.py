
import unittest
from unittest.mock import MagicMock

class TestQAChainCaching(unittest.TestCase):
    def setUp(self):
        self.session_state = {}
        self.get_qa_chain = MagicMock(return_value="mock_qa_chain")
        self.vectorstore = "mock_vectorstore"
        self.session_state["vectorstore"] = self.vectorstore

    def simulate_chat_logic(self):
        """Simulate the logic in app.py for retrieving qa_chain"""
        if "qa_chain" not in self.session_state:
            self.session_state["qa_chain"] = self.get_qa_chain(self.session_state["vectorstore"])
        return self.session_state["qa_chain"]

    def simulate_process_document(self):
        """Simulate the logic in app.py for processing a document"""
        # Assume vectorstore is updated
        self.session_state["vectorstore"] = "new_vectorstore"

        # Invalidate qa_chain
        if "qa_chain" in self.session_state:
            del self.session_state["qa_chain"]

    def test_caching_behavior(self):
        # First call: get_qa_chain should be called
        chain1 = self.simulate_chat_logic()
        self.assertEqual(chain1, "mock_qa_chain")
        self.get_qa_chain.assert_called_once_with("mock_vectorstore")

        # Second call: get_qa_chain should NOT be called again
        chain2 = self.simulate_chat_logic()
        self.assertEqual(chain2, "mock_qa_chain")
        self.get_qa_chain.assert_called_once()  # Still called once

    def test_invalidation_behavior(self):
        # Setup initial state
        self.simulate_chat_logic()
        self.get_qa_chain.assert_called_once()

        # Process new document
        self.simulate_process_document()
        self.assertNotIn("qa_chain", self.session_state)

        # Next chat call should trigger new get_qa_chain
        chain3 = self.simulate_chat_logic()
        self.assertEqual(chain3, "mock_qa_chain")
        self.assertEqual(self.get_qa_chain.call_count, 2)
        # Verify the second call was with the new vectorstore
        self.get_qa_chain.assert_called_with("new_vectorstore")

if __name__ == "__main__":
    unittest.main()
