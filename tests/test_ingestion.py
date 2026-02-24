import unittest
from unittest.mock import MagicMock
import sys
import os

# Mock dependencies
sys.modules["langchain_community.document_loaders"] = MagicMock()
sys.modules["langchain_text_splitters"] = MagicMock()

# Now import the module under test
from src.ingestion import secure_temp_file

class TestSecureTempFile(unittest.TestCase):
    def test_secure_temp_file_creation_and_deletion(self):
        content = b"test content"
        file_path = None

        with secure_temp_file(content) as path:
            file_path = path
            # Check file exists
            self.assertTrue(os.path.exists(file_path))
            # Check content
            with open(file_path, "rb") as f:
                self.assertEqual(f.read(), content)

        # Check file is deleted
        self.assertFalse(os.path.exists(file_path))

    def test_secure_temp_file_cleanup_on_exception(self):
        content = b"test content"
        file_path = None

        try:
            with secure_temp_file(content) as path:
                file_path = path
                self.assertTrue(os.path.exists(file_path))
                raise ValueError("Something went wrong")
        except ValueError:
            pass

        # Check file is deleted
        if file_path:
            self.assertFalse(os.path.exists(file_path))

if __name__ == '__main__':
    unittest.main()
