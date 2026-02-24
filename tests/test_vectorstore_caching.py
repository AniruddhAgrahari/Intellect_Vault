import sys
from unittest.mock import MagicMock, patch

# Mock the modules before importing from src.vectorstore
sys.modules["langchain_google_genai"] = MagicMock()
sys.modules["langchain_community.embeddings"] = MagicMock()
sys.modules["langchain_community.vectorstores"] = MagicMock()
sys.modules["langchain_core.documents"] = MagicMock()
sys.modules["tenacity"] = MagicMock()
sys.modules["langchain_google_genai._common"] = MagicMock()

import src.vectorstore

def test_hf_embeddings_caching():
    # Clear cache before test
    src.vectorstore.get_hf_embeddings.cache_clear()

    with patch("src.vectorstore.HuggingFaceEmbeddings") as mock_hf_class:
        mock_hf_class.return_value = MagicMock()
        # Call it twice
        emb1 = src.vectorstore.get_hf_embeddings()
        emb2 = src.vectorstore.get_hf_embeddings()

        # Verify it was only instantiated once
        assert mock_hf_class.call_count == 1
        # Verify both calls returned the same instance
        assert emb1 is emb2

def test_google_embeddings_caching():
    # Clear cache before test
    src.vectorstore.get_google_embeddings.cache_clear()

    with patch("src.vectorstore.BatchedGoogleEmbeddings") as mock_google_class:
        mock_google_class.return_value = MagicMock()
        # Call it twice
        emb1 = src.vectorstore.get_google_embeddings()
        emb2 = src.vectorstore.get_google_embeddings()

        # Verify it was only instantiated once
        assert mock_google_class.call_count == 1
        # Verify both calls returned the same instance
        assert emb1 is emb2

def test_different_params_not_cached():
    # Clear cache before test
    src.vectorstore.get_hf_embeddings.cache_clear()

    with patch("src.vectorstore.HuggingFaceEmbeddings") as mock_hf_class:
        mock_hf_class.side_effect = [MagicMock(), MagicMock()]
        emb1 = src.vectorstore.get_hf_embeddings(model_name="model1")
        emb2 = src.vectorstore.get_hf_embeddings(model_name="model2")

        assert mock_hf_class.call_count == 2
        assert emb1 is not emb2
