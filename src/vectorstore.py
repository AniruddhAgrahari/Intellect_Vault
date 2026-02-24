import time
from functools import lru_cache
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_google_genai._common import GoogleGenerativeAIError

class BatchedGoogleEmbeddings:
    """Wrapper for Google embeddings with batching and throttling."""
    
    def __init__(self, model="models/embedding-001", batch_size=5, delay=2):
        self.embeddings = GoogleGenerativeAIEmbeddings(model=model)
        self.batch_size = batch_size
        self.delay = delay
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=10, min=10, max=60),
        retry=retry_if_exception_type(GoogleGenerativeAIError)
    )
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents in batches with retry logic."""
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            print(f"Processing batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1}")
            
            try:
                batch_embeddings = self.embeddings.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
                
                # Throttle between batches (except for the last one)
                if i + self.batch_size < len(texts):
                    time.sleep(self.delay)
            except GoogleGenerativeAIError as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    print(f"⚠️ Quota exceeded. Retrying with exponential backoff...")
                    raise  # Let tenacity handle the retry
                else:
                    raise
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self.embeddings.embed_query(text)


@lru_cache(maxsize=2)
def get_embeddings(use_local: bool = True):
    """
    Returns the appropriate embedding model based on the use_local flag.
    """
    if use_local:
        print("🔧 Using local embeddings (no API required)")
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
    else:
        print("🌐 Using Google Gemini embeddings with batching and retry logic")
        return BatchedGoogleEmbeddings(
            model="models/embedding-001",
            batch_size=5,
            delay=2
        )


def get_vectorstore(documents: List[Document], persist_directory="chroma_db", use_local=True):
    """
    Creates or loads a Chroma vector store from documents.

    Args:
        documents: List of documents to embed
        persist_directory: Directory to persist the vector store
        use_local: If True, use local HuggingFace embeddings instead of Google API (default: True)
    """
    embeddings = get_embeddings(use_local)
    
    try:
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        return vectorstore
    except GoogleGenerativeAIError as e:
        if "429" in str(e) or "quota" in str(e).lower():
            print("❌ Google API quota exhausted even after retries.")
            print("🔄 Falling back to local embeddings...")
            return get_vectorstore(documents, persist_directory, use_local=True)
        else:
            raise


def load_existing_vectorstore(persist_directory="chroma_db", use_local=True):
    """Loads an existing Chroma vector store."""
    embeddings = get_embeddings(use_local)
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
