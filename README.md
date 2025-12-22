# 🧠 Intellect-Vault: Advanced RAG System

Intellect-Vault is a high-performance "Chat with your Documents" platform built using **Google Gemini**, **LangChain**, and **ChromaDB**. It allows users to upload complex PDFs and interact with them through natural language queries—delivering accurate, citation-backed answers instantly.

## 🚀 Features
- **Hybrid RAG:** Local embeddings (HuggingFace) + High-speed LLM (Groq Llama 3).
- **Source Attribution:** Every answer is cited directly from the PDF source.
- **Zero API Costs:** Built to run efficiently without expensive embedding quotas.
- **Context-Aware:** Handles large documents using recursive character chunking.

## 🏗️ System Architecture

```mermaid
graph TD
    A[User Uploads PDF] --> B[Streamlit UI]
    B --> C[PyPDFLoader & Text Splitting]
    C --> D[Google Gemini Embeddings]
    D --> E[ChromaDB Vector Store]
    F[User Asks Question] --> G[LangChain Conversational Chain]
    G --> H[Semantic Search in ChromaDB]
    H --> I[Retrieved Context + Prompt]
    I --> J[Google Gemini 1.5 Pro]
    J --> K[Answer + Citations]
    K --> B
```

1. **User -> Streamlit**: Minimalist and responsive chat interface.
2. **FastAPI (Implicit via Streamlit)**: Handles backend processing.
3. **Gemini Embeddings**: Converts text chunks into high-dimensional vectors.
4. **ChromaDB**: Stores vectors locally for instant retrieval.
5. **Gemini LLM**: Orchesrates the final response with grounding from the document.

## 📊 Key Metrics

- **Context Window**: 1M+ tokens (via Gemini 1.5 Pro), capable of handling massive technical manuals.
- **Semantic Search Accuracy**: High precision retrieval using `models/embedding-001`.
- **Response Grounding**: Zero-hallucination policy via strict prompt engineering and metadata mapping.

## 🛠️ Setup & Installation

1.  **Clone the repository**
2.  **Create a virtual environment**: `python -m venv venv`
3.  **Install dependencies**: `pip install -r requirements.txt`
4.  **Set up API Key**: Get a Google AI Studio API Key and enter it in the app sidebar or `.env`.
5.  **Run the App**: `streamlit run app.py`

---
*Built for performance. Optimized for intelligence.*