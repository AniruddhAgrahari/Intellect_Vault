import streamlit as st
import os
from dotenv import load_dotenv
from src.ingestion import load_pdf, split_documents
from src.vectorstore import get_vectorstore, load_existing_vectorstore
from src.retrieval import get_qa_chain
import uuid

# Configuration
MAX_SIZE = 10 * 1024 * 1024  # 10MB

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Intellect-Vault", layout="wide")

st.title("🧠 Intellect-Vault: Chat with your Documents")
st.markdown("Upload your PDF and ask questions based on its content.")

# Sidebar for settings and file upload
with st.sidebar:
    st.header("Settings")
    groq_api_key = st.text_input("Enter Groq API Key", type="password")
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
    
    st.divider()
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file and groq_api_key:
        if uploaded_file.size > MAX_SIZE:
            st.error(f"File size exceeds the {MAX_SIZE / (1024 * 1024):.0f}MB limit. Please upload a smaller file.")
        elif st.button("Process Document"):
            with st.status("Vectorizing Document...", expanded=True) as status:
                st.write("Loading PDF...")
                temp_filename = f"temp_{uuid.uuid4()}.pdf"
                try:
                    with open(temp_filename, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    st.write("Splitting text into chunks...")
                    docs = load_pdf(temp_filename)
                    for doc in docs:
                        doc.metadata["source"] = uploaded_file.name

                    chunks = split_documents(docs)

                    st.write("Generating embeddings and indexing...")
                    vectorstore = get_vectorstore(chunks)

                    st.session_state.vectorstore = vectorstore
                    st.session_state.processed = True

                    status.update(label="Document processed and indexed!", state="complete", expanded=False)
                finally:
                    if os.path.exists(temp_filename):
                        os.remove(temp_filename)

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("View Sources Used"):
                for source in message["sources"]:
                    st.info(f"**Source:** {source['file']}, **Page:** {source['page']}\n\n{source['content']}")

if prompt := st.chat_input("Ask a question about your document..."):
    if not groq_api_key:
        st.error("Please enter your Groq API Key in the sidebar.")
    elif "vectorstore" not in st.session_state:
        st.error("Please upload and process a document first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing document..."):
                qa_chain = get_qa_chain(st.session_state.vectorstore)
                response = qa_chain({
                    "question": prompt, 
                    "chat_history": st.session_state.chat_history
                })
                
                answer = response["answer"]
                sources = []
                if "source_documents" in response:
                    for doc in response["source_documents"]:
                        sources.append({
                            "file": doc.metadata.get("source", "Unknown"),
                            "page": doc.metadata.get("page", "Unknown"),
                            "content": doc.page_content
                        })
                
                st.markdown(answer)
                if sources:
                    with st.expander("View Sources Used"):
                        for source in sources:
                            st.info(f"**Source:** {source['file']}, **Page:** {source['page']}\n\n{source['content']}")
                
                st.session_state.chat_history.append((prompt, answer))
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })
