from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

def get_qa_chain(vectorstore):
    """Creates a ConversationalRetrievalChain with source citation instructions."""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Custom prompt to enforce source citation
    template = """
    You are a professional assistant for document analysis. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    CRITICAL: For every fact you provide, you MUST cite the source document name and page number 
    found in the context metadata. Use the format: "[Source: filename, Page: #]".
    
    Chat History:
    {chat_history}
    
    Context:
    {context}
    
    Question: {question}
    
    Helpful Answer:"""
    
    QA_PROMPT = PromptTemplate(
        template=template, 
        input_variables=["chat_history", "context", "question"]
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT}
    )
    return qa_chain
