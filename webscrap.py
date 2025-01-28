import requests
from bs4 import BeautifulSoup
import streamlit as st
import os
import time
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document  
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

def scrape_web_page(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        paragraphs = soup.find_all('p')
        text = "\n".join([para.get_text() for para in paragraphs])
        return text
    except Exception as e:
        return f"Error: {e}"

st.title("Web Scraping and RAG")

url = st.text_input("Enter the website URL to analyze")

if url:
    if "vector" not in st.session_state or st.session_state.get("last_url") != url:
        scraped_text = scrape_web_page(url)
   
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.docs = st.session_state.text_splitter.split_text(scraped_text)
        
        # Convert the text chunks into Document objects and embed
        st.session_state.documents = [Document(page_content=doc) for doc in st.session_state.docs]
        st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        st.session_state.vector = FAISS.from_documents(st.session_state.documents, st.session_state.embeddings)
        st.session_state.last_url = url
        
    # Initialize Groq model
    llm = ChatGroq(groq_api_key="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", model_name="gemma2-9b-it")

    # Define prompt template
    prompt = ChatPromptTemplate.from_template('''
    Answer the questions based on the provided context.
    Please provide accurate response based on the question.
    You can also add the context on your own to enhance user experience.
    Also add some emojis or experssions to make it more user appealing.
    Behave more like human than ai.
    Be talkative
    <context>
    {context}
    Question: {input}
    ''')

    # Create document chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Take user input for prompt
    prompt_input = st.text_input("Input your prompt here")

    if prompt_input:
        start = time.process_time()
        response = retrieval_chain.invoke({"input": prompt_input})
        st.write("Response time: ", time.process_time() - start)
        st.write(response['answer'])

        # Show similar documents
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write("----------------------------")
