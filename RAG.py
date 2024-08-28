import streamlit as st
import os
import sys
import time
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
from exception import CustomException

def configure():
    load_dotenv()

configure()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")

llm = ChatGroq(api_key=GROQ_API_KEY, model="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Question:{input}
    """
)

def create_vector_embedding():
    try: 
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        loader=PyPDFDirectoryLoader("research_papers") #Data ingestion
        docs=loader.load() #Document loading
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        final_document=text_splitter.split_documents(docs)
        vectors=FAISS.from_documents(final_document,embeddings)
        return vectors
    except Exception as e:
        raise CustomException(e,sys)
      


if __name__=="__main__":
    print("vector embedding started")
    vectors = create_vector_embedding()
    print("vector embedding completed")

    document_chain=create_stuff_documents_chain(llm=llm, prompt=prompt)
    retriever=vectors.as_retriever()
    retrival_chain=create_retrieval_chain(retriever,document_chain)
    response=retrival_chain.invoke({'input':'what is llm?'})
    print(response['answer'])