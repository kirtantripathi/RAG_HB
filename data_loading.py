import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import time
import streamlit as st
from langchain_core.documents import Document

def preprocess(text):
    text = text.replace("#","").replace("---","")
    print(text,end="\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    return(text)


def load_document(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    text = preprocess(text)
    file_name = os.path.basename(file_path).replace(".txt","")
    return Document(page_content=text,metadata={"source": file_name})
    

def load_documents(folder_path):
    documents = []
    for i in os.walk(folder_path):
        for j in i[2]:
            documents.append(load_document(os.path.join(i[0],j)))
            print(os.path.join(i[0],j))
    return documents


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " "]
    )
    return splitter.split_documents(documents)


def generate_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_and_save_vectorstore(chunks, embed_model, save_path="basic_cleaning_hash_and_dash_removed_faiss_index"):
    vector_db = FAISS.from_documents(chunks, embed_model)
    vector_db.save_local(save_path)
    print(f"Vectorstore saved at: {save_path}")


def load_data(folder_path):
    print("Loading documents...")
    documents = load_documents(folder_path)

    print("Splitting into chunks per document...")
    splitted_documents = chunk_documents(documents)
    print(f"\n\n\nTotal chunks created: {len(splitted_documents)}")

    print("Generating embeddings...")
    embed_model = generate_embeddings()

    print("Saving FAISS vectorstore...")
    build_and_save_vectorstore(splitted_documents, embed_model)

load_data("Data")
