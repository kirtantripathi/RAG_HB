from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from fastapi.responses import JSONResponse
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

key = open("keys.txt").read()
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# db = FAISS.load_local("faiss_index", embed_model, allow_dangerous_deserialization=True)
model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, api_key=key)

prompt_template = PromptTemplate(template="""You are an AI Assistant for Sunridge Institute of Technology (SIT).Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question:{query}
Context:{context}
Answer:
""", input_variables=["context","query"])

db = FAISS.load_local(
    folder_path="faiss_index",
    embeddings=embed_model,
    index_name="index",
    allow_dangerous_deserialization=True
)

app = FastAPI()

class Query(BaseModel):
    message: str

@app.get("/")
def home():
    return {"message":"welcome vats!!"}

@app.post("/ask")
def ask(request: Query):
    results = db.similarity_search(request.message, k=3)
    print(results)
    formatted = [
        {
            "content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in results
    ]
    p = prompt_template.invoke({"context":results,"query":request.message})
    response = model.invoke(p)
    return JSONResponse(content={"results": response.content})