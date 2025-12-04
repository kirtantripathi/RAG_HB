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
import cohere

key = open("keys.txt").read()
cohere_key = open("cohere_key.txt").read()
co = cohere.ClientV2(api_key=cohere_key)
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# db = FAISS.load_local("faiss_index", embed_model, allow_dangerous_deserialization=True)
model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, api_key=key)

prompt_template = PromptTemplate(template="""You are an AI Assistant for Sunridge Institute of Technology (SIT).Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question:{query}
Context:{context}
Answer:
""", input_variables=["context","query"])

metadata_filtering_prompt = PromptTemplate(template="""You are an Question enhancer for Sunridge Institute of Technology (SIT) related questions. Please enhance this question such that the question can be able to retrive the actual context from the text chunks using semantic sreach. Only return the enhanced question.

Rules:
- If the query is clearly a greeting, chit-chat, short acknowledgement (like "hi", "hello", "thanks", "ok"), **do NOT enhance it. Return it exactly as is.**
- If the query already contains enough context or is a direct question, **keep it mostly unchanged** and only clarify slightly if needed.
- If the query is missing detail but clearly refers to something about SIT, rewrite it so that it includes implied context (departments, campus details, academics, policies, etc.)
- NEVER invent content or assumptions that are not present in the original query.

Output:
Return ONLY the final query text.

Original Query: {query}""",input_variables=["query"])
# metadata_filtering_prompt = PromptTemplate(template="""You are an Question enhancer for Sunridge Institute of Technology (SIT) related questions. Please enhance this question such that the question can be able to retrive the actual context from the text chunks using semantic sreach. Only return the enhanced question. \n Question: {query}""",input_variables=["query"])

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
async def ask(request: Query):
    print("Original Query:",request.message)
    enhanced_query_prompt = metadata_filtering_prompt.invoke({"query":request.message})
    enhanced_query = model.invoke(enhanced_query_prompt)
    print("Enhanced Query:",enhanced_query.content)

    results = db.similarity_search(enhanced_query.content, k=10)
    # results2 = db.similarity_search_with_relevance_scores(request.message, k =3)
    
    list_of_docs = []
    for doc in results:
        list_of_docs.append(doc.page_content)
    
    response = co.rerank(
        model="rerank-v3.5",
        query=request.message,
        documents=list_of_docs,
        top_n=5,
    )
    filtered_docs = []
    for i in response.results:
        print(i.index)
        filtered_docs.append(list_of_docs[i.index])
    print(filtered_docs)

    # print(results)
    context_text = "\n\n".join(
        f"[{i+1}] {doc}" for i, doc in enumerate(filtered_docs)
    )
    
    p = prompt_template.invoke({"context":context_text,"query":request.message})
    response = model.invoke(p)
    return JSONResponse(content={"results": response.content})