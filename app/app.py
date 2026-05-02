import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# Load embeddings
embeddings = HuggingFaceEmbeddings()

# Load vectorstore
with open("../data/sample.txt", "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]

vectorstore = FAISS.from_texts(
    texts,
    embeddings
)

retriever = vectorstore.as_retriever()

# Load LLM
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=256
)

llm = HuggingFacePipeline(pipeline=pipe)

# Create QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# UI
st.title("🤖 RAG Chatbot")

query = st.text_input("Ask your question:")

if st.button("Ask"):
    if query:
        answer = qa.run(query)
        st.success(answer)