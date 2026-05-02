# 🤖 RAG Chatbot (Retrieval-Augmented Generation)

## 📌 Overview
This project builds an AI-powered chatbot using **Retrieval-Augmented Generation (RAG)**.

The chatbot can answer questions based on custom documents by combining:
- Vector search (semantic retrieval)
- Large Language Models (LLMs)

---

## 🎯 Objective
- Load and process custom documents
- Convert text into embeddings
- Store embeddings in a vector database (FAISS)
- Retrieve relevant context
- Generate accurate answers using an LLM
- Deploy as a Streamlit chatbot

---

## 🧠 How It Works

1️⃣ Document Loading  
- Load text data from files  

2️⃣ Text Splitting  
- Break large text into smaller chunks  

3️⃣ Embedding Creation  
- Convert text into numerical vectors using HuggingFace  

4️⃣ Vector Store  
- Store embeddings in FAISS for fast similarity search  

5️⃣ Retrieval  
- Find relevant text chunks based on user query  

6️⃣ Generation  
- Use LLM to generate answers from retrieved data  

---

## ⚙️ Tech Stack

- Python  
- LangChain  
- FAISS (Vector Database)  
- HuggingFace Transformers  
- Sentence Transformers  
- Streamlit  


## 📁 Project Structure
