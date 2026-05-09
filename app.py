# import streamlit as st

# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_classic.chains import RetrievalQA
# from langchain_community.llms import HuggingFacePipeline
# from transformers import pipeline

# # Load embeddings
# embeddings = HuggingFaceEmbeddings()

# # Load vectorstore
# with open("../data/sample.txt", "r", encoding="utf-8") as f:
#     texts = [line.strip() for line in f if line.strip()]

# vectorstore = FAISS.from_texts(
#     texts,
#     embeddings
# )

# retriever = vectorstore.as_retriever()

# # Load LLM
# pipe = pipeline(
#     "text2text-generation",
#     model="google/flan-t5-base",
#     max_length=256
# )

# llm = HuggingFacePipeline(pipeline=pipe)

# # Create QA chain
# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever
# )

# # UI
# st.title("🤖 RAG Chatbot")

# query = st.text_input("Ask your question:")

# if st.button("Ask"):
#     if query:
#         answer = qa.run(query)
#         st.success(answer)



# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# import streamlit as st

# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_classic.chains import RetrievalQA
# from langchain_community.llms import HuggingFacePipeline
# from transformers import pipeline

# # Page config
# st.set_page_config(page_title="RAG Chatbot", layout="wide")

# st.title("🤖 AI Chatbot (RAG)")

# # Load embeddings
# embeddings = HuggingFaceEmbeddings()

# # Load vectorstore
# import os
# if os.path.exists("vectorstore"):
#     vectorstore = FAISS.load_local(
#         "vectorstore",
#         embeddings,
#         allow_dangerous_deserialization=True
#     )
# else:
#     with open("../data/sample.txt", "r", encoding="utf-8") as f:
#         texts = [line.strip() for line in f if line.strip()]
#     vectorstore = FAISS.from_texts(texts, embeddings)

# retriever = vectorstore.as_retriever()

# # Load model
# pipe = pipeline(
#     "text2text-generation",
#     model="google/flan-t5-base",
#     max_length=256
# )

# llm = HuggingFacePipeline(pipeline=pipe)

# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever
# )

# # 🧠 Chat memory
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat history
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# # User input
# query = st.chat_input("Ask your question...")

# if query:
#     # Show user message
#     st.chat_message("user").markdown(query)
#     st.session_state.messages.append({"role": "user", "content": query})

#     # Generate response
#     response = qa.run(query)

#     # Show assistant message
#     with st.chat_message("assistant"):
#         st.markdown(response)

#     st.session_state.messages.append({"role": "assistant", "content": response})



# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# add PDF READER....................................................................

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from transformers import pipeline
import tempfile

st.set_page_config(page_title="PDF Chatbot", layout="wide")

st.title("📄 Chat with your PDF")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings()

    # Create vectorstore
    vectorstore = FAISS.from_documents(texts, embeddings)

    retriever = vectorstore.as_retriever()

    # LLM
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=256,
        truncation=True
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    # Chat memory
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    query = st.chat_input("Ask something from your PDF...")

    if query:
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        response = qa.invoke({"query": query})["result"]

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
