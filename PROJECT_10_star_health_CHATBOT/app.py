import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

# ----------------------------
# Load API Key
# ----------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("PROJECT_10_star_health_CHATBOT/GOOGLE_API_KEY.env")

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="📚 AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Document Chatbot")
st.markdown(
    "Ask questions from your uploaded documents using **Gemini + ChromaDB**."
)

# ----------------------------
# Load Embedding Model
# ----------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# ----------------------------
# Load Vector Database
# ----------------------------

persist_directory = "vectorstore"

vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)

retriever = vectordb.as_retriever(
    search_kwargs={"k":4}
)

# ----------------------------
# LLM
# ----------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=GOOGLE_API_KEY
)

# ----------------------------
# Prompt
# ----------------------------

template = """
You are an intelligent AI assistant.

Use ONLY the provided context.

If the answer isn't available,
say

"I couldn't find that information in the document."

Context:
{context}

Question:
{question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

# ----------------------------
# Session State
# ----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# Chat History
# ----------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------------------
# User Input
# ----------------------------

question = st.chat_input("Ask your question...")

if question:

    st.session_state.messages.append(
        {"role":"user","content":question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Thinking..."):

        result = qa.invoke({"query":question})

        answer = result["result"]

        with st.chat_message("assistant"):
            st.markdown(answer)

            if result["source_documents"]:

                with st.expander("📄 Source Chunks"):

                    for doc in result["source_documents"]:

                        st.write(doc.page_content)
                        st.divider()

        st.session_state.messages.append(
            {"role":"assistant","content":answer}
        )
