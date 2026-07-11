import os
import streamlit as st
import streamlit as st
import langchain
import langchain_core

st.write("LangChain:", langchain.__version__)
st.write("LangChain Core:", langchain_core.__version__)

from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(
    page_title="Star Health Insurance Chatbot",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Star Health Insurance RAG Chatbot")
st.write("Ask anything about Star Health Insurance.")

# -----------------------------
# API KEY
# -----------------------------
api_key = st.sidebar.text_input(
    "Enter Google Gemini API Key",
    type="password"
)

if not api_key:
    st.info("Please enter your Gemini API Key.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

# -----------------------------
# Load Documents
# -----------------------------
@st.cache_resource
def load_chain():

    loader = UnstructuredHTMLLoader("PROJECT_10_star_health_CHATBOT/starhealth.html")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    vectorstore = Chroma.from_documents(
        docs,
        embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k":4}
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are an expert assistant for Star Health Insurance.

Answer ONLY using the provided context.

If the answer is unavailable, say:

"I couldn't find that information in the provided policy."

Context:
{context}

Question:
{input}
"""
    )

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    return retrieval_chain

chain = load_chain()

# -----------------------------
# Chat
# -----------------------------
question = st.text_input("Ask a question")

if question:

    with st.spinner("Thinking..."):

        response = chain.invoke(
            {
                "input": question
            }
        )

    st.success(response["answer"])
