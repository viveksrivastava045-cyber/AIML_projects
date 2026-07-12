import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="🏥 Health Insurance RAG Chatbot",
    page_icon="🏥",
    layout="center"
)

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}
.stTextInput > div > div > input {
    border-radius: 10px;
}
.chat-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #ffffff;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Header
# ----------------------------
st.title("🏥 Health Insurance RAG Chatbot")
st.markdown(
    """
    Ask questions about health insurance policies and get intelligent answers using
    **RAG (Retrieval Augmented Generation)** powered by LangChain + OpenAI.
    """
)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("⚙️ Configuration")

openai_api_key = st.sidebar.text_input(
    "Enter OpenAI API Key",
    type="password"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Health Insurance HTML File",
    type=["html", "htm"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("👨‍💻 Developer: VIVEK SRIVASTAVA")

st.sidebar.link_button(
    "🔗 LinkedIn Profile",
    "www.linkedin.com/in/vivek-srivastava-0a878a329"
)

st.sidebar.link_button(
    "💻 GitHub Profile",
    "viveksrivastava045-cyber"
)

# ----------------------------
# Build RAG System
# ----------------------------
@st.cache_resource
def build_rag(html_path, api_key):

    os.environ["OPENAI_API_KEY"] = api_key

    loader = UnstructuredHTMLLoader(file_path=html_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    splits = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(
        """
        You are a Health Insurance Assistant.

        Use the following context to answer the question.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    )

    return retriever, llm, prompt


# ----------------------------
# File Processing
# ----------------------------
if uploaded_file and openai_api_key:

    temp_path = "starhealth.html"

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("🔄 Building Knowledge Base..."):
        retriever, llm, prompt = build_rag(
            temp_path,
            openai_api_key
        )

    st.success("✅ Knowledge Base Ready!")

    query = st.text_input(
        "💬 Ask a question about the insurance policy"
    )

    if query:

        with st.spinner("🤖 Thinking..."):

            docs = retriever.invoke(query)

            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            formatted_prompt = prompt.format(
                context=context,
                question=query
            )

            response = llm.invoke(formatted_prompt)

        st.markdown("### 📝 Answer")

        st.markdown(
            f"""
            <div class="chat-box">
            {response.content}
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("📚 Retrieved Context"):
            for i, doc in enumerate(docs, start=1):
                st.write(f"**Chunk {i}**")
                st.write(doc.page_content[:1000])

else:
    st.info(
        "👈 Upload an HTML insurance document and enter your OpenAI API key."
    )

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown(
    """
    Developed with ❤️ using Streamlit, LangChain, ChromaDB & OpenAI
    """
)
