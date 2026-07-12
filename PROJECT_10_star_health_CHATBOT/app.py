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
    st.title("🏥 Health Insurance RAG Chatbot")
    # page_title="🏥 Health Insurance RAG Chatbot",
    page_icon="🏥",
    layout="wide"
)

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.stApp{
    background: linear-gradient(135deg,#edf4ff,#f8fbff,#ffffff);
}

/* Header */

.main-title{
    text-align:center;
    padding:25px;
    border-radius:20px;
    background:linear-gradient(90deg,#0072ff,#00c6ff);
    color:white;
    box-shadow:0px 10px 30px rgba(0,0,0,.15);
    margin-bottom:20px;
}

.main-title h1{
    font-size:42px;
    font-weight:800;
    margin:0;
}

.main-title p{
    font-size:18px;
    margin-top:8px;
}

/* Sidebar */

section[data-testid="stSidebar"]{
    background:#0f172a;
}

section[data-testid="stSidebar"] *{
    color:white;
}

/* Inputs */

.stTextInput input{
    border-radius:12px;
    border:2px solid #dbeafe;
    padding:12px;
    font-size:17px;
}

.stTextInput input:focus{
    border:2px solid #2563eb;
}

/* Buttons */

.stButton button{

    width:100%;
    border:none;
    border-radius:12px;

    background:linear-gradient(90deg,#2563eb,#06b6d4);

    color:white;

    font-weight:700;

    padding:12px;

    transition:.3s;

}

.stButton button:hover{

transform:scale(1.02);

box-shadow:0px 10px 20px rgba(0,0,0,.2);

}

/* Answer Card */

.answer-box{

padding:20px;

background:white;

border-radius:18px;

box-shadow:0 8px 20px rgba(0,0,0,.08);

border-left:8px solid #2563eb;

font-size:18px;

line-height:1.8;

}

/* Info Card */

.info-card{

background:white;

padding:20px;

border-radius:18px;

box-shadow:0px 6px 18px rgba(0,0,0,.08);

margin-top:15px;

}

/* Footer */

.footer{

text-align:center;

padding:20px;

font-size:15px;

color:#64748b;

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
with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/2966/2966485.png",
        width=80
    )

    st.title("Configuration")

    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password"
    )

    uploaded_file = st.file_uploader(
        "Upload HTML Policy",
        type=["html","htm"]
    )

    st.divider()

    st.markdown("### 👨‍💻 Developer")

    st.write("**VIVEK KUMAR SRIVASTAVA**")

    col1,col2=st.columns(2)

    with col1:
        st.link_button(
            "GitHub",
            "https://github.com/viveksrivastva045-cyber"
        )

    with col2:
        st.link_button(
            "LinkedIn",
            "https://www.linkedin.com/in/vivek-srivastava-0a878a329/"
        )
# st.sidebar.header("⚙️ Configuration")

# openai_api_key = st.sidebar.text_input(
#     "Enter OpenAI API Key",
#     type="password"
# )

# uploaded_file = st.sidebar.file_uploader(
#     "Upload Health Insurance HTML File",
#     type=["html", "htm"]
# )

# st.sidebar.markdown("---")
# st.sidebar.subheader("👨‍💻 Developer: Richeek Pandey")

# st.sidebar.link_button(
#     "🔗 LinkedIn Profile",
#     "https://www.linkedin.com/in/richeek-pandey-9954783a9/"
# )

# st.sidebar.link_button(
#     "💻 GitHub Profile",
#     "https://github.com/richeekpandey07"
# )

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

    temp_path = "Types of Health Insurance Plans.html"

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
