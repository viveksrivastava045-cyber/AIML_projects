import streamlit as st
import os
import hashlib
import tempfile
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Health Insurance RAG Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# GLOBAL STYLING — gradients, glass cards, chat bubbles
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Animated full-page gradient background */
    .stApp {
        background: linear-gradient(-45deg, #0f2027, #2c5364, #6a11cb, #2575fc);
        background-size: 400% 400%;
        animation: gradientShift 18s ease infinite;
    }

    @keyframes gradientShift {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    /* Sidebar gradient */
    section[data-testid="stSidebar"] {
        background: linear-gradient(160deg, #1f1c2c, #928dab);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    section[data-testid="stSidebar"] * {
        color: #f5f5f5 !important;
    }

    /* Hero title */
    .hero-title {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 2.6rem;
        text-align: center;
        background: linear-gradient(90deg, #f7971e, #ffd200, #21d4fd, #b721ff);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 8s ease infinite;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        text-align: center;
        color: rgba(255,255,255,0.85);
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    /* Glassmorphism card */
    .glass-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
    }

    /* Status pill */
    .status-pill {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }
    .status-ready {
        background: linear-gradient(90deg, #11998e, #38ef7d);
        color: #06251f;
    }
    .status-waiting {
        background: linear-gradient(90deg, #f7971e, #ffd200);
        color: #3a2a00;
    }

    /* Chat bubbles */
    .chat-user {
        background: linear-gradient(135deg, #6a11cb, #2575fc);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 85%;
        margin-left: auto;
        box-shadow: 0 4px 14px rgba(37, 117, 252, 0.35);
    }
    .chat-bot {
        background: rgba(255, 255, 255, 0.92);
        color: #1a1a2e;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 85%;
        margin-right: auto;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
        line-height: 1.55;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #6a11cb, #2575fc);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(106, 17, 203, 0.45);
    }

    /* Input fields */
    .stTextInput > div > div > input,
    .stTextArea textarea {
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.25) !important;
        background: rgba(255,255,255,0.08) !important;
        color: white !important;
    }

    /* Footer */
    .footer-text {
        text-align: center;
        color: rgba(255,255,255,0.75);
        font-size: 0.9rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.15);
    }

    div[data-testid="stExpander"] {
        background: rgba(255,255,255,0.06);
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.12);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="hero-title">🏥 Health Insurance RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Ask anything about your policy — powered by '
    'Retrieval Augmented Generation with LangChain &amp; OpenAI</div>',
    unsafe_allow_html=True,
)

# =========================================================
# SESSION STATE
# =========================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of dicts: {question, answer, sources}
if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False
if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

# =========================================================
# SIDEBAR — Configuration
# =========================================================
with st.sidebar:
    st.header("⚙️ Configuration")

    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Your key is used only for this session and never stored.",
    )

    uploaded_file = st.file_uploader(
        "Upload Health Insurance HTML File",
        type=["html", "htm"],
    )

    with st.expander("🔧 Retrieval Settings"):
        chunk_size = st.slider("Chunk size", 300, 2000, 1000, step=100)
        chunk_overlap = st.slider("Chunk overlap", 0, 400, 200, step=50)
        top_k = st.slider("Chunks to retrieve (k)", 1, 10, 4)

    st.markdown("---")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.subheader("👨‍💻 Developer")
    st.markdown("**Vivek Srivastava**")
    st.link_button("🔗 LinkedIn Profile", "https://www.linkedin.com/in/vivek-srivastava-0a878a329")
    st.link_button(
        "💻 GitHub Profile",
        "https://github.com/viveksrivastava045-cyber/AIML_projects/blob/main/PROJECT_10_star_health_CHATBOT",
    )

# =========================================================
# RAG PIPELINE BUILDER (cached per file content + settings)
# =========================================================
@st.cache_resource(show_spinner=False)
def build_rag(file_bytes, file_hash, api_key, chunk_size, chunk_overlap, top_k):
    """Builds retriever + llm + prompt. Cached per unique file content & settings."""
    os.environ["OPENAI_API_KEY"] = api_key

    # Write to a uniquely named temp file so re-uploads never collide
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, f"{file_hash}.html")
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)

    loader = UnstructuredHTMLLoader(file_path=tmp_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splits = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template(
        """
        You are a knowledgeable and friendly Health Insurance Assistant.
        Answer the question using ONLY the context provided below.
        If the answer isn't in the context, say you don't have that information
        in the uploaded document instead of guessing.

        Context:
        {context}

        Question:
        {question}

        Answer clearly and concisely, using bullet points where helpful:
        """
    )

    return retriever, llm, prompt


# =========================================================
# MAIN AREA
# =========================================================
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    if uploaded_file and openai_api_key:
        st.markdown('<span class="status-pill status-ready">✅ Ready</span>', unsafe_allow_html=True)
        st.write(f"**File:** {uploaded_file.name}")
    elif uploaded_file or openai_api_key:
        st.markdown('<span class="status-pill status-waiting">⏳ Almost there</span>', unsafe_allow_html=True)
        missing = "OpenAI API key" if not openai_api_key else "HTML file"
        st.write(f"Please also provide: **{missing}**")
    else:
        st.markdown('<span class="status-pill status-waiting">⏳ Waiting for input</span>', unsafe_allow_html=True)
        st.write("Upload a policy document and enter your API key in the sidebar to get started.")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.write("**💡 Try asking:**")
    st.write("- What does this policy cover?")
    st.write("- What is the waiting period for pre-existing diseases?")
    st.write("- What is excluded from this plan?")
    st.markdown("</div>", unsafe_allow_html=True)

retriever = llm = prompt = None

if uploaded_file and openai_api_key:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    try:
        with st.spinner("🔄 Building knowledge base from your document..."):
            retriever, llm, prompt = build_rag(
                file_bytes, file_hash, openai_api_key, chunk_size, chunk_overlap, top_k
            )
        st.session_state.rag_ready = True
        st.session_state.file_hash = file_hash
    except Exception as e:
        st.session_state.rag_ready = False
        st.error(f"⚠️ Failed to build knowledge base: {e}")

st.markdown("### 💬 Chat")

# Render existing chat history as styled bubbles
for turn in st.session_state.chat_history:
    st.markdown(f'<div class="chat-user">🙋 {turn["question"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="chat-bot">🤖 {turn["answer"]}</div>', unsafe_allow_html=True)
    if turn.get("sources"):
        with st.expander("📚 View retrieved context for this answer"):
            for i, chunk in enumerate(turn["sources"], start=1):
                st.markdown(f"**Chunk {i}**")
                st.write(chunk[:1000])

# Chat input pinned at the bottom
if st.session_state.rag_ready:
    query = st.chat_input("Ask a question about the insurance policy...")
    if query:
        try:
            with st.spinner("🤖 Thinking..."):
                docs = retriever.invoke(query)
                context = "\n\n".join(doc.page_content for doc in docs)
                formatted_prompt = prompt.format(context=context, question=query)
                response = llm.invoke(formatted_prompt)

            st.session_state.chat_history.append(
                {
                    "question": query,
                    "answer": response.content,
                    "sources": [d.page_content for d in docs],
                }
            )
            st.rerun()
        except Exception as e:
            st.error(f"⚠️ Something went wrong while answering: {e}")
else:
    st.chat_input("Upload a document and enter your API key to start chatting...", disabled=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    '<div class="footer-text">Built with ❤️ using Streamlit, LangChain, ChromaDB &amp; OpenAI</div>',
    unsafe_allow_html=True,
)
#------------------------------
