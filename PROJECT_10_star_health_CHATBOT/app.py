import streamlit as st
import os
import hashlib
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Star Health RAG Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Design tokens & global styles
# ----------------------------
# Palette:
#   --bg-deep      #08211F  deep clinical teal-black (page background base)
#   --bg-mid       #0E3733  mid teal (gradient stop)
#   --bg-glow      #145C52  emerald glow (gradient stop / accents)
#   --accent       #2FD9A8  signal emerald (primary accent, "vitals" color)
#   --accent-amber #F2B705  amber (used once, for the assistant's pulse dot)
#   --surface      rgba(255,255,255,0.055) glass card surface
#   --text-light   #EAF6F3
#   --text-muted   #9FC2BC
#
# Type: Fraunces (display/editorial) + Inter (body/UI) + JetBrains Mono (retrieved evidence / data)
# Signature element: an animated ECG "vitals" line under the header that draws in once on load —
# a literal reading of "checking the pulse of your policy" rather than a generic gradient hero.

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;0,9..144,700;1,9..144,500&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
:root {
    --bg-deep: #08211F;
    --bg-mid: #0E3733;
    --bg-glow: #145C52;
    --accent: #2FD9A8;
    --accent-amber: #F2B705;
    --surface: rgba(255,255,255,0.055);
    --surface-strong: rgba(255,255,255,0.09);
    --border-soft: rgba(47,217,168,0.22);
    --text-light: #EAF6F3;
    --text-muted: #9FC2BC;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ---------- Page background ---------- */
.stApp {
    background:
        radial-gradient(1200px 600px at 12% -10%, rgba(47,217,168,0.16), transparent 60%),
        radial-gradient(900px 500px at 100% 0%, rgba(20,92,82,0.35), transparent 55%),
        linear-gradient(160deg, var(--bg-deep) 0%, var(--bg-mid) 55%, #0A2624 100%);
    color: var(--text-light);
}

[data-testid="stHeader"] { background: transparent; }

/* ---------- Sidebar ("policy card") ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(190deg, #0B2E2B 0%, #0A2422 100%);
    border-right: 1px solid var(--border-soft);
}
[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }

.sidebar-card {
    background: var(--surface);
    border: 1px solid var(--border-soft);
    border-radius: 16px;
    padding: 16px 18px;
    margin-bottom: 14px;
}
.sidebar-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 6px;
}
.sidebar-dev {
    font-family: 'Fraunces', serif;
    font-size: 17px;
    color: var(--text-light);
}

/* ---------- Header ---------- */
.header-wrap { padding: 8px 4px 6px 4px; }
.header-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 6px;
}
.header-title {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 2.6rem;
    line-height: 1.08;
    color: var(--text-light);
    margin: 0;
}
.header-title em { color: var(--accent); font-style: italic; font-weight: 500; }
.header-sub {
    color: var(--text-muted);
    font-size: 15.5px;
    max-width: 620px;
    margin-top: 10px;
    line-height: 1.55;
}

/* Signature: animated vitals / ECG line */
.vitals-line { width: 100%; max-width: 720px; margin: 18px 0 6px 0; }
.vitals-line path {
    fill: none;
    stroke: var(--accent);
    stroke-width: 2.4;
    stroke-linecap: round;
    stroke-linejoin: round;
    stroke-dasharray: 900;
    stroke-dashoffset: 900;
    animation: draw-vitals 2.1s ease-out forwards;
    filter: drop-shadow(0 0 5px rgba(47,217,168,0.55));
}
@keyframes draw-vitals {
    to { stroke-dashoffset: 0; }
}
@media (prefers-reduced-motion: reduce) {
    .vitals-line path { animation: none; stroke-dashoffset: 0; }
}

/* ---------- Status pill ---------- */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12.5px;
    padding: 7px 14px;
    border-radius: 999px;
    border: 1px solid var(--border-soft);
    background: var(--surface);
    color: var(--text-light);
    margin: 10px 0 18px 0;
}
.status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 8px var(--accent);
}
.status-dot.idle { background: #7a8f8c; box-shadow: none; }

/* ---------- Suggested question chips ---------- */
div[data-testid="column"] .stButton button {
    background: var(--surface);
    border: 1px solid var(--border-soft);
    border-radius: 12px;
    color: var(--text-light);
    font-size: 13.5px;
    padding: 10px 12px;
    width: 100%;
    transition: all 0.15s ease;
}
div[data-testid="column"] .stButton button:hover {
    border-color: var(--accent);
    background: var(--surface-strong);
    color: var(--accent);
}

/* ---------- Chat messages ---------- */
[data-testid="stChatMessage"] {
    background: var(--surface);
    border: 1px solid var(--border-soft);
    border-radius: 16px;
    padding: 4px 6px;
    margin-bottom: 4px;
}

/* ---------- Chat input ---------- */
[data-testid="stChatInput"] textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border-soft) !important;
    color: var(--text-light) !important;
    border-radius: 14px !important;
}

/* ---------- Expander (retrieved evidence) ---------- */
[data-testid="stExpander"] {
    border: 1px solid var(--border-soft);
    border-radius: 12px;
    background: rgba(0,0,0,0.16);
}
.evidence-chunk {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12.5px;
    color: var(--text-muted);
    background: rgba(255,255,255,0.03);
    border-left: 2px solid var(--accent);
    padding: 10px 12px;
    border-radius: 0 8px 8px 0;
    margin-bottom: 10px;
    white-space: pre-wrap;
}
.evidence-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 4px;
    display: block;
}

/* ---------- Divider ---------- */
.soft-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-soft), transparent);
    margin: 22px 0;
}

/* ---------- Footer ---------- */
.footer-text {
    text-align: center;
    color: var(--text-muted);
    font-size: 12.5px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.02em;
    padding: 18px 0 6px 0;
}

/* Links */
a, a:visited { color: var(--accent); }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Header
# ----------------------------
st.markdown('<div class="header-wrap">', unsafe_allow_html=True)
st.markdown('<div class="header-eyebrow">RAG · Retrieval-Augmented Health Insurance Assistant</div>', unsafe_allow_html=True)
st.markdown('<h1 class="header-title">Ask your policy, <em>get a straight answer</em>.</h1>', unsafe_allow_html=True)
st.markdown(
    '<div class="header-sub">Upload a health insurance policy document and ask questions in plain '
    'language. Every answer is grounded in the exact clauses retrieved from your document — '
    'expand any reply to see the evidence.</div>',
    unsafe_allow_html=True,
)
st.markdown("""
<svg class="vitals-line" viewBox="0 0 720 60" xmlns="http://www.w3.org/2000/svg">
  <path d="M0,30 L140,30 L160,8 L180,52 L200,30 L230,30 L245,18 L260,42 L275,30 L320,30
           L500,30 L520,8 L540,52 L560,30 L590,30 L605,18 L620,42 L635,30 L720,30" />
</svg>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-eyebrow">Configuration</div>', unsafe_allow_html=True)
    openai_api_key = st.text_input(
        "OpenAI API key",
        type="password",
        placeholder="sk-...",
        help="Your key is used only for this session and is never stored.",
    )
    uploaded_file = st.file_uploader(
        "Health insurance HTML document",
        type=["html", "htm"],
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file and openai_api_key:
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-eyebrow">Session</div>', unsafe_allow_html=True)
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-eyebrow">Developer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-dev">Vivek Srivastava</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.link_button("🔗 LinkedIn Profile", "https://www.linkedin.com/in/vivek-srivastava-0a878a329", use_container_width=True)
    st.link_button(
        "💻 GitHub Repository",
        "https://github.com/viveksrivastava045-cyber/AIML_projects/blob/main/PROJECT_10_star_health_CHATBOT",
        use_container_width=True,
    )

# ----------------------------
# Build RAG System
# ----------------------------
@st.cache_resource(show_spinner=False)
def build_rag(html_path, file_hash, api_key):
    os.environ["OPENAI_API_KEY"] = api_key

    loader = UnstructuredHTMLLoader(file_path=html_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    splits = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template(
        """
        You are a Health Insurance Assistant.

        Use the following context to answer the question. If the answer is not
        contained in the context, say so plainly instead of guessing.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    )

    return retriever, llm, prompt, len(splits)


def answer_question(query, retriever, llm, prompt):
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    formatted_prompt = prompt.format(context=context, question=query)
    response = llm.invoke(formatted_prompt)
    return response.content, docs


SUGGESTED_QUESTIONS = [
    "What is the sum insured under this policy?",
    "Is maternity coverage included?",
    "What is the waiting period for pre-existing conditions?",
    "What are the key exclusions of this policy?",
]

if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# Main area
# ----------------------------
if uploaded_file and openai_api_key:

    temp_path = os.path.join("/tmp", uploaded_file.name)
    file_bytes = uploaded_file.getbuffer()
    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    file_hash = hashlib.sha256(bytes(file_bytes)).hexdigest()

    try:
        with st.spinner("🔬 Reading document and building knowledge base..."):
            retriever, llm, prompt, chunk_count = build_rag(temp_path, file_hash, openai_api_key)
    except Exception as e:
        st.error(f"Couldn't build the knowledge base from this file: {e}")
        st.stop()

    st.markdown(
        f'<div class="status-pill"><span class="status-dot"></span>'
        f'Knowledge base ready · {chunk_count} chunks indexed from '
        f'<b>{uploaded_file.name}</b></div>',
        unsafe_allow_html=True,
    )

    # Suggested questions (only before the first message, to keep the UI calm)
    if not st.session_state.messages:
        st.caption("Try asking:")
        cols = st.columns(len(SUGGESTED_QUESTIONS))
        clicked_question = None
        for col, q in zip(cols, SUGGESTED_QUESTIONS):
            with col:
                if st.button(q, key=f"chip_{q}"):
                    clicked_question = q
    else:
        clicked_question = None

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📚 View retrieved evidence"):
                    for i, chunk in enumerate(msg["sources"], start=1):
                        st.markdown(f'<span class="evidence-label">Chunk {i}</span>', unsafe_allow_html=True)
                        st.markdown(f'<div class="evidence-chunk">{chunk[:1000]}</div>', unsafe_allow_html=True)

    typed_question = st.chat_input("Ask a question about the policy...")
    user_query = clicked_question or typed_question

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("🩺 Checking the policy..."):
                try:
                    answer, docs = answer_question(user_query, retriever, llm, prompt)
                except Exception as e:
                    answer, docs = f"Something went wrong while answering: {e}", []
            st.markdown(answer)
            sources = [d.page_content for d in docs]
            if sources:
                with st.expander("📚 View retrieved evidence"):
                    for i, chunk in enumerate(sources, start=1):
                        st.markdown(f'<span class="evidence-label">Chunk {i}</span>', unsafe_allow_html=True)
                        st.markdown(f'<div class="evidence-chunk">{chunk[:1000]}</div>', unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        st.rerun()

else:
    st.markdown(
        f'<div class="status-pill"><span class="status-dot idle"></span>'
        f'Waiting for API key and document upload</div>',
        unsafe_allow_html=True,
    )
    st.info("👈 Enter your OpenAI API key and upload a health insurance HTML document in the sidebar to begin.")

# ----------------------------
# Footer
# ----------------------------
st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="footer-text">Built with Streamlit · LangChain · ChromaDB · OpenAI</div>',
    unsafe_allow_html=True,
)
