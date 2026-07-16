import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Eye Gender Detection | Deep Learning",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# Custom CSS
# ------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Outfit:wght@600;700;800&display=swap" rel="stylesheet">
<style>
    :root {
        --bg: #0B1020;
        --bg-soft: #121933;
        --panel: rgba(18, 25, 51, 0.78);
        --panel-strong: rgba(15, 23, 42, 0.92);
        --surface: #F8FAFC;
        --surface-2: #E2E8F0;
        --text: #E5EEF8;
        --text-muted: #A8B3C7;
        --text-dark: #0F172A;
        --primary: #14B8A6;
        --primary-strong: #0F766E;
        --secondary: #38BDF8;
        --accent: #8B5CF6;
        --success: #16A34A;
        --warning: #F59E0B;
        --female-bg: linear-gradient(135deg, rgba(244, 114, 182, 0.18), rgba(139, 92, 246, 0.18));
        --female-border: rgba(244, 114, 182, 0.35);
        --male-bg: linear-gradient(135deg, rgba(56, 189, 248, 0.18), rgba(20, 184, 166, 0.16));
        --male-border: rgba(56, 189, 248, 0.35);
        --shadow: 0 18px 50px rgba(2, 8, 23, 0.35);
        --radius-lg: 22px;
        --radius-md: 16px;
        --radius-sm: 12px;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        color: var(--text);
        background:
            radial-gradient(circle at top left, rgba(56, 189, 248, 0.18), transparent 28%),
            radial-gradient(circle at top right, rgba(139, 92, 246, 0.18), transparent 24%),
            linear-gradient(145deg, #08111F 0%, #0B1020 42%, #111C35 100%);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #111827 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.16);
    }
    section[data-testid="stSidebar"] * {
        color: #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] p {
        color: #CBD5E1 !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(148, 163, 184, 0.18);
    }

    .hero {
        position: relative;
        overflow: hidden;
        border-radius: 28px;
        padding: 2.5rem 2rem 2rem 2rem;
        margin-bottom: 1.2rem;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(17, 24, 39, 0.78));
        border: 1px solid rgba(148, 163, 184, 0.14);
        box-shadow: var(--shadow);
        text-align: center;
    }
    .hero::before {
        content: "";
        position: absolute;
        inset: auto -60px -70px auto;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, rgba(20, 184, 166, 0.35), transparent 65%);
        pointer-events: none;
    }
    .hero::after {
        content: "";
        position: absolute;
        inset: -70px auto auto -60px;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, rgba(139, 92, 246, 0.22), transparent 65%);
        pointer-events: none;
    }
    .hero-title {
        position: relative;
        z-index: 1;
        font-family: 'Outfit', sans-serif;
        font-size: clamp(2rem, 2.7vw, 3.4rem);
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.05;
        color: #F8FAFC;
        margin-bottom: 0.55rem;
    }
    .hero-title .highlight {
        background: linear-gradient(90deg, #67E8F9 0%, #2DD4BF 45%, #A78BFA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-subtitle {
        position: relative;
        z-index: 1;
        max-width: 760px;
        margin: 0 auto;
        color: var(--text-muted);
        font-size: 1rem;
        line-height: 1.7;
    }

    .badge-row {
        position: relative;
        z-index: 1;
        margin-top: 1rem;
    }
    .badge {
        display: inline-block;
        margin: 0.3rem 0.35rem 0 0.35rem;
        padding: 0.5rem 0.95rem;
        border-radius: 999px;
        background: rgba(148, 163, 184, 0.12);
        border: 1px solid rgba(148, 163, 184, 0.18);
        color: #E2E8F0;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.01em;
    }

    .section-title, h3 {
        font-family: 'Outfit', sans-serif !important;
        color: #F8FAFC !important;
        letter-spacing: -0.02em;
    }

    .card {
        background: var(--panel);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: var(--radius-lg);
        padding: 1.35rem;
        box-shadow: var(--shadow);
        height: 100%;
    }
    .card h4, .card h5 {
        font-family: 'Outfit', sans-serif;
        color: #F8FAFC;
        margin-bottom: 0.35rem;
        letter-spacing: -0.02em;
    }
    .card p, .card li, .card .stCaption {
        color: var(--text-muted);
    }

    div[data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(30, 41, 59, 0.78));
        border: 2px dashed rgba(45, 212, 191, 0.45);
        border-radius: 20px;
        padding: 1rem;
    }
    div[data-testid="stFileUploaderDropzone"] * {
        color: #DDEAF7 !important;
    }
    div[data-testid="stFileUploaderDropzone"]:hover {
        border-color: #67E8F9;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(17, 24, 39, 0.92));
    }
    div[data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(90deg, var(--primary), var(--secondary)) !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        font-weight: 700 !important;
    }

    .result-box {
        text-align: center;
        padding: 1.5rem;
        border-radius: 20px;
        margin-top: 0.5rem;
        border: 1px solid transparent;
    }
    .result-female {
        background: var(--female-bg);
        border-color: var(--female-border);
    }
    .result-male {
        background: var(--male-bg);
        border-color: var(--male-border);
    }
    .result-label {
        font-family: 'Outfit', sans-serif;
        font-size: clamp(1.8rem, 2.1vw, 2.4rem);
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 0.2rem;
        letter-spacing: -0.03em;
    }
    .result-sub {
        font-size: 0.98rem;
        color: #D7E4F3;
    }

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 16px;
        padding: 0.9rem 1rem;
        box-shadow: 0 10px 24px rgba(2, 8, 23, 0.25);
    }
    div[data-testid="stMetricLabel"] {
        color: #B6C2D3 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #7DD3FC;
        font-family: 'Outfit', sans-serif;
    }

    div[data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, var(--primary), var(--secondary), var(--accent)) !important;
    }

    .stButton > button {
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        padding: 0.6rem 1.2rem;
        box-shadow: 0 10px 22px rgba(20, 184, 166, 0.24);
    }
    .stButton > button:hover {
        color: white;
        transform: translateY(-1px);
        box-shadow: 0 14px 28px rgba(20, 184, 166, 0.32);
    }

    div[data-testid="stAlert"] {
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.12);
    }

    .footer-note {
        text-align: center;
        color: #94A3B8;
        font-size: 0.9rem;
        padding-top: 1rem;
    }

    .dev-name {
        display: inline-block;
        padding: 0.45rem 0.85rem;
        border-radius: 999px;
        background: rgba(20, 184, 166, 0.14);
        color: #DFFCF7;
        border: 1px solid rgba(45, 212, 191, 0.24);
        font-weight: 700;
        margin-bottom: 0.7rem;
    }

    .muted-text {
        color: var(--text-muted);
    }

    @media (max-width: 768px) {
        .hero {
            padding: 1.6rem 1rem 1.4rem 1rem;
            border-radius: 22px;
        }
        .hero-subtitle {
            font-size: 0.95rem;
        }
        .card {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Hero Header
# ------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-title">👁️ <span class="highlight">Eye Gender Detection</span></div>
    <div class="hero-subtitle">Upload a clear close-up eye image and get a polished deep learning prediction with readable confidence details and a cleaner visual experience.</div>
    <div class="badge-row">
        <span class="badge">TensorFlow</span>
        <span class="badge">Streamlit</span>
        <span class="badge">CNN Classifier</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------
# Sidebar
# ------------------------------
with st.sidebar:
    st.header("ℹ️ About this app")
    st.write(
        "This app uses a trained convolutional neural network to classify "
        "whether an uploaded eye image belongs to a **male** or **female**."
    )
    st.markdown("---")
    st.subheader("📌 How to use")
    st.markdown(
        "1. Upload a clear, close-up eye image (jpg/png)\n"
        "2. Wait a moment while the model analyzes it\n"
        "3. Review the predicted label and confidence score"
    )
    st.markdown("---")
    st.subheader("⚙️ Model info")
    st.write("Input size: `299 x 299`")
    st.write("Framework: `TensorFlow / Keras`")
    st.markdown("---")
    confidence_threshold = st.slider(
        "Minimum confidence to trust prediction (%)",
        min_value=50, max_value=99, value=60, step=1
    )

# ------------------------------
# Load Model
# ------------------------------
# MODEL_PATH = "PROJECT_09_MALE_FEMALE_EYE_DETECTION/my_model.keras"

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model = tf.keras.models.load_model("PROJECT_09_MALE_FEMALE_EYE_DETECTION/my_model.keras")
        return model, None
    except Exception as e:
        return None, str(e)

with st.spinner("Loading model..."):
    model, load_error = load_model()

if load_error:
    st.error(
        f"❌ Could not load the model from `{MODEL_PATH}`.\n\n"
        f"Error: {load_error}\n\n"
        "Please check that the model file exists at this path in the repo."
    )
    st.stop()

# ------------------------------
# Prediction Function
# ------------------------------
def predict(img: Image.Image):
    img = img.convert("RGB").resize((299, 299))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    prediction = model.predict(arr, verbose=0)
    probability = float(prediction[0][0])
    if probability > 0.5:
        label = "Female Eye"
        emoji = "👩"
        confidence = probability
    else:
        label = "Male Eye"
        emoji = "👨"
        confidence = 1 - probability
    return label, emoji, confidence

# ------------------------------
# Upload & Predict
# ------------------------------
st.markdown("### 📤 Upload Your Image")
uploaded = st.file_uploader(
    "Drag and drop or browse a jpg/jpeg/png eye image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded is not None:
    try:
        image = Image.open(uploaded)
    except Exception:
        st.error("⚠️ Couldn't read this file as an image. Please upload a valid jpg/jpeg/png file.")
        st.stop()

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🖼️ Uploaded Image")
        st.image(image, use_container_width=True)
        st.caption(f"File: {uploaded.name} • {image.size[0]}×{image.size[1]} px")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🔮 Prediction")

        progress_bar = st.progress(0, text="Analyzing image...")
        for pct in (22, 47, 74, 92):
            time.sleep(0.12)
            progress_bar.progress(pct, text="Analyzing image...")

        label, emoji, confidence = predict(image)
        progress_bar.progress(100, text="Analysis complete")
        time.sleep(0.15)
        progress_bar.empty()

        result_class = "result-female" if label == "Female Eye" else "result-male"
        st.markdown(f"""
        <div class="result-box {result_class}">
            <div class="result-label">{emoji} {label}</div>
            <div class="result-sub">Model confidence: {confidence*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        st.metric("Confidence Score", f"{confidence*100:.2f}%")
        st.progress(float(confidence))

        if confidence * 100 < confidence_threshold:
            st.warning(
                f"⚠️ Confidence is below your selected threshold of {confidence_threshold}%. "
                "Try a brighter, sharper, and more centered close-up eye image for a stronger result."
            )
        else:
            st.success("✅ Prediction completed with good confidence.")

        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("👆 Upload an eye image above to get started.")

st.markdown("---")

# ------------------------------
# Developer Corner
# ------------------------------
st.markdown("### 👨‍💻 Developer Corner")
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 👤 Developer")
    st.markdown('<div class="dev-name">Vivek Srivastava</div>', unsafe_allow_html=True)
    st.write("B.Tech IT")
    st.write("Machine Learning & Data Science Enthusiast")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 🔗 Connect with Me")
    st.markdown("[💼 LinkedIn](https://www.linkedin.com/in/vivek-srivastava-0a878a329)")
    st.markdown("[💻 GitHub](https://github.com/viveksrivastava045-cyber/AIML_projects)")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="footer-note">
    Made with ❤️ using TensorFlow & Streamlit
</div>
""", unsafe_allow_html=True)
