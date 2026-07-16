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
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    :root {
        --ink: #1E1B4B;
        --muted: #6B7280;
        --primary: #6D28D9;
        --primary-dark: #4C1D95;
        --accent: #F472B6;
        --bg-a: #F5F3FF;
        --bg-b: #FDF2F8;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Overall page background */
    .stApp {
        background: linear-gradient(160deg, var(--bg-a) 0%, #FFFFFF 45%, var(--bg-b) 100%);
    }

    /* Hide default streamlit branding clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: rgba(0,0,0,0);}

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--primary-dark), #2E1065);
    }
    section[data-testid="stSidebar"] * {
        color: #EDE9FE !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.15);
    }

    /* Hero header */
    .hero {
        text-align: center;
        padding: 2rem 1rem 1.2rem 1rem;
    }
    .hero-title {
        font-family: 'Poppins', sans-serif;
        font-size: 48px;
        font-weight: 800;
        background: linear-gradient(90deg, var(--primary), var(--accent));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero-subtitle {
        font-size: 17px;
        color: var(--muted);
        font-weight: 500;
    }

    /* Badge */
    .badge {
        display: inline-block;
        background: white;
        color: var(--primary);
        padding: 5px 14px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        margin: 6px 4px 0 0;
        box-shadow: 0 2px 8px rgba(109, 40, 217, 0.15);
        border: 1px solid #E9D5FF;
    }

    /* Card container */
    .card {
        background: rgba(255,255,255,0.85);
        backdrop-filter: blur(6px);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 8px 30px rgba(109, 40, 217, 0.10);
        border: 1px solid #EDE9FE;
    }
    .card h4 {
        font-family: 'Poppins', sans-serif;
        color: var(--ink);
    }

    /* Section headers written with st.markdown("### ...") */
    h3 {
        font-family: 'Poppins', sans-serif !important;
        color: var(--ink) !important;
    }

    /* File uploader restyle */
    div[data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(135deg, #F5F3FF, #FDF2F8);
        border: 2px dashed #C4B5FD;
        border-radius: 18px;
    }
    div[data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--primary);
    }
    div[data-testid="stFileUploaderDropzone"] button {
        background: var(--primary) !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
    }

    /* Result box */
    .result-box {
        text-align: center;
        padding: 1.8rem;
        border-radius: 20px;
        margin-top: 0.5rem;
    }
    .result-female {
        background: linear-gradient(135deg, #FDE7F3, #FBCFE8);
        border: 1px solid #F9A8D4;
    }
    .result-male {
        background: linear-gradient(135deg, #DBEAFE, #BFDBFE);
        border: 1px solid #93C5FD;
    }
    .result-label {
        font-family: 'Poppins', sans-serif;
        font-size: 34px;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .result-sub {
        font-size: 14px;
        color: #475569;
    }

    /* Metric restyle */
    div[data-testid="stMetric"] {
        background: white;
        border-radius: 14px;
        padding: 0.8rem 1rem;
        border: 1px solid #EDE9FE;
        box-shadow: 0 4px 14px rgba(109, 40, 217, 0.08);
    }
    div[data-testid="stMetricValue"] {
        color: var(--primary);
        font-family: 'Poppins', sans-serif;
    }

    /* Progress bar */
    div[data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, var(--primary), var(--accent)) !important;
    }

    /* Buttons in general */
    .stButton > button {
        background: linear-gradient(90deg, var(--primary), var(--accent));
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
    }
    .stButton > button:hover {
        opacity: 0.9;
        color: white;
    }

    /* Alerts (info/success/warning/error) rounder + softer */
    div[data-testid="stAlert"] {
        border-radius: 14px;
    }

    /* Slider accent */
    div[data-testid="stSlider"] span {
        color: var(--primary);
    }

    /* Footer */
    .footer-note {
        text-align: center;
        color: #94A3B8;
        font-size: 13px;
        padding-top: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Hero Header
# ------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-title">👁️ Eye Gender Detection</div>
    <div class="hero-subtitle">Upload a close-up eye image and let a deep learning model predict male / female</div>
    <span class="badge">TensorFlow</span>
    <span class="badge">Streamlit</span>
    <span class="badge">CNN Classifier</span>
</div>
""", unsafe_allow_html=True)

st.write("")

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
        "2. Wait for the model to process it\n"
        "3. View the predicted label and confidence score"
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
MODEL_PATH = "PROJECT_09_MALE_FEMALE_EYE_DETECTION/my_model.keras"

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
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
        st.error("⚠️ Couldn't read this file as an image. Please upload a valid jpg/jpeg/png.")
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
        for pct in (25, 55, 80):
            time.sleep(0.12)
            progress_bar.progress(pct, text="Analyzing image...")

        label, emoji, confidence = predict(image)
        progress_bar.progress(100, text="Done")
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
                f"⚠️ Confidence is below your set threshold of {confidence_threshold}%. "
                "Try a clearer, well-lit close-up eye image for a more reliable result."
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
st.subheader("👨‍💻 Developer Corner")
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 👤 Developer")
    st.write("**Vivek Srivastava**")
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
