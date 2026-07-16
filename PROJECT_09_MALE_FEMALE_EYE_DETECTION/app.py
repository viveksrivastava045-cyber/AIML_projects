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
<style>
    /* Overall page */
    .main {
        background: linear-gradient(180deg, #F8FAFC 0%, #EEF2FF 100%);
    }

    /* Hide default streamlit branding clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Hero header */
    .hero {
        text-align: center;
        padding: 1.6rem 1rem 1rem 1rem;
    }
    .hero-title {
        font-size: 44px;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        font-size: 17px;
        color: #64748B;
        font-weight: 500;
    }

    /* Card container */
    .card {
        background: white;
        border-radius: 16px;
        padding: 1.4rem;
        box-shadow: 0 4px 18px rgba(30, 58, 138, 0.08);
        border: 1px solid #E5E9F5;
    }

    /* Result box */
    .result-box {
        text-align: center;
        padding: 1.5rem;
        border-radius: 16px;
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
        font-size: 32px;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .result-sub {
        font-size: 14px;
        color: #475569;
    }

    /* Badge */
    .badge {
        display: inline-block;
        background: #EEF2FF;
        color: #4338CA;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        margin: 4px 4px 0 0;
    }

    /* Footer */
    .footer-note {
        text-align: center;
        color: #94A3B8;
        font-size: 13px;
        padding-top: 1rem;
    }

    div[data-testid="stMetricValue"] {
        color: #1E3A8A;
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
    st.markdown("[💻 GitHub](https://github.com/viveksrivastava045-cyber/AIML_projects/edit/main/PROJECT_09_MALE_FEMALE_EYE_DETECTION)")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="footer-note">
    Made with ❤️ using TensorFlow & Streamlit
</div>
""", unsafe_allow_html=True)
# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image

# # ------------------------------
# # Page Configuration
# # ------------------------------
# st.set_page_config(
#     page_title="Male/Female Eye Detection",
#     page_icon="👁️",
#     layout="wide"
# )

# # ------------------------------
# # Custom CSS
# # ------------------------------
# st.markdown("""
# <style>
# .main{
#     background:#F8FAFC;
# }
# .title{
#     text-align:center;
#     font-size:42px;
#     font-weight:bold;
#     color:#1E3A8A;
# }
# .subtitle{
#     text-align:center;
#     font-size:18px;
#     color:gray;
# }
# .result{
#     font-size:30px;
#     font-weight:bold;
#     text-align:center;
# }
# </style>
# """, unsafe_allow_html=True)

# st.markdown("<div class='title'>👁️ Male & Female Eye Detection</div>", unsafe_allow_html=True)
# st.markdown("<div class='subtitle'>Deep Learning | TensorFlow | Streamlit</div>", unsafe_allow_html=True)

# # ------------------------------
# # Load Model
# # ------------------------------
# @st.cache_resource
# def load_model():
#     return tf.keras.models.load_model("PROJECT_09_MALE_FEMALE_EYE_DETECTION/my_model.keras")

# model = load_model()

# # ------------------------------
# # Prediction Function
# # ------------------------------
# def predict(img):

#     img = img.resize((299,299))

#     img = np.array(img)

#     if img.shape[-1] == 4:
#         img = img[:,:,:3]

#     img = img.astype("float32") / 255.0

#     img = np.expand_dims(img, axis=0)

#     prediction = model.predict(img)

#     probability = float(prediction[0][0])

#     if probability > 0.5:
#         label = "👩 Female Eye"
#         confidence = probability
#     else:
#         label = "👨 Male Eye"
#         confidence = 1 - probability

#     return label, confidence


# # ------------------------------
# # Upload Image
# # ------------------------------
# uploaded = st.file_uploader(
#     "Upload an Eye Image",
#     type=["jpg","jpeg","png"]
# )

# if uploaded:

#     image = Image.open(uploaded)

#     col1,col2 = st.columns(2)

#     with col1:
#         st.image(image, caption="Uploaded Image", use_container_width=True)

#     with st.spinner("Predicting..."):

#         label, confidence = predict(image)

#     with col2:

#         st.success("Prediction Completed")

#         st.markdown(
#             f"<div class='result'>{label}</div>",
#             unsafe_allow_html=True
#         )

#         st.metric(
#             "Confidence",
#             f"{confidence*100:.2f}%"
#         )

#         st.progress(float(confidence))

# st.markdown("---")
# # ----------------------------
# # Developer Corner
# # ----------------------------

# st.markdown("---")
# st.subheader("👨‍💻 Developer Corner")

# col1, col2 = st.columns(2)

# with col1:
#     st.markdown("### 👤 Developer")
#     st.write("**Vivek Srivastava**")
#     st.write("B.Tech IT ")
#     st.write("Machine Learning & Data Science Enthusiast")

# with col2:
#     st.markdown("### 🔗 Connect with Me")
#     st.markdown("[💼 LinkedIn](https://www.linkedin.com/in/vivek-srivastava-0a878a329)")
#     st.markdown("[💻 GitHub](https://github.com/viveksrivastava045-cyber/AIML_projects/edit/main/PROJECT_03_CANADA_per_capita_income)")

# st.markdown("---")
# st.markdown(
# """
# <center>
# Made with ❤️ using TensorFlow & Streamlit
# </center>
# """,
# unsafe_allow_html=True
# )
