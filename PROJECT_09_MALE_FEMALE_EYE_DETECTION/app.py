import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Male/Female Eye Detection",
    page_icon="👁️",
    layout="wide"
)

st.markdown("""
<style>

/* Hide Streamlit Branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main App */
.stApp{
    background-color:#FFFFFF;
}

/* Reduce top spacing */
.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    max-width:1200px;
}

/* Hero Banner */
.hero{
    background:linear-gradient(90deg,#2563EB,#1D4ED8);
    padding:35px;
    border-radius:20px;
    text-align:center;
    color:white;
    box-shadow:0px 8px 20px rgba(37,99,235,0.25);
    margin-bottom:30px;
}

.hero h1{
    font-size:42px;
    margin:0;
    font-weight:700;
}

.hero p{
    font-size:18px;
    margin-top:8px;
    opacity:0.95;
}

/* Cards */
.card{
    background:#FFFFFF;
    padding:25px;
    border-radius:18px;
    border:1px solid #E5E7EB;
    box-shadow:0 6px 20px rgba(0,0,0,.06);
    margin-top:20px;
}

/* Prediction Card */
.result-card{
    background:#F8FAFC;
    border-left:6px solid #2563EB;
    border-radius:18px;
    padding:30px;
    text-align:center;
    box-shadow:0 6px 18px rgba(0,0,0,.05);
}

.result-card h2{
    color:#374151;
    margin-bottom:15px;
}

.result{
    font-size:36px;
    font-weight:700;
    color:#2563EB;
    margin-bottom:15px;
}

/* Confidence */
.confidence{
    font-size:20px;
    font-weight:600;
    color:#111827;
}

/* Image */
.stImage img{
    border-radius:18px;
    border:2px solid #E5E7EB;
    box-shadow:0px 6px 18px rgba(0,0,0,.08);
}

/* File Uploader */
[data-testid="stFileUploader"]{
    border:2px dashed #2563EB;
    border-radius:18px;
    padding:20px;
    background:#F8FAFC;
}

/* Buttons */
.stButton>button{
    background:#2563EB;
    color:white;
    border:none;
    border-radius:10px;
    font-weight:600;
    transition:0.3s;
}

.stButton>button:hover{
    background:#1D4ED8;
    transform:translateY(-2px);
}

/* Progress Bar */
.stProgress > div > div > div > div{
    background:#2563EB;
}

/* Metrics */
[data-testid="metric-container"]{
    background:white;
    border-radius:15px;
    border:1px solid #E5E7EB;
    padding:15px;
    box-shadow:0px 4px 12px rgba(0,0,0,.05);
}

/* Developer Card */
.developer{
    background:white;
    border-radius:20px;
    padding:25px;
    border:1px solid #E5E7EB;
    box-shadow:0px 6px 18px rgba(0,0,0,.06);
    margin-top:30px;
}

.developer h2{
    color:#2563EB;
    text-align:center;
}

.developer p{
    font-size:17px;
    color:#374151;
    line-height:1.8;
}

/* Links */
a{
    color:#2563EB;
    text-decoration:none;
    font-weight:600;
}

a:hover{
    color:#1D4ED8;
    text-decoration:underline;
}

/* Footer */
.footer{
    text-align:center;
    margin-top:30px;
    color:#6B7280;
    font-size:15px;
}

/* Responsive */
@media (max-width:768px){

.hero h1{
    font-size:30px;
}

.hero p{
    font-size:16px;
}

.result{
    font-size:28px;
}

.card{
    padding:18px;
}

}


</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class='hero'>
<h1>👁️ Male & Female Eye Detection</h1>
<p>Deep Learning • TensorFlow • Streamlit</p>
</div>
""", unsafe_allow_html=True)
  
# ------------------------------
# Load Model
# ------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("PROJECT_09_MALE_FEMALE_EYE_DETECTION/my_model.keras")

model = load_model()

# ------------------------------
# Prediction Function
# ------------------------------
def predict(img):

    img = img.resize((299,299))

    img = np.array(img)

    if img.shape[-1] == 4:
        img = img[:,:,:3]

    img = img.astype("float32") / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    probability = float(prediction[0][0])

    if probability > 0.5:
        label = "👩 Female Eye"
        confidence = probability
    else:
        label = "👨 Male Eye"
        confidence = 1 - probability

    return label, confidence

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.header("About")

st.sidebar.info(
"""
### Model Information

- CNN Model
- Image Size : 299x299
- Binary Classification
- Classes:
  - Male Eye
  - Female Eye
"""
)

# ------------------------------
# Upload Image
# ------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.subheader("📤 Upload Eye Image")

uploaded = st.file_uploader(
    "Choose an image",
    type=["jpg","jpeg","png"]
)

st.markdown("</div>", unsafe_allow_html=True)

if uploaded:

    image = Image.open(uploaded)

    col1,col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Predicting..."):

        label, confidence = predict(image)

    with col2:

        st.success("Prediction Completed")

        st.markdown(
            f"<div class='result'>{label}</div>",
            unsafe_allow_html=True
        )

        st.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )

        st.progress(float(confidence))

# ----------------------------
# Developer Corner
# ----------------------------

st.markdown("---")
st.subheader("👨‍💻 Developer Corner")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 👤 Developer")
    st.write("**Vivek Srivastava**")
    st.write("B.Tech IT ")
    st.write("Machine Learning & Data Science Enthusiast")

with col2:
    st.markdown("### 🔗 Connect with Me")
    st.markdown("[💼 LinkedIn](https://www.linkedin.com/in/vivek-srivastava-0a878a329)")
    st.markdown("[💻 GitHub](https://github.com/viveksrivastava045-cyber/AIML_projects/edit/main/PROJECT_03_CANADA_per_capita_income)")

st.markdown("---")
st.markdown("---")

st.markdown(
"""
<center>
Made with ❤️ using TensorFlow & Streamlit
</center>
""",
unsafe_allow_html=True
)

