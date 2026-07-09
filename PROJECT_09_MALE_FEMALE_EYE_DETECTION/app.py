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

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

.stApp{
    background: meganta;
}

.hero{
    padding:25px;
    border-radius:20px;
    background:linear-gradient(90deg,#4F46E5,#2563EB);
    color:yellow;
    text-align:center;
    box-shadow:0px 8px 25px rgba(0,0,0,.15);
}

.hero h1{
    font-size:42px;
    margin-bottom:5px;
}

.hero p{
    font-size:18px;
    opacity:.9;
}

.card{
    background:white;
    padding:25px;
    border-radius:18px;
    box-shadow:0px 8px 20px rgba(0,0,0,.08);
    margin-top:20px;
}

.result-card{
    background:#F8FAFC;
    border-left:8px solid #2563EB;
    padding:25px;
    border-radius:15px;
    text-align:center;
}

.result{
    font-size:34px;
    font-weight:700;
    color:#2563EB;
}

.footer-card{
    background:white;
    border-radius:18px;
    padding:20px;
    box-shadow:0px 6px 18px rgba(0,0,0,.08);
}

a{
    text-decoration:none;
    font-weight:bold;
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

