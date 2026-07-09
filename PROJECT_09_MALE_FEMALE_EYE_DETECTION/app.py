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

# ------------------------------
# Custom CSS
# ------------------------------
st.markdown("""
<style>
.main{
    background:#F8FAFC;
}
.title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#1E3A8A;
}
.subtitle{
    text-align:center;
    font-size:18px;
    color:gray;
}
.result{
    font-size:30px;
    font-weight:bold;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>👁️ Male & Female Eye Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Deep Learning | TensorFlow | Streamlit</div>", unsafe_allow_html=True)

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
# Upload Image
# ------------------------------
uploaded = st.file_uploader(
    "Upload an Eye Image",
    type=["jpg","jpeg","png"]
)

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

st.markdown("---")
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
st.markdown(
"""
<center>
Made with ❤️ using TensorFlow & Streamlit
</center>
""",
unsafe_allow_html=True
)
