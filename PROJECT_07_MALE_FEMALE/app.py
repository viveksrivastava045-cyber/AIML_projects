import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Gender Recognition CNN",
    page_icon="🧑",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = tf.keras.models.load_model("PROJECT_07_MALE_FEMALE/gender_cnn_model.keras")

labels = np.load("labels.npy", allow_pickle=True).item()

class_names = list(labels.keys())

# -----------------------------
# Title
# -----------------------------
st.title("🧑 Gender Recognition using CNN")

st.write(
    "Upload an image and the trained CNN model will predict the gender."
)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image,width=250)

    img = image.resize((128,128))
    img = np.array(img)/255.0
    img = np.expand_dims(img,axis=0)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        gender = "Male"
        confidence = prediction
    else:
        gender = "Female"
        confidence = 1-prediction

    st.success(f"Prediction : {gender}")

    st.metric(
        "Confidence",
        f"{confidence*100:.2f}%"
    )

st.markdown("---")

st.header("Developer")

st.write("**Vivek Srivastava**")

st.markdown("[💼 LinkedIn](https://www.linkedin.com/)")

st.markdown("[💻 GitHub](https://github.com/)")
