import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Gender Recognition using CNN",
    page_icon="🧑",
    layout="wide"
)

# ---------------------------------
# Load CNN Model
# ---------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("PROJECT_07_MALE_FEMALE/gender_cnn_model.keras")

model = load_model()

# ---------------------------------
# Title
# ---------------------------------
st.title("🧑 Gender Recognition using CNN")

st.write(
    "Upload an image and the CNN model will predict whether the person is Male or Female."
)

# ---------------------------------
# Image Upload
# ---------------------------------
uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=300)

    # Preprocess Image
    img = image.resize((128, 128))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img, verbose=0)[0][0]

    if prediction >= 0.5:
        gender = "Male"
        confidence = prediction
    else:
        gender = "Female"
        confidence = 1 - prediction

    st.success(f"### Prediction: {gender}")

    st.metric(
        label="Confidence",
        value=f"{confidence*100:.2f}%"
    )

# ---------------------------------
# Footer
# ---------------------------------
st.markdown("---")

st.header("👨‍💻 Developer")

st.write("**Vivek Srivastava**")

st.markdown(
    "[💼 LinkedIn](https://www.linkedin.com/in/YOUR-LINKEDIN)"
)

st.markdown(
    "[💻 GitHub](https://github.com/YOUR-GITHUB)"
)
