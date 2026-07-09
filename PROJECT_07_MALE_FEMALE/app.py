import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Gender Recognition using CNN",
    page_icon="🧑",
    layout="centered"
)

# -----------------------------------
# Load Model
# -----------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("PROJECT_07_MALE_FEMALE/gender_cnn_model.keras")

try:
    model = load_model()
except Exception as e:
    st.error("Unable to load the model.")
    st.exception(e)
    st.stop()

# -----------------------------------
# Model Information
# -----------------------------------
st.title("🧑 Gender Recognition using CNN")

st.write(
    "Upload an image and the trained CNN model will predict the gender."
)

st.info(f"Model Input Shape: {model.input_shape}")
st.info(f"Model Output Shape: {model.output_shape}")

# Determine expected image size
try:
    IMG_SIZE = model.input_shape[1]
except:
    IMG_SIZE = 128

# -----------------------------------
# Upload Image
# -----------------------------------
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=300)

    # -----------------------------
    # Preprocess
    # -----------------------------
    img = image.resize((IMG_SIZE, IMG_SIZE))

    img = np.asarray(img, dtype=np.float32)

    img /= 255.0

    img = np.expand_dims(img, axis=0)

    st.write("Input Shape:", img.shape)

    # -----------------------------
    # Prediction
    # -----------------------------
    try:

        prediction = model.predict(img, verbose=0)

        st.write("Raw Prediction:", prediction)

        # Binary Output
        if prediction.shape[-1] == 1:

            score = float(prediction.squeeze())

            if score >= 0.5:
                gender = "Male"
                confidence = score
            else:
                gender = "Female"
                confidence = 1 - score

        # Two-class Softmax Output
        elif prediction.shape[-1] == 2:

            index = int(np.argmax(prediction))

            gender = ["Female", "Male"][index]

            confidence = float(prediction[0][index])

        else:
            st.error("Unsupported model output.")
            st.stop()

        st.success(f"Prediction: **{gender}**")

        st.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )

    except Exception as e:
        st.error("Prediction Failed")
        st.exception(e)

# -----------------------------------
# Footer
# -----------------------------------
st.markdown("---")

st.header("👨‍💻 Developer")

st.write("**Vivek Srivastava**")

st.markdown(
    "[💼 LinkedIn](https://www.linkedin.com/in/viveksrivastava)"
)

st.markdown(
    "[💻 GitHub](https://github.com/YOUR-GITHUB)"
)
