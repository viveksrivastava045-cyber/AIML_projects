import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Set up page configurations
st.set_page_config(
    page_title="Gender Classification CNN",
    page_icon="👤",
    layout="centered"
)

st.title("👤 Gender Classification AI App")
st.write("Upload an image of a face, and the CNN model will predict the gender.")

# Cache the model so it doesn't reload on every user interaction
@st.cache_resource
def load_gender_model():
    # Make sure 'gender_cnn_model.keras' is in the same directory as app.py
    model = tf.keras.models.load_model('PROJECT_07_MALE_FEMALE/gender_cnn_model.keras')
    return model

try:
    model = load_gender_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model. Make sure 'gender_cnn_model.keras' is in the root directory. Details: {e}")

# Image upload widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width = 128)
    st.write("")
    
    with st.spinner('Analyzing image...'):
        # 1. Resize image to 64x64 as required by the model architecture
        # Using ImageOps.fit crops and scales smoothly
        img_resized = ImageOps.fit(image, (64, 64), Image.Resampling.LANCZOS)
        
        # 2. Convert to RGB if it's grayscale or RGBA
        if img_resized.mode != 'RGB':
            img_resized = img_resized.convert('RGB')
            
        # 3. Convert image to numpy array
        img_array = np.array(img_resized)
        
        # 4. Normalize pixel values to [0, 1] (Standard CNN practice)
        img_array = img_array.astype('float32') / 255.0
        
        # 5. Add batch dimension: [64, 64, 3] -> [1, 64, 64, 3]
        img_batch = np.expand_dims(img_array, axis=0)
        
        # 6. Run prediction
        prediction = model.predict(img_batch)[0][0]
        
        # Determine labels based on prediction (Adjust threshold if necessary)
        # Note: Swap 'Male' and 'Female' targets below if your dataset mapping was inverted
        if prediction >= 0.5:
            confidence = prediction * 100
            st.metric(label="Predicted Gender", value="Male", delta=f"{confidence:.2f}% Confidence")
        else:
            confidence = (1 - prediction) * 100
            st.metric(label="Predicted Gender", value="Female", delta=f"{confidence:.2f}% Confidence")
            
        # Optional progress bar representation
        st.write("Probability Breakdown:")
        st.progress(float(prediction))
        st.caption("0.0 (Female) 👤 ---------------------------------------------------- 👤 1.0 (Male)")


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
st.caption("Made with ❤️ using Python, Scikit-Learn & Streamlit")
