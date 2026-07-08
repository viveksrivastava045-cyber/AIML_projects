import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image

# 1. Load the trained model
@st.cache_resource
def load_model():
    # Make sure 'male_female_model.pkl' is in the same directory
    return joblib.load("PROJECT_05_Male_female/male_female_model.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file 'male_female_model.pkl' not found. Please ensure it is in the same directory as this script.")
    st.stop()

# 2. App configuration & UI
st.set_page_config(page_title="Gender Classification App", page_icon="👤")
st.title("👤 Male / Female Classification App")
st.write("Upload an image, and the trained Logistic Regression model will predict the gender.")

# Class labels matching your training setup
CLASSES = ["Male", "Female"]
IMG_SIZE = 64

# 3. File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Process button
    if st.button("Predict Gender"):
        with st.spinner("Analyzing image..."):
            # Convert PIL Image to NumPy array (RGB)
            img_array = np.array(image)
            
            # OpenCV expects BGR, convert if necessary (matches cv2.imread behavior)
            if len(img_array.shape) == 3:  # Color image
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else: # Grayscale image
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                
            # Preprocessing exactly like your training script
            resized_img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
            flattened_img = resized_img.flatten().reshape(1, -1) # Reshape for a single sample prediction
            
            # Predict
            prediction = model.predict(flattened_img)[0]
            predicted_class = CLASSES[prediction]
            
            # Optional: Get prediction probability
            try:
                probabilities = model.predict_proba(flattened_img)[0]
                confidence = probabilities[prediction] * 100
                conf_text = f" ({confidence:.2f}% confidence)"
            except:
                conf_text = ""
            
            # Output Results
            st.success(f"**Prediction:** {predicted_class}{conf_text}")
