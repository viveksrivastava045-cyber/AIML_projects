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

/* -------------------------
Hide Streamlit Branding
--------------------------*/

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

/* -------------------------
Background
--------------------------*/

.stApp{

background:linear-gradient(
135deg,
#4F46E5 0%,
#1E1B8F 45%,
#0284C7 75%,
#22D3EE 100%
);

background-attachment:fixed;

}

/* -------------------------
Main Container
--------------------------*/

.block-container{

padding-top:2rem;
padding-bottom:2rem;
max-width:1200px;

}

/* -------------------------
Hero Banner
--------------------------*/

.hero{

background:rgba(255,255,255,.08);

backdrop-filter:blur(20px);

border:1px solid rgba(255,255,255,.15);

padding:35px;

border-radius:25px;

box-shadow:0 15px 40px rgba(0,0,0,.25);

text-align:center;

}

.hero h1{

font-size:52px;

font-weight:700;

margin:0;

color:#FFFFFF;

}

.hero p{

margin-top:12px;

font-size:22px;

color:#E0F2FE;

}

/* -------------------------
Cards
--------------------------*/

.card{

background:rgba(255,255,255,.08);

backdrop-filter:blur(18px);

border-radius:25px;

padding:25px;

border:1px solid rgba(255,255,255,.15);

box-shadow:0 15px 35px rgba(0,0,0,.20);

margin-top:20px;

}

/* -------------------------
Prediction Card
--------------------------*/

.result-card{

background:rgba(255,255,255,.10);

backdrop-filter:blur(20px);

padding:30px;

border-radius:20px;

border:1px solid rgba(255,255,255,.15);

text-align:center;

box-shadow:0 15px 30px rgba(0,0,0,.20);

}

/* -------------------------
Gradient Text
--------------------------*/

.gradient-text{

background:linear-gradient(
90deg,
#FDE68A,
#FFFFFF,
#93C5FD
);

-webkit-background-clip:text;

-webkit-text-fill-color:transparent;

font-weight:800;

}

/* -------------------------
Upload Heading
--------------------------*/

.upload-title{

font-size:34px;

font-weight:700;

color:white;

}

/* -------------------------
File Uploader
--------------------------*/

[data-testid="stFileUploader"]{

background:rgba(255,255,255,.08);

border:2px dashed rgba(255,255,255,.40);

border-radius:22px;

padding:18px;

}

[data-testid="stFileUploader"] section{

background:transparent !important;

}

[data-testid="stFileUploaderDropzone"]{

background:transparent !important;

border:none !important;

}

/* -------------------------
Image
--------------------------*/

.stImage img{

border-radius:20px;

border:3px solid rgba(255,255,255,.20);

box-shadow:0px 10px 25px rgba(0,0,0,.30);

}

/* -------------------------
Success Box
--------------------------*/

.stAlert{

background:rgba(34,197,94,.15)!important;

border:1px solid rgba(34,197,94,.35)!important;

color:white!important;

}

/* -------------------------
Metric
--------------------------*/

[data-testid="metric-container"]{

background:rgba(255,255,255,.08);

border:1px solid rgba(255,255,255,.15);

border-radius:20px;

color:white;

box-shadow:0px 8px 25px rgba(0,0,0,.18);

}

/* -------------------------
Progress Bar
--------------------------*/

.stProgress>div>div>div{

background:linear-gradient(
90deg,
#22C55E,
#14B8A6,
#06B6D4
);

}

/* -------------------------
Buttons
--------------------------*/

.stButton>button{

background:linear-gradient(
90deg,
#3B82F6,
#2563EB
);

color:white;

border:none;

border-radius:12px;

font-weight:600;

transition:.3s;

}

.stButton>button:hover{

transform:translateY(-3px);

box-shadow:0 10px 20px rgba(0,0,0,.30);

}

/* -------------------------
Developer Card
--------------------------*/

.developer{

background:rgba(255,255,255,.08);

backdrop-filter:blur(18px);

border-radius:22px;

padding:30px;

border:1px solid rgba(255,255,255,.15);

box-shadow:0 12px 30px rgba(0,0,0,.20);

color:white;

}

.developer a{

color:#93C5FD;

text-decoration:none;

font-weight:700;

}

.developer a:hover{

color:white;

}

/* -------------------------
Footer
--------------------------*/

.footer{

text-align:center;

color:#E0F2FE;

font-size:15px;

margin-top:30px;

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
        st.markdown(f"""
<div class="prediction-card">

<h2>🎯 Prediction Result</h2>

<div class="prediction-label">
{label}
</div>

<div class="prediction-confidence">

{confidence*100:.2f}% Confidence

</div>

</div>
""", unsafe_allow_html=True)

st.progress(confidence)



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

