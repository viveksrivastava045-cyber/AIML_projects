import streamlit as st
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# ---------------------------------------------
# PAGE CONFIG
# ---------------------------------------------
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="wide"
)
# ---------------------------------------------
# ABOUT
# ---------------------------------------------
st.write("")

st.info("""
### About Project

This application predicts the Iris flower species using the K-Nearest Neighbors (KNN) Machine Learning algorithm.

**Dataset:** Iris Dataset

**Algorithm:** KNN Classifier

**Libraries:** Scikit-Learn • Pandas • Streamlit
""")

# ---------------------------------------------
# CUSTOM CSS
# ---------------------------------------------
st.markdown("""
<style>

.main{
    background:#F8FAFC;
}

.stButton>button{
    width:100%;
    border-radius:12px;
    background:linear-gradient(to right,#4F46E5,#2563EB);
    color:white;
    font-size:18px;
    font-weight:bold;
    height:55px;
}

.stButton>button:hover{
    background:linear-gradient(to right,#2563EB,#1E40AF);
}

.metric-card{
    padding:18px;
    border-radius:15px;
    background:white;
    box-shadow:0px 4px 15px rgba(0,0,0,0.08);
    text-align:center;
}

.header{
background: linear-gradient(90deg,#4F46E5,#3B82F6);
padding:30px;
border-radius:15px;
text-align:center;
color:white;
}

.footer{
text-align:center;
color:grey;
padding:25px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------
# HEADER
# ---------------------------------------------
st.markdown("""
<div class="header">
<h1>🌸 Iris Flower Classification</h1>
<h4>Machine Learning using K-Nearest Neighbors (KNN)</h4>
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------------------------------------
# LOAD DATA
# ---------------------------------------------
iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["Species"] = [iris.target_names[i] for i in iris.target]

# ---------------------------------------------
# TRAIN MODEL
# ---------------------------------------------
X = iris.data
y = iris.target

X_train,X_test,y_train,y_test=train_test_split(
X,
y,
test_size=0.3,
random_state=42
)

model=KNeighborsClassifier(n_neighbors=5)

model.fit(X_train,y_train)

accuracy=model.score(X_test,y_test)

# ---------------------------------------------
# SIDEBAR
# ---------------------------------------------
st.sidebar.image(
"https://upload.wikimedia.org/wikipedia/commons/5/56/Iris_versicolor_3.jpg",
use_container_width=True
)

st.sidebar.title("🌼 Flower Measurements")

sl=st.sidebar.slider("Sepal Length",4.0,8.0,5.8)

sw=st.sidebar.slider("Sepal Width",2.0,5.0,3.0)

pl=st.sidebar.slider("Petal Length",1.0,7.0,4.3)

pw=st.sidebar.slider("Petal Width",0.1,3.0,1.3)

# ---------------------------------------------
# PREDICT
# ---------------------------------------------
input_data=np.array([[sl,sw,pl,pw]])

prediction=model.predict(input_data)

prob=model.predict_proba(input_data)[0]

species=iris.target_names[prediction[0]]

# ---------------------------------------------
# KPI CARDS
# ---------------------------------------------
c1,c2,c3=st.columns(3)

with c1:
    st.metric("Dataset Size",len(df))

with c2:
    st.metric("Features",4)

with c3:
    st.metric("Accuracy",f"{accuracy*100:.2f}%")

st.write("")

# ---------------------------------------------
# MAIN CONTENT
# ---------------------------------------------
left,right=st.columns([1.2,1])

with left:

    with st.expander("📊 View Dataset"):
        st.dataframe(df,use_container_width=True)

    st.subheader("Prediction Confidence")

    prob_df=pd.DataFrame({
        "Species":iris.target_names,
        "Probability":prob
    })

    st.bar_chart(
        prob_df.set_index("Species")
    )

with right:

    st.subheader("🌼 Predict Flower")

    st.write("")

    if st.button("Predict Species"):

        st.success(f"### Prediction : {species.title()}")

        st.write("### Input Values")

        st.table(pd.DataFrame({
            "Feature":[
                "Sepal Length",
                "Sepal Width",
                "Petal Length",
                "Petal Width"
            ],
            "Value":[sl,sw,pl,pw]
        }))
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
st.caption("Made with ❤️ using Python, Scikit-Learn & Streamlit")

st.markdown("""
<div class='footer'>
Made with ❤️ using Streamlit
</div>
""",unsafe_allow_html=True)
