import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Canada Per Capita Income Prediction",
    page_icon="📈",
    layout="wide"
)

# ----------------------------
# Title
# ----------------------------
st.title("📈 Canada Per Capita Income Prediction")
st.write(
    "This machine learning app predicts **Canada's Per Capita Income (US$)** using **Linear Regression**."
)

# ----------------------------
# Load Dataset
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("canada_per_capita_income.csv")

df = load_data()

# ----------------------------
# Show Dataset
# ----------------------------
st.subheader("Dataset")
st.dataframe(df, use_container_width=True)

# ----------------------------
# Train Model
# ----------------------------
X = df[['year']]
y = df['per capita income (US$)']

model = LinearRegression()
model.fit(X, y)

# ----------------------------
# Dataset Visualization
# ----------------------------
st.subheader("Data Visualization")

fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(df['year'], df['per capita income (US$)'], color='blue')
ax.plot(df['year'], model.predict(X), color='red')
ax.set_xlabel("Year")
ax.set_ylabel("Per Capita Income (US$)")
ax.set_title("Linear Regression Fit")

st.pyplot(fig)

# ----------------------------
# Prediction
# ----------------------------
st.subheader("Predict Income")

year = st.number_input(
    "Enter Year",
    min_value=1960,
    max_value=2050,
    value=2020,
    step=1
)

if st.button("Predict"):
    prediction = model.predict([[year]])

    st.success(
        f"Predicted Per Capita Income in {year}: **${prediction[0]:,.2f}**"
    )

# ----------------------------
# Model Information
# ----------------------------
st.subheader("Model Information")

st.write(f"**Coefficient:** {model.coef_[0]:.4f}")
st.write(f"**Intercept:** {model.intercept_:.2f}")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("Developed using ❤️ with Streamlit")
# ----------------------------
# Developer Corner
# ----------------------------
st.markdown("---")
st.subheader("👨‍💻 Developer Corner")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 👤 Developer")
    st.write("**Vivek Srivastava**")
    st.write("Machine Learning & Data Science Enthusiast")

with col2:
    st.markdown("### 🔗 Connect with Me")
    st.markdown("[💼 LinkedIn](https://# ----------------------------
# Developer Corner
# ----------------------------
st.markdown("---")
st.subheader("👨‍💻 Developer Corner")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 👤 Developer")
    st.write("**Vivek Srivastava**")
    st.write("B.Tech CSE (AI & ML)")
    st.write("Machine Learning & Data Science Enthusiast")

with col2:
    st.markdown("### 🔗 Connect with Me")
    st.markdown("[💼 LinkedIn](https://# ----------------------------
# Developer Corner
# ----------------------------
st.markdown("---")
st.subheader("👨‍💻 Developer Corner")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 👤 Developer")
    st.write("**Vivek Srivastava**")
    st.write("B.Tech CSE (AI & ML)")
    st.write("Machine Learning & Data Science Enthusiast")

with col2:
    st.markdown("### 🔗 Connect with Me")
    st.markdown("[💼 LinkedIn](https://www.linkedin.com/in/vivek-srivastava-0a878a329)")
    st.markdown("[💻 GitHub](https://github.com/viveksrivastava045-cyber/AIML_projects.git)")


st.markdown("---")
st.caption("Made with ❤️ using Python, Scikit-Learn & Streamlit")
