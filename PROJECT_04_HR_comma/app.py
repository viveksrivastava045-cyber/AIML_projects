import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Employee Retention Predictor",
    page_icon="👨‍💼",
    layout="wide"
)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("PROJECT_04_HR_comma/model.pkl")

model = load_model()

# -------------------------------
# Title
# -------------------------------
st.title("👨‍💼 Employee Retention Prediction")
st.markdown(
    "Predict whether an employee is likely to **leave the company** using a Logistic Regression model."
)

st.divider()

# -------------------------------
# Layout
# -------------------------------
col1, col2 = st.columns([1, 1])

with col1:

    satisfaction = st.slider(
        "Satisfaction Level",
        min_value=0.00,
        max_value=1.00,
        value=0.50,
        step=0.01,
    )

    hours = st.slider(
        "Average Monthly Hours",
        min_value=80,
        max_value=320,
        value=200,
    )

with col2:

    promotion = st.selectbox(
        "Promotion in Last 5 Years",
        ["No", "Yes"]
    )

    salary = st.selectbox(
        "Salary Level",
        ["Low", "Medium", "High"]
    )

st.divider()

# -------------------------------
# Feature Engineering
# -------------------------------

promotion = 1 if promotion == "Yes" else 0

salary_high = 1 if salary == "High" else 0
salary_low = 1 if salary == "Low" else 0
salary_medium = 1 if salary == "Medium" else 0

input_df = pd.DataFrame(
    [[
        satisfaction,
        hours,
        promotion,
        salary_high,
        salary_low,
        salary_medium
    ]],
    columns=[
        "satisfaction_level",
        "average_montly_hours",
        "promotion_last_5years",
        "salary_high",
        "salary_low",
        "salary_medium",
    ],
)

# -------------------------------
# Prediction
# -------------------------------

if st.button("🔍 Predict", use_container_width=True):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    st.divider()

    if prediction == 1:
        st.error("⚠️ This employee is likely to leave the company.")
        confidence = probability[1] * 100
    else:
        st.success("✅ This employee is likely to stay with the company.")
        confidence = probability[0] * 100

    st.metric(
        "Prediction Confidence",
        f"{confidence:.2f}%"
    )

    st.progress(confidence / 100)

    st.subheader("Input Summary")

    st.table(input_df)



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
    st.markdown("[💻 GitHub](https://github.com/viveksrivastava045-cyber/AIML_projects/edit/main/PROJECT_04_HR_comma)")

st.markdown("---")
st.caption("Made with ❤️ using Python, Scikit-Learn & Streamlit")
