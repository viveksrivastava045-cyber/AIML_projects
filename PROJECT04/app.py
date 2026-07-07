import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# ------------------------------------
# Page Configuration
# ------------------------------------
st.set_page_config(
    page_title="Employee Retention Prediction",
    page_icon="👨‍💼",
    layout="wide"
)

# ------------------------------------
# Title
# ------------------------------------
st.title("👨‍💼 Employee Retention Prediction")
st.write("""
This application predicts whether an employee is likely to **leave the company**
using a Logistic Regression model.
""")

# ------------------------------------
# Load Dataset
# ------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("HR_comma_sep.csv")

df = load_data()

# ------------------------------------
# Prepare Data
# ------------------------------------
subdf = df[['satisfaction_level',
            'average_montly_hours',
            'promotion_last_5years',
            'salary']]

salary_dummies = pd.get_dummies(subdf['salary'], prefix='salary')

X = pd.concat(
    [subdf.drop('salary', axis=1), salary_dummies],
    axis=1
)

# Ensure all salary columns exist
for col in ['salary_high', 'salary_low', 'salary_medium']:
    if col not in X.columns:
        X[col] = 0

y = df['left']

# ------------------------------------
# Train Model
# ------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# ------------------------------------
# Sidebar Inputs
# ------------------------------------
st.sidebar.header("Employee Information")

satisfaction = st.sidebar.slider(
    "Satisfaction Level",
    0.0,
    1.0,
    0.5,
    0.01
)

hours = st.sidebar.slider(
    "Average Monthly Hours",
    50,
    350,
    200
)

promotion = st.sidebar.selectbox(
    "Promotion in Last 5 Years",
    [0, 1]
)

salary = st.sidebar.selectbox(
    "Salary",
    ["low", "medium", "high"]
)

# ------------------------------------
# Create Input Data
# ------------------------------------
input_df = pd.DataFrame({
    'satisfaction_level': [satisfaction],
    'average_montly_hours': [hours],
    'promotion_last_5years': [promotion],
    'salary_high': [1 if salary == 'high' else 0],
    'salary_low': [1 if salary == 'low' else 0],
    'salary_medium': [1 if salary == 'medium' else 0]
})

# Match training column order
input_df = input_df[X.columns]

# ------------------------------------
# Prediction
# ------------------------------------
if st.button("Predict"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Employee is likely to leave the company.")
    else:
        st.success("✅ Employee is likely to stay in the company.")

    st.write(f"**Probability of Staying:** {probability[0]*100:.2f}%")
    st.write(f"**Probability of Leaving:** {probability[1]*100:.2f}%")

# ------------------------------------
# Dataset Preview
# ------------------------------------
st.markdown("---")

if st.checkbox("Show Dataset"):
    st.dataframe(df)

st.markdown("---")
st.caption("Developed using Streamlit & Scikit-Learn")