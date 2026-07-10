import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# -------------------------------------------------------
# Page Config
# -------------------------------------------------------

st.set_page_config(
    page_title="Employee Retention Predictor",
    page_icon="👨‍💼",
    layout="wide"
)

st.title("👨‍💼 Employee Retention Prediction")
st.markdown("Predict whether an employee is likely to leave the company.")

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("HR_comma_sep.csv")

df = load_data()

# -------------------------------------------------------
# Data Preprocessing
# -------------------------------------------------------

subdf = df[['satisfaction_level',
            'average_montly_hours',
            'promotion_last_5years',
            'salary']]

salary_dummies = pd.get_dummies(subdf.salary, prefix="salary")

X = pd.concat([subdf, salary_dummies], axis=1)

X.drop('salary', axis=1, inplace=True)

y = df.left

# -------------------------------------------------------
# Train Model
# -------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

st.sidebar.header("Model Information")

st.sidebar.metric(
    "Accuracy",
    f"{accuracy*100:.2f}%"
)

st.sidebar.write("Developed using Logistic Regression")

# -------------------------------------------------------
# User Inputs
# -------------------------------------------------------

st.header("Employee Details")

col1, col2 = st.columns(2)

with col1:

    satisfaction = st.slider(
        "Satisfaction Level",
        0.0,
        1.0,
        0.5,
        0.01
    )

    monthly_hours = st.slider(
        "Average Monthly Hours",
        80,
        320,
        200
    )

with col2:

    promotion = st.selectbox(
        "Promotion in Last 5 Years",
        [0, 1]
    )

    salary = st.selectbox(
        "Salary Level",
        ["low", "medium", "high"]
    )

salary_low = 1 if salary == "low" else 0
salary_medium = 1 if salary == "medium" else 0

input_data = pd.DataFrame(
    [[
        satisfaction,
        monthly_hours,
        promotion,
        salary_low,
        salary_medium
    ]],
    columns=X.columns
)

# -------------------------------------------------------
# Prediction
# -------------------------------------------------------

if st.button("Predict"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    if prediction == 1:

        st.error("⚠️ Employee is likely to leave the company.")

    else:

        st.success("✅ Employee is likely to stay with the company.")

    st.subheader("Prediction Probability")

    st.progress(float(max(probability)))

    st.write(
        pd.DataFrame({
            "Outcome": ["Stay", "Leave"],
            "Probability": probability
        })
    )

# -------------------------------------------------------
# Dataset Preview
# -------------------------------------------------------

with st.expander("View Dataset"):

    st.dataframe(df.head())

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
