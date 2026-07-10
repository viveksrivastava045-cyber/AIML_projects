import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="Employee Retention Prediction",
    page_icon="👨‍💼",
    layout="wide"
)

st.title("👨‍💼 Employee Retention Prediction")
st.write("Predict whether an employee will leave the company.")

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("PROJECT_04_HR_comma/HR_comma_sep.csv")

df = load_data()

# --------------------------------------------------
# Preprocessing (Exactly Same as Notebook)
# --------------------------------------------------

subdf = df[
    [
        "satisfaction_level",
        "average_montly_hours",
        "promotion_last_5years",
        "salary",
    ]
]

salary_dummies = pd.get_dummies(
    subdf["salary"],
    prefix="salary"
)

df_with_dummies = pd.concat(
    [subdf, salary_dummies],
    axis=1
)

df_with_dummies.drop(
    "salary",
    axis=1,
    inplace=True
)

X = df_with_dummies
y = df["left"]

# Ensure bool dummy columns become integers
X = X.astype(int, errors="ignore")

# --------------------------------------------------
# Train Model
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=0.30,
    random_state=42
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("Model")

st.sidebar.metric(
    "Accuracy",
    f"{accuracy*100:.2f}%"
)

# --------------------------------------------------
# Input Form
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    satisfaction = st.slider(
        "Satisfaction Level",
        0.00,
        1.00,
        0.50,
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
        "Salary",
        ["low", "medium", "high"]
    )

# --------------------------------------------------
# Salary Encoding (IMPORTANT)
# --------------------------------------------------

salary_high = 1 if salary == "high" else 0
salary_low = 1 if salary == "low" else 0
salary_medium = 1 if salary == "medium" else 0

# --------------------------------------------------
# Prediction Data
# --------------------------------------------------

input_data = pd.DataFrame({
    "satisfaction_level": [satisfaction],
    "average_montly_hours": [monthly_hours],
    "promotion_last_5years": [promotion],
    "salary_high": [salary_high],
    "salary_low": [salary_low],
    "salary_medium": [salary_medium],
})

# Match training columns exactly
input_data = input_data.reindex(columns=X.columns)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    if prediction == 1:

        st.error("⚠️ Employee is likely to leave.")

    else:

        st.success("✅ Employee is likely to stay.")

    st.subheader("Prediction Probability")

    result = pd.DataFrame({
        "Outcome": ["Stay", "Leave"],
        "Probability": probability
    })

    st.dataframe(result, use_container_width=True)

# --------------------------------------------------
# Dataset
# --------------------------------------------------

with st.expander("Dataset Preview"):

    st.dataframe(df.head(), use_container_width=True)
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
