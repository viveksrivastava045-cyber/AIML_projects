import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Iris Flower Classification using KNN",
    page_icon="🌸",
    layout="wide"
)

# -----------------------------------
# Load Dataset
# -----------------------------------
@st.cache_data
def load_data():
    iris = load_iris()

    df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )

    df["Species"] = [iris.target_names[i] for i in iris.target]

    return iris, df

iris, df = load_data()

# -----------------------------------
# Train Model
# -----------------------------------
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=1
)

model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# -----------------------------------
# Title
# -----------------------------------
st.title("🌸 Iris Flower Classification")
st.markdown("### Machine Learning using K-Nearest Neighbors (KNN)")

# -----------------------------------
# Sidebar
# -----------------------------------
st.sidebar.header("Enter Flower Measurements")

sepal_length = st.sidebar.slider(
    "Sepal Length (cm)",
    float(df.iloc[:,0].min()),
    float(df.iloc[:,0].max()),
    5.1
)

sepal_width = st.sidebar.slider(
    "Sepal Width (cm)",
    float(df.iloc[:,1].min()),
    float(df.iloc[:,1].max()),
    3.5
)

petal_length = st.sidebar.slider(
    "Petal Length (cm)",
    float(df.iloc[:,2].min()),
    float(df.iloc[:,2].max()),
    1.4
)

petal_width = st.sidebar.slider(
    "Petal Width (cm)",
    float(df.iloc[:,3].min()),
    float(df.iloc[:,3].max()),
    0.2
)

# -----------------------------------
# Prediction
# -----------------------------------
prediction = model.predict([[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]])

species = iris.target_names[prediction[0]]

# -----------------------------------
# Layout
# -----------------------------------
col1, col2 = st.columns([1,1])

with col1:
    st.subheader("Dataset Preview")
    st.dataframe(df, use_container_width=True)

    st.metric(
        "Model Accuracy",
        f"{accuracy*100:.2f}%"
    )

with col2:
    st.subheader("Prediction")

    st.write("Input Values")

    input_df = pd.DataFrame({
        "Feature":[
            "Sepal Length",
            "Sepal Width",
            "Petal Length",
            "Petal Width"
        ],
        "Value":[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    })

    st.table(input_df)

    st.success(f"Predicted Flower: **{species.title()}**")

    probabilities = model.predict_proba([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])[0]

    prob_df = pd.DataFrame({
        "Species": iris.target_names,
        "Probability": probabilities
    })

    st.subheader("Prediction Probabilities")
    st.bar_chart(prob_df.set_index("Species"))

st.markdown("---")
st.write("Developed using **Scikit-Learn + Streamlit**")