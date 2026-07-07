import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Outlier Removal using Percentiles",
    page_icon="📊",
    layout="wide"
)

# -------------------------------
# Header
# -------------------------------
st.title("📊 Outlier Detection & Removal using Percentiles")
st.markdown("""
This app demonstrates how to remove **outliers** from Airbnb NYC listing prices
using the **1st and 99.9th percentile**.
""")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("AB_NYC_2019.csv")

try:
    df = load_data()
except FileNotFoundError:
    st.error("AB_NYC_2019.csv not found. Please place it in the project folder.")
    st.stop()

# -------------------------------
# Dataset Preview
# -------------------------------
st.header("Dataset Preview")

st.dataframe(df.head())

st.write("### Dataset Shape")
st.write(df.shape)

# -------------------------------
# Original Statistics
# -------------------------------
st.header("Price Statistics (Original Data)")
st.dataframe(df["price"].describe())

# -------------------------------
# Histogram Before Cleaning
# -------------------------------
fig_before = px.histogram(
    df,
    x="price",
    nbins=100,
    title="Price Distribution Before Removing Outliers"
)

st.plotly_chart(fig_before, use_container_width=True)

# -------------------------------
# Percentile Calculation
# -------------------------------
min_threshold, max_threshold = df.price.quantile([0.01, 0.999])

st.header("Calculated Percentiles")

col1, col2 = st.columns(2)

with col1:
    st.metric("1st Percentile", f"{min_threshold:.2f}")

with col2:
    st.metric("99.9th Percentile", f"{max_threshold:.2f}")

# -------------------------------
# Outliers
# -------------------------------
st.header("Detected Outliers")

lower = df[df.price < min_threshold]
upper = df[df.price > max_threshold]

st.write("Lower Outliers:", lower.shape[0])
st.write("Upper Outliers:", upper.shape[0])

# -------------------------------
# Remove Outliers
# -------------------------------
df_clean = df[
    (df.price > min_threshold) &
    (df.price < max_threshold)
]

st.header("Dataset After Removing Outliers")

st.write("Original Shape:", df.shape)
st.write("Cleaned Shape:", df_clean.shape)

st.dataframe(df_clean.head())

# -------------------------------
# Statistics After Cleaning
# -------------------------------
st.header("Price Statistics (Cleaned Data)")
st.dataframe(df_clean["price"].describe())

# -------------------------------
# Histogram After Cleaning
# -------------------------------
fig_after = px.histogram(
    df_clean,
    x="price",
    nbins=100,
    title="Price Distribution After Removing Outliers"
)

st.plotly_chart(fig_after, use_container_width=True)

# -------------------------------
# Download Cleaned Dataset
# -------------------------------
csv = df_clean.to_csv(index=False).encode("utf-8")

st.download_button(
    "📥 Download Cleaned Dataset",
    csv,
    "AB_NYC_Cleaned.csv",
    "text/csv"
)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("## 👨‍💻 Connect with Me")

c1, c2 = st.columns(2)

with c1:
    st.markdown(
        """
        **GitHub**  
        https://github.com/YOUR_GITHUB_USERNAME
        """
    )

with c2:
    st.markdown(
        """
        **LinkedIn**  
        https://linkedin.com/in/YOUR_LINKEDIN_USERNAME
        """
    )

st.markdown("---")
st.caption("Made with ❤️ using Streamlit")