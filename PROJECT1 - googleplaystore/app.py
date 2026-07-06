import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Google Play Store Analysis", layout="wide")

st.title("📱 Google Play Store Data Analysis Dashboard")
st.markdown("Interactive dashboard to analyze factors affecting app performance.")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("googleplaystore_v2.csv")
    return df

df = load_data()

# ---------------- Sidebar ----------------
st.sidebar.header("Filters")

category = st.sidebar.multiselect(
    "Category",
    df["Category"].dropna().unique(),
    default=df["Category"].dropna().unique()
)

df = df[df["Category"].isin(category)]

# ---------------- Dataset ----------------
st.header("Dataset Preview")

st.dataframe(df.head())

st.write("Number of Apps:", len(df))

# ---------------- Statistics ----------------
st.header("Dataset Statistics")

st.dataframe(df.describe())

# ---------------- Rating Distribution ----------------
st.header("App Rating Distribution")

fig, ax = plt.subplots()

df["Rating"].dropna().hist(
    bins=20,
    ax=ax
)

ax.set_xlabel("Rating")
ax.set_ylabel("Count")

st.pyplot(fig)

# ---------------- Installs vs Rating ----------------
st.header("Installs vs Rating")

fig, ax = plt.subplots()

ax.scatter(
    df["Installs"],
    df["Rating"],
    alpha=0.4
)

ax.set_xlabel("Installs")
ax.set_ylabel("Rating")

st.pyplot(fig)

# ---------------- Price vs Rating ----------------
st.header("Price vs Rating")

fig, ax = plt.subplots()

ax.scatter(
    df["Price"],
    df["Rating"],
    alpha=0.4,
)

ax.set_xlabel("Price ($)")
ax.set_ylabel("Rating")

st.pyplot(fig)

# ---------------- Size vs Rating ----------------
st.header("Size vs Rating")

fig, ax = plt.subplots()

ax.scatter(
    df["Size"],
    df["Rating"],
    alpha=0.4,
)

ax.set_xlabel("Size (MB)")
ax.set_ylabel("Rating")

st.pyplot(fig)

# ---------------- Category Average Rating ----------------
st.header("Average Rating by Category")

avg_rating = (
    df.groupby("Category")["Rating"]
    .mean()
    .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(10,6))

avg_rating.plot.bar(ax=ax)

plt.xticks(rotation=90)

st.pyplot(fig)

# ---------------- Correlation ----------------
st.header("Correlation Matrix")

corr = df.select_dtypes(include="number").corr()

st.dataframe(corr)

# ---------------- Business Insights ----------------
st.header("Business Insights")

st.success("""
• Apps with higher installs generally have better visibility, but not always higher ratings.

• Free apps dominate the Play Store.

• Price alone does not improve ratings.

• Large app size does not guarantee better performance.

• Categories differ significantly in average ratings.
""")
