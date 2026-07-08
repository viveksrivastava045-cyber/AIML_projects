import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Canada Per Capita Income Prediction",
    page_icon="📈",
    layout="wide"
)

# --------------------------------------------------
# Title
# --------------------------------------------------
# st.title("📈 Canada Per Capita Income Prediction")
# st.markdown(
#     """
# This app uses **Linear Regression** to predict the **Per Capita Income (US$)** of Canada
# based on historical data.
# """
# )
st.markdown("""
<div style="
padding:35px;
background:linear-gradient(135deg,#2563EB,#7C3AED);
border-radius:18px;
text-align:center;
box-shadow:0px 8px 20px rgba(0,0,0,0.4);">

<h1 style="
color:white;
font-size:52px;
font-weight:800;">
📈 Canada Per Capita Income Predictor
</h1>

<p style="
color:#E5E7EB;
font-size:20px;">
Predict Future Income using <b>Machine Learning</b> 🤖
</p>

</div>
""", unsafe_allow_html=True)
# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
df = pd.read_csv("canada_per_capita_income.csv")

# Display Dataset
st.subheader("📊 Dataset")
st.dataframe(df, use_container_width=True)

# --------------------------------------------------
# Train Model
# --------------------------------------------------
X = df[['year']]
y = df['per capita income (US$)']

model = LinearRegression()
model.fit(X, y)

# --------------------------------------------------
# Model Information
# --------------------------------------------------
st.subheader("📌 Model Information")

col1, col2 = st.columns(2)

with col1:
    st.metric("Slope", f"{model.coef_[0]:.2f}")

with col2:
    st.metric("Intercept", f"{model.intercept_:.2f}")

# --------------------------------------------------
# Prediction
# --------------------------------------------------
st.subheader("🔮 Predict Income")

year = st.number_input(
    "Enter Year",
    min_value=1960,
    max_value=2050,
    value=2020,
    step=1
)

prediction = model.predict([[year]])

st.success(
    f"Estimated Per Capita Income in **{year}** is **${prediction[0]:,.2f}**"
)

# --------------------------------------------------
# Regression Graph
# --------------------------------------------------
st.subheader("📉 Regression Plot")

fig, ax = plt.subplots(figsize=(10,5))

ax.scatter(
    df['year'],
    df['per capita income (US$)'],
    color="blue",
    label="Actual Data"
)

ax.plot(
    df['year'],
    model.predict(df[['year']]),
    color="red",
    linewidth=3,
    label="Regression Line"
)

ax.set_xlabel("Year")
ax.set_ylabel("Per Capita Income (US$)")
ax.set_title("Canada Per Capita Income")
ax.legend()

st.pyplot(fig)

# --------------------------------------------------
# Predict Future Years
# --------------------------------------------------
st.subheader("📈 Future Predictions")

future_years = np.arange(year, year+11).reshape(-1,1)
future_income = model.predict(future_years)

future_df = pd.DataFrame({
    "Year": future_years.flatten(),
    "Predicted Income (US$)": future_income
})

st.dataframe(future_df, use_container_width=True)


# --------------------------------------------------
# Footer
# --------------------------------------------------


st.markdown("---")

st.markdown("""
<div style="
background: linear-gradient(135deg, #1E293B, #0F172A);
padding:25px;
border-radius:15px;
border:2px solid #38BDF8;
text-align:center;
box-shadow:0px 4px 15px rgba(0,0,0,0.3);
">

<h2 style="color:#38BDF8;">👨‍💻 Richeek Pandey</h2>

<h4 style="color:#F8FAFC;">
📊 Aspiring Data Scientist | Machine Learning Enthusiast
</h4>

<p style="color:#CBD5E1; font-size:16px;">
This application was built using
<b style="color:#FACC15;">Python</b>,
<b style="color:#22C55E;">Scikit-learn</b>,
<b style="color:#F97316;">Pandas</b>,
<b style="color:#A855F7;">Matplotlib</b> &
<b style="color:#EF4444;">Streamlit</b>.
</p>

<p style="font-size:18px;">
<a href="https://www.linkedin.com/in/richeek-pandey-9954783a9" target="_blank"
style="text-decoration:none;color:#38BDF8;">
💼 LinkedIn
</a>

&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;

<a href="https://github.com/richeekpandey07" target="_blank"
style="text-decoration:none;color:#22C55E;">
💻 GitHub
</a>
</p>

<p style="color:#FACC15; font-size:17px;">
⭐ Thank you for exploring my project!
</p>

</div>
""", unsafe_allow_html=True)
