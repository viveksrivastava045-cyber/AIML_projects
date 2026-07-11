import streamlit as st
from transformers import pipeline

# 1. Page Configuration
st.set_page_config(
    page_title="KGF 2 Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# 2. Model Loading with Caching
# @st.cache_resource ensures the model is only downloaded/loaded once into memory
@st.cache_resource
def load_classifier():
    # Initializes the default sentiment-analysis pipeline
    return pipeline("sentiment-analysis")

classifier = load_classifier()

# 3. App UI Header & Image
st.title("🎬 KGF 2 Review Sentiment Analyzer")

# --- NEW: Added the image here ---
# use_column_width=True ensures the image scales nicely to fit the app's width
st.image("PROJECT_11_KGF_REVIEW/ChatGPT Image Jul 11, 2026, 03_46_39 PM.png", use_column_width=True)
# ---------------------------------

st.markdown("Analyze the sentiment of audience reviews using LLMs. Type a review below to see if it's **POSITIVE** or **NEGATIVE**.")

# 4. User Input
default_review = "KGF 2 is an amazing movie with powerful action and excellent performance."
sample_review = st.text_area("Enter Movie Review:", value=default_review, height=150)

# 5. Prediction Logic
if st.button("Analyze Sentiment", type="primary"):
    if sample_review.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        with st.spinner("Analyzing sentiment..."):
            # Generate sentiment prediction
            sentiment_result = classifier(sample_review)
            
            # Extract label and confidence score
            sentiment = sentiment_result[0]['label']
            confidence = sentiment_result[0]['score']
            
            # 6. Display Results
            st.markdown("### Result")
            if sentiment == "POSITIVE":
                st.success(f"**Sentiment:** {sentiment} 📈")
                st.info(f"**Confidence Score:** {confidence:.4f}")
            else:
                st.error(f"**Sentiment:** {sentiment} 📉")
                st.info(f"**Confidence Score:** {confidence:.4f}")

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
    st.write("## Vivek Srivastava")
    st.write("B.Tech IT ")
    st.write("Machine Learning & Data Science Enthusiast")

with col2:
    st.markdown("### 🔗 Connect with Me")
    st.markdown("[💼 LinkedIn](https://www.linkedin.com/in/vivek-srivastava-0a878a329)")
    st.markdown("[💻 GitHub](https://github.com/viveksrivastava045-cyber/AIML_projects/edit/main/PROJECT_03_CANADA_per_capita_income)")

st.markdown("---")
st.caption("Made with ❤️ using Python, Scikit-Learn & Streamlit")
