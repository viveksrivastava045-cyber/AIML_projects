import streamlit as st
import pandas as pd
from transformers import pipeline

st.set_page_config(
    page_title="KGF 2 Review Analysis",
    page_icon="🎬",
    layout="wide"
)

# --------------------------
# Cache Models
# --------------------------
@st.cache_resource
def load_models():
    sentiment = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )

    translator = pipeline(
        "translation_en_to_es",
        model="Helsinki-NLP/opus-mt-en-es"
    )

    qa = pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2"
    )

    return sentiment, summarizer, translator, qa


sentiment_model, summarizer, translator, qa = load_models()

# --------------------------
# Sidebar
# --------------------------
st.sidebar.title("🎬 KGF 2 NLP Project")

option = st.sidebar.radio(
    "Select Task",
    [
        "Sentiment Analysis",
        "Summarization",
        "Translation",
        "Question Answering"
    ]
)

# --------------------------
# Header
# --------------------------
st.title("🎬 Analyzing Netflix KGF 2 Reviews with LLMs")
st.markdown(
"""
Analyze movie reviews using Hugging Face Large Language Models.

- Sentiment Analysis
- Review Summarization
- English → Spanish Translation
- Question Answering
"""
)

review = st.text_area(
    "Enter Movie Review",
    height=180
)

# --------------------------
# Sentiment
# --------------------------
if option == "Sentiment Analysis":

    if st.button("Analyze Sentiment"):

        if review:

            result = sentiment_model(review)[0]

            st.success(f"Prediction : {result['label']}")
            st.info(f"Confidence : {result['score']:.2%}")

# --------------------------
# Summary
# --------------------------
elif option == "Summarization":

    if st.button("Generate Summary"):

        if review:

            summary = summarizer(
                review,
                max_length=60,
                min_length=20,
                do_sample=False
            )

            st.write(summary[0]["summary_text"])

# --------------------------
# Translation
# --------------------------
elif option == "Translation":

    if st.button("Translate"):

        if review:

            translated = translator(review)

            st.success(translated[0]["translation_text"])

# --------------------------
# QA
# --------------------------
elif option == "Question Answering":

    question = st.text_input("Ask a Question")

    if st.button("Answer"):

        if review and question:

            answer = qa(
                question=question,
                context=review
            )

            st.success(answer["answer"])