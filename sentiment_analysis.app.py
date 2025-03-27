import streamlit as st
import joblib
import numpy as np
svm_model = joblib.load("svm_model (1).pkl")
vectorizer = joblib.load("tfidf_vectorizer (2).pkl")
def predict_sentiment(text):
    """Predicts the sentiment of the given text."""
    text_transformed = vectorizer.transform([text])
    prediction = svm_model.predict(text_transformed)[0]
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_map[prediction]
st.title("Twitter Sentiment Analysis")
st.write("Enter a tweet and let the model predict its sentiment.")
user_input = st.text_area("Enter tweet here:")
if st.button("Analyze Sentiment"):
    if user_input.strip():
        sentiment = predict_sentiment(user_input)
        st.write(f"**Predicted Sentiment:** {sentiment}")
    else:
        st.warning("Please enter some text for analysis.")
