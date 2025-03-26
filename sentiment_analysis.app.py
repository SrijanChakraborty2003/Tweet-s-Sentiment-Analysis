import streamlit as st
import joblib
import numpy as np

# Load the trained model and vectorizer
svm_model = joblib.load("svm_model (1).pkl")
vectorizer = joblib.load("tfidf_vectorizer (2).pkl")

# Function to predict sentiment
def predict_sentiment(text):
    """Predicts the sentiment of the given text."""
    text_transformed = vectorizer.transform([text])  # Convert input text to TF-IDF
    prediction = svm_model.predict(text_transformed)[0]  # Predict sentiment

    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_map[prediction]

# Streamlit UI
st.title("Twitter Sentiment Analysis")
st.write("Enter a tweet and let the model predict its sentiment.")

# User input
user_input = st.text_area("Enter tweet here:")

if st.button("Analyze Sentiment"):
    if user_input.strip():  # Ensure input is not empty
        sentiment = predict_sentiment(user_input)
        st.write(f"**Predicted Sentiment:** {sentiment}")
    else:
        st.warning("Please enter some text for analysis.")
