import streamlit as st
import joblib
svm_model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer (1).pkl")
st.title("📊 Twitter Sentiment Analysis")
st.write("Analyze the sentiment of a tweet using an SVM model.")
user_input = st.text_area("Enter a tweet:")
if st.button("Analyze Sentiment"):
    if user_input.strip():
        input_tfidf = vectorizer.transform([user_input])
        prediction = svm_model.predict(input_tfidf)
        sentiment_map = {0: "Positive 😊", 1: "Neutral 😐", 2: "Negative 😠", 3: "Mixed 🤔"}
        st.write(f"**Predicted Sentiment:** {sentiment_map[prediction[0]]}")
    else:
        st.warning("Please enter a valid tweet.")
