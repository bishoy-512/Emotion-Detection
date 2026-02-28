# app.py
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load your trained model
model = joblib.load("emotion_model.pkl")

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Simple preprocessing function
def preprocess(text):
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Lowercase and tokenize
    tokens = text.lower().split()
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Streamlit app title
st.title("Emotion Detection App 😃")
st.write("Type a sentence and I will detect the emotion (Joy, Fear, Anger)")

# Input box
user_input = st.text_area("Enter your text here:")

# Predict button
if st.button("Predict Emotion"):
    if user_input.strip() != "":
        # Preprocess the input text
        cleaned_text = preprocess(user_input)
        # Predict
        prediction = model.predict([cleaned_text])[0]

        emotion_map = {
            0: "Joy 😂",
            1: "Fear 😨",
            2: "Anger 😡"
        }

        st.success(f"Predicted Emotion: {emotion_map[prediction]}")
    else:
        st.warning("Please enter some text.")
