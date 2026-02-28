import streamlit as st
import joblib
import spacy

# Load trained model
model = joblib.load(r'D:\NLP\Emotions\emotion_model.pkl')

# Load spacy model
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)
    return " ".join(filtered_tokens)

# UI
st.title("Emotion Detection App 😃")
st.write("Type a sentence and I will detect the emotion (Joy, Fear, Anger)")

user_input = st.text_area("Enter your text here:")

if st.button("Predict Emotion"):
    if user_input.strip() != "":
        
        # Apply same preprocessing
        clean_text = preprocess(user_input)

        prediction = model.predict([clean_text])[0]

        emotion_map = {
            0: "Joy 😂",
            1: "Fear 😨",
            2: "Anger 😡"
        }

        st.success(f"Predicted Emotion: {emotion_map[prediction]}")

    else:
        st.warning("Please enter some text.")