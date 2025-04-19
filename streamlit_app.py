import streamlit as st
from gtts import gTTS
import tempfile
import joblib

# Define sentiment labels
label_map = {
    0: "Angry",
    1: "Happy",
    2: "Neutral",
    3: "Sad"
}

# Tokenizer used during training
def my_tokenizer(s):
    return s.split(" ")

# Load trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit app interface
st.title("Kannada Sentiment Analyzer with Text-to-Speech")

user_input = st.text_area("Enter a Kannada sentence:")

if st.button("Analyze and Speak") and user_input.strip():
    input_vec = vectorizer.transform([user_input])
    pred = model.predict(input_vec)[0]
    sentiment = label_map.get(pred, "Unknown")
    st.success(f"Predicted Sentiment: **{sentiment}**")

    # Convert input to speech using gTTS (Kannada)
    tts = gTTS(text=user_input, lang='kn')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        st.audio(fp.name, format="audio/mp3")
