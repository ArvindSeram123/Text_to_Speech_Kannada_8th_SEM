import streamlit as st
from gtts import gTTS
from pydub import AudioSegment
import tempfile
import joblib

# Define your sentiment label mapping
label_map = {
    0: "Angry",
    1: "Happy",
    2: "Neutral",
    3: "Sad"
}

# Define your playback speeds
speed_map = {
    "Happy": 1.5,     # Faster
    "Neutral": 1.0,   # Normal
    "Sad": 0.7        # Slower
}

# Define tokenizer again for vectorizer to load
def my_tokenizer(s):
    return s.split(' ')

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit UI
st.title("Kannada Sentiment Analyzer + Audio Playback")

user_input = st.text_area("Enter Kannada sentence:")

if st.button("Analyze and Speak") and user_input.strip():
    # Predict sentiment
    input_vec = vectorizer.transform([user_input])
    pred = model.predict(input_vec)[0]
    sentiment = label_map.get(pred, "Unknown")

    st.success(f"Predicted Sentiment: **{sentiment}**")

    # Convert to speech
    tts = gTTS(text=user_input, lang='kn')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as mp3_fp:
        tts.save(mp3_fp.name)

        # Load with pydub
        sound = AudioSegment.from_file(mp3_fp.name, format="mp3")

        # Change speed
        speed = speed_map.get(sentiment, 1.0)
        slowed = sound._spawn(sound.raw_data, overrides={
            "frame_rate": int(sound.frame_rate * speed)
        }).set_frame_rate(44100)

        # Export and play
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_fp:
            slowed.export(wav_fp.name, format="wav")
            st.audio(wav_fp.name, format="audio/wav")
