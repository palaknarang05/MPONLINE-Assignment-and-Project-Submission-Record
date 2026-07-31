import streamlit as st
import tensorflow as tf
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="AI Sentiment Chatbot",
    page_icon="🤖",
    layout="centered"
)

# ---------------------- LOAD CSS ----------------------
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()



# ---------------------- LOAD MODEL ----------------------
model = tf.keras.models.load_model("sentiment_model.keras")

# ---------------------- LOAD TOKENIZER ----------------------
with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)



MAX_LENGTH = 100

# ---------------------- TEXT CLEANING ----------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = " ".join(text.split())
    return text
# ---------------------- CHATBOT RESPONSE ----------------------
def chatbot_response(sentiment):

    if sentiment == "Positive":
        responses = [
            " That's wonderful! I'm happy to hear that.",
            " Great! Keep smiling and enjoy your day.",
            " Amazing! Wishing you continued happiness.",
            " That sounds fantastic!",
            " Keep up the positive energy!"
        ]
    else:
        responses = [
            " I'm sorry you're feeling this way.",
            " Things will get better. Stay strong.",
            " Remember, every difficult moment is temporary.",
            " Take a deep breath. You're doing your best.",
            " I'm here for you. Tomorrow is a new day."
        ]

    import random
    return random.choice(responses)

# ---------------------- TITLE ----------------------
st.title("🤖 AI Sentiment Chatbot")

st.write("Analyze the sentiment of your message using an LSTM model trained on the Sentiment140 Twitter dataset.")

# ---------------------- INPUT ----------------------
text = st.text_area(
    "Type your message",
    height=150,
    placeholder="Example: I got my dream internship today!"
)

# ---------------------- BUTTON ----------------------
if st.button("Analyze Sentiment"):

    if text.strip() == "":
        st.warning("Please enter a message.")
    else:

        cleaned = clean_text(text)

        sequence = tokenizer.texts_to_sequences([cleaned])

        padded = pad_sequences(sequence, maxlen=MAX_LENGTH)

        prediction = model.predict(padded, verbose=0)[0][0]

        confidence = prediction if prediction >= 0.5 else 1 - prediction

        if prediction >= 0.5:
            sentiment = "Positive "
            st.success(f"Sentiment: {sentiment}")
        else:
            sentiment = "Negative "
            st.error(f"Sentiment: {sentiment}")

        st.progress(float(confidence))

        st.write(f"**Confidence:** {confidence*100:.2f}%")

        st.markdown("---")

        st.subheader(" Chatbot Response")

        st.info(chatbot_response(sentiment.split()[0]))