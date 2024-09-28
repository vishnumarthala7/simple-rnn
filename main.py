import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


# Load word index and reverse word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}
model = load_model('sample_rnn_imdb.h5')

# Function to decode encoded review back into human-readable text
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3, '?') for i in encoded_review])

# Preprocess the text into encoded and padded form for model input
def preprocess_text(text):
    words = text.lower().split()  # Split words in lowercase
    encoded_review = [word_index.get(word, 2) + 3 for word in words]  # Encode words, fallback to 2 if not found
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)  # Pad sequences to maxlen of 500
    return padded_review

# Predict sentiment function
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
    return sentiment, prediction[0][0]

# Streamlit app
import streamlit as st
st.cache_data.clear()

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative")

user_input = st.text_area('Movie Review')

if st.button('Classify'):
    if user_input.strip() != '':  # Check if the input is not empty
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)
        sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
        st.write(f'Sentiment: {sentiment}')
        st.write(f'Prediction Score: {prediction[0][0]:.4f}')
    else:
        st.write("Please enter a valid movie review.")
