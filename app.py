import numpy as np
import tensorflow as tf 
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense
from tensorflow.keras.models import load_model
import streamlit as st


word_index = imdb.get_word_index()
reverse_word_index= {value:key for key,value in word_index.items()}

## load the model
model = load_model("Models/model.h5")

def decode_review(review):
    return " ".join([reverse_word_index.get(word_index-3,"?") for word_index in review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500,padding="pre")
    return padded_review

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)

    Sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    return Sentiment, prediction[0][0]

st.title("IMDB Movie Review Prediction")
st.write("Enter a movie review to classify it as positive or negative")

user_input = st.text_input("Enter the review:")

if st.button("Classify"):
    Sentiment, score = predict_sentiment(user_input)
    st.write(f"Sentiment: {Sentiment}")
    st.write(f"Prediction Score: {score}")

else:
    st.write("Please enter the movie review:")


