import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

## Load the imdb dataset and word index 
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key,value in word_index.items()}

## Load the pretrained model with Relu activation function
model = load_model('simple_rnn_imdb.h5')

## Step 2: Helper Functions
# Function to decode the reviews
def decode_review(encoded_review):
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])
    return decoded_review

## Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

### Step 3: Creating prediction function
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction > 0.5 else 'Negative'
    return sentiment, prediction[0][0]


## Design the Streamlit App
import streamlit as st
st.title('IMDB Movie Review Sentiment Analysis')
st.write("Enter a movie review to classify it as positive or negitive.")

user_input = st.text_area("Enter your review here:")

if st.button("Classify"):
    preprocess_input = preprocess_text(user_input)

    ## Make the Prediction
    prediction = model.predict(preprocess_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    # Display the result on the screen
    st.write(f"The sentiment of the review is: {sentiment}")
    st.write(f"The Prediction Score is: {prediction[0][0]}")
else:
    st.write("Please enter a movie review")

