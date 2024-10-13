import os
import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import tensorflow as tf  # Import TensorFlow for logging control

# Set TensorFlow logging level to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents, words, classes, and model
with open('intents.json') as json_file:
    intents = json.load(json_file)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    """Tokenize and lemmatize the input sentence."""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    """Create a bag of words representation for the input sentence."""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    """Predict the class of the input sentence."""
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    result.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in result:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    """Get a random response from the intents."""
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def chat():
    """Start the chatbot interaction loop."""
    print("Bot is running!")
    print("Type 'exit' or 'quit' to stop the bot.")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Bot: Goodbye!")
            break  # Exit the loop if the user types 'exit' or 'quit'

        intents_list = predict_class(user_input)
        if intents_list:
            response = get_response(intents_list, intents)
            print("Bot:", response)
        else:
            print("Bot: I didn't understand that.")

if __name__ == "__main__":
    chat()
