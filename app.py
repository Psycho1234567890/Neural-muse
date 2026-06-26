from flask import Flask, render_template, request
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
import os

app = Flask(__name__)

# ------------------------------
# Load tokenizer
# ------------------------------
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# ------------------------------
# Model parameters (must match training)
# ------------------------------
total_words = len(tokenizer.word_index) + 1
MAX_SEQUENCE_LEN = 12   # from your model's input shape (see error logs)

# ------------------------------
# Recreate the model architecture (exactly as in train.py)
# ------------------------------
model = Sequential([
    Embedding(total_words, 100, input_length=MAX_SEQUENCE_LEN),
    SimpleRNN(128, return_sequences=True),
    Dropout(0.2),
    SimpleRNN(64),
    Dense(64, activation='relu'),
    Dense(total_words, activation='softmax')
])

# Build the model with the correct input shape before loading weights
model.build(input_shape=(None, MAX_SEQUENCE_LEN))

# Load the weights from the .keras file (saved locally)
model.load_weights("simple_rnn_model.keras")

# No need to compile – we're only doing inference

# ------------------------------
# Text generation function
# ------------------------------
def generate_text(seed_text, next_words=20, temperature=0.8):
    result = seed_text
    for _ in range(next_words):
        tokens = tokenizer.texts_to_sequences([result])[0]
        tokens = pad_sequences([tokens], maxlen=MAX_SEQUENCE_LEN, padding="pre")
        prediction = model.predict(tokens, verbose=0)[0]

        # Temperature scaling
        prediction = np.log(prediction + 1e-8) / temperature
        prediction = np.exp(prediction)
        prediction = prediction / np.sum(prediction)

        # Sample from the distribution
        word_id = np.random.choice(len(prediction), p=prediction)

        # Find the word corresponding to the ID
        word = ""
        for w, index in tokenizer.word_index.items():
            if index == word_id:
                word = w
                break
        if word == "":
            break
        result += " " + word
    return result

# ------------------------------
# Flask routes
# ------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    output = ""
    seed_text = ""
    if request.method == "POST":
        seed_text = request.form.get("text", "").strip()
        temperature = float(request.form.get("temperature", 0.8))
        word_count = int(request.form.get("word_count", 20))
        if seed_text:
            output = generate_text(seed_text, next_words=word_count, temperature=temperature)
    return render_template("index.html", output=output, seed_text=seed_text)

# ------------------------------
# Run the app
# ------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)