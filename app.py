from flask import Flask, render_template, request
import tensorflow as tf
import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load model and tokenizer
model = tf.keras.models.load_model("simple_rnn_model.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

def generate_text(seed_text, next_words=20, temperature=0.8):
    result = seed_text
    for _ in range(next_words):
        tokens = tokenizer.texts_to_sequences([result])[0]
        tokens = pad_sequences([tokens], maxlen=50, padding="pre")
        prediction = model.predict(tokens, verbose=0)[0]
        # Apply temperature
        prediction = np.log(prediction + 1e-8) / temperature
        prediction = np.exp(prediction)
        prediction = prediction / np.sum(prediction)
        word_id = np.random.choice(len(prediction), p=prediction)
        # Find word
        word = ""
        for w, index in tokenizer.word_index.items():
            if index == word_id:
                word = w
                break
        if word == "":
            break
        result += " " + word
    return result

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)