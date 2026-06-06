# ============================================================
#   Toxic Comment Classification - Flask Backend
#   Connects the LSTM model to the royal frontend
# ============================================================

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import re
import os

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import warnings
warnings.filterwarnings('ignore')

# ============================================================
#   CONFIGURATION
# ============================================================
MODEL_PATH     = 'outputs/lstm_toxic_model.h5'
TOKENIZER_PATH = 'outputs/tokenizer.pkl'
MAX_SEQ_LENGTH = 200
THRESHOLD      = 0.7

NEGATION_WORDS = [
    'not', "n't", 'never', 'no', 'nobody', 'nothing',
    'neither', 'nor', 'nowhere', 'hardly', 'barely', 'dont',
    'wont', 'cant', 'isnt', 'arent', 'wasnt', 'werent',
    'hasnt', 'havent', 'hadnt', 'didnt', 'doesnt'
]

STRONG_TOXIC_WORDS = [
    'idiot', 'stupid', 'dumb', 'moron', 'hate', 'kill',
    'ugly', 'loser', 'worthless', 'trash', 'garbage',
    'pathetic', 'freak', 'jerk', 'fool', 'useless'
]

# ============================================================
#   FLASK APP INIT
# ============================================================
app = Flask(__name__)

# ── Load model & tokenizer once at startup ───────────────────
print("⏳ Loading model and tokenizer...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"❌ Model not found at '{MODEL_PATH}'. "
        "Please run toxic_detector.py first."
    )
if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(
        f"❌ Tokenizer not found at '{TOKENIZER_PATH}'. "
        "Please run toxic_detector.py first."
    )

model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

print("✅ Model and tokenizer loaded successfully!")

# ============================================================
#   HELPER FUNCTIONS
# ============================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def has_negation_before_toxic_word(text):
    words = text.lower().split()
    for i, word in enumerate(words):
        clean_word = re.sub(r'[^a-z]', '', word)
        if clean_word in NEGATION_WORDS or "n't" in word:
            window = words[i+1: i+5]
            for w in window:
                clean_w = re.sub(r'[^a-z]', '', w)
                if clean_w in STRONG_TOXIC_WORDS:
                    return True
    return False


def predict_comment(comment):
    cleaned  = clean_text(comment)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded   = pad_sequences(sequence, maxlen=MAX_SEQ_LENGTH,
                             padding='post', truncating='post')
    prob = model.predict(padded, verbose=0)[0][0]

    if has_negation_before_toxic_word(comment):
        prob = prob * 0.35

    if prob >= THRESHOLD:
        label      = 'TOXIC'
        confidence = round(float(prob) * 100, 2)
    else:
        label      = 'NON-TOXIC'
        confidence = round((1 - float(prob)) * 100, 2)

    return label, confidence

# ============================================================
#   ROUTES
# ============================================================
@app.route('/')
def index():
    """Serve the royal frontend page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Receive comment from frontend via POST request,
    run prediction and return JSON result.
    """
    data    = request.get_json()
    comment = data.get('comment', '').strip()

    if not comment:
        return jsonify({'error': 'No comment provided'}), 400

    label, confidence = predict_comment(comment)

    return jsonify({
        'label'     : label,
        'confidence': confidence,
        'comment'   : comment
    })


# ============================================================
#   RUN SERVER
# ============================================================
if __name__ == '__main__':
    print("\n🚀 Flask server starting...")
    print("   Open your browser and go to: http://localhost:5000")
    print("   Press CTRL+C to stop the server\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
