# ============================================================
#   Toxic Comment Classification - Prediction Script
#   Load the saved LSTM model and predict new comments
# ============================================================

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
#   CONFIGURATION  (must match toxic_detector.py settings)
# ============================================================
MODEL_PATH     = 'outputs/lstm_toxic_model.h5'
TOKENIZER_PATH = 'outputs/tokenizer.pkl'
MAX_SEQ_LENGTH = 200
THRESHOLD      = 0.7    # Increased to 0.7 to reduce false positives

# ── Negation words that flip the meaning of a sentence ───────
NEGATION_WORDS = [
    'not', "n't", 'never', 'no', 'nobody', 'nothing',
    'neither', 'nor', 'nowhere', 'hardly', 'barely', 'dont',
    'wont', 'cant', 'isnt', 'arent', 'wasnt', 'werent',
    'hasnt', 'havent', 'hadnt', 'didnt', 'doesnt'
]

# ── Strong toxic words that if negated should be non-toxic ────
STRONG_TOXIC_WORDS = [
    'idiot', 'stupid', 'dumb', 'moron', 'hate', 'kill',
    'ugly', 'loser', 'worthless', 'trash', 'garbage',
    'pathetic', 'freak', 'jerk', 'fool', 'useless'
]

# ============================================================
#   LOAD MODEL & TOKENIZER
# ============================================================
def load_resources():
    """Load the saved model and tokenizer from disk."""

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"❌ Model not found at '{MODEL_PATH}'\n"
            "   Please run toxic_detector.py first to train the model."
        )

    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(
            f"❌ Tokenizer not found at '{TOKENIZER_PATH}'\n"
            "   Please run toxic_detector.py first to train the model."
        )

    print("⏳ Loading model and tokenizer...")
    model     = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)

    print("✅ Model and tokenizer loaded successfully!\n")
    return model, tokenizer


# ============================================================
#   TEXT CLEANING  (same as training)
# ============================================================
def clean_text(text):
    """Clean raw input text before prediction."""
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================================================
#   NEGATION DETECTION
# ============================================================
def has_negation_before_toxic_word(text):
    """
    Check if a negation word appears just before a toxic word.
    Example: 'you are not an idiot' → negation detected → reduce toxicity
    Returns True if negation is found near a toxic word.
    """
    words = text.lower().split()
    for i, word in enumerate(words):
        # Clean punctuation from word for matching
        clean_word = re.sub(r'[^a-z]', '', word)
        if clean_word in NEGATION_WORDS or "n't" in word:
            # Check the next 4 words after the negation word
            window = words[i+1 : i+5]
            for w in window:
                clean_w = re.sub(r'[^a-z]', '', w)
                if clean_w in STRONG_TOXIC_WORDS:
                    return True
    return False


# ============================================================
#   PREDICT FUNCTION
# ============================================================
def predict_comment(comment, model, tokenizer):
    """
    Predict whether a comment is TOXIC or NON-TOXIC.
    Includes negation handling and raised threshold for accuracy.

    Returns:
        label      : 'TOXIC' or 'NON-TOXIC'
        confidence : float percentage (e.g. 94.3)
    """
    cleaned  = clean_text(comment)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded   = pad_sequences(sequence, maxlen=MAX_SEQ_LENGTH,
                             padding='post', truncating='post')

    prob = model.predict(padded, verbose=0)[0][0]

    # ── Negation check ────────────────────────────────────────
    # If negation is detected before a toxic word, reduce the
    # toxic probability significantly to handle cases like
    # "you are not an idiot" correctly
    if has_negation_before_toxic_word(comment):
        prob = prob * 0.35   # reduce toxic probability by 65%

    if prob >= THRESHOLD:
        label      = 'TOXIC'
        confidence = round(float(prob) * 100, 2)
    else:
        label      = 'NON-TOXIC'
        confidence = round((1 - float(prob)) * 100, 2)

    return label, confidence


# ============================================================
#   DISPLAY RESULT
# ============================================================
def display_result(comment, label, confidence):
    """Pretty-print the prediction result."""
    print("─" * 55)
    print(f"  💬 Comment   : {comment[:80]}{'...' if len(comment)>80 else ''}")
    print(f"  🏷️  Prediction : {label}")
    print(f"  📊 Confidence : {confidence}%")

    if label == 'TOXIC':
        print("  ☠️  Status    : This comment contains harmful content")
    else:
        print("  ✅ Status    : This comment appears to be clean")

    print("─" * 55)


# ============================================================
#   MAIN — Interactive Loop
# ============================================================
def main():
    print("\n" + "═" * 55)
    print("   👑  TOXIC COMMENT CLASSIFICATION  👑")
    print("   Powered by Bidirectional LSTM")
    print("═" * 55 + "\n")

    model, tokenizer = load_resources()

    # ── Demo predictions ──────────────────────────────────────
    demo_comments = [
        "You are so stupid and completely worthless!",
        "Thank you for the wonderful explanation, it helped a lot!",
        "I will destroy you, you absolute idiot.",
        "Have a great day, stay safe and keep smiling.",
        "You are not an idiot, you are actually very smart!",
        "He is never a loser, he always tries his best.",
    ]

    print("── Demo Predictions ────────────────────────────────\n")
    for comment in demo_comments:
        label, confidence = predict_comment(comment, model, tokenizer)
        display_result(comment, label, confidence)
        print()

    # ── Interactive mode ──────────────────────────────────────
    print("\n── Interactive Mode (type 'quit' to exit) ──────────\n")

    while True:
        user_input = input("  Enter a comment: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n  👋 Goodbye! Stay respectful online.\n")
            break

        if not user_input:
            print("  ⚠️  Please enter a comment.\n")
            continue

        label, confidence = predict_comment(user_input, model, tokenizer)
        print()
        display_result(user_input, label, confidence)
        print()


if __name__ == '__main__':
    main()
