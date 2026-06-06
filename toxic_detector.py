# ============================================================
#   Toxic Comment Classification - LSTM Model Training
#   Dataset  : Jigsaw Toxic Comment Classification (Kaggle)
#   Framework: TensorFlow / Keras
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout,
    SpatialDropout1D, Bidirectional
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import re
import warnings
warnings.filterwarnings('ignore')

# ── Reproducibility ──────────────────────────────────────────
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
#   1.  CONFIGURATION  (change paths here if needed)
# ============================================================
DATA_PATH       = 'data/train.csv'       # Path to Jigsaw train.csv
MODEL_SAVE_PATH = 'outputs/lstm_toxic_model.h5'
TOKENIZER_PATH  = 'outputs/tokenizer.pkl'
HISTORY_IMG     = 'outputs/training_history.png'
CONFUSION_IMG   = 'outputs/confusion_matrix.png'

MAX_VOCAB_SIZE  = 20000   # Maximum number of unique words
MAX_SEQ_LENGTH  = 200     # Maximum length of each comment
EMBEDDING_DIM   = 128     # Embedding vector size
LSTM_UNITS      = 128     # Number of LSTM units
DROPOUT_RATE    = 0.3
BATCH_SIZE      = 64
EPOCHS          = 3     # Reduced from 10 for faster training

os.makedirs('outputs', exist_ok=True)

# ============================================================
#   2.  LOAD DATASET
# ============================================================
print("\n📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"   Dataset shape : {df.shape}")
print(f"   Columns       : {list(df.columns)}")
print(df.head(3))

# ── Create binary label ──────────────────────────────────────
# Jigsaw has multiple toxicity columns; we merge them into one
toxic_cols = ['toxic', 'severe_toxic', 'obscene',
              'threat', 'insult', 'identity_hate']

df['label'] = (df[toxic_cols].sum(axis=1) > 0).astype(int)

print(f"\n   Label distribution:")
print(df['label'].value_counts())
print(f"   0 = Non-Toxic | 1 = Toxic")

# ── Keep only needed columns ─────────────────────────────────
df = df[['comment_text', 'label']].dropna()

# ── Sample 50,000 rows for faster training ───────────────────
# (Remove this line if you want to train on full dataset)
df = df.sample(50000, random_state=42).reset_index(drop=True)
print(f"\n   ✅ Using 50,000 samples for faster training")
print(f"   Label distribution after sampling:")
print(df['label'].value_counts())

# ============================================================
#   3.  TEXT CLEANING
# ============================================================
def clean_text(text):
    """Remove noise from raw comment text."""
    text = str(text).lower()                          # lowercase
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # remove URLs
    text = re.sub(r'<.*?>', '', text)                  # remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)               # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()           # remove extra spaces
    return text

print("\n🧹 Cleaning text...")
df['clean_text'] = df['comment_text'].apply(clean_text)
print("   Cleaning done!")
print(df[['comment_text', 'clean_text', 'label']].head(3))

# ============================================================
#   4.  TOKENIZATION & PADDING
# ============================================================
print("\n🔡 Tokenizing text...")

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(df['clean_text'])

sequences = tokenizer.texts_to_sequences(df['clean_text'])
padded    = pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH,
                          padding='post', truncating='post')

print(f"   Vocabulary size : {len(tokenizer.word_index)}")
print(f"   Padded shape    : {padded.shape}")

# ── Save tokenizer ───────────────────────────────────────────
with open(TOKENIZER_PATH, 'wb') as f:
    pickle.dump(tokenizer, f)
print(f"   ✅ Tokenizer saved → {TOKENIZER_PATH}")

# ============================================================
#   5.  TRAIN / TEST SPLIT
# ============================================================
X = padded
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Split:")
print(f"   Train samples : {X_train.shape[0]}")
print(f"   Test  samples : {X_test.shape[0]}")

# ============================================================
#   6.  BUILD LSTM MODEL
# ============================================================
print("\n🏗️  Building LSTM model...")

model = Sequential([
    Embedding(input_dim=MAX_VOCAB_SIZE,
              output_dim=EMBEDDING_DIM,
              input_length=MAX_SEQ_LENGTH),
    SpatialDropout1D(0.2),
    Bidirectional(LSTM(LSTM_UNITS, dropout=0.2, recurrent_dropout=0.2)),
    Dense(64, activation='relu'),
    Dropout(DROPOUT_RATE),
    Dense(1, activation='sigmoid')   # Binary: toxic or not
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================================
#   7.  CALLBACKS
# ============================================================
early_stop = EarlyStopping(
    monitor='val_loss', patience=3,
    restore_best_weights=True, verbose=1
)

checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH, monitor='val_accuracy',
    save_best_only=True, verbose=1
)

# ============================================================
#   8.  TRAIN MODEL
# ============================================================
print("\n🚀 Training started...")

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

print("\n✅ Training complete!")

# ============================================================
#   9.  EVALUATE MODEL
# ============================================================
print("\n📈 Evaluating on test set...")

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"   Test Loss     : {loss:.4f}")
print(f"   Test Accuracy : {accuracy*100:.2f}%")

y_pred_prob = model.predict(X_test, verbose=0)
y_pred      = (y_pred_prob > 0.5).astype(int).flatten()

print("\n📋 Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=['Non-Toxic', 'Toxic']
))

# ============================================================
#   10. PLOT TRAINING HISTORY
# ============================================================
print("\n📉 Saving training history plot...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#07071a')

for ax in axes:
    ax.set_facecolor('#0d0d2b')
    ax.tick_params(colors='#e8dfc8')
    for spine in ax.spines.values():
        spine.set_edgecolor('#f0c040')

# Accuracy
axes[0].plot(history.history['accuracy'],
             color='#f0c040', linewidth=2, label='Train Accuracy')
axes[0].plot(history.history['val_accuracy'],
             color='#00e676', linewidth=2, label='Val Accuracy')
axes[0].set_title('Model Accuracy', color='#f0c040', fontsize=14, pad=12)
axes[0].set_xlabel('Epoch', color='#e8dfc8')
axes[0].set_ylabel('Accuracy', color='#e8dfc8')
axes[0].legend(facecolor='#0d0d2b', labelcolor='#e8dfc8')

# Loss
axes[1].plot(history.history['loss'],
             color='#ff6b6b', linewidth=2, label='Train Loss')
axes[1].plot(history.history['val_loss'],
             color='#ffaa00', linewidth=2, label='Val Loss')
axes[1].set_title('Model Loss', color='#f0c040', fontsize=14, pad=12)
axes[1].set_xlabel('Epoch', color='#e8dfc8')
axes[1].set_ylabel('Loss', color='#e8dfc8')
axes[1].legend(facecolor='#0d0d2b', labelcolor='#e8dfc8')

plt.tight_layout()
plt.savefig(HISTORY_IMG, dpi=150, bbox_inches='tight',
            facecolor='#07071a')
plt.close()
print(f"   ✅ Saved → {HISTORY_IMG}")

# ============================================================
#   11. CONFUSION MATRIX
# ============================================================
print("\n🔲 Saving confusion matrix...")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor('#07071a')
ax.set_facecolor('#0d0d2b')

sns.heatmap(
    cm, annot=True, fmt='d',
    cmap='YlOrRd',
    xticklabels=['Non-Toxic', 'Toxic'],
    yticklabels=['Non-Toxic', 'Toxic'],
    linewidths=1, linecolor='#07071a',
    ax=ax
)

ax.set_title('Confusion Matrix', color='#f0c040', fontsize=15, pad=14)
ax.set_xlabel('Predicted Label', color='#e8dfc8', fontsize=12)
ax.set_ylabel('True Label',      color='#e8dfc8', fontsize=12)
ax.tick_params(colors='#e8dfc8')

plt.tight_layout()
plt.savefig(CONFUSION_IMG, dpi=150, bbox_inches='tight',
            facecolor='#07071a')
plt.close()
print(f"   ✅ Saved → {CONFUSION_IMG}")

print("\n🎉 All done! Your model is trained and saved.")
print(f"   Model     → {MODEL_SAVE_PATH}")
print(f"   Tokenizer → {TOKENIZER_PATH}")
