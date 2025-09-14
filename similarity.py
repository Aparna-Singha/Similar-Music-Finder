from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import librosa
import tensorflow_hub as hub
import tensorflow as tf
from typing import Any
import os

print("Current working directory:", os.getcwd())
print("Files in current directory:", os.listdir('.'))

yamnet_model: Any = hub.load('https://tfhub.dev/google/yamnet/1')

def extract_embedding(file_path):
    try:
        waveform, sr = librosa.load(file_path, sr=16000)
    except Exception as e:
        print(f"💔 Error loading {file_path}: {e}")
        return None

    target_length = 16000 * 30  # 30 seconds
    if len(waveform) < target_length:
        waveform = np.pad(waveform, (0, target_length - len(waveform)), mode='constant')
    else:
        waveform = waveform[:target_length]
    
    waveform = waveform.astype(np.float32)  
    
    scores, embeddings, spectrogram = yamnet_model(waveform)
    song_embedding = tf.reduce_mean(embeddings, axis=0)
    return song_embedding.numpy()

# Load database CSV
db = pd.read_csv('/Users/aparnasingha/Documents/Music Prediction 3/extract_songs/extracted_songs.csv')

# Confirm embedding columns (assuming 1024 dimensions)
embedding_columns = [f'embedding_{i}' for i in range(1024)]

# Extract user embedding
user_embedding = extract_embedding('/Users/aparnasingha/Documents/Music Prediction 3/extract_songs/song.mp3')

if user_embedding is not None:
    user_embedding = user_embedding.reshape(1, -1)
    cosine_similarities = cosine_similarity(user_embedding, db[embedding_columns].values)
    db['similarity'] = cosine_similarities[0]
    top_matches = db.sort_values(by='similarity', ascending=False).head(10)
    print("❤️")
    print(top_matches[["path", "similarity"]])
else:
    print("💔 Failed to extract embedding for the user file.")
