import os 
import pandas as pd
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import librosa
from typing import Any


yamnet_model: Any = hub.load('https://tfhub.dev/google/yamnet/1')


def extract_embedding(file_path):
    try:
        waveform, sr = librosa.load(file_path, sr=16000)
    except Exception as e:
        print(f"💔 Error loading audio {file_path}: {repr(e)}")
        return None

    target_length = 16000 * 30  # 30 seconds
    if len(waveform) < target_length:
        waveform = np.pad(waveform, (0, target_length - len(waveform)), mode='constant')
    else:
        waveform = waveform[:target_length]

    waveform = waveform.astype(np.float32)

    try:
        scores, embeddings, spectrogram = yamnet_model(waveform)
        song_embedding = tf.reduce_mean(embeddings, axis=0)
        return song_embedding.numpy()
    except Exception as e:
        print(f"💔 Error during model inference for {file_path}: {repr(e)}")
        return None


extract_list = []
dataset_path = './fma_small'

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.mp3') or file.endswith('.wav'):
            path = os.path.join(root, file)
            embedding = extract_embedding(path)
            if embedding is not None:
                extract_list.append([path] + embedding.tolist() + [None, None, None, None])
                print(f"❤️ Extracted: {path}")
            else:
                print(f"💔 Skipped: {path}")

# Save to CSV
embedding_dim = 1024
columns = ['path'] + [f'embedding_{i}' for i in range(embedding_dim)] + ['title', 'artist', 'album', 'duration']

df = pd.DataFrame(extract_list, columns=columns)
df.to_csv('extracted_songs.csv', index=False)
print("Saved extracted_songs.csv")
