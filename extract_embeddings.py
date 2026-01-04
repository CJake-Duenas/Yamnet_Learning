import os
import glob
import numpy as np
import librosa
import tensorflow_hub as hub
import tensorflow as tf

YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
SAMPLE_RATE = 16000

def load_audio(path, sr=SAMPLE_RATE):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    audio = audio.astype('float32')
    return audio

def get_embedding(model, wav):
    scores, embeddings, spectrogram = model(wav)
    emb = tf.reduce_mean(embeddings, axis=0).numpy()
    return emb

def main(dataset_dir="dataset", out_npz="embeddings.npz"):
    print("Loading YAMNet model...")
    model = hub.load(YAMNET_HANDLE)
    print("Model loaded.")

    X = []
    y = []

    # 0 = not cry, 1 = cry
    for label_idx, folder_name in [(0, "not_cry"), (1, "cry")]:
        folder_path = os.path.join(dataset_dir, folder_name)
        files = glob.glob(os.path.join(folder_path, "*.wav"))
        print(f"Found {len(files)} files in {folder_name}")

        for f in files:
            print("Processing:", f)
            audio = load_audio(f)
            audio_tf = tf.constant(audio)
            emb = get_embedding(model, audio_tf)

            X.append(emb)
            y.append(label_idx)

    X = np.array(X)
    y = np.array(y)

    np.savez(out_npz, X=X, y=y)
    print("Saved embeddings to", out_npz)

if __name__ == "__main__":
    main()
