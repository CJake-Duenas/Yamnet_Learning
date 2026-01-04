import queue
import threading
import time
import numpy as np
import sounddevice as sd
import librosa
import tensorflow_hub as hub
import tensorflow as tf
import joblib
import tkinter as tk
from tkinter import ttk
import soundfile as sf

# -------------------------------------------------------
# SETTINGS
# -------------------------------------------------------
SAMPLE_RATE = 16000        # Force this — stable for all systems
CHUNK_SEC = 1.0
CHUNK = int(SAMPLE_RATE * CHUNK_SEC)

YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
MODEL_PATH = "cry_clf.joblib"
THRESH = 0.77
SMOOTH_WINDOW = 5

ALERT_SOUND = "cheer.wav"  # Must be in Yamnet_Learning folder
ALERT_DURATION = 1.0       # seconds
ALERT_COOLDOWN = 3.0       # seconds before another alert

q = queue.Queue()
last_alert_time = 0


# -------------------------------------------------------
# AUDIO PLAYER (non-blocking)
# -------------------------------------------------------
def play_sound(filepath, duration=1.0):
    try:
        audio, sr = sf.read(filepath, dtype='float32')
        # Trim to 1 second
        desired_samples = int(sr * duration)
        audio = audio[:desired_samples]

        sd.play(audio, sr)
    except Exception as e:
        print("Error playing sound:", e)


# -------------------------------------------------------
# TKINTER UI
# -------------------------------------------------------
class DetectorUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Baby Cry Detector")
        self.root.geometry("450x200")

        self.label = tk.Label(
            self.root,
            text="Listening...",
            font=("Arial", 32),
            fg="green"
        )
        self.label.pack(expand=True)

        self.root.update()

    def show_listening(self):
        self.label.config(text="Listening...", fg="green")
        self.root.update()

    def show_alert(self):
        self.label.config(text="Cry detected!", fg="red")
        self.root.update()


ui = DetectorUI()  # Start UI immediately


# -------------------------------------------------------
# AUDIO CALLBACK
# -------------------------------------------------------
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)

    mono = np.mean(indata, axis=1)
    q.put(mono.copy())


# -------------------------------------------------------
# WORKER THREAD
# -------------------------------------------------------
def worker():
    global last_alert_time

    print("Loading YAMNet...")
    yamnet = hub.load(YAMNET_HANDLE)

    print("Loading classifier...")
    model_bundle = joblib.load(MODEL_PATH)
    scaler = model_bundle["scaler"]
    clf = model_bundle["clf"]

    print("Ready. Listening...")

    history = []

    while True:
        seg = q.get()
        if seg is None:
            break

        # Always resample from mic to 16k
        seg_rs = librosa.resample(
            seg.astype(np.float32),
            orig_sr=SAMPLE_RATE,     # we FORCE the mic to output 16k
            target_sr=SAMPLE_RATE
        )

        wav_tf = tf.constant(seg_rs)
        _, embeddings, _ = yamnet(wav_tf)
        emb = tf.reduce_mean(embeddings, axis=0).numpy().reshape(1, -1)

        emb_s = scaler.transform(emb)
        prob = clf.predict_proba(emb_s)[0, 1]

        history.append(prob)
        if len(history) > SMOOTH_WINDOW:
            history.pop(0)

        avg_prob = float(np.mean(history))

        print(f"prob={prob:.3f}, avg={avg_prob:.3f}")

        # ---------------------------------------------------
        # ALERT SECTION
        # ---------------------------------------------------
        if avg_prob >= THRESH:
            now = time.time()

            if now - last_alert_time > ALERT_COOLDOWN:
                last_alert_time = now

                ui.show_alert()
                play_sound(ALERT_SOUND, ALERT_DURATION)

                # Reset UI after alert
                ui.root.after(int(ALERT_DURATION * 1000), ui.show_listening)


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    print("Starting microphone stream...")

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    # Force mic to 16k
    sd.default.samplerate = SAMPLE_RATE
    sd.default.channels = 1

    try:
        with sd.InputStream(
            callback=audio_callback,
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK
        ):
            ui.root.mainloop()

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        q.put(None)
        t.join()


if __name__ == "__main__":
    main()
