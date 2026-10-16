"""
Live sign recognition from your laptop webcam, with spoken output.

Uses the model trained by train_sign_model.py.

Setup:
    pip install pyttsx3        (plus the packages from training)
    Linux only: sudo apt install espeak-ng

Usage:
    python recognize_live.py --model export/sign_model.keras
    python recognize_live.py --model export/sign_model.tflite --flip

Keys: q = quit

IMPORTANT: use --flip (mirror the webcam) if your training clips were
recorded mirrored/selfie-style, and leave it off if they weren't. MediaPipe
labels hands Left/Right assuming a mirrored image, so this must match
training or the two hand slots get swapped.
"""
import argparse
import json
import os
import queue
import threading
import time
from collections import Counter, deque

import cv2
import numpy as np
import mediapipe as mp

from train_sign_model import (NUM_FEATURES, NUM_FRAMES, draw_landmarks,
                              ensure_hand_model, frame_features, make_landmarker)


class Speaker:
    """Text-to-speech on a background thread so the video never freezes."""

    def __init__(self):
        self.q = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        import pyttsx3
        engine = pyttsx3.init()
        while True:
            text = self.q.get()
            if text is None:
                break
            engine.say(text)
            engine.runAndWait()

    def say(self, text):
        self.q.put(text)

    def close(self):
        self.q.put(None)


def load_predictor(path):
    """Returns f(x[1, NUM_FRAMES, NUM_FEATURES]) -> probabilities[num_classes]."""
    import tensorflow as tf
    if path.endswith(".tflite"):
        interp = tf.lite.Interpreter(model_path=path)
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]["index"]
        out = interp.get_output_details()[0]["index"]

        def predict(x):
            interp.set_tensor(inp, x)
            interp.invoke()
            return interp.get_tensor(out)[0]
        return predict

    model = tf.keras.models.load_model(path)
    return lambda x: model(x, training=False).numpy()[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="export/sign_model.keras")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--flip", action="store_true", help="mirror the webcam image")
    ap.add_argument("--window", type=float, default=None,
                    help="seconds of video per prediction (default: typical "
                         "training clip length from model_info.json)")
    ap.add_argument("--threshold", type=float, default=0.8,
                    help="min average confidence to accept a phrase")
    ap.add_argument("--votes", type=int, default=6,
                    help="how many of the last 8 predictions must agree")
    ap.add_argument("--no-speech", action="store_true", help="print only, no audio")
    args = ap.parse_args()

    info_path = os.path.join(os.path.dirname(args.model), "model_info.json")
    info = json.load(open(info_path))
    labels = info["labels"]
    assert info["num_frames"] == NUM_FRAMES and info["num_features"] == NUM_FEATURES, \
        "Model was trained with a different feature layout."
    window = args.window or info.get("clip_seconds", 2.0)
    print(f"Phrases: {labels}\nWindow: {window:.2f}s  (press q to quit)")

    predict = load_predictor(args.model)
    speaker = None if args.no_speech else Speaker()
    ensure_hand_model()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam.")

    buffer = deque()                 # (time, features, hand_visible)
    votes = deque(maxlen=8)          # (label_index, confidence)
    last_spoken, last_hand_time = None, time.monotonic()
    shown_label, shown_conf, spoken_text = "-", 0.0, ""
    t0, last_ts, frame_no = time.monotonic(), -1, 0

    with make_landmarker() as landmarker:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if args.flip:
                bgr = cv2.flip(bgr, 1)

            now = time.monotonic() - t0
            ts = max(int(now * 1000), last_ts + 1)       # must strictly increase
            last_ts = ts
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

            visible = bool(result.hand_landmarks)
            if visible:
                last_hand_time = time.monotonic()
            buffer.append((now, frame_features(result), visible))
            while buffer and buffer[0][0] < now - window:
                buffer.popleft()
            draw_landmarks(bgr, result)

            # Hands gone for a moment -> forget state so a phrase can be repeated
            if time.monotonic() - last_hand_time > 1.0:
                votes.clear()
                last_spoken = None
                shown_label, shown_conf = "-", 0.0

            # Predict every 3rd frame once the window is mostly full
            frame_no += 1
            span = buffer[-1][0] - buffer[0][0]
            hand_frac = sum(b[2] for b in buffer) / len(buffer)
            if frame_no % 3 == 0 and span >= 0.8 * window and hand_frac >= 0.3:
                idx = np.linspace(0, len(buffer) - 1, NUM_FRAMES).astype(int)
                seq = np.stack([buffer[i][1] for i in idx])[None].astype(np.float32)
                probs = predict(seq)
                k = int(np.argmax(probs))
                votes.append((k, float(probs[k])))
                shown_label, shown_conf = labels[k], float(probs[k])

                if len(votes) == votes.maxlen:
                    top, n = Counter(v[0] for v in votes).most_common(1)[0]
                    avg = np.mean([v[1] for v in votes if v[0] == top])
                    if n >= args.votes and avg >= args.threshold:
                        name = labels[top]
                        if name == "none":
                            last_spoken = None          # pause between signs
                        elif name != last_spoken:
                            last_spoken = name
                            spoken_text = name.replace("_", " ")
                            print(f"[{time.strftime('%H:%M:%S')}] {spoken_text} "
                                  f"({avg:.0%})")
                            if speaker:
                                speaker.say(spoken_text)
                            votes.clear()

            cv2.putText(bgr, f"seeing: {shown_label} ({shown_conf:.0%})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(bgr, f"said: {spoken_text}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("Sign recognition", bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    if speaker:
        speaker.close()


if __name__ == "__main__":
    main()