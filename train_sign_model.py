"""
Train a sign-phrase classifier from video clips, on a laptop.

Pipeline:
  clips/<phrase>/*.mp4  --(hand_landmarker.task)-->  landmark sequences
                        --(LSTM training)-->  sign_model.tflite + labels.txt

Folder layout:
    clips/
      hello/       a.mp4  b.mp4 ...
      thank_you/   ...
      none/        (idle hands / random movement - strongly recommended)

Usage:
    pip install mediapipe opencv-python numpy tensorflow scikit-learn
    python train_sign_model.py --clips clips --out export
    python train_sign_model.py --clips clips --preview   # check detection first

Landmark extraction results are cached, so re-training after tweaking
hyperparameters doesn't re-process the videos.

Outputs (in --out): sign_model.keras (for Python), sign_model.tflite (for the
future Android app), labels.txt, model_info.json.
Then run recognize_live.py to use the model with your webcam + speech.
"""
import argparse
import json
import os
import urllib.request
import zipfile

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---- Feature layout (the Android app must reproduce this exactly) ----------
NUM_FRAMES = 30                      # frames per sequence
NUM_HANDS = 2                        # slot 0 = Left, slot 1 = Right
FEATS_PER_HAND = 21 * 3 + 3          # 63 wrist-relative coords + 3 absolute wrist
NUM_FEATURES = NUM_HANDS * FEATS_PER_HAND   # 132

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hand_landmarker.task")
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


# ---------------------------------------------------------------------------
# Landmark extraction
# ---------------------------------------------------------------------------
def _is_hand_landmarker(path):
    """A real hand_landmarker.task is a zip containing hand_detector.tflite."""
    try:
        with zipfile.ZipFile(path) as z:
            return "hand_detector.tflite" in z.namelist()
    except Exception:
        return False


def ensure_hand_model():
    """Make sure MODEL_PATH holds the genuine Hand Landmarker model."""
    if os.path.exists(MODEL_PATH) and not _is_hand_landmarker(MODEL_PATH):
        print(f"{MODEL_PATH} is not a valid Hand Landmarker model; replacing it.")
        os.remove(MODEL_PATH)
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand_landmarker.task ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        if not _is_hand_landmarker(MODEL_PATH):
            raise SystemExit("Downloaded file is not a valid Hand Landmarker model. "
                             "Download it manually (see MODEL_URL) and retry.")
    print(f"Using hand model: {os.path.abspath(MODEL_PATH)}")


def make_landmarker():
    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=NUM_HANDS,
    )
    return vision.HandLandmarker.create_from_options(opts)


def hand_features(landmarks):
    """21 landmarks -> 66 numbers: wrist-relative, scale-normalized, + wrist abs."""
    pts = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
    wrist = pts[0].copy()
    rel = pts - wrist
    rel /= (np.linalg.norm(rel[9]) + 1e-6)   # wrist -> middle finger base
    return np.concatenate([rel.flatten(), wrist])


def frame_features(result):
    vec = np.zeros(NUM_FEATURES, dtype=np.float32)
    for lm, handed in zip(result.hand_landmarks, result.handedness):
        slot = 0 if handed[0].category_name == "Left" else 1
        vec[slot * FEATS_PER_HAND:(slot + 1) * FEATS_PER_HAND] = hand_features(lm)
    return vec


def draw_landmarks(bgr, result):
    h, w = bgr.shape[:2]
    for lm in result.hand_landmarks:
        for p in lm:
            cv2.circle(bgr, (int(p.x * w), int(p.y * h)), 3, (0, 255, 0), -1)


def video_to_sequence(path, preview=False):
    """Returns (sequence[NUM_FRAMES, NUM_FEATURES], duration_seconds) or None."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    with make_landmarker() as landmarker:   # fresh state for every clip
        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(img, int(i * 1000 / fps))
            frames.append(frame_features(result))
            if preview:
                draw_landmarks(bgr, result)
                cv2.imshow("preview (q to skip clip)", bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            i += 1
    cap.release()
    if preview:
        cv2.destroyAllWindows()
    if len(frames) < 5:
        return None
    idx = np.linspace(0, len(frames) - 1, NUM_FRAMES).astype(int)
    return np.stack(frames)[idx], len(frames) / fps


def build_dataset(clips_dir, cache_dir, preview=False):
    ensure_hand_model()
    os.makedirs(cache_dir, exist_ok=True)
    labels = sorted(d for d in os.listdir(clips_dir)
                    if os.path.isdir(os.path.join(clips_dir, d)))
    if len(labels) < 2:
        raise SystemExit("Need at least 2 phrase folders inside the clips directory.")

    X, y, durations = [], [], []
    for li, label in enumerate(labels):
        folder = os.path.join(clips_dir, label)
        count = 0
        for fn in sorted(os.listdir(folder)):
            if not fn.lower().endswith(VIDEO_EXTS):
                continue
            path = os.path.join(folder, fn)
            cache_file = os.path.join(
                cache_dir, f"{label}__{fn}__{int(os.path.getmtime(path))}.npz")
            if os.path.exists(cache_file) and not preview:
                c = np.load(cache_file)
                seq, dur = c["seq"], float(c["duration"])
            else:
                out = video_to_sequence(path, preview)
                if out is None:
                    print(f"  skip (too short/unreadable): {label}/{fn}")
                    continue
                seq, dur = out
                if not seq.any():
                    print(f"  skip (no hands detected): {label}/{fn}")
                    continue
                np.savez(cache_file, seq=seq, duration=dur)
            X.append(seq)
            y.append(li)
            durations.append(dur)
            count += 1
        print(f"{label}: {count} clips")
        if count < 5:
            print(f"  warning: '{label}' has very few clips; expect poor accuracy")
    return np.stack(X).astype(np.float32), np.array(y), labels, np.array(durations)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def augment(X):
    """Light jitter + small time shift (training data only)."""
    X = X + np.random.normal(0, 0.01, X.shape).astype(np.float32)
    return np.roll(X, np.random.randint(-3, 4), axis=1)


def build_model(num_classes):
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(NUM_FRAMES, NUM_FEATURES)),
        tf.keras.layers.Masking(mask_value=0.0),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])


def train_and_export(X, y, labels, durations, out_dir, epochs):
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    # Hold out validation and test sets (stratified so every phrase appears)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

    X_tr = np.concatenate([X_tr] + [augment(X_tr) for _ in range(4)])
    y_tr = np.concatenate([y_tr] * 5)

    model = build_model(len(labels))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        X_tr, y_tr, validation_data=(X_val, y_val), epochs=epochs, batch_size=32,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            patience=15, restore_best_weights=True)],
    )

    print("\nTest set results:")
    pred = model.predict(X_te).argmax(axis=1)
    print(classification_report(
        y_te, pred, labels=list(range(len(labels))),
        target_names=labels, zero_division=0))

    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "sign_model.keras"))   # for Python use

    # Typical signing duration (ignoring the idle class): the live recognizer
    # uses this as its sliding-window length so it matches the training clips.
    real = np.array([labels[i] != "none" for i in y])
    clip_seconds = float(np.median(durations[real] if real.any() else durations))

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    try:
        tflite = converter.convert()
    except Exception as e:
        print(f"Plain conversion failed ({e}); retrying with SELECT_TF_OPS ...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter._experimental_lower_tensor_list_ops = False
        tflite = converter.convert()

    with open(os.path.join(out_dir, "sign_model.tflite"), "wb") as f:
        f.write(tflite)
    with open(os.path.join(out_dir, "labels.txt"), "w") as f:
        f.write("\n".join(labels))
    with open(os.path.join(out_dir, "model_info.json"), "w") as f:
        json.dump({"num_frames": NUM_FRAMES, "num_features": NUM_FEATURES,
                   "clip_seconds": round(clip_seconds, 2),
                   "labels": labels}, f, indent=2)
    print(f"\nSaved to {out_dir}/: sign_model.keras, sign_model.tflite, "
          f"labels.txt, model_info.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default="C:/Users/Louie/Documents/Stuff/STEAMCongress/clips", help="folder of phrase subfolders")
    ap.add_argument("--out", default="C:/Users/Louie/Documents/Stuff/STEAMCongress/exports", help="output folder")
    ap.add_argument("--cache", default="landmark_cache", help="landmark cache folder")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--preview", action="store_true",
                    help="show detected landmarks on each clip, then exit (no training)")
    args = ap.parse_args()

    X, y, labels, durations = build_dataset(args.clips, args.cache, preview=args.preview)
    if args.preview:
        print("Preview done. Re-run without --preview to train.")
        return
    print(f"\nDataset: {X.shape[0]} sequences, shape {X.shape[1:]}, "
          f"{len(labels)} classes\n")
    train_and_export(X, y, labels, durations, args.out, args.epochs)


if __name__ == "__main__":
    main()