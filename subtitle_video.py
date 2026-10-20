"""
Add subtitles to a video of someone signing, using the model trained by
train_sign_model.py (the same model the live recognizer and the app use).

Outputs, next to each input video:
    <name>_subtitled.mp4   video with the recognized phrases burned in
    <name>.srt             subtitle file (works as soft subtitles in VLC etc.)
and prints a transcript with timestamps.

Usage:
    python subtitle_video.py my_signing_video.mp4
    python subtitle_video.py a.mp4 b.mp4 --flip
    python subtitle_video.py clip.mp4 --threshold 0.6 --debug

How it works: landmarks are extracted from every frame, then a window about as
long as your training clips slides across the video. Each window is classified
exactly like a training clip (resampled to 30 frames). Runs of windows that agree
on the same phrase become one subtitle.

--flip: mirror frames for DETECTION only (the output video is never mirrored).
Use it if your training clips were recorded mirrored/selfie-style, so that
MediaPipe's Left/Right labels match training. Leave it off otherwise.

Optional: install ffmpeg to keep the original audio and get a more widely
playable mp4. Without it you still get the .srt and a silent subtitled video.
"""
import argparse
import json
import os
import shutil
import subprocess

import cv2
import numpy as np
import mediapipe as mp

from recognize_live import load_predictor
from train_sign_model import (NUM_FEATURES, NUM_FRAMES, ensure_hand_model,
                              frame_features, make_landmarker)

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# 1. Landmarks for every frame
# ---------------------------------------------------------------------------
def extract_all(path, flip):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    feats, hands = [], []
    with make_landmarker() as landmarker:       # fresh tracking state
        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if flip:
                bgr = cv2.flip(bgr, 1)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                int(i * 1000 / fps))
            feats.append(frame_features(result))
            hands.append(bool(result.hand_landmarks))
            i += 1
            if i % 100 == 0:
                print(f"  analyzed {i}/{total or '?'} frames", end="\r")
    cap.release()
    print(f"  analyzed {len(feats)} frames        ")
    return np.array(feats, dtype=np.float32), np.array(hands), fps


# ---------------------------------------------------------------------------
# 2. Sliding-window classification
# ---------------------------------------------------------------------------
def classify_windows(feats, hands, fps, window_s, stride_s, predict, min_hand_frac=0.3):
    """Returns a list of (start_frame, end_frame, label_index, confidence)."""
    n = len(feats)
    win = max(5, int(round(window_s * fps)))
    stride = max(1, int(round(stride_s * fps)))
    starts = list(range(0, max(1, n - win + 1), stride))
    if n > win and starts[-1] != n - win:
        starts.append(n - win)                   # make sure the end is covered
    out = []
    for s in starts:
        e = min(n, s + win)
        if hands[s:e].mean() < min_hand_frac:    # nobody signing here
            continue
        idx = np.linspace(s, e - 1, NUM_FRAMES).astype(int)
        probs = predict(feats[idx][None].astype(np.float32))
        k = int(np.argmax(probs))
        out.append((s, e, k, float(probs[k])))
    return out


# ---------------------------------------------------------------------------
# 3. Windows -> subtitle segments
# ---------------------------------------------------------------------------
def build_segments(windows, labels, threshold, min_run, fps):
    """Returns a list of [start_sec, end_sec, label, mean_confidence]."""
    segs, run = [], []

    def flush():
        if len(run) >= min_run:
            s = min(w[0] for w in run)
            e = max(w[1] for w in run)
            conf = float(np.mean([w[3] for w in run]))
            segs.append([s / fps, e / fps, labels[run[0][2]], conf])
        run.clear()

    for w in windows:
        s, e, k, c = w
        accepted = labels[k] != "none" and c >= threshold
        if accepted and run and run[-1][2] == k and s <= run[-1][1]:
            run.append(w)
        else:
            flush()
            if accepted:
                run.append(w)
    flush()

    # Different phrases must not overlap on screen
    for i in range(len(segs) - 1):
        if segs[i][1] > segs[i + 1][0]:
            segs[i][1] = segs[i + 1][0]
    return [s for s in segs if s[1] > s[0]]


def pretty(label):
    return label.replace("_", " ")


# ---------------------------------------------------------------------------
# 4. Output: SRT + burned-in video
# ---------------------------------------------------------------------------
def srt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def write_srt(segs, path, phrases):
    with open(path, "w", encoding="utf-8") as f:
        for i, (s, e, label, _) in enumerate(segs, 1):
            f.write(f"{i}\n{srt_time(s)} --> {srt_time(e)}\n"
                    f"{phrases.get(label, pretty(label))}\n\n")


def draw_caption(frame, text):
    h, w = frame.shape[:2]
    scale = max(0.6, min(2.0, w / 800))
    thick = max(1, int(scale * 2))
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    x = max(0, (w - tw) // 2)
    y = h - int(0.08 * h)
    pad = int(10 * scale)
    overlay = frame.copy()
    cv2.rectangle(overlay, (max(0, x - pad), y - th - pad),
                  (min(w, x + tw + pad), y + base + pad), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, text, (x, y), FONT, scale, (255, 255, 255), thick, cv2.LINE_AA)


def render_video(src, dst, segs, phrases):
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        for s, e, label, _ in segs:
            if s <= t < e:
                draw_caption(frame, phrases.get(label, pretty(label)))
                break
        writer.write(frame)
        i += 1
    cap.release()
    writer.release()


def add_audio(silent_video, original, final):
    """Re-encode to widely playable H.264 and copy the original audio (ffmpeg)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    cmd = [ffmpeg, "-y", "-i", silent_video, "-i", original,
           "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", final]
    return subprocess.run(cmd, capture_output=True).returncode == 0


# ---------------------------------------------------------------------------
def process(video, args, predict, labels, window_s, phrases):
    print(f"\n{video}")
    feats, hands, fps = extract_all(video, args.flip)
    if len(feats) < 5:
        print("  video too short or unreadable, skipping")
        return
    windows = classify_windows(feats, hands, fps, window_s, args.stride, predict)

    if args.debug:
        print("  window start-end (s)   prediction")
        for s, e, k, c in windows:
            print(f"  {s / fps:6.2f}-{e / fps:6.2f}        {labels[k]} ({c:.0%})")

    segs = build_segments(windows, labels, args.threshold, args.min_run, fps)

    base, _ = os.path.splitext(video)
    srt_path, out_path = base + ".srt", base + "_subtitled.mp4"
    write_srt(segs, srt_path, phrases)

    if segs:
        print("  Transcript:")
        for s, e, label, c in segs:
            print(f"    {srt_time(s)} -> {srt_time(e)}  "
                  f"{phrases.get(label, pretty(label))}  ({c:.0%})")
    else:
        print("  No signs recognized. Try --threshold 0.6, check --flip, "
              "or run with --debug to see raw predictions.")

    tmp = base + "_tmp_noaudio.mp4"
    render_video(video, tmp, segs, phrases)
    if add_audio(tmp, video, out_path):
        os.remove(tmp)
        print(f"  Saved {out_path} (with audio) and {srt_path}")
    else:
        os.replace(tmp, out_path)
        print(f"  Saved {out_path} (no audio; install ffmpeg to keep it) "
              f"and {srt_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", help="one or more video files")
    ap.add_argument("--model", default="C:/Users/Louie/Documents/Stuff/STEAMCongress/exports/sign_model.keras")
    ap.add_argument("--flip", action="store_true",
                    help="mirror frames for detection (match your training clips)")
    ap.add_argument("--window", type=float, default=None,
                    help="window length in seconds (default: typical training clip length)")
    ap.add_argument("--stride", type=float, default=0.25,
                    help="seconds between windows (smaller = finer timing, slower)")
    ap.add_argument("--threshold", type=float, default=0.8,
                    help="min confidence for a window to count")
    ap.add_argument("--min-run", type=int, default=2,
                    help="min consecutive agreeing windows to make a subtitle")
    ap.add_argument("--debug", action="store_true", help="print every window's prediction")
    args = ap.parse_args()

    model_dir = os.path.dirname(args.model)
    info = json.load(open(os.path.join(model_dir, "model_info.json")))
    labels = info["labels"]
    assert info["num_frames"] == NUM_FRAMES and info["num_features"] == NUM_FEATURES, \
        "Model was trained with a different feature layout."
    window_s = args.window or info.get("clip_seconds", 2.0)

    phrases = {}
    pf = os.path.join(model_dir, "phrases.json")
    if os.path.exists(pf):
        phrases = json.load(open(pf, encoding="utf-8"))

    print(f"Phrases: {labels}\nWindow: {window_s:.2f}s, stride {args.stride}s")
    ensure_hand_model()
    predict = load_predictor(args.model)
    for v in args.videos:
        process(v, args, predict, labels, window_s, phrases)


if __name__ == "__main__":
    main()