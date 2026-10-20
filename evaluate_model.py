"""
Check how well the trained model does on your own clips, and find out WHICH
clips it gets wrong and why (underfitting vs overfitting vs similar signs).

Uses the landmark cache created by train_sign_model.py, so it is fast and needs
no video processing. It also rebuilds the same train/validation/test split that
training used (same seed), so it can tell you whether a given clip was actually
used for training.

Usage:
    python evaluate_model.py
    python evaluate_model.py --find april        # inspect clips with 'april' in the name/folder
    python evaluate_model.py --clips "D:\\my_clips" --cache landmark_cache

NOTE: the split is only reproduced exactly if your clips have not changed
since the last training run. If you added/removed/replaced clips, retrain first.
"""
import argparse
import json
import os
from collections import Counter

import numpy as np

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def load_cached(clips_dir, cache_dir, labels):
    folders = sorted(d for d in os.listdir(clips_dir)
                     if os.path.isdir(os.path.join(clips_dir, d)))
    index = {l: i for i, l in enumerate(labels)}
    X, y, names, missing = [], [], [], 0
    for folder in folders:
        if folder not in index:
            print(f"warning: folder '{folder}' is not in the model's label list "
                  f"(added after training?) - skipped")
            continue
        for fn in sorted(os.listdir(os.path.join(clips_dir, folder))):
            if not fn.lower().endswith(VIDEO_EXTS):
                continue
            path = os.path.join(clips_dir, folder, fn)
            cache_file = os.path.join(
                cache_dir, f"{folder}__{fn}__{int(os.path.getmtime(path))}.npz")
            if not os.path.exists(cache_file):
                missing += 1          # skipped during training (no hands / too short)
                continue
            X.append(np.load(cache_file)["seq"])
            y.append(index[folder])
            names.append(f"{folder}/{fn}")
    if missing:
        print(f"note: {missing} clips have no cached landmarks "
              f"(skipped in training, or the file changed since)")
    return np.stack(X).astype(np.float32), np.array(y), names


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default=r"C:\Users\Louie\Documents\Stuff\STEAMCongress\clips")
    ap.add_argument("--cache", default="landmark_cache")
    ap.add_argument("--model", default="C:/Users/Louie/Documents/Stuff/STEAMCongress/exports/sign_model.keras")
    ap.add_argument("--find", default=None, help="text to look for in clip names")
    args = ap.parse_args()

    from sklearn.model_selection import train_test_split
    import tensorflow as tf

    labels = json.load(open(os.path.join(os.path.dirname(args.model),
                                         "model_info.json")))["labels"]
    X, y, names = load_cached(args.clips, args.cache, labels)
    print(f"{len(X)} clips, {len(set(y.tolist()))} phrases")

    # Same split as train_sign_model.py
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.3, stratify=y, random_state=42)
    va, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=42)
    split_of = {}
    for name, ids in (("train", tr), ("validation", va), ("test", te)):
        for i in ids:
            split_of[i] = name

    model = tf.keras.models.load_model(args.model)
    probs = model.predict(X, batch_size=64, verbose=0)
    pred = probs.argmax(axis=1)

    def acc(ids):
        return float((pred[ids] == y[ids]).mean()) if len(ids) else float("nan")

    print("\nAccuracy by split:")
    print(f"  train      {acc(tr):6.1%}  ({len(tr)} clips)")
    print(f"  validation {acc(va):6.1%}  ({len(va)} clips)")
    print(f"  test       {acc(te):6.1%}  ({len(te)} clips)")
    gap = acc(tr) - acc(te)
    if acc(tr) < 0.9:
        print("  -> low TRAIN accuracy: the model is underfitting (too small / "
              "stopped too early / too many similar classes).")
    elif gap > 0.25:
        print("  -> big train/test gap: overfitting; it needs more varied clips "
              "per phrase.")

    counts = Counter(y.tolist())
    print("\nPhrases with the fewest clips:")
    for c, n in sorted(counts.items(), key=lambda t: t[1])[:10]:
        print(f"  {labels[c]:<25} {n} clips")

    per = sorted(((labels[c], counts[c], float((pred[y == c] == c).mean()))
                  for c in counts), key=lambda t: t[2])
    print("\nWeakest phrases (accuracy over all their clips):")
    for name, n, a in per[:15]:
        print(f"  {name:<25} {a:6.1%}  ({n} clips)")

    conf = Counter((labels[t], labels[p]) for t, p in zip(y, pred) if t != p)
    print("\nMost common mix-ups (true -> predicted):")
    for (t, p), n in conf.most_common(15):
        print(f"  {t:<22} -> {p:<22} x{n}")

    if args.find:
        print(f"\nClips matching '{args.find}':")
        for i, name in enumerate(names):
            if args.find.lower() in name.lower():
                top3 = np.argsort(probs[i])[::-1][:3]
                guess = ", ".join(f"{labels[j]} {probs[i][j]:.0%}" for j in top3)
                ok = "OK " if pred[i] == y[i] else "BAD"
                print(f"  [{ok}] {name:<40} {split_of[i]:<10} {guess}")


if __name__ == "__main__":
    main()