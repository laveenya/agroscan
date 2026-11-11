"""Evaluation helpers: top-1/top-3, macro-F1, confidence interval, confidence table, confusion matrix."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score


def predict_all(model, ds, tta=False):
    probs, ys = [], []
    for x, y in ds:
        p = model(x, training=False).numpy().astype("float32")
        if tta:
            p = (p + model(tf.image.flip_left_right(x), training=False).numpy().astype("float32")) / 2
        probs.append(p)
        ys.append(y.numpy().argmax(1))
    return np.concatenate(probs), np.concatenate(ys)


def bootstrap_ci(y_true, y_pred, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    accs = [(y_true[i] == y_pred[i]).mean() for i in (rng.integers(0, len(y_true), len(y_true)) for _ in range(n))]
    return np.percentile(accs, [2.5, 97.5])


def summarize(probs, y_true, n_classes):
    y_pred = probs.argmax(1)
    lo, hi = bootstrap_ci(y_true, y_pred)
    return {
        "top1": float((y_pred == y_true).mean()),
        "top3": float((np.argsort(-probs, 1)[:, :3] == y_true[:, None]).any(1).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0)),
        "ci_low": float(lo), "ci_high": float(hi), "images": int(len(y_true)),
    }


def selective_table(probs, y_true, thresholds=(0.0, 0.5, 0.6, 0.7, 0.8, 0.9)):
    conf, pred = probs.max(1), probs.argmax(1)
    rows = []
    for t in thresholds:
        keep = conf >= t
        rows.append({"min_confidence": t, "answered_share": round(float(keep.mean()), 3),
                     "accuracy_on_answered": round(float((pred[keep] == y_true[keep]).mean()), 3) if keep.any() else None})
    return pd.DataFrame(rows)


def save_confusion_matrix(probs, y_true, class_names, path):
    n = len(class_names)
    cm = confusion_matrix(y_true, probs.argmax(1), labels=list(range(n)))
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    for r in range(n):
        for c in range(n):
            ax.text(c, r, str(cm[r][c]), ha="center", va="center", fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)
