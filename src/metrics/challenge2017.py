from __future__ import annotations

import numpy as np


CHALLENGE_LABELS = ["N", "A", "O", "~"]


def confusion_matrix_4class(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_f1_from_confusion(cm: np.ndarray) -> np.ndarray:
    f1 = np.zeros(4, dtype=np.float64)
    for idx in range(4):
        tp = float(cm[idx, idx])
        row_sum = float(cm[idx, :].sum())
        col_sum = float(cm[:, idx].sum())
        denom = row_sum + col_sum
        f1[idx] = (2.0 * tp / denom) if denom > 0 else 0.0
    return f1


def challenge_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, dict[str, float], np.ndarray]:
    cm = confusion_matrix_4class(y_true, y_pred)
    per_class = per_class_f1_from_confusion(cm)
    metrics = {
        "F1n": float(per_class[0]),
        "F1a": float(per_class[1]),
        "F1o": float(per_class[2]),
        "F1p": float(per_class[3]),
    }
    macro = float(per_class.mean())
    return macro, metrics, cm
