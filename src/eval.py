from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report

from src.data.physionet2017 import INDEX_TO_LABEL
from src.metrics.challenge2017 import CHALLENGE_LABELS, challenge_macro_f1


def render_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CHALLENGE_LABELS,
        yticklabels=CHALLENGE_LABELS,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Reference")
    ax.set_title("PhysioNet 2017 Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate out-of-fold predictions.")
    parser.add_argument("--predictions_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/reports"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.predictions_csv)
    if not {"y_true", "y_pred", "record_id"}.issubset(frame.columns):
        raise ValueError("predictions_csv must contain record_id,y_true,y_pred columns")

    y_true = frame["y_true"].to_numpy()
    y_pred = frame["y_pred"].to_numpy()
    macro_f1, per_class, cm = challenge_macro_f1(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3],
        target_names=[INDEX_TO_LABEL[i] for i in range(4)],
        digits=4,
    )

    summary_text = [
        f"Macro F1: {macro_f1:.5f}",
        f"F1n: {per_class['F1n']:.5f}",
        f"F1a: {per_class['F1a']:.5f}",
        f"F1o: {per_class['F1o']:.5f}",
        f"F1p: {per_class['F1p']:.5f}",
        "",
        "Classification report",
        report,
    ]
    (args.output_dir / "evaluation.txt").write_text("\n".join(summary_text), encoding="utf-8")
    render_confusion_matrix(cm, args.output_dir / "confusion_matrix.png")
    print("\n".join(summary_text))


if __name__ == "__main__":
    main()
