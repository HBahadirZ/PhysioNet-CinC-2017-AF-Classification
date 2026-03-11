from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


LABEL_TO_INDEX = {"N": 0, "A": 1, "O": 2, "~": 3}
INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}


@dataclass(frozen=True)
class RecordMeta:
    record_id: str
    label: str
    mat_path: Path
    hea_path: Path


def _load_reference(data_dir: Path) -> pd.DataFrame:
    reference_path = data_dir / "REFERENCE.csv"
    if not reference_path.exists():
        raise FileNotFoundError(f"Missing labels file: {reference_path}")
    df = pd.read_csv(reference_path, header=None, names=["record_id", "label"])
    invalid = set(df["label"].unique()) - set(LABEL_TO_INDEX.keys())
    if invalid:
        raise ValueError(f"Unexpected labels in REFERENCE.csv: {sorted(invalid)}")
    return df


def load_record_table(data_dir: Path, max_records: int | None = None) -> pd.DataFrame:
    records_file = data_dir / "RECORDS"
    if not records_file.exists():
        raise FileNotFoundError(f"Missing RECORDS file: {records_file}")

    records_df = pd.read_csv(records_file, header=None, names=["record_id"])
    labels_df = _load_reference(data_dir)
    merged = records_df.merge(labels_df, on="record_id", how="inner")
    if max_records is not None:
        merged = merged.iloc[:max_records].copy()

    merged["mat_path"] = merged["record_id"].apply(lambda rec: data_dir / f"{rec}.mat")
    merged["hea_path"] = merged["record_id"].apply(lambda rec: data_dir / f"{rec}.hea")
    merged["label_idx"] = merged["label"].map(LABEL_TO_INDEX)
    return merged


def validate_dataset_files(records: pd.DataFrame) -> dict[str, int]:
    missing_mat = int((~records["mat_path"].apply(Path.exists)).sum())
    missing_hea = int((~records["hea_path"].apply(Path.exists)).sum())
    class_counts = records["label"].value_counts().to_dict()
    return {
        "num_records": int(len(records)),
        "missing_mat": missing_mat,
        "missing_hea": missing_hea,
        "count_N": int(class_counts.get("N", 0)),
        "count_A": int(class_counts.get("A", 0)),
        "count_O": int(class_counts.get("O", 0)),
        "count_tilde": int(class_counts.get("~", 0)),
    }


def load_signal_from_mat(mat_path: Path) -> np.ndarray:
    contents = loadmat(mat_path)
    if "val" not in contents:
        raise ValueError(f"Unexpected matrix format: {mat_path}")
    signal = np.asarray(contents["val"]).astype(np.float32).squeeze()
    if signal.ndim != 1:
        signal = signal.reshape(-1).astype(np.float32)
    return signal
