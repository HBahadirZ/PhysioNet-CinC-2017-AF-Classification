from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def build_stratified_folds(
    records: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 1337,
) -> pd.DataFrame:
    folds = records.copy().reset_index(drop=True)
    folds["fold"] = -1
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold_idx, (_, val_idx) in enumerate(splitter.split(folds["record_id"], folds["label_idx"])):
        folds.loc[val_idx, "fold"] = fold_idx

    if (folds["fold"] < 0).any():
        raise RuntimeError("Failed to assign all fold indices.")
    return folds
