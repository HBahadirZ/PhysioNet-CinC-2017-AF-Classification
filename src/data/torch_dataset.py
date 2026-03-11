from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.physionet2017 import load_signal_from_mat
from src.features.augment import apply_train_augmentations
from src.features.preprocess import bandpass_filter, crop_or_pad, normalize_signal


class PhysioNetTorchDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        clip_samples: int,
        fs: int,
        train: bool,
        random_crop: bool,
        apply_bandpass: bool,
        lowcut_hz: float,
        highcut_hz: float,
        seed: int = 1337,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.clip_samples = clip_samples
        self.fs = fs
        self.train = train
        self.random_crop = random_crop
        self.apply_bandpass = apply_bandpass
        self.lowcut_hz = lowcut_hz
        self.highcut_hz = highcut_hz
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        mat_path = Path(row["mat_path"])
        signal = load_signal_from_mat(mat_path)
        if self.apply_bandpass:
            signal = bandpass_filter(signal, fs=self.fs, lowcut=self.lowcut_hz, highcut=self.highcut_hz)
        signal = normalize_signal(signal)
        signal = crop_or_pad(
            signal,
            target_length=self.clip_samples,
            random_crop=self.train and self.random_crop,
            rng=self.rng,
        )
        if self.train:
            signal = apply_train_augmentations(signal, fs=self.fs, rng=self.rng)
            signal = normalize_signal(signal)

        x = torch.from_numpy(signal).unsqueeze(0)
        y = torch.tensor(int(row["label_idx"]), dtype=torch.long)
        return x, y
