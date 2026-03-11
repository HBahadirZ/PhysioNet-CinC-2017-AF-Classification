from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(signal: np.ndarray, fs: int, lowcut: float, highcut: float, order: int = 3) -> np.ndarray:
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal).astype(np.float32)


def normalize_signal(signal: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = signal.mean()
    std = signal.std()
    return ((signal - mean) / (std + eps)).astype(np.float32)


def crop_or_pad(
    signal: np.ndarray,
    target_length: int,
    random_crop: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    length = signal.shape[0]
    if length == target_length:
        return signal.astype(np.float32)
    if length > target_length:
        if random_crop:
            if rng is None:
                rng = np.random.default_rng()
            start = int(rng.integers(0, length - target_length + 1))
        else:
            start = (length - target_length) // 2
        return signal[start : start + target_length].astype(np.float32)

    padded = np.zeros(target_length, dtype=np.float32)
    offset = (target_length - length) // 2
    padded[offset : offset + length] = signal.astype(np.float32)
    return padded
