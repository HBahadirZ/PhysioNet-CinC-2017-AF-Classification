from __future__ import annotations

import numpy as np


def random_time_shift(signal: np.ndarray, max_shift: int, rng: np.random.Generator) -> np.ndarray:
    if max_shift <= 0:
        return signal
    shift = int(rng.integers(-max_shift, max_shift + 1))
    return np.roll(signal, shift).astype(np.float32)


def random_amplitude_scale(signal: np.ndarray, low: float, high: float, rng: np.random.Generator) -> np.ndarray:
    factor = float(rng.uniform(low, high))
    return (signal * factor).astype(np.float32)


def add_gaussian_noise(signal: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    if std <= 0:
        return signal
    noise = rng.normal(0.0, std, size=signal.shape).astype(np.float32)
    return (signal + noise).astype(np.float32)


def add_baseline_wander(signal: np.ndarray, fs: int, max_amp: float, rng: np.random.Generator) -> np.ndarray:
    if max_amp <= 0:
        return signal
    t = np.arange(signal.shape[0], dtype=np.float32) / float(fs)
    freq = float(rng.uniform(0.05, 0.5))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    amp = float(rng.uniform(0.0, max_amp))
    baseline = amp * np.sin(2.0 * np.pi * freq * t + phase)
    return (signal + baseline).astype(np.float32)


def apply_train_augmentations(signal: np.ndarray, fs: int, rng: np.random.Generator) -> np.ndarray:
    out = signal
    out = random_time_shift(out, max_shift=int(0.2 * fs), rng=rng)
    out = random_amplitude_scale(out, low=0.85, high=1.15, rng=rng)
    out = add_gaussian_noise(out, std=0.01, rng=rng)
    out = add_baseline_wander(out, fs=fs, max_amp=0.05, rng=rng)
    return out.astype(np.float32)
