from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    data_dir: Path = Path("training2017")
    output_dir: Path = Path("outputs")
    num_folds: int = 5
    seed: int = 1337
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    num_workers: int = 0
    clip_seconds: int = 30
    sampling_rate: int = 300
    train_random_crop: bool = True
    max_records: int | None = None
    device: str = "cuda"
    label_smoothing: float = 0.0
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    apply_bandpass: bool = True
    lowcut_hz: float = 0.5
    highcut_hz: float = 40.0

    @property
    def clip_samples(self) -> int:
        return self.clip_seconds * self.sampling_rate


def ensure_output_dirs(cfg: TrainConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "reports").mkdir(parents=True, exist_ok=True)
