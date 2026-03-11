from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import TrainConfig, ensure_output_dirs
from src.data.physionet2017 import load_record_table, validate_dataset_files
from src.data.splits import build_stratified_folds
from src.data.torch_dataset import PhysioNetTorchDataset
from src.metrics.challenge2017 import challenge_macro_f1
from src.models.cnn1d import ECGResNet1D
from src.utils.repro import seed_everything


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def build_dataloaders(cfg: TrainConfig, fold_frame: pd.DataFrame, fold: int) -> tuple[DataLoader, DataLoader, pd.DataFrame]:
    train_frame = fold_frame[fold_frame["fold"] != fold].copy().reset_index(drop=True)
    val_frame = fold_frame[fold_frame["fold"] == fold].copy().reset_index(drop=True)

    train_ds = PhysioNetTorchDataset(
        frame=train_frame,
        clip_samples=cfg.clip_samples,
        fs=cfg.sampling_rate,
        train=True,
        random_crop=cfg.train_random_crop,
        apply_bandpass=cfg.apply_bandpass,
        lowcut_hz=cfg.lowcut_hz,
        highcut_hz=cfg.highcut_hz,
        seed=cfg.seed + fold,
    )
    val_ds = PhysioNetTorchDataset(
        frame=val_frame,
        clip_samples=cfg.clip_samples,
        fs=cfg.sampling_rate,
        train=False,
        random_crop=False,
        apply_bandpass=cfg.apply_bandpass,
        lowcut_hz=cfg.lowcut_hz,
        highcut_hz=cfg.highcut_hz,
        seed=cfg.seed + 100 + fold,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, val_frame


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_targets.append(y.cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_targets)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def optimize_class_scales(y_true: np.ndarray, probs: np.ndarray, seed: int, n_iter: int = 600) -> np.ndarray:
    rng = np.random.default_rng(seed)
    best_scales = np.ones(4, dtype=np.float32)
    best_pred = np.argmax(probs, axis=1)
    best_score, _, _ = challenge_macro_f1(y_true, best_pred)
    for _ in range(n_iter):
        candidate = rng.uniform(0.7, 1.5, size=4).astype(np.float32)
        pred = np.argmax(probs * candidate[None, :], axis=1)
        score, _, _ = challenge_macro_f1(y_true, pred)
        if score > best_score:
            best_score = score
            best_scales = candidate
    return best_scales


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PhysioNet 2017 classifier.")
    parser.add_argument("--data_dir", type=Path, default=Path("training2017"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--clip_seconds", type=int, default=30)
    parser.add_argument("--sampling_rate", type=int, default=300)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_bandpass", action="store_true")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--use_focal_loss", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--optimize_thresholds", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_folds=args.num_folds,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        num_workers=args.num_workers,
        clip_seconds=args.clip_seconds,
        sampling_rate=args.sampling_rate,
        max_records=args.max_records,
        device=args.device,
        label_smoothing=args.label_smoothing,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        apply_bandpass=(not args.no_bandpass),
    )
    ensure_output_dirs(cfg)
    seed_everything(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Config: {cfg}")

    records = load_record_table(cfg.data_dir, max_records=cfg.max_records)
    quality = validate_dataset_files(records)
    print("Dataset quality:", json.dumps(quality, indent=2))
    records = records[records["mat_path"].apply(Path.exists)].copy().reset_index(drop=True)
    fold_frame = build_stratified_folds(records, n_splits=cfg.num_folds, seed=cfg.seed)

    oof_rows: list[pd.DataFrame] = []
    fold_scores: list[float] = []

    for fold in range(cfg.num_folds):
        print(f"\n===== Fold {fold} / {cfg.num_folds - 1} =====")
        train_loader, val_loader, val_frame = build_dataloaders(cfg, fold_frame, fold)
        model = ECGResNet1D(num_classes=4).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
        )

        train_labels = fold_frame[fold_frame["fold"] != fold]["label_idx"].to_numpy()
        class_counts = np.bincount(train_labels, minlength=4).astype(np.float32)
        class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
        class_weights = class_weights / class_weights.mean()
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

        if cfg.use_focal_loss:
            criterion: nn.Module = FocalLoss(alpha=class_weights_tensor, gamma=cfg.focal_gamma)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=cfg.label_smoothing)

        best_score = -1.0
        best_state = None
        no_improve = 0

        for epoch in range(cfg.epochs):
            tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_probs, val_targets = predict_probs(model, val_loader, device)
            val_pred = np.argmax(val_probs, axis=1)
            val_f1, _, _ = challenge_macro_f1(val_targets, val_pred)
            scheduler.step(val_f1)
            print(
                f"fold={fold} epoch={epoch + 1}/{cfg.epochs} "
                f"train_loss={tr_loss:.5f} val_macro_f1={val_f1:.5f}"
            )
            if val_f1 > best_score:
                best_score = val_f1
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg.patience:
                    print("Early stopping triggered.")
                    break

        if best_state is None:
            raise RuntimeError("No best model state captured.")
        model.load_state_dict(best_state)

        fold_ckpt_path = cfg.output_dir / "checkpoints" / f"fold_{fold}.pt"
        torch.save({"state_dict": model.state_dict(), "config": cfg.__dict__}, fold_ckpt_path)

        val_probs, val_targets = predict_probs(model, val_loader, device)
        val_pred = np.argmax(val_probs, axis=1)
        fold_f1, _, _ = challenge_macro_f1(val_targets, val_pred)
        fold_scores.append(fold_f1)

        fold_oof = val_frame[["record_id"]].copy()
        fold_oof["y_true"] = val_targets
        fold_oof["y_pred"] = val_pred
        for idx in range(4):
            fold_oof[f"prob_{idx}"] = val_probs[:, idx]
        fold_oof["fold"] = fold
        oof_rows.append(fold_oof)

        print(f"Fold {fold} best macro F1: {fold_f1:.5f}")

    oof = pd.concat(oof_rows, ignore_index=True)
    scales = np.ones(4, dtype=np.float32)
    if args.optimize_thresholds:
        scales = optimize_class_scales(
            y_true=oof["y_true"].to_numpy(),
            probs=oof[[f"prob_{i}" for i in range(4)]].to_numpy(),
            seed=cfg.seed,
        )
        oof_scaled_pred = np.argmax(oof[[f"prob_{i}" for i in range(4)]].to_numpy() * scales[None, :], axis=1)
        oof["y_pred"] = oof_scaled_pred

    final_macro_f1, per_class, _ = challenge_macro_f1(oof["y_true"].to_numpy(), oof["y_pred"].to_numpy())
    oof_path = cfg.output_dir / "reports" / "oof_predictions.csv"
    oof.to_csv(oof_path, index=False)

    summary = {
        "fold_scores": fold_scores,
        "mean_fold_f1": float(np.mean(fold_scores)),
        "std_fold_f1": float(np.std(fold_scores)),
        "oof_macro_f1": float(final_macro_f1),
        "per_class": per_class,
        "class_scales": scales.tolist(),
    }
    summary_path = cfg.output_dir / "reports" / "train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nTraining complete.")
    print(json.dumps(summary, indent=2))
    print(f"OOF predictions saved to: {oof_path}")


if __name__ == "__main__":
    main()
