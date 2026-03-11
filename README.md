# PhysioNet/CinC 2017 AF Classification

Personal machine learning project for 4-class ECG rhythm classification (`N`, `A`, `O`, `~`) on the PhysioNet/CinC 2017 challenge dataset.

Challenge page: https://archive.physionet.org/challenge/2017/

## What this project does

- Loads and validates PhysioNet 2017 training records (`RECORDS`, `REFERENCE.csv`, `.mat`, `.hea`)
- Trains a residual 1D CNN with stratified cross-validation
- Scores runs using challenge-style macro F1 (`(F1n + F1a + F1o + F1p) / 4`)
- Exports fold checkpoints, out-of-fold predictions, and evaluation reports
- Supports quick ablation sweeps for hyperparameter tuning

## Project structure

- `src/train.py`: training loop, cross-validation, checkpointing, threshold optimization
- `src/eval.py`: evaluation report + confusion matrix image
- `src/data/`: data parser, dataset checks, fold generation, torch dataset
- `src/features/`: ECG preprocessing and augmentations
- `src/models/cnn1d.py`: residual 1D CNN baseline
- `src/metrics/challenge2017.py`: challenge metric implementation
- `experiments/run_ablation.py`: configurable ablation runner

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset setup

Download and unzip PhysioNet 2017 training data into:

- `training2017/RECORDS`
- `training2017/REFERENCE.csv`
- waveform files (`*.mat`, `*.hea`)

Data is intentionally excluded from git via `.gitignore`.

## Training

Baseline run:

```bash
python -m src.train --data_dir training2017 --output_dir outputs/baseline --epochs 30 --num_folds 5 --optimize_thresholds
```

Quick smoke run:

```bash
python -m src.train --data_dir training2017 --output_dir outputs/smoke --epochs 1 --num_folds 2 --max_records 256
```

## Evaluation

```bash
python -m src.eval --predictions_csv outputs/baseline/reports/oof_predictions.csv --output_dir outputs/baseline/reports
```

## Ablations

```bash
python experiments/run_ablation.py --data_dir training2017 --base_output outputs/ablations
```

## Current results snapshot

From local runs in this repository:

- `outputs/baseline_fast`: macro F1 `0.3856`
- `outputs/smoke`: macro F1 `0.2930`
- `outputs/ablations_quick` best run: macro F1 `0.2515`

These are development runs (CPU fallback on this machine), not final challenge submissions.

## Reproducibility

- Fixed random seed (`--seed`, default `1337`)
- Deterministic torch backend settings in `src/utils/repro.py`
- Per-fold checkpoints and summaries under `outputs/...`

