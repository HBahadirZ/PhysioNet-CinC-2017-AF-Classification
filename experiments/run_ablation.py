from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run training ablations.")
    parser.add_argument("--python", type=str, default="python")
    parser.add_argument("--data_dir", type=Path, default=Path("training2017"))
    parser.add_argument("--base_output", type=Path, default=Path("outputs/ablations"))
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--max_runs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.base_output.mkdir(parents=True, exist_ok=True)

    grid = {
        "lr": [1e-3, 5e-4],
        "batch_size": [64],
        "use_focal_loss": [False, True],
        "label_smoothing": [0.0, 0.05],
    }
    keys = list(grid.keys())
    runs = []
    all_values = list(itertools.product(*(grid[key] for key in keys)))
    if args.max_runs is not None:
        all_values = all_values[: args.max_runs]

    for values in all_values:
        params = dict(zip(keys, values))
        run_name = "_".join(f"{k}-{str(v).replace('.', 'p')}" for k, v in params.items())
        out_dir = args.base_output / run_name
        cmd = [
            args.python,
            "-m",
            "src.train",
            "--data_dir",
            str(args.data_dir),
            "--output_dir",
            str(out_dir),
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--num_folds",
            str(args.num_folds),
            "--optimize_thresholds",
        ]
        if args.max_records is not None:
            cmd += ["--max_records", str(args.max_records)]
        cmd += ["--lr", str(params["lr"]), "--batch_size", str(params["batch_size"])]
        cmd += ["--label_smoothing", str(params["label_smoothing"])]
        if params["use_focal_loss"]:
            cmd += ["--use_focal_loss"]

        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        summary_file = out_dir / "reports" / "train_summary.json"
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        runs.append({"run_name": run_name, "params": params, "summary": summary})

    runs = sorted(runs, key=lambda r: r["summary"]["oof_macro_f1"], reverse=True)
    leaderboard_path = args.base_output / "leaderboard.json"
    leaderboard_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(f"Saved ablation leaderboard to {leaderboard_path}")
    if runs:
        print("Best run:", runs[0]["run_name"], "OOF F1:", runs[0]["summary"]["oof_macro_f1"])


if __name__ == "__main__":
    main()
