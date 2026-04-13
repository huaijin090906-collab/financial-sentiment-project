"""Batch runner for transformer baselines (FT only).

Usage on Colab:
    python scripts/run_transformer_baselines.py
    python scripts/run_transformer_baselines.py --configs configs/transformer_finbert_fpb.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cli.train_transformer import run_experiment
from src.utils.config import load_yaml_config
from src.utils.paths import ensure_dir


DEFAULT_CONFIGS = [
    "configs/transformer_bert_fpb.yaml",
    "configs/transformer_bert_tfns.yaml",
    "configs/transformer_finbert_fpb.yaml",
    "configs/transformer_finbert_tfns.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run transformer baselines.")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Config files to run. Defaults to all four transformer baseline configs.",
    )
    return parser.parse_args()


def _result_rows(result: dict) -> list[dict]:
    rows = []
    for eval_name, payload in result["eval_sets"].items():
        rows.append(
            {
                "run_name": result["run_name"],
                "model_name": result["model_name"],
                "train_path": result["train_path"],
                "eval_set": eval_name,
                "accuracy": payload["accuracy"],
                "macro_f1": payload["macro_f1"],
                "weighted_f1": payload["weighted_f1"],
                "macro_precision": payload["macro_precision"],
                "weighted_precision": payload["weighted_precision"],
                "macro_recall": payload["macro_recall"],
                "weighted_recall": payload["weighted_recall"],
                "trainer_log_history_path": result["trainer_log_history_path"],
                "loss_curve_path": result["loss_curve_path"],
                "metric_curve_path": result["metric_curve_path"],
                "metrics_path": result["metrics_path"],
                "prediction_path": payload["prediction_path"],
                "confusion_matrix_path": payload["confusion_matrix_path"],
                "confusion_figure_path": payload["confusion_figure_path"],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    config_paths = args.configs or DEFAULT_CONFIGS

    summary_rows: list[dict] = []
    full_results: list[dict] = []

    for config_path in config_paths:
        print(f"\n{'='*60}")
        print(f"Running: {config_path}")
        print(f"{'='*60}")
        config = load_yaml_config(config_path)
        result = run_experiment(config)
        full_results.append(result)
        summary_rows.extend(_result_rows(result))

    summary_dir = ensure_dir("outputs/metrics/transformer_baselines")
    summary_csv_path = summary_dir / "summary.csv"
    summary_json_path = summary_dir / "summary.json"

    pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)
    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(full_results, handle, indent=2)

    print(f"\nSummary CSV saved to: {summary_csv_path}")
    print(f"Summary JSON saved to: {summary_json_path}")


if __name__ == "__main__":
    main()
