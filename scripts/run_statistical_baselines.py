from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cli.train_baseline import run_experiment
from src.utils.config import load_yaml_config
from src.utils.paths import ensure_dir


BASE_CONFIGS = [
    ("configs/baseline.yaml", "fpb"),
    ("configs/baseline_tfns.yaml", "tfns"),
]

MODEL_SPECS = [
    {"type": "logreg", "name_suffix": "logreg"},
    {"type": "linear_svm", "name_suffix": "linear_svm"},
    {"type": "naive_bayes", "name_suffix": "naive_bayes", "alpha": 1.0},
]


def _prepare_config(config_path: str, train_domain: str, model_spec: dict[str, object]) -> dict[str, object]:
    config = deepcopy(load_yaml_config(config_path))
    base_name = config.get("experiment", {}).get("name", train_domain)
    config["experiment"]["name"] = f"{base_name}_{model_spec['name_suffix']}"

    config.setdefault("model", {})
    config["model"]["type"] = model_spec["type"]
    if "alpha" in model_spec:
        config["model"]["alpha"] = model_spec["alpha"]

    output_suffix = "statistical_baselines"
    config.setdefault("output", {})
    config["output"]["metrics_dir"] = f"outputs/metrics/{output_suffix}"
    config["output"]["predictions_dir"] = f"outputs/predictions/{output_suffix}"
    config["output"]["figures_dir"] = f"outputs/figures/{output_suffix}"
    config["output"]["logs_dir"] = f"outputs/logs/{output_suffix}"
    config["output"]["checkpoints_dir"] = f"outputs/checkpoints/{output_suffix}"
    return config


def _result_rows(result: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    eval_sets = result["eval_sets"]
    for eval_name, payload in eval_sets.items():
        rows.append(
            {
                "run_name": result["run_name"],
                "model_type": result["model_type"],
                "train_path": result["train_path"],
                "eval_set": eval_name,
                "accuracy": payload["accuracy"],
                "macro_f1": payload["macro_f1"],
                "weighted_f1": payload["weighted_f1"],
                "macro_precision": payload["macro_precision"],
                "weighted_precision": payload["weighted_precision"],
                "macro_recall": payload["macro_recall"],
                "weighted_recall": payload["weighted_recall"],
                "metrics_path": result["metrics_path"],
                "prediction_path": payload["prediction_path"],
                "confusion_matrix_path": payload["confusion_matrix_path"],
            }
        )
    return rows


def main() -> None:
    summary_rows: list[dict[str, object]] = []
    full_results: list[dict[str, object]] = []

    for config_path, train_domain in BASE_CONFIGS:
        for model_spec in MODEL_SPECS:
            config = _prepare_config(config_path, train_domain, model_spec)
            print(
                f"Running statistical baseline: train={train_domain} "
                f"model={model_spec['type']}"
            )
            result = run_experiment(config)
            full_results.append(result)
            summary_rows.extend(_result_rows(result))

    summary_dir = ensure_dir("outputs/metrics/statistical_baselines")
    summary_csv_path = summary_dir / "summary.csv"
    summary_json_path = summary_dir / "summary.json"

    pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)
    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(full_results, handle, indent=2)

    print(f"Summary CSV saved to: {summary_csv_path}")
    print(f"Summary JSON saved to: {summary_json_path}")


if __name__ == "__main__":
    main()
