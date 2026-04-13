from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.dataset import load_labeled_csv
from src.evaluation.metrics import (
    build_confusion_matrix_frame,
    compute_classification_metrics,
)
from src.evaluation.plots import plot_confusion_matrix_heatmap
from src.models.rule_based import LoughranMcDonaldRuleBaseline
from src.utils.config import dump_yaml_config, load_yaml_config
from src.utils.paths import ensure_dir, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Loughran-McDonald lexicon rule-based baseline."
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    return parser.parse_args()


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _get_run_name(config: dict[str, Any]) -> str:
    base_name = config.get("experiment", {}).get("name", "rule_based")
    return f"{base_name}_{_timestamp()}"


def _prepare_output_dirs(output_config: dict[str, Any]) -> dict[str, Path]:
    return {
        "metrics": ensure_dir(output_config.get("metrics_dir", "outputs/metrics")),
        "predictions": ensure_dir(output_config.get("predictions_dir", "outputs/predictions")),
        "figures": ensure_dir(output_config.get("figures_dir", "outputs/figures")),
        "logs": ensure_dir(output_config.get("logs_dir", "outputs/logs")),
    }


def _run_dirs(output_dirs: dict[str, Path], run_name: str) -> dict[str, Path]:
    return {
        "figures": ensure_dir(output_dirs["figures"] / run_name),
        "logs": ensure_dir(output_dirs["logs"] / run_name),
    }


def _save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def _format_eval_title(eval_name: str) -> str:
    return f"LM Lexicon | {eval_name.replace('_', ' ').title()}"


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    output_config = config.get("output", {})

    text_column = data_config.get("text_column", "text")
    label_column = data_config.get("label_column", "label")
    label_map = data_config.get("label_map", {})
    eval_sets = data_config.get("eval_sets", [])
    if not eval_sets:
        raise ValueError("Config must define at least one data.eval_sets entry.")

    baseline = LoughranMcDonaldRuleBaseline.from_csv(
        path_str=model_config["lexicon_path"],
        positive_column=model_config.get("positive_column", "Positive"),
        negative_column=model_config.get("negative_column", "Negative"),
        word_column=model_config.get("word_column", "Word"),
        lexicon_threshold=model_config.get("lexicon_threshold", 0),
        min_count_diff=model_config.get("min_count_diff", 1),
    )

    run_name = _get_run_name(config)
    output_dirs = _prepare_output_dirs(output_config)
    run_dirs = _run_dirs(output_dirs, run_name)

    print("=" * 80)
    print("Starting rule-based baseline - Model: LM Lexicon")
    print(
        f"Lexicon: {model_config['lexicon_path']} | Eval sets: {[spec['name'] for spec in eval_sets]}"
    )
    print(f"Model config: {model_config}")
    print("=" * 80)

    config_snapshot_path = run_dirs["logs"] / f"{run_name}_config.yaml"
    dump_yaml_config(
        {key: value for key, value in config.items() if key != "_config_path"},
        config_snapshot_path,
    )

    aggregate_metrics: dict[str, Any] = {
        "run_name": run_name,
        "model_type": "lm_lexicon_rule",
        "config_path": config.get("_config_path"),
        "lexicon_path": str(resolve_path(model_config["lexicon_path"])),
        "eval_sets": {},
    }

    print(f"Evaluating {len(eval_sets)} dataset(s)...")
    for idx, eval_spec in enumerate(eval_sets, start=1):
        eval_name = eval_spec["name"]
        print(f"[{idx}/{len(eval_sets)}] Evaluating {eval_name}...")
        eval_frame = load_labeled_csv(
            eval_spec["path"],
            text_column=text_column,
            label_column=label_column,
            label_map=label_map,
        )

        rule_predictions = baseline.predict_frame(eval_frame["text"].tolist())
        prediction_frame = pd.concat(
            [eval_frame.reset_index(drop=True), rule_predictions.reset_index(drop=True)],
            axis=1,
        )

        prediction_path = output_dirs["predictions"] / f"{run_name}__{eval_name}.csv"
        prediction_frame.to_csv(prediction_path, index=False)

        metric_payload = compute_classification_metrics(
            eval_frame["label"].tolist(),
            prediction_frame["prediction"].tolist(),
        )
        metric_payload["size"] = int(len(eval_frame))
        metric_payload["path"] = str(resolve_path(eval_spec["path"]))
        metric_payload["prediction_path"] = str(prediction_path)

        confusion_path = output_dirs["metrics"] / f"{run_name}__{eval_name}_confusion_matrix.csv"
        confusion_frame = build_confusion_matrix_frame(
            eval_frame["label"].tolist(),
            prediction_frame["prediction"].tolist(),
        )
        confusion_frame.to_csv(confusion_path, index=True)
        metric_payload["confusion_matrix_path"] = str(confusion_path)

        figure_path = run_dirs["figures"] / f"{run_name}__{eval_name}_confusion_matrix.png"
        plot_confusion_matrix_heatmap(
            confusion_frame,
            _format_eval_title(eval_name),
            figure_path,
        )
        metric_payload["confusion_figure_path"] = str(figure_path)

        aggregate_metrics["eval_sets"][eval_name] = metric_payload
        print(
            "  "
            + f"{eval_name}: accuracy={metric_payload['accuracy']:.4f} "
            + f"macro_f1={metric_payload['macro_f1']:.4f} "
            + f"weighted_f1={metric_payload['weighted_f1']:.4f}"
        )

    metrics_path = output_dirs["metrics"] / f"{run_name}.json"
    _save_json(aggregate_metrics, metrics_path)
    aggregate_metrics["metrics_path"] = str(metrics_path)
    return aggregate_metrics


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    results = run_experiment(config)

    print(f"Run: {results['run_name']}")
    print(f"Model type: {results['model_type']}")
    print(f"Metrics saved to: {results['metrics_path']}")
    for eval_name, payload in results["eval_sets"].items():
        print(
            f"[{eval_name}] accuracy={payload['accuracy']:.4f} "
            f"macro_f1={payload['macro_f1']:.4f} "
            f"weighted_f1={payload['weighted_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
