from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.dataset import load_labeled_csv
from src.evaluation.metrics import (
    build_confusion_matrix_frame,
    compute_classification_metrics,
)
from src.evaluation.plots import (
    plot_confusion_matrix_heatmap,
    plot_eval_metric_summary,
)
from src.models.baseline import build_baseline_pipeline
from src.utils.config import dump_yaml_config, load_yaml_config
from src.utils.paths import ensure_dir, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression baseline.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    return parser.parse_args()


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _get_run_name(config: dict[str, Any]) -> str:
    base_name = config.get("experiment", {}).get("name", "baseline")
    return f"{base_name}_{_timestamp()}"


def _load_eval_sets(data_config: dict[str, Any]) -> list[dict[str, Any]]:
    eval_sets = data_config.get("eval_sets", [])
    if not eval_sets:
        raise ValueError("Config must define at least one data.eval_sets entry.")
    return eval_sets


def _prepare_output_dirs(output_config: dict[str, Any]) -> dict[str, Path]:
    return {
        "metrics": ensure_dir(output_config.get("metrics_dir", "outputs/metrics")),
        "predictions": ensure_dir(output_config.get("predictions_dir", "outputs/predictions")),
        "figures": ensure_dir(output_config.get("figures_dir", "outputs/figures")),
        "logs": ensure_dir(output_config.get("logs_dir", "outputs/logs")),
        "checkpoints": ensure_dir(output_config.get("checkpoints_dir", "outputs/checkpoints")),
    }


def _run_dirs(output_dirs: dict[str, Path], run_name: str) -> dict[str, Path]:
    return {
        "figures": ensure_dir(output_dirs["figures"] / run_name),
        "logs": ensure_dir(output_dirs["logs"] / run_name),
    }


def _get_model_type(model_config: dict[str, Any]) -> str:
    return str(model_config.get("type", "logreg"))


def _build_score_frame(pipeline: Any, texts: pd.Series) -> pd.DataFrame:
    classifier = pipeline.named_steps["classifier"]

    if hasattr(classifier, "predict_proba"):
        probabilities = pipeline.predict_proba(texts)
        return pd.DataFrame(
            probabilities,
            columns=[f"prob_{label}" for label in pipeline.classes_],
        )

    if hasattr(classifier, "decision_function"):
        decisions = pipeline.decision_function(texts)
        if getattr(decisions, "ndim", 1) == 1:
            decisions = decisions.reshape(-1, 1)
        return pd.DataFrame(
            decisions,
            columns=[f"score_{label}" for label in pipeline.classes_],
        )

    return pd.DataFrame(index=range(len(texts)))


def _format_model_title(model_type: str) -> str:
    title_map = {
        "logreg": "LogReg",
        "linear_svm": "Linear SVM",
        "naive_bayes": "Naive Bayes",
    }
    return title_map.get(model_type, model_type.replace("_", " ").title())


def _format_eval_title(model_type: str, train_path: str, eval_name: str) -> str:
    train_domain = Path(train_path).parent.name.upper()
    eval_title = eval_name.replace("_", " ").title()
    return f"{_format_model_title(model_type)} | {train_domain} Train | {eval_title}"


def _format_summary_title(model_type: str, train_path: str) -> str:
    train_domain = Path(train_path).parent.name.upper()
    return f"{_format_model_title(model_type)} | {train_domain} Train | Eval Summary"


def _save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    output_config = config.get("output", {})

    text_column = data_config.get("text_column", "text")
    label_column = data_config.get("label_column", "label")
    label_map = data_config.get("label_map", {})
    model_type = _get_model_type(model_config)
    eval_sets = _load_eval_sets(data_config)

    train_frame = load_labeled_csv(
        data_config["train_path"],
        text_column=text_column,
        label_column=label_column,
        label_map=label_map,
    )

    pipeline = build_baseline_pipeline(model_config)

    run_name = _get_run_name(config)
    output_dirs = _prepare_output_dirs(output_config)
    run_dirs = _run_dirs(output_dirs, run_name)

    print("=" * 80)
    print(
        f"Starting statistical baseline - Model: {_format_model_title(model_type)}"
    )
    print(
        f"Train set: {Path(data_config['train_path']).parent.name.upper()} | "
        f"Train size: {len(train_frame)} | Eval sets: {[spec['name'] for spec in eval_sets]}"
    )
    print(f"Model config: {model_config}")
    print("=" * 80)
    print("Fitting model...")
    pipeline.fit(train_frame["text"], train_frame["label"])
    print("Fitting complete.")

    config_snapshot_path = run_dirs["logs"] / f"{run_name}_config.yaml"
    dump_yaml_config(
        {key: value for key, value in config.items() if key != "_config_path"},
        config_snapshot_path,
    )

    model_path = output_dirs["checkpoints"] / f"{run_name}.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(pipeline, handle)

    aggregate_metrics: dict[str, Any] = {
        "run_name": run_name,
        "model_type": model_type,
        "train_size": int(len(train_frame)),
        "train_path": str(resolve_path(data_config["train_path"])),
        "config_path": config.get("_config_path"),
        "model_path": str(model_path),
        "eval_summary_csv_path": "",
        "eval_summary_figure_path": "",
        "eval_sets": {},
    }

    eval_summary_rows: list[dict[str, Any]] = []
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
        predictions = pipeline.predict(eval_frame["text"])
        score_frame = _build_score_frame(pipeline, eval_frame["text"])

        prediction_frame = pd.concat(
            [
                eval_frame.reset_index(drop=True),
                pd.DataFrame({"prediction": predictions}),
                score_frame.reset_index(drop=True),
            ],
            axis=1,
        )

        prediction_path = output_dirs["predictions"] / f"{run_name}__{eval_name}.csv"
        prediction_frame.to_csv(prediction_path, index=False)

        metric_payload = compute_classification_metrics(
            eval_frame["label"].tolist(),
            list(predictions),
        )
        metric_payload["size"] = int(len(eval_frame))
        metric_payload["path"] = str(resolve_path(eval_spec["path"]))
        metric_payload["prediction_path"] = str(prediction_path)

        confusion_path = output_dirs["metrics"] / f"{run_name}__{eval_name}_confusion_matrix.csv"
        confusion_frame = build_confusion_matrix_frame(
            eval_frame["label"].tolist(),
            list(predictions),
        )
        confusion_frame.to_csv(confusion_path, index=True)
        metric_payload["confusion_matrix_path"] = str(confusion_path)

        figure_path = run_dirs["figures"] / f"{run_name}__{eval_name}_confusion_matrix.png"
        plot_confusion_matrix_heatmap(
            confusion_frame,
            _format_eval_title(model_type, data_config["train_path"], eval_name),
            figure_path,
        )
        metric_payload["confusion_figure_path"] = str(figure_path)

        aggregate_metrics["eval_sets"][eval_name] = metric_payload
        eval_summary_rows.append(
            {
                "eval_set": eval_name,
                "accuracy": metric_payload["accuracy"],
                "macro_f1": metric_payload["macro_f1"],
                "weighted_f1": metric_payload["weighted_f1"],
                "macro_precision": metric_payload["macro_precision"],
                "weighted_precision": metric_payload["weighted_precision"],
                "macro_recall": metric_payload["macro_recall"],
                "weighted_recall": metric_payload["weighted_recall"],
            }
        )
        print(
            "  "
            + f"{eval_name}: accuracy={metric_payload['accuracy']:.4f} "
            + f"macro_f1={metric_payload['macro_f1']:.4f} "
            + f"weighted_f1={metric_payload['weighted_f1']:.4f}"
        )

    eval_summary_frame = pd.DataFrame(eval_summary_rows)
    eval_summary_csv_path = run_dirs["logs"] / f"{run_name}_eval_summary.csv"
    eval_summary_frame.to_csv(eval_summary_csv_path, index=False)

    eval_summary_figure_path = run_dirs["figures"] / f"{run_name}_eval_summary.png"
    plot_eval_metric_summary(
        eval_summary_frame[["eval_set", "accuracy", "macro_f1", "weighted_f1"]],
        _format_summary_title(model_type, data_config["train_path"]),
        eval_summary_figure_path,
    )
    aggregate_metrics["eval_summary_csv_path"] = str(eval_summary_csv_path)
    aggregate_metrics["eval_summary_figure_path"] = str(eval_summary_figure_path)

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
    print(f"Train size: {results['train_size']}")
    print(f"Eval summary saved to: {results['eval_summary_csv_path']}")
    print(f"Eval summary figure saved to: {results['eval_summary_figure_path']}")
    print(f"Metrics saved to: {results['metrics_path']}")
    for eval_name, payload in results["eval_sets"].items():
        print(
            f"[{eval_name}] accuracy={payload['accuracy']:.4f} "
            f"macro_f1={payload['macro_f1']:.4f} "
            f"weighted_f1={payload['weighted_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
