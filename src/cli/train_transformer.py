from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import EarlyStoppingCallback

from src.data.dataset import load_labeled_csv
from src.evaluation.metrics import (
    build_confusion_matrix_frame,
    compute_classification_metrics,
)
from src.evaluation.plots import (
    plot_confusion_matrix_heatmap,
    plot_training_loss_curve,
    plot_training_metric_curve,
)
from src.models.transformer import (
    BlockProgressCallback,
    ID2LABEL,
    SentimentDataset,
    build_model,
    build_tokenizer,
    build_training_args,
    compute_hf_metrics,
    predict_from_model,
)
from src.utils.config import dump_yaml_config, load_yaml_config
from src.utils.paths import ensure_dir, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a transformer for sentiment classification.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    return parser.parse_args()


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _get_run_name(config: dict[str, Any]) -> str:
    base_name = config.get("experiment", {}).get("name", "transformer")
    return f"{base_name}_{_timestamp()}"


def _prepare_output_dirs(output_config: dict[str, Any]) -> dict[str, Path]:
    return {
        "metrics": ensure_dir(output_config.get("metrics_dir", "outputs/metrics")),
        "predictions": ensure_dir(output_config.get("predictions_dir", "outputs/predictions")),
        "figures": ensure_dir(output_config.get("figures_dir", "outputs/figures")),
        "logs": ensure_dir(output_config.get("logs_dir", "outputs/logs")),
        "checkpoints": ensure_dir(output_config.get("checkpoints_dir", "outputs/checkpoints")),
    }


def _save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def _format_model_title(model_name: str) -> str:
    short = model_name.split("/")[-1]
    return short.replace("-", " ").title()


def _format_eval_title(model_name: str, train_path: str, eval_name: str) -> str:
    train_domain = Path(train_path).parent.name.upper()
    eval_title = eval_name.replace("_", " ").title()
    return f"{_format_model_title(model_name)} | {train_domain} Train | {eval_title}"


def _format_training_title(model_name: str, train_path: str, suffix: str) -> str:
    train_domain = Path(train_path).parent.name.upper()
    return f"{_format_model_title(model_name)} | {train_domain} Train | {suffix}"


def _run_dirs(output_dirs: dict[str, Path], run_name: str) -> dict[str, Path]:
    return {
        "figures": ensure_dir(output_dirs["figures"] / run_name),
        "logs": ensure_dir(output_dirs["logs"] / run_name),
    }


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    output_config = config.get("output", {})

    model_name = model_config["name"]
    max_length = model_config.get("max_length", 128)

    text_column = data_config.get("text_column", "text")
    label_column = data_config.get("label_column", "label")
    label_map = data_config.get("label_map", {})

    train_frame = load_labeled_csv(
        data_config["train_path"],
        text_column=text_column,
        label_column=label_column,
        label_map=label_map,
    )
    val_frame = load_labeled_csv(
        data_config["val_path"],
        text_column=text_column,
        label_column=label_column,
        label_map=label_map,
    )

    tokenizer = build_tokenizer(model_name)
    model = build_model(model_name)

    train_dataset = SentimentDataset(
        train_frame["text"].tolist(),
        train_frame["label"].tolist(),
        tokenizer,
        max_length=max_length,
    )
    val_dataset = SentimentDataset(
        val_frame["text"].tolist(),
        val_frame["label"].tolist(),
        tokenizer,
        max_length=max_length,
    )

    run_name = _get_run_name(config)
    output_dirs = _prepare_output_dirs(output_config)
    run_dirs = _run_dirs(output_dirs, run_name)

    hf_output_dir = str(output_dirs["checkpoints"] / run_name)
    training_args = build_training_args(hf_output_dir, training_config)

    from transformers import Trainer

    callbacks = []
    patience = training_config.get("early_stopping_patience")
    if patience:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
    callbacks.append(
        BlockProgressCallback(
            description=f"FT {_format_model_title(model_name)}"
        )
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_hf_metrics,
        callbacks=callbacks,
    )

    trainer.train()

    config_snapshot_path = run_dirs["logs"] / f"{run_name}_config.yaml"
    dump_yaml_config(
        {key: value for key, value in config.items() if key != "_config_path"},
        config_snapshot_path,
    )

    log_history_path = run_dirs["logs"] / f"{run_name}_trainer_log_history.json"
    _save_json({"log_history": trainer.state.log_history}, log_history_path)

    loss_curve_path = run_dirs["figures"] / f"{run_name}_loss_curve.png"
    metric_curve_path = run_dirs["figures"] / f"{run_name}_eval_metrics_curve.png"
    plot_training_loss_curve(
        trainer.state.log_history,
        _format_training_title(model_name, data_config["train_path"], "Loss"),
        loss_curve_path,
    )
    plot_training_metric_curve(
        trainer.state.log_history,
        _format_training_title(model_name, data_config["train_path"], "Eval Metrics"),
        metric_curve_path,
    )

    best_model_dir = output_dirs["checkpoints"] / f"{run_name}_best"
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))

    aggregate_metrics: dict[str, Any] = {
        "run_name": run_name,
        "model_name": model_name,
        "train_size": int(len(train_frame)),
        "val_size": int(len(val_frame)),
        "train_path": str(resolve_path(data_config["train_path"])),
        "config_path": config.get("_config_path"),
        "best_model_dir": str(best_model_dir),
        "trainer_log_history_path": str(log_history_path),
        "loss_curve_path": str(loss_curve_path),
        "metric_curve_path": str(metric_curve_path),
        "eval_sets": {},
    }

    eval_sets = data_config.get("eval_sets", [])
    for eval_spec in eval_sets:
        eval_name = eval_spec["name"]
        eval_frame = load_labeled_csv(
            eval_spec["path"],
            text_column=text_column,
            label_column=label_column,
            label_map=label_map,
        )
        eval_dataset = SentimentDataset(
            eval_frame["text"].tolist(),
            eval_frame["label"].tolist(),
            tokenizer,
            max_length=max_length,
        )

        eval_batch_size = training_config.get("eval_batch_size", 32)
        predicted_labels, probabilities = predict_from_model(
            trainer.model, eval_dataset, batch_size=eval_batch_size
        )

        probability_frame = pd.DataFrame(
            probabilities,
            columns=[f"prob_{ID2LABEL[i]}" for i in range(probabilities.shape[1])],
        )
        prediction_frame = pd.concat(
            [
                eval_frame.reset_index(drop=True),
                pd.DataFrame({"prediction": predicted_labels}),
                probability_frame.reset_index(drop=True),
            ],
            axis=1,
        )

        prediction_path = output_dirs["predictions"] / f"{run_name}__{eval_name}.csv"
        prediction_frame.to_csv(prediction_path, index=False)

        metric_payload = compute_classification_metrics(
            eval_frame["label"].tolist(),
            predicted_labels,
        )
        metric_payload["size"] = int(len(eval_frame))
        metric_payload["path"] = str(resolve_path(eval_spec["path"]))
        metric_payload["prediction_path"] = str(prediction_path)

        confusion_path = output_dirs["metrics"] / f"{run_name}__{eval_name}_confusion_matrix.csv"
        confusion_frame = build_confusion_matrix_frame(
            eval_frame["label"].tolist(),
            predicted_labels,
        )
        confusion_frame.to_csv(confusion_path, index=True)
        metric_payload["confusion_matrix_path"] = str(confusion_path)

        figure_path = run_dirs["figures"] / f"{run_name}__{eval_name}_confusion_matrix.png"
        plot_confusion_matrix_heatmap(
            confusion_frame,
            _format_eval_title(model_name, data_config["train_path"], eval_name),
            figure_path,
        )
        metric_payload["confusion_figure_path"] = str(figure_path)

        aggregate_metrics["eval_sets"][eval_name] = metric_payload

    metrics_path = output_dirs["metrics"] / f"{run_name}.json"
    _save_json(aggregate_metrics, metrics_path)
    aggregate_metrics["metrics_path"] = str(metrics_path)

    return aggregate_metrics


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    results = run_experiment(config)

    print(f"Run: {results['run_name']}")
    print(f"Model: {results['model_name']}")
    print(f"Train size: {results['train_size']}")
    print(f"Best model saved to: {results['best_model_dir']}")
    print(f"Loss curve saved to: {results['loss_curve_path']}")
    print(f"Eval metric curve saved to: {results['metric_curve_path']}")
    print(f"Metrics saved to: {results['metrics_path']}")
    for eval_name, payload in results["eval_sets"].items():
        print(
            f"[{eval_name}] accuracy={payload['accuracy']:.4f} "
            f"macro_f1={payload['macro_f1']:.4f} "
            f"weighted_f1={payload['weighted_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
