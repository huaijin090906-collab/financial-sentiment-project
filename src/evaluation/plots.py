from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


def plot_confusion_matrix_heatmap(
    matrix_frame: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix_frame,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        linewidths=0.5,
        square=True,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def _history_frame(log_history: list[dict]) -> pd.DataFrame:
    if not log_history:
        return pd.DataFrame()
    return pd.DataFrame(log_history)


def plot_training_loss_curve(
    log_history: list[dict],
    title: str,
    output_path: Path,
) -> None:
    history = _history_frame(log_history)
    if history.empty or "step" not in history.columns:
        return

    train_history = history[history["loss"].notna()] if "loss" in history.columns else pd.DataFrame()
    eval_history = history[history["eval_loss"].notna()] if "eval_loss" in history.columns else pd.DataFrame()
    if train_history.empty and eval_history.empty:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    if not train_history.empty:
        plt.plot(train_history["step"], train_history["loss"], label="Train Loss", linewidth=2)
    if not eval_history.empty:
        plt.plot(eval_history["step"], eval_history["eval_loss"], label="Eval Loss", linewidth=2)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_training_metric_curve(
    log_history: list[dict],
    title: str,
    output_path: Path,
) -> None:
    history = _history_frame(log_history)
    if history.empty or "step" not in history.columns:
        return

    metric_columns = [
        column
        for column in ("eval_accuracy", "eval_macro_f1", "eval_weighted_f1")
        if column in history.columns and history[column].notna().any()
    ]
    if not metric_columns:
        return

    eval_history = history[history[metric_columns].notna().any(axis=1)]
    if eval_history.empty:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    label_map = {
        "eval_accuracy": "Eval Accuracy",
        "eval_macro_f1": "Eval Macro-F1",
        "eval_weighted_f1": "Eval Weighted-F1",
    }
    for column in metric_columns:
        plt.plot(eval_history["step"], eval_history[column], label=label_map[column], linewidth=2)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_eval_metric_summary(
    metric_frame: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    if metric_frame.empty:
        return

    plot_frame = metric_frame.melt(
        id_vars=["eval_set"],
        value_vars=["accuracy", "macro_f1", "weighted_f1"],
        var_name="metric",
        value_name="score",
    )

    label_map = {
        "accuracy": "Accuracy",
        "macro_f1": "Macro-F1",
        "weighted_f1": "Weighted-F1",
    }
    plot_frame["metric"] = plot_frame["metric"].map(label_map)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    sns.barplot(data=plot_frame, x="eval_set", y="score", hue="metric")
    plt.title(title)
    plt.xlabel("Eval Set")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.0)
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
