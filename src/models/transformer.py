from __future__ import annotations

import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainingArguments,
)

from src.data.dataset import CANONICAL_LABELS

LABEL2ID = {label: idx for idx, label in enumerate(CANONICAL_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
NUM_LABELS = len(CANONICAL_LABELS)


class SentimentDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[str] | None,
        tokenizer: AutoTokenizer,
        max_length: int = 128,
    ):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = (
            torch.tensor([LABEL2ID[label] for label in labels], dtype=torch.long)
            if labels is not None
            else None
        )

    def __len__(self) -> int:
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


def build_tokenizer(model_name: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_name)


def build_model(model_name: str) -> AutoModelForSequenceClassification:
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )


def build_training_args(
    output_dir: str,
    training_config: dict[str, Any],
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config.get("epochs", 4),
        per_device_train_batch_size=training_config.get("train_batch_size", 16),
        per_device_eval_batch_size=training_config.get("eval_batch_size", 32),
        learning_rate=training_config.get("learning_rate", 2e-5),
        weight_decay=training_config.get("weight_decay", 0.01),
        warmup_ratio=training_config.get("warmup_ratio", 0.1),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=training_config.get("logging_steps", 50),
        fp16=training_config.get("fp16", torch.cuda.is_available()),
        report_to="none",
        seed=training_config.get("seed", 42),
        disable_tqdm=training_config.get("disable_tqdm", True),
    )


def compute_hf_metrics(eval_pred: Any) -> dict[str, float]:
    """Metric function compatible with HuggingFace Trainer."""
    from sklearn.metrics import accuracy_score, f1_score

    logits, label_ids = eval_pred
    predictions = np.argmax(logits, axis=-1)
    labels = list(range(NUM_LABELS))
    return {
        "accuracy": float(accuracy_score(label_ids, predictions)),
        "macro_f1": float(
            f1_score(label_ids, predictions, labels=labels, average="macro")
        ),
        "weighted_f1": float(
            f1_score(label_ids, predictions, labels=labels, average="weighted")
        ),
    }


class BlockProgressCallback(TrainerCallback):
    """Single-line console progress updates for notebook-friendly training logs."""

    def __init__(self, description: str = "Training", min_interval_seconds: float = 0.5) -> None:
        self.description = description
        self.min_interval_seconds = min_interval_seconds
        self.current_step = 0
        self.epoch_start_step = 0
        self.current_epoch_index = 0
        self.total_epochs = 0
        self.total_steps = 0
        self.latest_train_logs: dict[str, Any] = {}
        self.last_render_time = 0.0
        self.last_line_length = 0

    def _epoch_total_steps(self) -> int:
        if self.total_epochs <= 0:
            return max(self.total_steps, 1)
        base_steps = self.total_steps // self.total_epochs
        remainder = self.total_steps % self.total_epochs
        epoch_idx = max(self.current_epoch_index - 1, 0)
        return max(base_steps + (1 if epoch_idx < remainder else 0), 1)

    def _render_line(self, message: str, *, finalize: bool = False) -> None:
        padded = message.ljust(self.last_line_length)
        sys.stdout.write("\r" + padded)
        if finalize:
            sys.stdout.write("\n")
            self.last_line_length = 0
        else:
            sys.stdout.flush()
            self.last_line_length = len(padded)

    def _build_progress_message(self, state: Any) -> str:
        epoch_steps = self._epoch_total_steps()
        epoch_step = max(int(state.global_step) - self.epoch_start_step, 0)
        percent = 100.0 * int(state.global_step) / max(self.total_steps, 1)
        message = (
            f"{self.description} | epoch {self.current_epoch_index}/{self.total_epochs}"
            f" | step {epoch_step}/{epoch_steps}"
            f" | total {int(state.global_step)}/{self.total_steps}"
            f" | {percent:5.1f}%"
        )
        if "loss" in self.latest_train_logs:
            message += f" | loss {self.latest_train_logs['loss']:.4f}"
        return message

    def on_train_begin(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> None:
        self.total_steps = int(state.max_steps or 0)
        self.total_epochs = max(int(round(args.num_train_epochs)), 1)
        self.current_step = 0
        self.epoch_start_step = 0
        self.current_epoch_index = 0
        self.latest_train_logs = {}
        self.last_render_time = 0.0
        self.last_line_length = 0
        print("=" * 80)
        print(f"Starting: {self.description}")
        print(
            f"Training config: epochs={self.total_epochs}, "
            f"batch_size={args.per_device_train_batch_size}, lr={args.learning_rate:.2e}"
        )
        print("=" * 80)

    def on_epoch_begin(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> None:
        self.current_epoch_index += 1
        self.epoch_start_step = int(state.global_step)
        self.last_render_time = 0.0
        self._render_line(self._build_progress_message(state))

    def on_step_end(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> None:
        target_step = int(state.global_step)
        if target_step <= self.current_step:
            return
        self.current_step = target_step
        now = time.monotonic()
        if now - self.last_render_time < self.min_interval_seconds and target_step < self.total_steps:
            return
        self.last_render_time = now
        self._render_line(self._build_progress_message(state))

    def on_log(self, args: TrainingArguments, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if not logs:
            return
        if "loss" in logs:
            self.latest_train_logs = dict(logs)

    def on_evaluate(self, args: TrainingArguments, state: Any, control: Any, metrics: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if not metrics:
            return
        self._render_line(self._build_progress_message(state), finalize=True)
        train_loss = self.latest_train_logs.get("loss")
        summary_parts = [f"Epoch {self.current_epoch_index}/{self.total_epochs}"]
        if train_loss is not None:
            summary_parts.append(f"Train Loss: {train_loss:.4f}")
        if "eval_loss" in metrics:
            summary_parts.append(f"Eval Loss: {metrics['eval_loss']:.4f}")
        if "eval_accuracy" in metrics:
            summary_parts.append(f"Eval Acc: {metrics['eval_accuracy']:.4f}")
        if "eval_macro_f1" in metrics:
            summary_parts.append(f"Macro-F1: {metrics['eval_macro_f1']:.4f}")
        if "eval_weighted_f1" in metrics:
            summary_parts.append(f"Weighted-F1: {metrics['eval_weighted_f1']:.4f}")
        print("  " + " | ".join(summary_parts))

    def on_train_end(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> None:
        if self.last_line_length:
            self._render_line(self._build_progress_message(state), finalize=True)


def predict_from_model(
    model: AutoModelForSequenceClassification,
    dataset: SentimentDataset,
    batch_size: int = 32,
) -> tuple[list[str], np.ndarray]:
    device = next(model.parameters()).device
    dataloader = DataLoader(dataset, batch_size=batch_size)
    logits_batches: list[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "labels"
            }
            outputs = model(**batch)
            logits_batches.append(outputs.logits.detach().cpu())

    logits = torch.cat(logits_batches, dim=0).numpy()
    probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    predicted_ids = np.argmax(logits, axis=-1)
    predicted_labels = [ID2LABEL[pid] for pid in predicted_ids]
    return predicted_labels, probabilities
