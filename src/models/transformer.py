from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainerCallback,
    Trainer,
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
    """Epoch-level tqdm progress bars with clean summaries."""

    def __init__(self, description: str = "Training") -> None:
        self.description = description
        self.progress_bar: tqdm | None = None
        self.current_step = 0
        self.epoch_start_step = 0
        self.current_epoch_index = 0
        self.total_epochs = 0
        self.total_steps = 0
        self.latest_train_logs: dict[str, Any] = {}

    def _epoch_total_steps(self) -> int:
        if self.total_epochs <= 0:
            return max(self.total_steps, 1)
        base_steps = self.total_steps // self.total_epochs
        remainder = self.total_steps % self.total_epochs
        epoch_idx = max(self.current_epoch_index - 1, 0)
        return max(base_steps + (1 if epoch_idx < remainder else 0), 1)

    def _close_bar(self) -> None:
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None

    def on_train_begin(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> None:
        self.total_steps = int(state.max_steps or 0)
        self.total_epochs = max(int(math.ceil(args.num_train_epochs)), 1)
        self.current_step = 0
        self.epoch_start_step = 0
        self.current_epoch_index = 0
        self.latest_train_logs = {}
        tqdm.write("=" * 80)
        tqdm.write(f"Starting transformer fine-tuning: {self.description}")
        tqdm.write(
            f"Training config: epochs={self.total_epochs}, "
            f"batch_size={args.per_device_train_batch_size}, lr={args.learning_rate:.2e}"
        )
        tqdm.write("=" * 80)

    def on_epoch_begin(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> None:
        self._close_bar()
        self.current_epoch_index += 1
        self.epoch_start_step = int(state.global_step)
        self.progress_bar = tqdm(
            total=self._epoch_total_steps(),
            desc="Training",
            dynamic_ncols=True,
            leave=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

    def on_step_end(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> None:
        if self.progress_bar is None:
            return
        target_step = int(state.global_step)
        progress_target = max(target_step - self.epoch_start_step, 0)
        current_progress = self.progress_bar.n
        step_delta = max(progress_target - current_progress, 0)
        if step_delta:
            self.progress_bar.update(step_delta)
            self.current_step = target_step

    def on_log(self, args: TrainingArguments, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if not logs:
            return
        if "loss" in logs:
            self.latest_train_logs = dict(logs)

    def on_evaluate(self, args: TrainingArguments, state: Any, control: Any, metrics: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if not metrics:
            return
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
        tqdm.write("  " + " - ".join(summary_parts))

    def on_epoch_end(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> None:
        self._close_bar()

    def on_train_end(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> None:
        self._close_bar()


def predict_from_model(
    model: AutoModelForSequenceClassification,
    dataset: SentimentDataset,
    batch_size: int = 32,
) -> tuple[list[str], np.ndarray]:
    trainer = Trainer(model=model)
    output = trainer.predict(dataset)
    logits = output.predictions
    probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    predicted_ids = np.argmax(logits, axis=-1)
    predicted_labels = [ID2LABEL[pid] for pid in predicted_ids]
    return predicted_labels, probabilities
