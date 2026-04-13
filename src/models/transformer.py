from __future__ import annotations

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
    """Custom tqdm-based progress display with clear progress bars."""

    def __init__(self, description: str = "Training") -> None:
        self.description = description
        self.progress_bar: tqdm | None = None
        self.current_step = 0

    def on_train_begin(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> None:
        total_steps = int(state.max_steps or 0)
        self.progress_bar = tqdm(
            total=total_steps,
            desc=self.description,
            dynamic_ncols=True,
            leave=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )
        self.current_step = 0

    def on_step_end(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> None:
        if self.progress_bar is None:
            return
        target_step = int(state.global_step)
        step_delta = max(target_step - self.current_step, 0)
        if step_delta:
            self.progress_bar.update(step_delta)
            self.current_step = target_step

    def on_log(self, args: TrainingArguments, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if self.progress_bar is None or not logs:
            return

        postfix: dict[str, str] = {}
        if "loss" in logs:
            postfix["loss"] = f"{logs['loss']:.4f}"
        if "learning_rate" in logs:
            postfix["lr"] = f"{logs['learning_rate']:.2e}"
        if postfix:
            self.progress_bar.set_postfix(postfix)

    def on_evaluate(self, args: TrainingArguments, state: Any, control: Any, metrics: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if not metrics:
            return

        summary_parts = []
        if "eval_loss" in metrics:
            summary_parts.append(f"eval_loss={metrics['eval_loss']:.4f}")
        if "eval_accuracy" in metrics:
            summary_parts.append(f"eval_acc={metrics['eval_accuracy']:.4f}")
        if "eval_macro_f1" in metrics:
            summary_parts.append(f"eval_macro_f1={metrics['eval_macro_f1']:.4f}")
        if "eval_weighted_f1" in metrics:
            summary_parts.append(f"eval_weighted_f1={metrics['eval_weighted_f1']:.4f}")
        if summary_parts:
            tqdm.write(" | ".join(summary_parts))

    def on_train_end(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> None:
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None


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
