from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


class PlainTextDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
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

    def __len__(self) -> int:
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {key: val[idx] for key, val in self.encodings.items()}


def build_mlm_model(model_name: str) -> AutoModelForMaskedLM:
    return AutoModelForMaskedLM.from_pretrained(model_name)


def build_mlm_training_args(
    output_dir: str,
    training_config: dict[str, Any],
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config.get("epochs", 5),
        per_device_train_batch_size=training_config.get("train_batch_size", 16),
        per_device_eval_batch_size=training_config.get("eval_batch_size", 32),
        learning_rate=training_config.get("learning_rate", 5e-5),
        weight_decay=training_config.get("weight_decay", 0.01),
        warmup_ratio=training_config.get("warmup_ratio", 0.1),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=training_config.get("logging_steps", 50),
        fp16=training_config.get("fp16", torch.cuda.is_available()),
        report_to="none",
        seed=training_config.get("seed", 42),
        disable_tqdm=training_config.get("disable_tqdm", True),
    )


def build_data_collator(
    tokenizer: AutoTokenizer,
    mlm_probability: float = 0.15,
) -> DataCollatorForLanguageModeling:
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
    )
