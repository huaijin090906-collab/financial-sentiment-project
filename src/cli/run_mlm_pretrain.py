from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer, Trainer
from transformers.trainer_callback import PrinterCallback

from src.data.dataset import load_labeled_csv
from src.models.mlm_pretrain import (
    PlainTextDataset,
    build_data_collator,
    build_mlm_model,
    build_mlm_training_args,
)
from src.models.transformer import BlockProgressCallback, configure_hf_loading_output
from src.utils.config import dump_yaml_config, load_yaml_config
from src.utils.paths import ensure_dir, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continued MLM pretraining on in-domain financial text."
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    return parser.parse_args()


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_texts(data_config: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Load plain text from configured CSV files (labels are ignored)."""
    text_column = data_config.get("text_column", "text")
    label_column = data_config.get("label_column", "label")

    all_texts: list[str] = []
    for csv_path_str in data_config.get("corpus_paths", []):
        csv_path = resolve_path(csv_path_str)
        frame = pd.read_csv(csv_path)
        texts = frame[text_column].dropna().astype(str).str.strip().tolist()
        all_texts.extend([t for t in texts if t])

    split_ratio = data_config.get("val_ratio", 0.05)
    split_idx = int(len(all_texts) * (1 - split_ratio))
    return all_texts[:split_idx], all_texts[split_idx:]


def run_pretraining(config: dict[str, Any]) -> dict[str, Any]:
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    output_config = config.get("output", {})

    model_name = model_config["name"]
    max_length = model_config.get("max_length", 128)
    mlm_probability = model_config.get("mlm_probability", 0.15)

    train_texts, val_texts = _load_texts(data_config)

    configure_hf_loading_output()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = build_mlm_model(model_name)

    train_dataset = PlainTextDataset(train_texts, tokenizer, max_length=max_length)
    val_dataset = PlainTextDataset(val_texts, tokenizer, max_length=max_length)

    data_collator = build_data_collator(tokenizer, mlm_probability=mlm_probability)

    experiment_name = config.get("experiment", {}).get("name", "mlm_pretrain")
    run_name = f"{experiment_name}_{_timestamp()}"

    checkpoints_dir = ensure_dir(output_config.get("checkpoints_dir", "outputs/checkpoints"))
    logs_dir = ensure_dir(output_config.get("logs_dir", "outputs/logs"))

    hf_output_dir = str(checkpoints_dir / run_name)
    training_args = build_mlm_training_args(hf_output_dir, training_config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[BlockProgressCallback(description="MLM Pretraining")],
    )
    trainer.remove_callback(PrinterCallback)

    trainer.train()

    save_dir = checkpoints_dir / f"{run_name}_best"
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    config_snapshot_path = logs_dir / f"{run_name}_config.yaml"
    dump_yaml_config(
        {key: value for key, value in config.items() if key != "_config_path"},
        config_snapshot_path,
    )

    result = {
        "run_name": run_name,
        "base_model": model_name,
        "train_texts": len(train_texts),
        "val_texts": len(val_texts),
        "saved_model_dir": str(save_dir),
    }

    print(f"MLM pretraining run: {run_name}")
    print(f"Base model: {model_name}")
    print(f"Train texts: {len(train_texts)}, Val texts: {len(val_texts)}")
    print(f"Pretrained model saved to: {save_dir}")

    return result


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    run_pretraining(config)


if __name__ == "__main__":
    main()
