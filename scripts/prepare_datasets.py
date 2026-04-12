from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.preprocess import (
    clean_dataframe,
    map_fpb_labels,
    map_tfns_labels,
)
from src.data.raw_datasets import load_fpb_raw, load_tfns_raw
from src.utils.paths import ensure_dir

RANDOM_SEED = 42


def save_split(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def split_summary(name: str, frame: pd.DataFrame) -> dict[str, object]:
    dist = frame["label"].value_counts().to_dict()
    return {
        "name": name,
        "rows": len(frame),
        "label_distribution": {str(k): int(v) for k, v in dist.items()},
    }


def prepare_fpb() -> dict[str, object]:
    print("Loading FPB raw data...")
    raw = load_fpb_raw("data/raw/fpb/Sentences_AllAgree.txt")
    valid = raw[raw["parse_ok"]].copy()

    mapped = map_fpb_labels(valid)
    cleaned, clean_stats = clean_dataframe(
        mapped, remove_urls=True, fix_encoding=True
    )

    # 70 / 15 / 15 stratified split
    train_frame, temp_frame = train_test_split(
        cleaned, test_size=0.30, random_state=RANDOM_SEED, stratify=cleaned["label"]
    )
    val_frame, test_frame = train_test_split(
        temp_frame, test_size=0.50, random_state=RANDOM_SEED, stratify=temp_frame["label"]
    )

    out_dir = ensure_dir("data/processed/fpb")
    save_split(train_frame, out_dir / "train.csv")
    save_split(val_frame, out_dir / "val.csv")
    save_split(test_frame, out_dir / "test.csv")

    report = {
        "dataset": "fpb",
        "cleaning": clean_stats,
        "splits": {
            "train": split_summary("train", train_frame),
            "val": split_summary("val", val_frame),
            "test": split_summary("test", test_frame),
        },
    }
    print(f"  FPB done: train={len(train_frame)}, val={len(val_frame)}, test={len(test_frame)}")
    return report


def prepare_tfns() -> dict[str, object]:
    print("Loading TFNS raw data...")
    raw = load_tfns_raw("data/raw/tfns/train.csv", "data/raw/tfns/validation.csv")

    raw_train = raw[raw["split"] == "train"].copy()
    raw_val = raw[raw["split"] == "validation"].copy()

    mapped_train = map_tfns_labels(raw_train)
    mapped_val = map_tfns_labels(raw_val)

    cleaned_train, train_clean_stats = clean_dataframe(
        mapped_train, remove_urls=True, fix_encoding=False
    )
    cleaned_val, val_clean_stats = clean_dataframe(
        mapped_val, remove_urls=True, fix_encoding=False
    )

    # Split 15% from official train as test (stratified)
    train_frame, test_frame = train_test_split(
        cleaned_train,
        test_size=0.15,
        random_state=RANDOM_SEED,
        stratify=cleaned_train["label"],
    )
    val_frame = cleaned_val

    out_dir = ensure_dir("data/processed/tfns")
    save_split(train_frame, out_dir / "train.csv")
    save_split(val_frame, out_dir / "val.csv")
    save_split(test_frame, out_dir / "test.csv")

    report = {
        "dataset": "tfns",
        "cleaning": {
            "train_split": train_clean_stats,
            "val_split": val_clean_stats,
        },
        "splits": {
            "train": split_summary("train", train_frame),
            "val": split_summary("val", val_frame),
            "test": split_summary("test", test_frame),
        },
    }
    print(f"  TFNS done: train={len(train_frame)}, val={len(val_frame)}, test={len(test_frame)}")
    return report


def main() -> None:
    fpb_report = prepare_fpb()
    tfns_report = prepare_tfns()

    full_report = {"fpb": fpb_report, "tfns": tfns_report}

    report_path = ensure_dir("outputs/preprocessing") / "processing_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    print(f"\nProcessing report saved to: {report_path}")


if __name__ == "__main__":
    main()
