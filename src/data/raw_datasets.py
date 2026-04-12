from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.paths import resolve_path


def load_fpb_raw(path_str: str) -> pd.DataFrame:
    """Load Financial PhraseBank raw text file without label remapping."""
    path = resolve_path(path_str)
    records: list[dict[str, object]] = []

    with path.open("r", encoding="latin-1") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                records.append(
                    {
                        "dataset": "fpb",
                        "split": "all_agree",
                        "line_number": line_number,
                        "text": "",
                        "label": None,
                        "parse_ok": False,
                        "raw_line": line,
                    }
                )
                continue

            if "@" not in line:
                records.append(
                    {
                        "dataset": "fpb",
                        "split": "all_agree",
                        "line_number": line_number,
                        "text": line,
                        "label": None,
                        "parse_ok": False,
                        "raw_line": line,
                    }
                )
                continue

            text, label = line.rsplit("@", 1)
            records.append(
                {
                    "dataset": "fpb",
                    "split": "all_agree",
                    "line_number": line_number,
                    "text": text,
                    "label": label,
                    "parse_ok": True,
                    "raw_line": line,
                }
            )

    return pd.DataFrame(records)


def load_tfns_raw(train_path_str: str, validation_path_str: str) -> pd.DataFrame:
    """Load TFNS train and validation CSVs without label remapping."""
    train_path = resolve_path(train_path_str)
    validation_path = resolve_path(validation_path_str)

    train_frame = pd.read_csv(train_path).copy()
    train_frame["dataset"] = "tfns"
    train_frame["split"] = "train"
    train_frame["source_file"] = str(train_path)

    validation_frame = pd.read_csv(validation_path).copy()
    validation_frame["dataset"] = "tfns"
    validation_frame["split"] = "validation"
    validation_frame["source_file"] = str(validation_path)

    combined = pd.concat([train_frame, validation_frame], ignore_index=True)
    combined["row_number"] = range(1, len(combined) + 1)
    return combined


def load_all_raw_datasets() -> dict[str, pd.DataFrame]:
    return {
        "fpb": load_fpb_raw("data/raw/fpb/Sentences_AllAgree.txt"),
        "tfns": load_tfns_raw(
            "data/raw/tfns/train.csv",
            "data/raw/tfns/validation.csv",
        ),
    }
