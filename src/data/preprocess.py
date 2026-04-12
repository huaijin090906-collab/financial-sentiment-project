from __future__ import annotations

import re
from typing import Any

import pandas as pd


TFNS_LABEL_MAP = {0: "negative", 1: "positive", 2: "neutral"}

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MULTI_SPACE_PATTERN = re.compile(r"\s{2,}")

_FPB_PLUS_BYTE = re.compile(r"\+(?=[\x80-\xff])")
_FPB_DOUBLE_PLUS = re.compile(r"\+\+(?=[A-Za-z])")
_REPLACEMENT_CHAR = "\ufffd"


def fix_fpb_encoding(text: str) -> str:
    text = _FPB_PLUS_BYTE.sub("", text)
    text = _FPB_DOUBLE_PLUS.sub("", text)
    text = text.replace(_REPLACEMENT_CHAR, "")
    return text


def clean_text(text: str, *, remove_urls: bool = True) -> str:
    text = text.strip()
    if remove_urls:
        text = URL_PATTERN.sub("", text)
    text = MULTI_SPACE_PATTERN.sub(" ", text)
    return text.strip()


def clean_dataframe(
    frame: pd.DataFrame,
    *,
    remove_urls: bool = True,
    fix_encoding: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply lightweight cleaning. Returns (cleaned_frame, stats)."""
    stats: dict[str, Any] = {"rows_before": len(frame)}

    cleaned = frame.copy()
    cleaned["text"] = cleaned["text"].astype(str)

    if fix_encoding:
        cleaned["text"] = cleaned["text"].map(fix_fpb_encoding)

    cleaned["text"] = cleaned["text"].map(
        lambda t: clean_text(t, remove_urls=remove_urls)
    )

    empty_mask = cleaned["text"] == ""
    stats["empty_after_clean"] = int(empty_mask.sum())
    cleaned = cleaned[~empty_mask].reset_index(drop=True)

    dup_mask = cleaned.duplicated(subset=["text", "label"], keep="first")
    stats["duplicates_removed"] = int(dup_mask.sum())
    cleaned = cleaned[~dup_mask].reset_index(drop=True)

    stats["rows_after"] = len(cleaned)
    return cleaned, stats


def map_fpb_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """FPB labels are already positive/neutral/negative."""
    result = frame[["text", "label"]].copy()
    result["label"] = result["label"].str.strip().str.lower()
    return result


def map_tfns_labels(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[["text", "label"]].copy()
    result["label"] = result["label"].map(TFNS_LABEL_MAP)
    unmapped = result["label"].isna().sum()
    if unmapped > 0:
        raise ValueError(f"TFNS has {unmapped} unmapped labels.")
    return result
