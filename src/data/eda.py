from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.utils.paths import ensure_dir


matplotlib.use("Agg")
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
TICKER_PATTERN = re.compile(r"\$[A-Za-z][A-Za-z.\-]*")
NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?\b")
PERCENT_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?%|\b\d+(?:[.,]\d+)?\s*%")
TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
def add_text_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["text"] = enriched["text"].fillna("").astype(str)
    enriched["text_stripped"] = enriched["text"].str.strip()
    enriched["char_length"] = enriched["text_stripped"].str.len()
    enriched["word_count"] = enriched["text_stripped"].str.split().str.len()
    enriched["has_url"] = enriched["text_stripped"].str.contains(URL_PATTERN, regex=True)
    enriched["has_ticker"] = enriched["text_stripped"].str.contains(
        TICKER_PATTERN, regex=True
    )
    enriched["has_number"] = enriched["text_stripped"].str.contains(
        NUMBER_PATTERN, regex=True
    )
    enriched["has_percent"] = enriched["text_stripped"].str.contains(
        PERCENT_PATTERN, regex=True
    )
    enriched["has_replacement_char"] = enriched["text_stripped"].str.contains(
        "\ufffd", regex=False
    )
    return enriched


def _round(value: float, digits: int = 4) -> float:
    if math.isnan(value):
        return 0.0
    return round(float(value), digits)


def build_basic_summary(frame: pd.DataFrame, dataset_name: str) -> dict[str, object]:
    total_rows = int(len(frame))
    label_counts = frame["label"].value_counts(dropna=False).to_dict()
    label_distribution = {
        str(label): {
            "count": int(count),
            "ratio": _round(count / total_rows if total_rows else 0.0),
        }
        for label, count in label_counts.items()
    }

    summary: dict[str, object] = {
        "dataset": dataset_name,
        "num_rows": total_rows,
        "num_columns": int(frame.shape[1]),
        "columns": [str(column) for column in frame.columns],
        "label_distribution": label_distribution,
        "missing_values": {column: int(value) for column, value in frame.isna().sum().items()},
        "empty_text_rows": int((frame["text_stripped"] == "").sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_texts": int(frame.duplicated(subset=["text_stripped"]).sum()),
        "parse_failures": int((~frame.get("parse_ok", pd.Series([True] * len(frame)))).sum()),
        "length_stats": {
            "char_length": {
                "mean": _round(frame["char_length"].mean()),
                "median": _round(frame["char_length"].median()),
                "min": int(frame["char_length"].min()),
                "max": int(frame["char_length"].max()),
            },
            "word_count": {
                "mean": _round(frame["word_count"].mean()),
                "median": _round(frame["word_count"].median()),
                "min": int(frame["word_count"].min()),
                "max": int(frame["word_count"].max()),
            },
        },
        "pattern_stats": {
            "rows_with_url": int(frame["has_url"].sum()),
            "rows_with_ticker": int(frame["has_ticker"].sum()),
            "rows_with_number": int(frame["has_number"].sum()),
            "rows_with_percent": int(frame["has_percent"].sum()),
            "rows_with_replacement_char": int(frame["has_replacement_char"].sum()),
        },
    }

    by_label = (
        frame.groupby("label", dropna=False)[["char_length", "word_count"]]
        .agg(["mean", "median"])
        .round(4)
    )
    summary["length_by_label"] = json.loads(by_label.to_json())
    return summary


def _normalize_text_for_tokens(text: str) -> str:
    text = URL_PATTERN.sub(" ", text)
    text = text.replace("\u2019", "'")
    return text.lower()


def _build_stopwords(remove_stopwords: bool, extra_stopwords: Iterable[str] | None) -> set[str]:
    stopwords: set[str] = set()
    if remove_stopwords:
        stopwords.update(ENGLISH_STOP_WORDS)
    if extra_stopwords:
        stopwords.update(word.strip().lower() for word in extra_stopwords if word.strip())
    return stopwords


def extract_top_ngrams(
    texts: Iterable[str],
    n: int,
    top_k: int,
    remove_stopwords: bool,
    extra_stopwords: Iterable[str] | None,
) -> pd.DataFrame:
    stopwords = _build_stopwords(remove_stopwords, extra_stopwords)
    counter: Counter[tuple[str, ...]] = Counter()

    for text in texts:
        tokens = [
            token
            for token in TOKEN_PATTERN.findall(_normalize_text_for_tokens(text))
            if len(token) >= 2 and token not in stopwords
        ]
        if len(tokens) < n:
            continue
        counter.update(tuple(tokens[idx : idx + n]) for idx in range(len(tokens) - n + 1))

    rows = [
        {"ngram": " ".join(ngram), "count": int(count)}
        for ngram, count in counter.most_common(top_k)
    ]
    return pd.DataFrame(rows)


def save_distribution_csv(frame: pd.DataFrame, path: Path) -> None:
    distribution = frame["label"].value_counts(dropna=False).rename_axis("label").reset_index(name="count")
    distribution["ratio"] = distribution["count"] / len(frame)
    distribution.to_csv(path, index=False)


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_label_distribution(frame: pd.DataFrame, dataset_name: str, path: Path) -> None:
    plt.figure(figsize=(7, 4))
    counts = frame["label"].astype(str).value_counts().reset_index()
    counts.columns = ["label", "count"]
    sns.barplot(data=counts, x="label", y="count", hue="label", palette="deep", legend=False)
    plt.title(f"{dataset_name.upper()} Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    _save_figure(path)


def plot_histogram(frame: pd.DataFrame, feature: str, dataset_name: str, path: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    sns.histplot(frame[feature], bins=30, kde=False)
    plt.title(f"{dataset_name.upper()} {feature.replace('_', ' ').title()} Histogram")
    plt.xlabel(feature.replace("_", " ").title())
    plt.ylabel("Count")
    _save_figure(path)


def plot_boxplot(frame: pd.DataFrame, feature: str, dataset_name: str, path: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    sns.boxplot(data=frame, x="label", y=feature, hue="label", palette="deep", legend=False)
    plt.title(f"{dataset_name.upper()} {feature.replace('_', ' ').title()} by Label")
    plt.xlabel("Label")
    plt.ylabel(feature.replace("_", " ").title())
    _save_figure(path)


def plot_top_words(top_words: pd.DataFrame, dataset_name: str, path: Path) -> None:
    if top_words.empty:
        return
    plot_frame = top_words.iloc[::-1]
    plt.figure(figsize=(9, 6))
    sns.barplot(data=plot_frame, x="count", y="ngram", hue="ngram", palette="viridis", legend=False)
    plt.title(f"{dataset_name.upper()} Top Words")
    plt.xlabel("Count")
    plt.ylabel("Word")
    _save_figure(path)


def write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_markdown(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def dataset_output_dir(dataset_name: str) -> Path:
    return ensure_dir(Path("outputs/eda") / dataset_name)
