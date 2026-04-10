from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.eda import (
    add_text_features,
    build_basic_summary,
    dataset_output_dir,
    extract_top_ngrams,
    plot_boxplot,
    plot_histogram,
    plot_label_distribution,
    plot_top_words,
    save_distribution_csv,
    write_json,
    write_markdown,
)
from src.data.raw_datasets import load_all_raw_datasets
from src.utils.paths import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDA on raw financial sentiment datasets.")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k words/ngrams to save.")
    parser.add_argument(
        "--remove-stopwords",
        action="store_true",
        help="Remove English stopwords during word/ngram counting.",
    )
    parser.add_argument(
        "--extra-stopwords-file",
        type=str,
        default=None,
        help="Optional file with one extra stopword per line.",
    )
    return parser.parse_args()


def load_extra_stopwords(path_str: str | None) -> list[str]:
    if not path_str:
        return []
    return [
        line.strip()
        for line in Path(path_str).expanduser().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_dataset_markdown(dataset_name: str, summary: dict[str, object]) -> str:
    label_distribution = summary["label_distribution"]
    length_stats = summary["length_stats"]
    pattern_stats = summary["pattern_stats"]

    lines = [
        f"# {dataset_name.upper()} EDA Summary",
        "",
        f"- Rows: {summary['num_rows']}",
        f"- Columns: {', '.join(summary['columns'])}",
        f"- Empty text rows: {summary['empty_text_rows']}",
        f"- Duplicate rows: {summary['duplicate_rows']}",
        f"- Duplicate texts: {summary['duplicate_texts']}",
        f"- Parse failures: {summary['parse_failures']}",
        "",
        "## Label Distribution",
    ]

    for label, payload in label_distribution.items():
        lines.append(f"- {label}: {payload['count']} ({payload['ratio']:.2%})")

    lines.extend(
        [
            "",
            "## Text Length",
            f"- Character length mean / median: {length_stats['char_length']['mean']} / {length_stats['char_length']['median']}",
            f"- Character length min / max: {length_stats['char_length']['min']} / {length_stats['char_length']['max']}",
            f"- Word count mean / median: {length_stats['word_count']['mean']} / {length_stats['word_count']['median']}",
            f"- Word count min / max: {length_stats['word_count']['min']} / {length_stats['word_count']['max']}",
            "",
            "## Pattern Stats",
            f"- Rows with URL: {pattern_stats['rows_with_url']}",
            f"- Rows with ticker: {pattern_stats['rows_with_ticker']}",
            f"- Rows with number: {pattern_stats['rows_with_number']}",
            f"- Rows with percent: {pattern_stats['rows_with_percent']}",
            f"- Rows with replacement character: {pattern_stats['rows_with_replacement_char']}",
        ]
    )
    return "\n".join(lines) + "\n"


def build_comparison_markdown(
    fpb_summary: dict[str, object], tfns_summary: dict[str, object]
) -> str:
    fpb_labels = fpb_summary["label_distribution"]
    tfns_labels = tfns_summary["label_distribution"]
    fpb_lengths = fpb_summary["length_stats"]
    tfns_lengths = tfns_summary["length_stats"]
    fpb_patterns = fpb_summary["pattern_stats"]
    tfns_patterns = tfns_summary["pattern_stats"]

    lines = [
        "# Cross-Dataset Comparison",
        "",
        "## Label Balance",
        f"- FPB is smaller ({fpb_summary['num_rows']} rows) and still noticeably imbalanced, with `neutral` dominating.",
        f"- TFNS is much larger ({tfns_summary['num_rows']} rows) and heavily skewed toward label `2`, while labels `0` and `1` are much less frequent.",
        "",
        "## Text Length",
        f"- FPB average text length: {fpb_lengths['word_count']['mean']} words, {fpb_lengths['char_length']['mean']} characters.",
        f"- TFNS average text length: {tfns_lengths['word_count']['mean']} words, {tfns_lengths['char_length']['mean']} characters.",
        "- This suggests FPB contains more sentence-like financial statements, while TFNS is closer to short headlines or alert-style snippets.",
        "",
        "## Writing Style",
        f"- FPB rows with URLs / tickers: {fpb_patterns['rows_with_url']} / {fpb_patterns['rows_with_ticker']}.",
        f"- TFNS rows with URLs / tickers: {tfns_patterns['rows_with_url']} / {tfns_patterns['rows_with_ticker']}.",
        "- TFNS contains much more platform-style noise such as URLs, ticker symbols, and short brokerage-action headlines.",
        "",
        "## Data Quality Risks",
        f"- FPB has replacement-character rows: {fpb_patterns['rows_with_replacement_char']}, indicating encoding cleanup may be needed later.",
        "- TFNS appears structurally clean, but the numeric labels should be verified against the dataset documentation before any later label mapping step.",
        "",
        "## Cross-Dataset Evaluation Risks",
        "- Domain shift is likely to be substantial because the two datasets differ in text length, writing style, and label format.",
        "- URL/ticker-heavy TFNS text may cause models trained on FPB to underperform on TFNS unless tokenization and preprocessing are handled carefully in a later step.",
        "- Severe class imbalance, especially in TFNS, may distort macro-F1 and cross-dataset transfer unless addressed later during training or sampling.",
    ]

    return "\n".join(lines) + "\n"


def run_dataset_eda(
    dataset_name: str,
    frame: pd.DataFrame,
    top_k: int,
    remove_stopwords: bool,
    extra_stopwords: list[str],
) -> dict[str, object]:
    output_dir = dataset_output_dir(dataset_name)
    enriched = add_text_features(frame)
    summary = build_basic_summary(enriched, dataset_name)

    save_distribution_csv(enriched, output_dir / "label_distribution.csv")

    top_words = extract_top_ngrams(
        enriched["text_stripped"], n=1, top_k=top_k, remove_stopwords=remove_stopwords, extra_stopwords=extra_stopwords
    )
    top_bigrams = extract_top_ngrams(
        enriched["text_stripped"], n=2, top_k=top_k, remove_stopwords=remove_stopwords, extra_stopwords=extra_stopwords
    )
    top_trigrams = extract_top_ngrams(
        enriched["text_stripped"], n=3, top_k=top_k, remove_stopwords=remove_stopwords, extra_stopwords=extra_stopwords
    )

    top_words.to_csv(output_dir / "top_words.csv", index=False)
    top_bigrams.to_csv(output_dir / "top_bigrams.csv", index=False)
    top_trigrams.to_csv(output_dir / "top_trigrams.csv", index=False)

    plot_label_distribution(enriched, dataset_name, output_dir / "label_distribution.png")
    plot_histogram(enriched, "char_length", dataset_name, output_dir / "char_length_histogram.png")
    plot_histogram(enriched, "word_count", dataset_name, output_dir / "word_count_histogram.png")
    plot_boxplot(enriched, "char_length", dataset_name, output_dir / "char_length_boxplot_by_label.png")
    plot_boxplot(enriched, "word_count", dataset_name, output_dir / "word_count_boxplot_by_label.png")
    plot_top_words(top_words, dataset_name, output_dir / "top_words.png")

    write_json(summary, output_dir / "summary.json")
    write_markdown(build_dataset_markdown(dataset_name, summary), output_dir / "summary.md")

    print(f"[{dataset_name}] rows={summary['num_rows']} output={output_dir}")
    return summary


def main() -> None:
    args = parse_args()
    extra_stopwords = load_extra_stopwords(args.extra_stopwords_file)

    datasets = load_all_raw_datasets()
    all_output_dir = ensure_dir("outputs/eda")

    fpb_summary = run_dataset_eda(
        "fpb",
        datasets["fpb"],
        top_k=args.top_k,
        remove_stopwords=args.remove_stopwords,
        extra_stopwords=extra_stopwords,
    )
    tfns_summary = run_dataset_eda(
        "tfns",
        datasets["tfns"],
        top_k=args.top_k,
        remove_stopwords=args.remove_stopwords,
        extra_stopwords=extra_stopwords,
    )

    comparison_text = build_comparison_markdown(fpb_summary, tfns_summary)
    write_markdown(comparison_text, all_output_dir / "comparison_summary.md")

    print(f"Comparison summary saved to: {all_output_dir / 'comparison_summary.md'}")


if __name__ == "__main__":
    main()
