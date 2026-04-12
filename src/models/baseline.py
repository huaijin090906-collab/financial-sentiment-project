from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


SUPPORTED_BASELINES = ("logreg", "linear_svm", "naive_bayes")


def _build_classifier(model_config: dict[str, Any]) -> Any:
    model_type = model_config.get("type", "logreg")

    if model_type == "logreg":
        return LogisticRegression(
            C=model_config.get("C", 1.0),
            max_iter=model_config.get("max_iter", 300),
            class_weight=model_config.get("class_weight"),
        )

    if model_type == "linear_svm":
        return LinearSVC(
            C=model_config.get("C", 1.0),
            max_iter=model_config.get("max_iter", 3000),
            class_weight=model_config.get("class_weight"),
        )

    if model_type == "naive_bayes":
        return MultinomialNB(alpha=model_config.get("alpha", 1.0))

    raise ValueError(
        f"Unsupported baseline type '{model_type}'. "
        f"Supported values: {SUPPORTED_BASELINES}."
    )


def build_baseline_pipeline(model_config: dict[str, Any]) -> Pipeline:
    ngram_range = tuple(model_config.get("ngram_range", [1, 2]))

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    sublinear_tf=True,
                    max_features=model_config.get("max_features"),
                    min_df=model_config.get("min_df", 1),
                    ngram_range=ngram_range,
                ),
            ),
            (
                "classifier",
                _build_classifier(model_config),
            ),
        ]
    )
