from __future__ import annotations

from typing import Any

import pandas as pd


SARCASM_MODEL = "YamenRM/sarcasm_model"


def _normalise_label(label: Any) -> str:
    value = str(label).strip().lower()

    if value in {
        "sarcastic",
        "label_1",
        "1",
        "true",
        "yes",
    }:
        return "sarcastic"

    return "not_sarcastic"


def load_sarcasm_detector(
    model_name: str = SARCASM_MODEL,
):
    try:
        from transformers import pipeline

        return pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            truncation=True,
            max_length=128,
        )

    except Exception:
        return None


def detect_sarcasm(
    text: Any,
    detector=None,
) -> dict:
    text = "" if pd.isna(text) else str(text).strip()

    if not text:
        return {
            "sarcasm_label": "not_sarcastic",
            "is_sarcastic": False,
            "sarcasm_score": 0.0,
        }

    if detector is None:
        detector = load_sarcasm_detector()

    if detector is None:
        return {
            "sarcasm_label": "unavailable",
            "is_sarcastic": False,
            "sarcasm_score": 0.0,
        }

    try:
        output = detector(text)

        if isinstance(output, list):
            output = output[0]

        label = _normalise_label(
            output.get("label", "not_sarcastic")
        )

        score = float(
            output.get("score", 0.0)
        )

        return {
            "sarcasm_label": label,
            "is_sarcastic": label == "sarcastic",
            "sarcasm_score": score,
        }

    except Exception:
        return {
            "sarcasm_label": "unavailable",
            "is_sarcastic": False,
            "sarcasm_score": 0.0,
        }


def analyze_sarcasm(
    dataframe: pd.DataFrame,
    detector=None,
    batch_size: int = 32,
) -> pd.DataFrame:

    df = dataframe.copy()

    if df.empty:
        df["sarcasm_label"] = pd.Series(dtype="object")
        df["is_sarcastic"] = pd.Series(dtype="bool")
        df["sarcasm_score"] = pd.Series(dtype="float64")
        return df

    if detector is None:
        detector = load_sarcasm_detector()

    if detector is None:
        df["sarcasm_label"] = "unavailable"
        df["is_sarcastic"] = False
        df["sarcasm_score"] = 0.0
        return df

    if "clean_text" in df.columns:
        texts = (
            df["clean_text"]
            .fillna("")
            .astype(str)
            .tolist()
        )
    else:
        texts = (
            df["comment_text"]
            .fillna("")
            .astype(str)
            .tolist()
        )

    labels = []
    flags = []
    scores = []

    for start in range(0, len(texts), batch_size):

        batch = texts[start:start + batch_size]

        try:
            outputs = detector(
                batch,
                batch_size=batch_size,
                truncation=True,
                max_length=128,
            )

        except TypeError:
            outputs = detector(batch)

        for output in outputs:

            if isinstance(output, list):
                output = max(
                    output,
                    key=lambda x: x.get("score", 0.0),
                )

            label = _normalise_label(
                output.get(
                    "label",
                    "not_sarcastic",
                )
            )

            score = float(
                output.get("score", 0.0)
            )

            labels.append(label)
            flags.append(label == "sarcastic")
            scores.append(score)

    df["sarcasm_label"] = labels
    df["is_sarcastic"] = flags
    df["sarcasm_score"] = scores

    return df


def sarcasm_summary(
    dataframe: pd.DataFrame,
) -> dict:

    if (
        dataframe is None
        or dataframe.empty
        or "sarcasm_label" not in dataframe.columns
    ):
        return {
            "total_comments": 0,
            "sarcastic_comments": 0,
            "not_sarcastic_comments": 0,
            "sarcasm_percentage": 0.0,
            "average_sarcasm_confidence": 0.0,
            "available": False,
        }

    valid = dataframe[
        dataframe["sarcasm_label"].isin(
            [
                "sarcastic",
                "not_sarcastic",
            ]
        )
    ].copy()

    if valid.empty:
        return {
            "total_comments": len(dataframe),
            "sarcastic_comments": 0,
            "not_sarcastic_comments": 0,
            "sarcasm_percentage": 0.0,
            "average_sarcasm_confidence": 0.0,
            "available": False,
        }

    sarcastic = int(
        (
            valid["sarcasm_label"]
            == "sarcastic"
        ).sum()
    )

    total = len(valid)

    confidence = pd.to_numeric(
        valid["sarcasm_score"],
        errors="coerce",
    ).fillna(0)

    return {
        "total_comments": int(total),
        "sarcastic_comments": sarcastic,
        "not_sarcastic_comments": int(
            total - sarcastic
        ),
        "sarcasm_percentage": round(
            sarcastic / total * 100,
            2,
        ),
        "average_sarcasm_confidence": round(
            float(confidence.mean()) * 100,
            2,
        ),
        "available": True,
    }


def sarcasm_by_sentiment(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    columns = [
        "vader_sentiment",
        "comments",
        "sarcastic_comments",
        "sarcasm_percentage",
    ]

    if (
        dataframe is None
        or dataframe.empty
        or "sarcasm_label" not in dataframe.columns
    ):
        return pd.DataFrame(columns=columns)

    valid = dataframe[
        dataframe["sarcasm_label"].isin(
            [
                "sarcastic",
                "not_sarcastic",
            ]
        )
    ].copy()

    if (
        valid.empty
        or "vader_sentiment" not in valid.columns
    ):
        return pd.DataFrame(columns=columns)

    result = (
        valid
        .groupby("vader_sentiment")
        .agg(
            comments=(
                "sarcasm_label",
                "size",
            ),
            sarcastic_comments=(
                "is_sarcastic",
                "sum",
            ),
        )
        .reindex(
            [
                "positive",
                "neutral",
                "negative",
            ],
            fill_value=0,
        )
        .reset_index()
    )

    result["comments"] = (
        result["comments"]
        .astype(int)
    )

    result["sarcastic_comments"] = (
        result["sarcastic_comments"]
        .astype(int)
    )

    result["sarcasm_percentage"] = (
        result["sarcastic_comments"]
        /
        result["comments"].replace(0, 1)
        * 100
    ).round(2)

    return result


def sarcastic_comment_examples(
    dataframe: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:

    if (
        dataframe is None
        or dataframe.empty
        or "sarcasm_label" not in dataframe.columns
    ):
        return pd.DataFrame()

    result = dataframe[
        dataframe["sarcasm_label"]
        == "sarcastic"
    ].copy()

    if result.empty:
        return result

    return (
        result
        .sort_values(
            "sarcasm_score",
            ascending=False,
        )
        .head(n)
    )