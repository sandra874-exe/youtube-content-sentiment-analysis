import re
import pandas as pd


VALID_SENTIMENTS = {"negative", "neutral", "positive"}


def clean_text(text):
    """Light cleaning that removes noise but keeps sentiment cues."""
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_for_ml(text):
    """Cleaning used before TF-IDF. VADER/Transformer use light cleaning."""
    text = clean_text(text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prepare_dataframe(df, text_candidates=None):
    """Find the text field and create model-ready text columns."""
    df = df.copy()

    candidates = text_candidates or [
        "comment",
        "clean_text",
        "processed_text",
    ]

    source_column = next(
        (column for column in candidates if column in df.columns),
        None,
    )

    if source_column is None:
        raise ValueError(
            "No text column found. Expected one of: "
            + ", ".join(candidates)
        )

    df["source_text"] = df[source_column].fillna("").astype(str)
    df["sentiment_text"] = df["source_text"].apply(clean_text)
    df["processed_text"] = df["source_text"].apply(clean_for_ml)

    df = df[df["sentiment_text"].str.len() > 0].copy()
    df["word_count"] = df["processed_text"].str.split().str.len()
    df["character_count"] = df["processed_text"].str.len()

    return df.reset_index(drop=True)


def normalize_sentiment_labels(df, label_column="human-sentiment"):
    df = df.copy()

    if label_column not in df.columns:
        raise ValueError(
            f"Missing label column: {label_column}"
        )

    df[label_column] = (
        df[label_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    aliases = {
        "pos": "positive",
        "neu": "neutral",
        "neg": "negative",
        "1": "positive",
        "0": "neutral",
        "-1": "negative",
    }
    df[label_column] = df[label_column].replace(aliases)

    df = df[df[label_column].isin(VALID_SENTIMENTS)].copy()
    return df.reset_index(drop=True)


def dataset_quality_summary(df):
    return {
        "rows": len(df),
        "empty_comments": int(df["sentiment_text"].eq("").sum()) if "sentiment_text" in df else 0,
        "duplicate_text": int(df.duplicated(subset=["sentiment_text"]).sum()) if "sentiment_text" in df else 0,
        "unique_creators": int(df["creator"].nunique()) if "creator" in df.columns else 0,
        "unique_videos": int(df["video_id"].nunique()) if "video_id" in df.columns else 0,
    }
