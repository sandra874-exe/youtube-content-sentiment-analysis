from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ============================================================
# CONSTANTS
# ============================================================

SENTIMENT_ORDER = ["positive", "neutral", "negative"]

_VADER = SentimentIntensityAnalyzer()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: Any) -> str:
    """Clean a YouTube comment using the project cleaning rules."""
    from html import unescape

    if pd.isna(text):
        return ""

    text = unescape(str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _ensure_dataframe(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        raise TypeError("Comments must be a pandas DataFrame or a list of dictionaries.")

    if df.empty:
        return df

    if "comment_text" not in df.columns:
        if "text" in df.columns:
            df["comment_text"] = df["text"]
        elif "clean_text" in df.columns:
            df["comment_text"] = df["clean_text"]
        else:
            raise ValueError("Input comments must contain 'comment_text'.")

    if "like_count" not in df.columns:
        df["like_count"] = 0

    df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce").fillna(0)
    df["clean_text"] = df["comment_text"].map(clean_text)
    return df


# ============================================================
# VADER
# ============================================================

def get_vader_sentiment(text: Any) -> dict:
    """
    Return VADER sentiment and all sentiment scores.
    """

    if pd.isna(text):
        text = ""

    text = str(text)

    scores = _VADER.polarity_scores(text)

    compound = scores["compound"]

    if compound >= 0.05:
        sentiment = "positive"
    elif compound <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return {
        "sentiment": sentiment,
        "positive_score": scores["pos"],
        "neutral_score": scores["neu"],
        "negative_score": scores["neg"],
        "compound_score": compound,
    }

def analyze_vader(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Run VADER on every comment and store sentiment + scores.
    """

    df = _ensure_dataframe(dataframe)

    if df.empty:
        df["vader_sentiment"] = pd.Series(dtype="object")
        df["vader_positive"] = pd.Series(dtype="float")
        df["vader_neutral"] = pd.Series(dtype="float")
        df["vader_negative"] = pd.Series(dtype="float")
        df["vader_compound"] = pd.Series(dtype="float")
        return df

    results = df["clean_text"].map(get_vader_sentiment)

    df["vader_sentiment"] = results.map(
        lambda x: x["sentiment"]
    )

    df["vader_positive"] = results.map(
        lambda x: x["positive_score"]
    )

    df["vader_neutral"] = results.map(
        lambda x: x["neutral_score"]
    )

    df["vader_negative"] = results.map(
        lambda x: x["negative_score"]
    )

    df["vader_compound"] = results.map(
        lambda x: x["compound_score"]
    )

    return df

# ============================================================
# TRANSFORMER
# ============================================================

def load_transformer(model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
    """Load the transformer pipeline. Returns None if unavailable."""
    try:
        from transformers import pipeline

        return pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            truncation=True,
            max_length=512,
        )
    except Exception:
        return None


def _normalise_transformer_label(label: Any) -> str:
    value = str(label).strip().lower()
    mapping = {
        "label_0": "negative",
        "label_1": "neutral",
        "label_2": "positive",
        "0": "negative",
        "1": "neutral",
        "2": "positive",
        "negative": "negative",
        "neutral": "neutral",
        "positive": "positive",
    }
    return mapping.get(value, value)


def analyze_transformer(
    dataframe: pd.DataFrame,
    classifier=None,
    batch_size: int = 32,
) -> pd.DataFrame:
    df = _ensure_dataframe(dataframe)

    if df.empty:
        df["transformer_sentiment"] = pd.Series(dtype="object")
        df["transformer_score"] = pd.Series(dtype="float64")
        return df

    if classifier is None:
        classifier = load_transformer()

    if classifier is None:
        df["transformer_sentiment"] = np.nan
        df["transformer_score"] = np.nan
        return df

    texts = df["clean_text"].fillna("").tolist()
    labels: List[str] = []
    scores: List[float] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        try:
            outputs = classifier(
            batch,
            batch_size=batch_size,
            truncation=True,
            max_length=512,
        )
        except TypeError:
            outputs = classifier(batch)

        for output in outputs:
            if isinstance(output, list):
                output = max(output, key=lambda x: x.get("score", 0))
            labels.append(_normalise_transformer_label(output.get("label", "neutral")))
            scores.append(float(output.get("score", 0.0)))

    df["transformer_sentiment"] = labels
    df["transformer_score"] = scores
    return df


# ============================================================
# SUMMARIES / ENGAGEMENT
# ============================================================

def sentiment_summary(
    dataframe: pd.DataFrame,
    sentiment_column: str = "vader_sentiment",
) -> pd.DataFrame:
    if dataframe is None or dataframe.empty:
        return pd.DataFrame(columns=["sentiment", "count", "percentage"])

    if sentiment_column not in dataframe.columns:
        raise ValueError(
            f"Sentiment column '{sentiment_column}' was not found."
        )

    counts = (
        dataframe[sentiment_column]
        .value_counts()
        .reindex(SENTIMENT_ORDER, fill_value=0)
    )

    total = int(counts.sum())

    result = pd.DataFrame({
        "sentiment": SENTIMENT_ORDER,
        "count": counts.astype(int).values,
    })

    result["percentage"] = (
        result["count"] / total * 100 if total else 0
    )

    result["percentage"] = result["percentage"].round(2)

    return result


def engagement_analysis(
    dataframe: pd.DataFrame,
    sentiment_column: str = "vader_sentiment",
) -> pd.DataFrame:
    if dataframe is None or dataframe.empty:
        return pd.DataFrame(columns=[
            "sentiment", "comments", "average_likes", "total_likes"
        ])

    if sentiment_column not in dataframe.columns:
        raise ValueError(
            f"Sentiment column '{sentiment_column}' was not found."
        )

    df = dataframe.copy()

    if "like_count" not in df.columns:
        df["like_count"] = 0

    df["like_count"] = pd.to_numeric(
        df["like_count"],
        errors="coerce",
    ).fillna(0)

    result = (
        df.groupby(sentiment_column)
        .agg(
            comments=("comment_text", "count"),
            average_likes=("like_count", "mean"),
            total_likes=("like_count", "sum"),
        )
        .reindex(SENTIMENT_ORDER, fill_value=0)
        .rename_axis("sentiment")
        .reset_index()
    )

    result["comments"] = (
        pd.to_numeric(
            result["comments"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    result["average_likes"] = (
        pd.to_numeric(
            result["average_likes"],
            errors="coerce",
        )
        .fillna(0)
        .round(2)
    )

    result["total_likes"] = (
        pd.to_numeric(
            result["total_likes"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    return result


def comment_statistics(dataframe: pd.DataFrame) -> Dict[str, float]:
    if dataframe is None or dataframe.empty:
        return {
            "total_comments": 0,
            "average_likes": 0.0,
            "total_likes": 0,
            "average_words": 0.0,
            "average_characters": 0.0,
        }

    df = dataframe.copy()
    df["like_count"] = pd.to_numeric(df.get("like_count", 0), errors="coerce").fillna(0)

    text = df["comment_text"].fillna("").astype(str)
    words = text.str.split().str.len()
    chars = text.str.len()

    return {
        "total_comments": int(len(df)),
        "average_likes": round(float(df["like_count"].mean()), 2),
        "total_likes": int(df["like_count"].sum()),
        "average_words": round(float(words.mean()), 2),
        "average_characters": round(float(chars.mean()), 2),
    }


# ============================================================
# KEYWORDS / TF-IDF
# ============================================================

def _text_for_sentiment(
    dataframe: pd.DataFrame,
    sentiment: Optional[str],
    sentiment_column: str = "vader_sentiment",
) -> List[str]:
    if dataframe is None or dataframe.empty:
        return []

    if sentiment_column not in dataframe.columns:
        raise ValueError(
            f"Sentiment column '{sentiment_column}' was not found."
        )

    df = dataframe

    if sentiment and sentiment.lower() != "all":
        df = df[
            df[sentiment_column]
            .astype(str)
            .str.lower()
            == sentiment.lower()
        ]

    if df.empty:
        return []

    return (
        df["clean_text"]
        .fillna(df["comment_text"].fillna(""))
        .astype(str)
        .tolist()
    )


def extract_keywords(
    dataframe: pd.DataFrame,
    sentiment: Optional[str] = None,
    top_n: int = 20,
    sentiment_column: str = "vader_sentiment",
) -> pd.DataFrame:
    texts = _text_for_sentiment(
        dataframe,
        sentiment,
        sentiment_column,
    )
    empty = pd.DataFrame(columns=["word", "frequency"])
    if not texts:
        return empty

    try:
        vectorizer = CountVectorizer(
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']{2,}\b",
        )
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return empty

    frequencies = np.asarray(matrix.sum(axis=0)).ravel()
    words = vectorizer.get_feature_names_out()
    result = pd.DataFrame({"word": words, "frequency": frequencies.astype(int)})
    return result.sort_values(
        ["frequency", "word"], ascending=[False, True]
    ).head(top_n).reset_index(drop=True)


def extract_tfidf(
    dataframe: pd.DataFrame,
    sentiment: Optional[str] = None,
    top_n: int = 20,
    sentiment_column: str = "vader_sentiment",
) -> pd.DataFrame:
    texts = _text_for_sentiment(
        dataframe,
        sentiment,
        sentiment_column,
    )
    empty = pd.DataFrame(columns=["term", "tfidf"])
    if not texts:
        return empty

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']{2,}\b",
        )
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return empty

    scores = np.asarray(matrix.mean(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    result = pd.DataFrame({"term": terms, "tfidf": scores})
    return result.sort_values(
        ["tfidf", "term"], ascending=[False, True]
    ).head(top_n).reset_index(drop=True)


# ============================================================
# EXAMPLES / MODEL AGREEMENT / INSIGHTS
# ============================================================

def get_comment_examples(
    dataframe: pd.DataFrame,
    sentiment: str,
    n: int = 5,
    sentiment_column: str = "vader_sentiment",
) -> pd.DataFrame:
    if (
        dataframe is None
        or dataframe.empty
        or sentiment_column not in dataframe.columns
    ):
        return pd.DataFrame()

    return dataframe[
        dataframe[sentiment_column]
        .astype(str)
        .str.lower()
        == sentiment.lower()
    ].head(n).copy()


def model_agreement(dataframe: pd.DataFrame) -> float:
    if dataframe is None or dataframe.empty:
        return 0.0
    if "transformer_sentiment" not in dataframe.columns:
        return 0.0

    valid = dataframe.dropna(subset=["vader_sentiment", "transformer_sentiment"])
    if valid.empty:
        return 0.0
    return round(
        float((valid["vader_sentiment"] == valid["transformer_sentiment"]).mean() * 100),
        2,
    )


def generate_insights(
    dataframe: pd.DataFrame,
    summary: Optional[pd.DataFrame] = None,
    sentiment_column: str = "vader_sentiment",
) -> List[str]:
    if dataframe is None or dataframe.empty:
        return ["There are not enough comments to generate audience insights."]

    if sentiment_column not in dataframe.columns:
        raise ValueError(
            f"Sentiment column '{sentiment_column}' was not found."
        )

    if summary is None:
        summary = sentiment_summary(
            dataframe,
            sentiment_column=sentiment_column,
        )

    total = len(dataframe)
    counts = dataframe[sentiment_column].value_counts()
    dominant = counts.idxmax() if not counts.empty else "neutral"
    dominant_pct = counts.get(dominant, 0) / total * 100

    insights = [
        f"{dominant.title()} sentiment is the largest audience mood, representing {dominant_pct:.1f}% of analysed comments.",
    ]

    if "like_count" in dataframe.columns:
        likes = pd.to_numeric(dataframe["like_count"], errors="coerce").fillna(0)
        by_sentiment = dataframe.assign(_likes=likes).groupby("vader_sentiment") ["_likes"].mean()
        if not by_sentiment.empty:
            highest = by_sentiment.idxmax()
            insights.append(
                f"Comments classified as {highest} receive the highest average likes ({by_sentiment.max():.1f})."
            )

    if "transformer_sentiment" in dataframe.columns:
        agreement = model_agreement(dataframe)
        insights.append(
            f"VADER and the Transformer model agree on {agreement:.1f}% of analysed comments."
        )

    if len(insights) < 4:
        negative_pct = counts.get("negative", 0) / total * 100
        positive_pct = counts.get("positive", 0) / total * 100
        if positive_pct >= negative_pct:
            insights.append(
                f"Positive comments exceed negative comments by {positive_pct - negative_pct:.1f} percentage points."
            )
        else:
            insights.append(
                f"Negative comments exceed positive comments by {negative_pct - positive_pct:.1f} percentage points."
            )

    return insights[:4]


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def run_complete_analysis(
    comments: Any,
    transformer_classifier=None,
    use_transformer: bool = True,
) -> Dict[str, Any]:
    """Run the complete pipeline and return the exact structure used by app.py."""
    df = _ensure_dataframe(comments)

    if df.empty:
        empty_summary = pd.DataFrame(columns=["sentiment", "count", "percentage"])
        empty_engagement = pd.DataFrame(columns=[
            "vader_sentiment", "comments", "average_likes", "total_likes"
        ])
        return {
            "comments": df,
            "summary": empty_summary,
            "statistics": comment_statistics(df),
            "insights": ["No comments were available for analysis."],
            "keywords": {"all": pd.DataFrame(columns=["word", "frequency"]), "positive": pd.DataFrame(columns=["word", "frequency"]), "neutral": pd.DataFrame(columns=["word", "frequency"]), "negative": pd.DataFrame(columns=["word", "frequency"])},
            "tfidf": {"all": pd.DataFrame(columns=["term", "tfidf"]), "positive": pd.DataFrame(columns=["term", "tfidf"]), "neutral": pd.DataFrame(columns=["term", "tfidf"]), "negative": pd.DataFrame(columns=["term", "tfidf"])},
            "engagement": empty_engagement,
            "model_agreement": 0.0,
        }

    df = analyze_vader(df)

    if use_transformer:
        df = analyze_transformer(
            df,
            classifier=transformer_classifier,
        )

    if (
        "transformer_sentiment" in df.columns
        and df["transformer_sentiment"].notna().any()
    ):
        df["primary_sentiment"] = df["transformer_sentiment"].fillna(
            df["vader_sentiment"]
        )
        df["primary_model"] = df["transformer_sentiment"].notna().map(
            {
                True: "Transformer",
                False: "VADER",
            }
        )
    else:
        df["primary_sentiment"] = df["vader_sentiment"]
        df["primary_model"] = "VADER"

    primary_column = "primary_sentiment"

    summary = sentiment_summary(
        df,
        sentiment_column=primary_column,
    )

    engagement = engagement_analysis(
        df,
        sentiment_column=primary_column,
    )

    statistics = comment_statistics(df)

    keywords = {
        "all": extract_keywords(
            df,
            None,
            20,
            primary_column,
        ),
        "positive": extract_keywords(
            df,
            "positive",
            20,
            primary_column,
        ),
        "neutral": extract_keywords(
            df,
            "neutral",
            20,
            primary_column,
        ),
        "negative": extract_keywords(
            df,
            "negative",
            20,
            primary_column,
        ),
    }

    tfidf = {
        "all": extract_tfidf(
            df,
            None,
            20,
            primary_column,
        ),
        "positive": extract_tfidf(
            df,
            "positive",
            20,
            primary_column,
        ),
        "neutral": extract_tfidf(
            df,
            "neutral",
            20,
            primary_column,
        ),
        "negative": extract_tfidf(
            df,
            "negative",
            20,
            primary_column,
        ),
    }

    return {
        "comments": df,
        "summary": summary,
        "statistics": statistics,
        "insights": generate_insights(
            df,
            summary,
            sentiment_column=primary_column,
        ),
        "keywords": keywords,
        "tfidf": tfidf,
        "engagement": engagement,
        "model_agreement": model_agreement(df),
        "primary_model": (
            "Transformer"
            if df["primary_model"].eq("Transformer").any()
            else "VADER"
        ),
    }
