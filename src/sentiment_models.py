from functools import lru_cache
from pathlib import Path

import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import (
    MODEL_DIR,
    TRANSFORMER_MODEL,
    VADER_POSITIVE_THRESHOLD,
    VADER_NEGATIVE_THRESHOLD,
)


# ============================================================
# VADER
# ============================================================

def vader_predict(texts):
    analyzer = SentimentIntensityAnalyzer()
    labels = []
    scores = []

    for text in texts:
        compound = analyzer.polarity_scores(str(text))["compound"]

        if compound >= VADER_POSITIVE_THRESHOLD:
            label = "positive"
        elif compound <= VADER_NEGATIVE_THRESHOLD:
            label = "negative"
        else:
            label = "neutral"

        labels.append(label)
        scores.append(float(compound))

    return labels, scores


# ============================================================
# TRANSFORMER
# ============================================================

@lru_cache(maxsize=1)
def load_transformer():
    from transformers import pipeline

    return pipeline(
        "sentiment-analysis",
        model=TRANSFORMER_MODEL,
        truncation=True,
        max_length=512,
    )


def transformer_predict(texts, classifier=None):
    classifier = classifier or load_transformer()

    labels = []
    scores = []

    label_map = {
        "label_0": "negative",
        "label_1": "neutral",
        "label_2": "positive",
        "negative": "negative",
        "neutral": "neutral",
        "positive": "positive",
    }

    # Hugging Face pipelines can process a list in batches.
    results = classifier(
        [str(text) for text in texts],
        batch_size=16,
    )

    for result in results:
        raw_label = str(result["label"]).lower()
        labels.append(label_map.get(raw_label, raw_label))
        scores.append(float(result["score"]))

    return labels, scores


# ============================================================
# TF-IDF
# ============================================================

def build_tfidf_vectorizer():
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        max_features=10000,
    )


def build_logistic_model():
    return Pipeline([
        ("tfidf", build_tfidf_vectorizer()),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])


def build_svm_model():
    return Pipeline([
        ("tfidf", build_tfidf_vectorizer()),
        (
            "classifier",
            LinearSVC(
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])


def save_model(model, filename):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / filename)


def load_saved_model(filename):
    path = MODEL_DIR / filename
    if not path.exists():
        return None
    return joblib.load(path)
