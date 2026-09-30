from pathlib import Path

import pandas as pd

from .config import COMMENTS_FILE, FINAL_FILE, PROCESSED_DIR
from .preprocessing import prepare_dataframe
from .sentiment_models import load_transformer, vader_predict, transformer_predict


def run_sentiment_analysis(classifier=None):
    if not COMMENTS_FILE.exists():
        raise FileNotFoundError(
            "youtube_comments.csv was not found. Run Creator Setup first."
        )

    df = pd.read_csv(COMMENTS_FILE)
    df = prepare_dataframe(df, text_candidates=["comment", "clean_text", "processed_text"])

    # VADER and RoBERTa use sentiment_text, preserving punctuation and emojis.
    vader_labels, vader_scores = vader_predict(df["sentiment_text"])
    df["vader_sentiment"] = vader_labels
    df["vader_score"] = vader_scores

    classifier = classifier or load_transformer()
    transformer_labels, transformer_scores = transformer_predict(
        df["sentiment_text"],
        classifier=classifier,
    )
    df["transformer_sentiment"] = transformer_labels
    df["transformer_score"] = transformer_scores

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FINAL_FILE, index=False)
    return df
