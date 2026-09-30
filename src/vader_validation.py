from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


INPUT_FILE = Path("data/processed/vader_validation_sample.csv")
OUTPUT_FILE = Path("data/processed/vader_validation_results.csv")
CONFUSION_MATRIX_FILE = Path(
    "data/processed/vader_confusion_matrix.png"
)

LABELS = ["positive", "neutral", "negative"]


def vader_label(text: str) -> str:
    analyzer = SentimentIntensityAnalyzer()
    compound = analyzer.polarity_scores(str(text))["compound"]

    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def main():
    print("Loading human-labelled validation data...")

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "comment_id",
        "clean_text",
        "human_sentiment",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df["comment_id"].duplicated().any():
        raise ValueError("Duplicate comment_id values found.")

    invalid_labels = set(df["human_sentiment"].dropna()) - set(LABELS)

    if invalid_labels:
        raise ValueError(
            f"Invalid human sentiment labels: {sorted(invalid_labels)}"
        )

    print(f"Validation comments: {len(df)}")
    print("\nHuman label distribution:")
    print(df["human_sentiment"].value_counts())

    analyzer = SentimentIntensityAnalyzer()

    scores = df["clean_text"].fillna("").astype(str).apply(
        analyzer.polarity_scores
    )

    df["vader_neg"] = scores.apply(lambda x: x["neg"])
    df["vader_neu"] = scores.apply(lambda x: x["neu"])
    df["vader_pos"] = scores.apply(lambda x: x["pos"])
    df["vader_compound"] = scores.apply(lambda x: x["compound"])

    def classify(compound):
        if compound >= 0.05:
            return "positive"
        if compound <= -0.05:
            return "negative"
        return "neutral"

    df["vader_sentiment"] = df["vader_compound"].apply(classify)

    # Keep the human labels unchanged.
    output_columns = [
        "comment_id",
        "domain",
        "creator",
        "video_id",
        "video_title",
        "clean_text",
        "human_sentiment",
        "vader_neg",
        "vader_neu",
        "vader_pos",
        "vader_compound",
        "vader_sentiment",
    ]

    df[output_columns].to_csv(
        OUTPUT_FILE,
        index=False,
    )

    y_true = df["human_sentiment"]
    y_pred = df["vader_sentiment"]

    accuracy = accuracy_score(y_true, y_pred)

    print("\n==============================")
    print("VADER VALIDATION")
    print("==============================")

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            zero_division=0,
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=LABELS,
        )
    )

    macro_report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    print(
        f"Macro Precision: "
        f"{macro_report['macro avg']['precision']:.4f}"
    )
    print(
        f"Macro Recall: "
        f"{macro_report['macro avg']['recall']:.4f}"
    )
    print(
        f"Macro F1: "
        f"{macro_report['macro avg']['f1-score']:.4f}"
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS,
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=LABELS,
    )

    disp.plot(
        ax=ax,
        values_format="d",
        colorbar=False,
    )

    ax.set_title("VADER Sentiment Confusion Matrix")

    plt.tight_layout()
    plt.savefig(
        CONFUSION_MATRIX_FILE,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"\nSaved validation results to: {OUTPUT_FILE}")
    print(
        f"Saved confusion matrix to: "
        f"{CONFUSION_MATRIX_FILE}"
    )


if __name__ == "__main__":
    main()
