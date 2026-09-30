from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


INPUT_FILE = Path("data/processed/vader_validation_sample.csv")
OUTPUT_FILE = Path("data/processed/transformer_validation_results.csv")
CONFUSION_MATRIX_FILE = Path(
    "data/processed/transformer_confusion_matrix.png"
)

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

LABELS = ["positive", "neutral", "negative"]


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

    print(f"\nLoading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    print(f"Using device: {device}")

    texts = df["clean_text"].fillna("").astype(str).tolist()

    predictions = []
    prediction_scores = []
    negative_scores = []
    neutral_scores = []
    positive_scores = []

    batch_size = 16

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.no_grad():
            outputs = model(**encoded)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )

        predicted_ids = probabilities.argmax(dim=-1)

        for probs, predicted_id in zip(
            probabilities,
            predicted_ids,
        ):
            predicted_id = int(predicted_id.item())

            raw_label = model.config.id2label[predicted_id]
            label = raw_label.lower()

            predictions.append(label)
            prediction_scores.append(
                float(probs[predicted_id].item())
            )

            negative_scores.append(
                float(probs[0].item())
            )
            neutral_scores.append(
                float(probs[1].item())
            )
            positive_scores.append(
                float(probs[2].item())
            )

        print(
            f"Processed {min(start + batch_size, len(texts))}"
            f"/{len(texts)}"
        )

    df["transformer_neg"] = negative_scores
    df["transformer_neu"] = neutral_scores
    df["transformer_pos"] = positive_scores
    df["transformer_score"] = prediction_scores
    df["transformer_sentiment"] = predictions

    output_columns = [
        "comment_id",
        "domain",
        "creator",
        "video_id",
        "video_title",
        "clean_text",
        "human_sentiment",
        "transformer_neg",
        "transformer_neu",
        "transformer_pos",
        "transformer_score",
        "transformer_sentiment",
    ]

    df[output_columns].to_csv(
        OUTPUT_FILE,
        index=False,
    )

    y_true = df["human_sentiment"]
    y_pred = df["transformer_sentiment"]

    accuracy = accuracy_score(y_true, y_pred)

    print("\n==============================")
    print("TRANSFORMER VALIDATION")
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

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    print(
        f"Macro Precision: "
        f"{report['macro avg']['precision']:.4f}"
    )
    print(
        f"Macro Recall: "
        f"{report['macro avg']['recall']:.4f}"
    )
    print(
        f"Macro F1: "
        f"{report['macro avg']['f1-score']:.4f}"
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

    ax.set_title(
        "Transformer Sentiment Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"\nSaved validation results to: {OUTPUT_FILE}"
    )
    print(
        f"Saved confusion matrix to: "
        f"{CONFUSION_MATRIX_FILE}"
    )


if __name__ == "__main__":
    main()
