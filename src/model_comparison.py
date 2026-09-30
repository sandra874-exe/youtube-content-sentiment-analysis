from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


VADER_FILE = Path(
    "data/processed/vader_validation_results.csv"
)

TRANSFORMER_FILE = Path(
    "data/processed/transformer_validation_results.csv"
)

OUTPUT_FILE = Path(
    "data/processed/model_comparison.csv"
)

LABELS = ["positive", "neutral", "negative"]


def calculate_metrics(y_true, y_pred):
    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro Precision": report["macro avg"]["precision"],
        "Macro Recall": report["macro avg"]["recall"],
        "Macro F1": report["macro avg"]["f1-score"],
    }


def main():
    print("Loading validation results...")

    vader = pd.read_csv(VADER_FILE)
    transformer = pd.read_csv(TRANSFORMER_FILE)

    required_vader = {
        "comment_id",
        "human_sentiment",
        "vader_sentiment",
    }

    required_transformer = {
        "comment_id",
        "human_sentiment",
        "transformer_sentiment",
    }

    missing_vader = required_vader - set(vader.columns)
    missing_transformer = required_transformer - set(
        transformer.columns
    )

    if missing_vader:
        raise ValueError(
            f"VADER file is missing: {sorted(missing_vader)}"
        )

    if missing_transformer:
        raise ValueError(
            "Transformer file is missing: "
            f"{sorted(missing_transformer)}"
        )

    if vader["comment_id"].duplicated().any():
        raise ValueError(
            "Duplicate comment_id values found in VADER results."
        )

    if transformer["comment_id"].duplicated().any():
        raise ValueError(
            "Duplicate comment_id values found in "
            "Transformer results."
        )

    comparison = vader[
        [
            "comment_id",
            "human_sentiment",
            "vader_sentiment",
        ]
    ].merge(
        transformer[
            [
                "comment_id",
                "human_sentiment",
                "transformer_sentiment",
            ]
        ],
        on="comment_id",
        how="inner",
        suffixes=("_vader", "_transformer"),
        validate="one_to_one",
    )

    print(
        f"Matched comments: {len(comparison)}"
    )

    if len(comparison) != len(vader) or len(comparison) != len(
        transformer
    ):
        raise ValueError(
            "VADER and Transformer results do not contain "
            "the same set of comment IDs."
        )

    if (
        comparison["human_sentiment_vader"]
        != comparison["human_sentiment_transformer"]
    ).any():
        raise ValueError(
            "Human sentiment labels differ between "
            "the two validation files."
        )

    y_true = comparison["human_sentiment_vader"]

    vader_metrics = calculate_metrics(
        y_true,
        comparison["vader_sentiment"],
    )

    transformer_metrics = calculate_metrics(
        y_true,
        comparison["transformer_sentiment"],
    )

    results = pd.DataFrame(
        [
            {
                "Model": "VADER",
                **vader_metrics,
            },
            {
                "Model": "Transformer",
                **transformer_metrics,
            },
        ]
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n==============================")
    print("MODEL COMPARISON")
    print("==============================")

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nVADER Confusion Matrix:")
    print(
        confusion_matrix(
            y_true,
            comparison["vader_sentiment"],
            labels=LABELS,
        )
    )

    print("\nTransformer Confusion Matrix:")
    print(
        confusion_matrix(
            y_true,
            comparison["transformer_sentiment"],
            labels=LABELS,
        )
    )

    agreement = (
        comparison["vader_sentiment"]
        == comparison["transformer_sentiment"]
    ).mean()

    print(
        f"\nVADER/Transformer agreement: "
        f"{agreement:.4f}"
    )

    print(
        f"\nSaved comparison to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
