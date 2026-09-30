from pathlib import Path

import pandas as pd


VADER_FILE = Path(
    "data/processed/vader_validation_results.csv"
)

TRANSFORMER_FILE = Path(
    "data/processed/transformer_validation_results.csv"
)

OUTPUT_FILE = Path(
    "data/processed/error_analysis.csv"
)


def main():
    print("Loading validation results...")

    vader = pd.read_csv(VADER_FILE)
    transformer = pd.read_csv(TRANSFORMER_FILE)

    required_vader = {
        "comment_id",
        "human_sentiment",
        "clean_text",
        "vader_sentiment",
    }

    required_transformer = {
        "comment_id",
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
            "domain",
            "creator",
            "video_id",
            "video_title",
            "clean_text",
            "human_sentiment",
            "vader_sentiment",
        ]
    ].merge(
        transformer[
            [
                "comment_id",
                "transformer_sentiment",
            ]
        ],
        on="comment_id",
        how="inner",
        validate="one_to_one",
    )

    print(
        f"Matched comments: {len(comparison)}"
    )

    if len(comparison) != len(vader) or len(comparison) != len(
        transformer
    ):
        raise ValueError(
            "The VADER and Transformer files do not contain "
            "the same set of comment IDs."
        )

    comparison["vader_correct"] = (
        comparison["vader_sentiment"]
        == comparison["human_sentiment"]
    )

    comparison["transformer_correct"] = (
        comparison["transformer_sentiment"]
        == comparison["human_sentiment"]
    )

    def classify_error(row):
        if row["vader_correct"] and row["transformer_correct"]:
            return "Both Correct"

        if row["vader_correct"] and not row["transformer_correct"]:
            return "VADER Correct / Transformer Wrong"

        if not row["vader_correct"] and row["transformer_correct"]:
            return "VADER Wrong / Transformer Correct"

        return "Both Wrong"

    comparison["error_category"] = comparison.apply(
        classify_error,
        axis=1,
    )

    output_columns = [
        "comment_id",
        "domain",
        "creator",
        "video_id",
        "video_title",
        "clean_text",
        "human_sentiment",
        "vader_sentiment",
        "transformer_sentiment",
        "vader_correct",
        "transformer_correct",
        "error_category",
    ]

    comparison[output_columns].to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n==============================")
    print("ERROR ANALYSIS")
    print("==============================")

    print("\nError category counts:")
    print(
        comparison["error_category"]
        .value_counts()
        .to_string()
    )

    print("\nError category percentages:")
    print(
        (
            comparison["error_category"]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
            .astype(str)
            + "%"
        ).to_string()
    )

    print("\nErrors by domain:")
    domain_summary = pd.crosstab(
        comparison["domain"],
        comparison["error_category"],
    )

    print(domain_summary.to_string())

    print(
        f"\nSaved error analysis to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
