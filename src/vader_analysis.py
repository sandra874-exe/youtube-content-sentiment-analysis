import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


analyzer = SentimentIntensityAnalyzer()


def get_vader_sentiment(text):

    scores = analyzer.polarity_scores(str(text))
    compound = scores["compound"]

    if compound >= 0.05:
        return "positive"

    elif compound <= -0.05:
        return "negative"

    else:
        return "neutral"


# =========================================================
# 1. LOAD HUMAN-LABELLED VALIDATION DATA
# =========================================================

validation_path = "data/processed/vader_validation_sample.csv"

validation_df = pd.read_csv(validation_path)

print("\nValidation dataset loaded successfully.")
print("Rows:", len(validation_df))

print("\nColumns:")
print(validation_df.columns.tolist())


# =========================================================
# 2. RUN VADER ON 120 HUMAN-LABELLED COMMENTS
# =========================================================

validation_df["vader_sentiment"] = validation_df[
    "clean_text"
].apply(get_vader_sentiment)


# =========================================================
# 3. NORMALIZE HUMAN LABELS
# =========================================================

human = validation_df[
    "human-sentiment"
].astype(str).str.lower().str.strip()

vader = validation_df[
    "vader_sentiment"
].str.lower().str.strip()


# =========================================================
# 4. SAVE VALIDATION RESULTS
# =========================================================

validation_df.to_csv(
    "data/processed/vader_validation_results.csv",
    index=False
)


# =========================================================
# 5. CALCULATE METRICS
# =========================================================

accuracy = accuracy_score(
    human,
    vader
)

precision = precision_score(
    human,
    vader,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    human,
    vader,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    human,
    vader,
    average="weighted",
    zero_division=0
)


print("\n======================================")
print("VADER VALIDATION RESULTS")
print("======================================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# =========================================================
# 6. CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:")

print(
    classification_report(
        human,
        vader,
        labels=["negative", "neutral", "positive"],
        zero_division=0
    )
)


# =========================================================
# 7. CONFUSION MATRIX
# =========================================================

labels = [
    "negative",
    "neutral",
    "positive"
]

cm = confusion_matrix(
    human,
    vader,
    labels=labels
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel("VADER Prediction")
plt.ylabel("Human Label")

plt.title(
    "VADER vs Human Sentiment"
)

plt.tight_layout()

plt.savefig(
    "data/processed/vader_confusion_matrix.png",
    dpi=300
)

plt.show()


# =========================================================
# 8. FIND VADER DISAGREEMENTS
# =========================================================

errors = validation_df[
    validation_df["human-sentiment"]
    .astype(str)
    .str.lower()
    .str.strip()
    != validation_df["vader_sentiment"]
]

errors.to_csv(
    "data/processed/vader_errors.csv",
    index=False
)

print(
    "\nTotal disagreements:",
    len(errors)
)


# =========================================================
# 9. APPLY VADER TO ALL 750 COMMENTS
# =========================================================

full_path = (
    "data/processed/"
    "youtube_comments_clean.csv"
)

full_df = pd.read_csv(full_path)

print(
    "\nFull dataset rows:",
    len(full_df)
)

full_df["vader_sentiment"] = full_df[
    "clean_text"
].apply(get_vader_sentiment)


# =========================================================
# 10. SAVE FULL VADER DATASET
# =========================================================

full_df.to_csv(
    "data/processed/"
    "youtube_comments_vader.csv",
    index=False
)


# =========================================================
# 11. FULL DATASET SENTIMENT DISTRIBUTION
# =========================================================

print("\nFull Dataset Sentiment Distribution:")

print(
    full_df["vader_sentiment"]
    .value_counts()
)

print(
    "\nVADER analysis completed successfully."
)

print(
    "\nSaved:"
)

print(
    "data/processed/vader_validation_results.csv"
)

print(
    "data/processed/vader_errors.csv"
)

print(
    "data/processed/vader_confusion_matrix.png"
)

print(
    "data/processed/youtube_comments_vader.csv"
)