import os
import pandas as pd
import matplotlib.pyplot as plt

from transformers import pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

INPUT_FILE = "data/processed/vader_validation_results.csv"
OUTPUT_FILE = "data/processed/transformer_validation_results.csv"
CONFUSION_MATRIX_FILE = "data/processed/transformer_confusion_matrix.png"

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

print("Loading validation data...")

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)
print("Columns:", list(df.columns))

text_column = "clean_text"
human_column = "human-sentiment"

# Remove missing values
df = df.dropna(subset=[text_column, human_column]).copy()

# Keep only valid human labels
valid_labels = ["positive", "neutral", "negative"]
df = df[df[human_column].isin(valid_labels)].copy()

print("Comments used for evaluation:", len(df))

print("\nLoading Transformer model...")
print("Model:", MODEL_NAME)

classifier = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    truncation=True,
    max_length=512
)

print("\nRunning Transformer sentiment analysis...")

results = []

texts = df[text_column].astype(str).tolist()

# Process in batches
for start in range(0, len(texts), 16):
    batch = texts[start:start + 16]

    predictions = classifier(
        batch,
        batch_size=16
    )

    results.extend(predictions)

    print(
        f"Processed {min(start + 16, len(texts))}/{len(texts)} comments"
    )

# Convert model labels to standard labels
def normalize_label(label):
    label = str(label).lower()

    if "negative" in label:
        return "negative"

    if "neutral" in label:
        return "neutral"

    if "positive" in label:
        return "positive"

    if label in ["label_0", "0"]:
        return "negative"

    if label in ["label_1", "1"]:
        return "neutral"

    if label in ["label_2", "2"]:
        return "positive"

    return label


df["transformer_sentiment"] = [
    normalize_label(result["label"])
    for result in results
]

df["transformer_score"] = [
    result["score"]
    for result in results
]

# Save predictions
df.to_csv(OUTPUT_FILE, index=False)

print("\nTransformer predictions saved to:")
print(OUTPUT_FILE)

# Evaluation
y_true = df[human_column]
y_pred = df["transformer_sentiment"]

accuracy = accuracy_score(y_true, y_pred)

print("\n==============================")
print("TRANSFORMER EVALUATION")
print("==============================")

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=valid_labels,
        zero_division=0
    )
)

# Confusion matrix
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=valid_labels
)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=valid_labels
)

disp.plot()

plt.title("Transformer Sentiment Confusion Matrix")
plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nConfusion matrix saved to:")
print(CONFUSION_MATRIX_FILE)

# Prediction counts
print("\nTransformer sentiment counts:")
print(df["transformer_sentiment"].value_counts())

print("\nHuman sentiment counts:")
print(df[human_column].value_counts())

print("\nFirst 10 predictions:")
print(
    df[
        [
            "clean_text",
            "human-sentiment",
            "transformer_sentiment",
            "transformer_score"
        ]
    ].head(10).to_string(index=False)
)

print("\nTransformer analysis completed successfully.")