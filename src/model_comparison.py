import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

VADER_FILE = "data/processed/vader_validation_results.csv"
TRANSFORMER_FILE = "data/processed/transformer_validation_results.csv"

OUTPUT_FILE = "data/processed/model_comparison.csv"
CHART_FILE = "data/processed/model_comparison.png"

print("Loading validation results...")

vader = pd.read_csv(VADER_FILE)
transformer = pd.read_csv(TRANSFORMER_FILE)

print("VADER shape:", vader.shape)
print("Transformer shape:", transformer.shape)

# Use the human labels as ground truth
y_true = vader["human-sentiment"]

# VADER predictions
y_vader = vader["vader_sentiment"]

# Transformer predictions
y_transformer = transformer["transformer_sentiment"]

models = {
    "VADER": y_vader,
    "Transformer": y_transformer
}

comparison = []

for model_name, predictions in models.items():

    accuracy = accuracy_score(y_true, predictions)

    precision = precision_score(
        y_true,
        predictions,
        labels=["positive", "neutral", "negative"],
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        labels=["positive", "neutral", "negative"],
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        labels=["positive", "neutral", "negative"],
        average="macro",
        zero_division=0
    )

    comparison.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": f1
    })

comparison_df = pd.DataFrame(comparison)

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(
    comparison_df.to_string(index=False)
)

comparison_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nComparison saved to:")
print(OUTPUT_FILE)

# Plot comparison
metrics = [
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1"
]

x = range(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    [i - width / 2 for i in x],
    comparison_df.loc[0, metrics],
    width,
    label="VADER"
)

plt.bar(
    [i + width / 2 for i in x],
    comparison_df.loc[1, metrics],
    width,
    label="Transformer"
)

plt.xticks(x, metrics)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("VADER vs Transformer Sentiment Performance")
plt.legend()

plt.tight_layout()

plt.savefig(
    CHART_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nComparison chart saved to:")
print(CHART_FILE)

print("\nModel comparison completed successfully.")