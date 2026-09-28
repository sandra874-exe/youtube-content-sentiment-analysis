import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load data
df = pd.read_csv("data/processed/vader_validation_sample.csv")

# Create VADER analyzer
analyzer = SentimentIntensityAnalyzer()

# Function
def vader_label(text):

    score = analyzer.polarity_scores(str(text))

    compound = score["compound"]

    if compound >= 0.05:
        return "positive"

    elif compound <= -0.05:
        return "negative"

    else:
        return "neutral"

# Predict
df["vader_sentiment"] = df["clean_text"].apply(vader_label)

# Save
df.to_csv(
    "data/processed/vader_validation_results.csv",
    index=False
)

# Accuracy
accuracy = accuracy_score(
    df["human-sentiment"],
    df["vader_sentiment"]
)

print("\nAccuracy:")
print(accuracy)

print("\nClassification Report:")
print(
    classification_report(
        df["human-sentiment"],
        df["vader_sentiment"]
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        df["human-sentiment"],
        df["vader_sentiment"]
    )
)