import pandas as pd

INPUT_FILE = "data/processed/transformer_validation_results.csv"
OUTPUT_FILE = "data/processed/error_analysis.csv"

print("Loading validation results...")

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)

# Ground truth
df["human"] = df["human-sentiment"]

# Predictions
df["vader"] = df["vader_sentiment"]
df["transformer"] = df["transformer_sentiment"]

# Determine correctness
df["vader_correct"] = df["vader"] == df["human"]
df["transformer_correct"] = df["transformer"] == df["human"]

# Categorize each comment
def classify(row):

    if row["vader_correct"] and row["transformer_correct"]:
        return "Both Correct"

    elif row["vader_correct"] and not row["transformer_correct"]:
        return "VADER Correct / Transformer Wrong"

    elif not row["vader_correct"] and row["transformer_correct"]:
        return "VADER Wrong / Transformer Correct"

    else:
        return "Both Wrong"


df["error_category"] = df.apply(classify, axis=1)

# Save complete analysis
columns = [
    "i",
    "domain",
    "creator",
    "video_id",
    "video_title",
    "clean_text",
    "human",
    "vader",
    "transformer",
    "vader_correct",
    "transformer_correct",
    "error_category"
]

df[columns].to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n==============================")
print("ERROR ANALYSIS")
print("==============================")

print("\nCategory counts:")

counts = df["error_category"].value_counts()

print(counts)

print("\nCategory percentages:")

percentages = (
    df["error_category"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print(percentages)

print("\n==============================")
print("VADER CORRECT / TRANSFORMER WRONG")
print("==============================")

vader_only = df[
    (df["vader_correct"] == True) &
    (df["transformer_correct"] == False)
]

print(
    vader_only[
        [
            "clean_text",
            "human",
            "vader",
            "transformer"
        ]
    ].to_string(index=False)
)

print("\n==============================")
print("VADER WRONG / TRANSFORMER CORRECT")
print("==============================")

transformer_only = df[
    (df["vader_correct"] == False) &
    (df["transformer_correct"] == True)
]

print(
    transformer_only[
        [
            "clean_text",
            "human",
            "vader",
            "transformer"
        ]
    ].to_string(index=False)
)

print("\n==============================")
print("BOTH WRONG")
print("==============================")

both_wrong = df[
    (df["vader_correct"] == False) &
    (df["transformer_correct"] == False)
]

print(
    both_wrong[
        [
            "clean_text",
            "human",
            "vader",
            "transformer"
        ]
    ].to_string(index=False)
)

print("\nError analysis saved to:")
print(OUTPUT_FILE)

print("\nError analysis completed successfully.")