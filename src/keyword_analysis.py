import pandas as pd
import re
import os
from collections import Counter

# ============================================================
# LOAD DATA
# ============================================================

input_path = "data/processed/youtube_comments_vader.csv"

df = pd.read_csv(input_path)

print("\nDataset loaded:", df.shape)


# ============================================================
# STOPWORDS
# ============================================================

stopwords = {
    "the", "and", "a", "an", "to", "of", "in", "is", "it",
    "this", "that", "for", "on", "with", "was", "are", "i",
    "you", "my", "me", "your", "we", "they", "he", "she",
    "but", "or", "so", "if", "at", "as", "be", "have",
    "has", "had", "do", "does", "did", "not", "no", "very",
    "just", "from", "about", "what", "who", "how", "why",
    "can", "could", "would", "will", "all", "more", "really",
    "get", "got", "like", "one", "im", "i'm", "its", "it's",
    "than", "then", "there", "their", "them", "our", "out",
    "up", "down", "was", "were", "been", "also", "dont",
    "don't", "didnt", "didn't", "cant", "can't"
}


# ============================================================
# WORD EXTRACTION
# ============================================================

def extract_words(text):

    text = str(text).lower()

    words = re.findall(
        r"\b[a-zA-Z]{3,}\b",
        text
    )

    words = [
        word
        for word in words
        if word not in stopwords
    ]

    return words


# ============================================================
# TOP WORDS FUNCTION
# ============================================================

def get_top_words(data, n=20):

    counter = Counter()

    for text in data["clean_text"]:

        words = extract_words(text)

        counter.update(words)

    return counter.most_common(n)


# ============================================================
# OVERALL SENTIMENT KEYWORDS
# ============================================================

positive_df = df[
    df["vader_sentiment"] == "positive"
]

negative_df = df[
    df["vader_sentiment"] == "negative"
]

neutral_df = df[
    df["vader_sentiment"] == "neutral"
]


print("\n======================================")
print("TOP POSITIVE KEYWORDS")
print("======================================")

positive_words = get_top_words(
    positive_df,
    25
)

for word, count in positive_words:
    print(f"{word}: {count}")


print("\n======================================")
print("TOP NEGATIVE KEYWORDS")
print("======================================")

negative_words = get_top_words(
    negative_df,
    25
)

for word, count in negative_words:
    print(f"{word}: {count}")


print("\n======================================")
print("TOP NEUTRAL KEYWORDS")
print("======================================")

neutral_words = get_top_words(
    neutral_df,
    25
)

for word, count in neutral_words:
    print(f"{word}: {count}")


# ============================================================
# SAVE KEYWORD TABLES
# ============================================================

os.makedirs(
    "data/processed/keywords",
    exist_ok=True
)

pd.DataFrame(
    positive_words,
    columns=["keyword", "frequency"]
).to_csv(
    "data/processed/keywords/positive_keywords.csv",
    index=False
)

pd.DataFrame(
    negative_words,
    columns=["keyword", "frequency"]
).to_csv(
    "data/processed/keywords/negative_keywords.csv",
    index=False
)

pd.DataFrame(
    neutral_words,
    columns=["keyword", "frequency"]
).to_csv(
    "data/processed/keywords/neutral_keywords.csv",
    index=False
)


# ============================================================
# DOMAIN-SPECIFIC KEYWORDS
# ============================================================

domains = df["domain"].dropna().unique()

for domain in domains:

    print("\n======================================")
    print(f"DOMAIN: {domain}")
    print("======================================")

    domain_df = df[
        df["domain"] == domain
    ]

    for sentiment in [
        "positive",
        "negative",
        "neutral"
    ]:

        sentiment_df = domain_df[
            domain_df["vader_sentiment"] == sentiment
        ]

        words = get_top_words(
            sentiment_df,
            15
        )

        print(f"\n{sentiment.upper()}:")

        for word, count in words:

            print(
                f"{word}: {count}"
            )

        filename = (
            f"data/processed/keywords/"
            f"{domain}_{sentiment}_keywords.csv"
        )

        pd.DataFrame(
            words,
            columns=["keyword", "frequency"]
        ).to_csv(
            filename,
            index=False
        )


print("\n======================================")
print("KEYWORD ANALYSIS COMPLETED")
print("======================================")