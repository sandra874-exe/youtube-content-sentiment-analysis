import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# LOAD DATA
# ============================================================

input_path = "data/processed/youtube_comments_vader.csv"

df = pd.read_csv(input_path)

print("Dataset loaded:", df.shape)


# ============================================================
# OUTPUT FOLDER
# ============================================================

output_folder = "data/processed/tfidf"

os.makedirs(output_folder, exist_ok=True)


# ============================================================
# TF-IDF FUNCTION
# ============================================================

def get_tfidf_keywords(texts, top_n=20):

    texts = texts.fillna("").astype(str)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    matrix = vectorizer.fit_transform(texts)

    scores = matrix.mean(axis=0).A1

    terms = vectorizer.get_feature_names_out()

    result = pd.DataFrame({
        "term": terms,
        "tfidf_score": scores
    })

    result = result.sort_values(
        "tfidf_score",
        ascending=False
    )

    return result.head(top_n)


# ============================================================
# OVERALL POSITIVE / NEGATIVE TF-IDF
# ============================================================

for sentiment in ["positive", "negative"]:

    subset = df[
        df["vader_sentiment"] == sentiment
    ]

    print("\n======================================")
    print(f"TF-IDF: {sentiment.upper()}")
    print("======================================")

    keywords = get_tfidf_keywords(
        subset["clean_text"],
        top_n=20
    )

    print(keywords.to_string(index=False))

    keywords.to_csv(
        f"{output_folder}/{sentiment}_tfidf.csv",
        index=False
    )


# ============================================================
# DOMAIN + SENTIMENT TF-IDF
# ============================================================

domains = df["domain"].dropna().unique()

for domain in domains:

    for sentiment in [
        "positive",
        "negative"
    ]:

        subset = df[
            (df["domain"] == domain) &
            (df["vader_sentiment"] == sentiment)
        ]

        if len(subset) < 3:

            print(
                f"\nSkipping {domain} - {sentiment}: "
                "not enough comments."
            )

            continue

        keywords = get_tfidf_keywords(
            subset["clean_text"],
            top_n=15
        )

        print("\n======================================")
        print(f"{domain.upper()} - {sentiment.upper()}")
        print("======================================")

        print(
            keywords.to_string(index=False)
        )

        safe_domain = (
            str(domain)
            .replace(" ", "_")
            .replace("/", "_")
        )

        filename = (
            f"{output_folder}/"
            f"{safe_domain}_{sentiment}_tfidf.csv"
        )

        keywords.to_csv(
            filename,
            index=False
        )


# ============================================================
# VISUALIZATION FUNCTION
# ============================================================

def create_bar_chart(
    csv_path,
    title,
    output_path
):

    data = pd.read_csv(csv_path)

    data = data.sort_values(
        "tfidf_score",
        ascending=True
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        data["term"],
        data["tfidf_score"]
    )

    plt.title(title)
    plt.xlabel("Average TF-IDF Score")
    plt.ylabel("Term")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.show()


# ============================================================
# OVERALL GRAPHS
# ============================================================

create_bar_chart(
    f"{output_folder}/positive_tfidf.csv",
    "Top TF-IDF Terms in Positive Comments",
    f"{output_folder}/positive_tfidf.png"
)

create_bar_chart(
    f"{output_folder}/negative_tfidf.csv",
    "Top TF-IDF Terms in Negative Comments",
    f"{output_folder}/negative_tfidf.png"
)


# ============================================================
# DOMAIN GRAPHS
# ============================================================

for domain in domains:

    safe_domain = (
        str(domain)
        .replace(" ", "_")
        .replace("/", "_")
    )

    for sentiment in ["positive", "negative"]:

        csv_path = (
            f"{output_folder}/"
            f"{safe_domain}_{sentiment}_tfidf.csv"
        )

        if not os.path.exists(csv_path):
            continue

        graph_path = (
            f"{output_folder}/"
            f"{safe_domain}_{sentiment}_tfidf.png"
        )

        create_bar_chart(
            csv_path,
            f"{domain} - {sentiment.title()} TF-IDF Terms",
            graph_path
        )


print("\n======================================")
print("TF-IDF ANALYSIS COMPLETED")
print("======================================")