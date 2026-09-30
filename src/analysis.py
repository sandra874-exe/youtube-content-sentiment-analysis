import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


def sentiment_summary(df):
    result = (
        df["transformer_sentiment"]
        .value_counts(normalize=True)
        .mul(100)
        .rename("percentage")
        .reset_index()
    )
    result.columns = ["sentiment", "percentage"]
    return result


def model_agreement(df):
    comparisons = []
    pairs = [
        ("VADER", "vader_sentiment", "RoBERTa", "transformer_sentiment"),
    ]

    if "tfidf_sentiment" in df.columns:
        pairs.extend([
            ("VADER", "vader_sentiment", "TF-IDF", "tfidf_sentiment"),
            ("RoBERTa", "transformer_sentiment", "TF-IDF", "tfidf_sentiment"),
        ])

    for left_name, left_col, right_name, right_col in pairs:
        valid = df[[left_col, right_col]].dropna()
        if valid.empty:
            continue
        comparisons.append({
            "comparison": f"{left_name} vs {right_name}",
            "agreement_percent": float((valid[left_col] == valid[right_col]).mean() * 100),
        })

    return pd.DataFrame(comparisons)


def engagement_summary(df):
    if "like_count" not in df.columns:
        return pd.DataFrame()

    return (
        df.groupby("transformer_sentiment")
        .agg(
            comments=("comment", "count"),
            average_likes=("like_count", "mean"),
            median_likes=("like_count", "median"),
        )
        .reset_index()
    )


def topic_model(df, n_topics=5, top_words=8):
    if len(df) < max(30, n_topics * 5):
        return pd.DataFrame(), pd.DataFrame()

    work = df[["processed_text"]].copy()
    work = work[work["processed_text"].str.len() > 0]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=15000,
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(work["processed_text"])

    if matrix.shape[1] < n_topics:
        return pd.DataFrame(), pd.DataFrame()

    nmf = NMF(
        n_components=n_topics,
        init="nndsvda",
        random_state=42,
        max_iter=500,
    )

    document_topic_matrix = nmf.fit_transform(matrix)
    topic_indices = document_topic_matrix.argmax(axis=1)

    terms = vectorizer.get_feature_names_out()
    topic_rows = []

    for topic_index, component in enumerate(nmf.components_, start=1):
        top_indices = component.argsort()[::-1][:top_words]
        for rank, term_index in enumerate(top_indices, start=1):
            topic_rows.append({
                "topic": f"Topic {topic_index}",
                "rank": rank,
                "keyword": terms[term_index],
                "weight": float(component[term_index]),
            })

    assignments = work.copy()
    assignments["topic"] = [f"Topic {i + 1}" for i in topic_indices]

    if "transformer_sentiment" in df.columns:
        assignments = assignments.join(
            df[["transformer_sentiment"]].loc[assignments.index]
        )

    return pd.DataFrame(topic_rows), assignments
