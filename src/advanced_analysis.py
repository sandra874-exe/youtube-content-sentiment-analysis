import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# LOAD DATA
# ============================================================

input_path = "data/processed/youtube_comments_vader.csv"

df = pd.read_csv(input_path)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs("data/processed/graphs", exist_ok=True)


# ============================================================
# 1. OVERALL SENTIMENT DISTRIBUTION
# ============================================================

sentiment_counts = df["vader_sentiment"].value_counts()

print("\nOverall Sentiment:")
print(sentiment_counts)

plt.figure(figsize=(7, 5))

sentiment_counts.plot(
    kind="bar"
)

plt.title("Overall YouTube Comment Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Number of Comments")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "data/processed/graphs/overall_sentiment.png",
    dpi=300
)

plt.show()


# ============================================================
# 2. SENTIMENT BY DOMAIN
# ============================================================

domain_sentiment = pd.crosstab(
    df["domain"],
    df["vader_sentiment"]
)

print("\nSentiment by Domain:")
print(domain_sentiment)

domain_sentiment.plot(
    kind="bar",
    figsize=(9, 6)
)

plt.title("Sentiment Distribution by Domain")
plt.xlabel("Domain")
plt.ylabel("Number of Comments")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "data/processed/graphs/sentiment_by_domain.png",
    dpi=300
)

plt.show()


# ============================================================
# 3. SENTIMENT PERCENTAGE BY DOMAIN
# ============================================================

domain_percentage = pd.crosstab(
    df["domain"],
    df["vader_sentiment"],
    normalize="index"
) * 100

print("\nSentiment Percentage by Domain:")
print(domain_percentage.round(2))

domain_percentage.plot(
    kind="bar",
    stacked=True,
    figsize=(9, 6)
)

plt.title("Percentage of Sentiment by Domain")
plt.xlabel("Domain")
plt.ylabel("Percentage")
plt.xticks(rotation=0)

plt.legend(title="Sentiment")

plt.tight_layout()

plt.savefig(
    "data/processed/graphs/sentiment_percentage_by_domain.png",
    dpi=300
)

plt.show()


# ============================================================
# 4. SENTIMENT BY VIDEO
# ============================================================

video_sentiment = pd.crosstab(
    df["video_title"],
    df["vader_sentiment"]
)

print("\nSentiment by Video:")
print(video_sentiment)

video_sentiment.plot(
    kind="bar",
    stacked=True,
    figsize=(14, 7)
)

plt.title("Sentiment Distribution by Video")
plt.xlabel("Video")
plt.ylabel("Number of Comments")

plt.xticks(
    rotation=70,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    "data/processed/graphs/sentiment_by_video.png",
    dpi=300
)

plt.show()


# ============================================================
# 5. SENTIMENT BY CREATOR
# ============================================================

creator_sentiment = pd.crosstab(
    df["creator"],
    df["vader_sentiment"]
)

print("\nSentiment by Creator:")
print(creator_sentiment)

creator_sentiment.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Sentiment Distribution by Creator")
plt.xlabel("Creator")
plt.ylabel("Number of Comments")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "data/processed/graphs/sentiment_by_creator.png",
    dpi=300
)

plt.show()


# ============================================================
# 6. COMMENT LENGTH BY SENTIMENT
# ============================================================

if "word_count" in df.columns:

    plt.figure(figsize=(9, 6))

    sns.boxplot(
        data=df,
        x="vader_sentiment",
        y="word_count"
    )

    plt.title("Comment Length by Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Word Count")

    plt.tight_layout()

    plt.savefig(
        "data/processed/graphs/comment_length_sentiment.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 7. FIND ENGAGEMENT COLUMN
# ============================================================

possible_like_columns = [
    "comment_like_count",
    "comment_likes",
    "like_count",
    "likes"
]

like_column = None

for column in possible_like_columns:

    if column in df.columns:
        like_column = column
        break


# ============================================================
# 8. ENGAGEMENT VS SENTIMENT
# ============================================================

if like_column:

    df[like_column] = pd.to_numeric(
        df[like_column],
        errors="coerce"
    )

    engagement = (
        df.groupby("vader_sentiment")[like_column]
        .agg(["count", "mean", "median", "max"])
        .round(2)
    )

    print("\nEngagement by Sentiment:")
    print(engagement)

    engagement["mean"].plot(
        kind="bar",
        figsize=(8, 5)
    )

    plt.title("Average Comment Likes by Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Average Comment Likes")
    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        "data/processed/graphs/engagement_by_sentiment.png",
        dpi=300
    )

    plt.show()

else:

    print("\nNo comment-like column found.")


# ============================================================
# 9. DOMAIN + SENTIMENT + ENGAGEMENT
# ============================================================

if like_column:

    domain_engagement = (
        df.groupby(
            ["domain", "vader_sentiment"]
        )[like_column]
        .mean()
        .round(2)
    )

    print("\nAverage Engagement by Domain and Sentiment:")
    print(domain_engagement)


# ============================================================
# 10. SAVE SUMMARY TABLES
# ============================================================

domain_sentiment.to_csv(
    "data/processed/domain_sentiment_counts.csv"
)

domain_percentage.to_csv(
    "data/processed/domain_sentiment_percentage.csv"
)

video_sentiment.to_csv(
    "data/processed/video_sentiment_counts.csv"
)

creator_sentiment.to_csv(
    "data/processed/creator_sentiment_counts.csv"
)


print("\n========================================")
print("ADVANCED ANALYSIS COMPLETED")
print("========================================")