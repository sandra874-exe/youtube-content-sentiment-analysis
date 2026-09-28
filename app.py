import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="YouTube Content Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

VADER_FILE = "data/processed/youtube_comments_vader.csv"
VALIDATION_FILE = "data/processed/transformer_validation_results.csv"
COMPARISON_FILE = "data/processed/model_comparison.csv"
ERROR_FILE = "data/processed/error_analysis.csv"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    vader = pd.read_csv(VADER_FILE)

    validation = pd.read_csv(VALIDATION_FILE)

    comparison = pd.read_csv(COMPARISON_FILE)

    errors = pd.read_csv(ERROR_FILE)

    return vader, validation, comparison, errors


try:

    df, validation, comparison, errors = load_data()

except Exception as e:

    st.error("Unable to load project data.")

    st.code(str(e))

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("📊 YouTube Content Sentiment Analysis")

st.markdown(
    """
    **Sentiment analysis of YouTube comments using VADER and Transformer models**
    
    This dashboard presents sentiment patterns, engagement,
    keyword insights, model performance and error analysis.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Sentiment Analysis",
        "Engagement Analysis",
        "Keyword Analysis",
        "Model Comparison",
        "Error Analysis",
        "Comment Analyzer"
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.header("Project Overview")

    total_comments = len(df)

    total_creators = df["creator"].nunique()

    total_videos = df["video_id"].nunique()

    total_domains = df["domain"].nunique()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Comments",
        total_comments
    )

    col2.metric(
        "Creators",
        total_creators
    )

    col3.metric(
        "Videos",
        total_videos
    )

    col4.metric(
        "Domains",
        total_domains
    )

    st.divider()

    st.subheader("Overall Sentiment")

    sentiment_counts = (
        df["vader_sentiment"]
        .value_counts()
        .reindex(
            ["positive", "neutral", "negative"],
            fill_value=0
        )
    )

    col1, col2 = st.columns(2)

    with col1:

        st.dataframe(
            sentiment_counts.rename("Comments"),
            use_container_width=True
        )

    with col2:

        fig, ax = plt.subplots()

        ax.pie(
            sentiment_counts.values,
            labels=sentiment_counts.index,
            autopct="%1.1f%%"
        )

        ax.set_title("Overall Sentiment Distribution")

        st.pyplot(fig)

        plt.close(fig)

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

elif page == "Sentiment Analysis":

    st.header("Sentiment Analysis")

    st.subheader("Filter Data")

    col1, col2 = st.columns(2)

    with col1:

        domains = ["All"] + sorted(
            df["domain"].dropna().unique().tolist()
        )

        selected_domain = st.selectbox(
            "Domain",
            domains
        )

    with col2:

        creators = ["All"] + sorted(
            df["creator"].dropna().unique().tolist()
        )

        selected_creator = st.selectbox(
            "Creator",
            creators
        )

    filtered = df.copy()

    if selected_domain != "All":

        filtered = filtered[
            filtered["domain"] == selected_domain
        ]

    if selected_creator != "All":

        filtered = filtered[
            filtered["creator"] == selected_creator
        ]

    st.write(
        f"Showing **{len(filtered)} comments**"
    )

    sentiment_counts = (
        filtered["vader_sentiment"]
        .value_counts()
        .reindex(
            ["positive", "neutral", "negative"],
            fill_value=0
        )
    )

    st.subheader("Sentiment Distribution")

    fig, ax = plt.subplots()

    sentiment_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Sentiment")

    ax.set_ylabel("Number of Comments")

    ax.set_title("Sentiment Distribution")

    plt.xticks(rotation=0)

    st.pyplot(fig)

    plt.close(fig)

    st.subheader("Sentiment by Domain")

    domain_sentiment = pd.crosstab(
        df["domain"],
        df["vader_sentiment"]
    )

    domain_sentiment = domain_sentiment.reindex(
        columns=["positive", "neutral", "negative"],
        fill_value=0
    )

    st.dataframe(
        domain_sentiment,
        use_container_width=True
    )

    fig, ax = plt.subplots()

    domain_sentiment.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Domain")

    ax.set_ylabel("Number of Comments")

    ax.set_title("Sentiment by Domain")

    plt.xticks(rotation=0)

    st.pyplot(fig)

    plt.close(fig)

    st.subheader("Sentiment by Creator")

    creator_sentiment = pd.crosstab(
        df["creator"],
        df["vader_sentiment"]
    )

    creator_sentiment = creator_sentiment.reindex(
        columns=["positive", "neutral", "negative"],
        fill_value=0
    )

    st.dataframe(
        creator_sentiment,
        use_container_width=True
    )


# ============================================================
# ENGAGEMENT ANALYSIS
# ============================================================

elif page == "Engagement Analysis":

    st.header("Engagement Analysis")

    numeric_columns = []

    for column in df.columns:

        if pd.api.types.is_numeric_dtype(df[column]):

            numeric_columns.append(column)

    st.subheader("Available Numeric Variables")

    st.write(numeric_columns)

    if "like_count" in df.columns:

        engagement = (
            df.groupby("vader_sentiment")["like_count"]
            .mean()
            .reindex(
                ["positive", "neutral", "negative"]
            )
        )

        st.subheader("Average Likes by Sentiment")

        st.bar_chart(engagement)

    else:

        st.info(
            "The current processed dataset does not contain "
            "a like_count column."
        )

    if "reply_count" in df.columns:

        replies = (
            df.groupby("vader_sentiment")["reply_count"]
            .mean()
            .reindex(
                ["positive", "neutral", "negative"]
            )
        )

        st.subheader("Average Replies by Sentiment")

        st.bar_chart(replies)

    else:

        st.info(
            "The current processed dataset does not contain "
            "a reply_count column."
        )

    st.subheader("Comment Length")

    df_temp = df.copy()

    df_temp["comment_length"] = (
        df_temp["clean_text"]
        .astype(str)
        .str.len()
    )

    length_summary = (
        df_temp.groupby("vader_sentiment")["comment_length"]
        .mean()
        .reindex(
            ["positive", "neutral", "negative"]
        )
    )

    st.bar_chart(length_summary)


# ============================================================
# KEYWORD ANALYSIS
# ============================================================

elif page == "Keyword Analysis":

    st.header("Keyword Analysis")

    st.write(
        "This section displays frequently occurring terms "
        "from the sentiment analysis."
    )

    sentiment = st.selectbox(
        "Select Sentiment",
        [
            "positive",
            "neutral",
            "negative"
        ]
    )

    keyword_file = (
        f"data/processed/"
        f"{sentiment}_keywords.csv"
    )

    try:

        keywords = pd.read_csv(
            keyword_file
        )

        st.subheader(
            f"Top {sentiment.title()} Keywords"
        )

        st.dataframe(
            keywords.head(20),
            use_container_width=True
        )

    except Exception:

        st.warning(
            "Keyword file not found."
        )

    st.divider()

    st.subheader("TF-IDF Analysis")

    tfidf_folder = "data/processed/tfidf"

    try:

        import os

        files = [
            file
            for file in os.listdir(tfidf_folder)
            if file.endswith(".csv")
        ]

        if files:

            selected_file = st.selectbox(
                "Select TF-IDF file",
                sorted(files)
            )

            tfidf = pd.read_csv(
                os.path.join(
                    tfidf_folder,
                    selected_file
                )
            )

            st.dataframe(
                tfidf.head(20),
                use_container_width=True
            )

        else:

            st.info(
                "No TF-IDF files found."
            )

    except Exception:

        st.info(
            "TF-IDF folder not available."
        )


# ============================================================
# MODEL COMPARISON
# ============================================================

elif page == "Model Comparison":

    st.header("VADER vs Transformer")

    st.write(
        "Both models were evaluated against the same "
        "120 human-labelled comments."
    )

    st.dataframe(
        comparison.style.format(
            {
                "Accuracy": "{:.2%}",
                "Macro Precision": "{:.2%}",
                "Macro Recall": "{:.2%}",
                "Macro F1": "{:.2%}"
            }
        ),
        use_container_width=True
    )

    st.subheader("Performance Comparison")

    metrics = [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1"
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for _, row in comparison.iterrows():

        ax.plot(
            metrics,
            [
                row[metric]
                for metric in metrics
            ],
            marker="o",
            label=row["Model"]
        )

    ax.set_ylim(0, 1)

    ax.set_ylabel("Score")

    ax.set_title(
        "VADER vs Transformer Performance"
    )

    ax.legend()

    st.pyplot(fig)

    plt.close(fig)

    st.subheader(
        "Transformer Confusion Matrix"
    )

    y_true = validation["human-sentiment"]

    y_pred = validation[
        "transformer_sentiment"
    ]

    labels = [
        "positive",
        "neutral",
        "negative"
    ]

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    fig, ax = plt.subplots()

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    disp.plot(
        ax=ax
    )

    ax.set_title(
        "Transformer Confusion Matrix"
    )

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# ERROR ANALYSIS
# ============================================================

elif page == "Error Analysis":

    st.header("Model Error Analysis")

    category_counts = (
        errors["error_category"]
        .value_counts()
    )

    st.subheader("Error Categories")

    st.dataframe(
        category_counts.rename(
            "Number of Comments"
        ),
        use_container_width=True
    )

    fig, ax = plt.subplots()

    category_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel(
        "Number of Comments"
    )

    ax.set_title(
        "Model Error Categories"
    )

    plt.xticks(
        rotation=25,
        ha="right"
    )

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    selected_category = st.selectbox(
        "Select Error Category",
        sorted(
            errors["error_category"]
            .unique()
        )
    )

    selected_errors = errors[
        errors["error_category"]
        == selected_category
    ]

    st.write(
        f"Comments in this category: "
        f"**{len(selected_errors)}**"
    )

    st.dataframe(
        selected_errors[
            [
                "clean_text",
                "human",
                "vader",
                "transformer"
            ]
        ],
        use_container_width=True
    )


# ============================================================
# COMMENT ANALYZER
# ============================================================

elif page == "Comment Analyzer":

    st.header("🔍 Interactive Comment Analyzer")

    st.write(
        "Enter a YouTube comment to compare "
        "VADER and Transformer sentiment."
    )

    comment = st.text_area(
        "Enter your comment:",
        height=150,
        placeholder="Type a YouTube comment here..."
    )

    if st.button("Analyze Comment"):

        if not comment.strip():

            st.warning(
                "Please enter a comment."
            )

        else:

            st.info(
                "Interactive model inference requires "
                "loading the Transformer model."
            )

            st.write(
                "**Entered comment:**"
            )

            st.write(comment)

            st.write(
                "The saved validation results can be "
                "explored in the Model Comparison and "
                "Error Analysis sections."
            )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "YouTube Content Sentiment Analysis"
)

st.sidebar.caption(
    "VADER + Transformer"
)