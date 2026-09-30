from pathlib import Path
import math

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import confusion_matrix

from src.youtube_api import (
    identify_youtube_url,
    get_video_info,
    get_channel_info,
    get_channel_videos,
    get_video_comments,
)

from src.sentiment_analyzer import (
    run_complete_analysis,
    load_transformer,
    get_vader_sentiment,
)

from src.sarcasm_analyzer import (
    load_sarcasm_detector,
    detect_sarcasm,
    sarcastic_comment_examples,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="YouTube Pulse",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background: #f8fafc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    [data-testid="stSidebar"] {
        background: #111827;
    }

    [data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }

    .hero {
        padding: 35px;
        border-radius: 24px;
        background:
            linear-gradient(
                135deg,
                #111827 0%,
                #312e81 55%,
                #7c3aed 100%
            );
        color: white;
        margin-bottom: 25px;
        box-shadow:
            0 15px 40px rgba(15, 23, 42, 0.20);
    }

    .hero h1 {
        font-size: 46px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .hero p {
        font-size: 18px;
        opacity: 0.9;
    }

    .metric-card {
        background: white;
        padding: 22px;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        box-shadow:
            0 6px 20px rgba(15, 23, 42, 0.06);
        min-height: 125px;
    }

    .metric-label {
        color: #64748b;
        font-size: 14px;
        font-weight: 600;
    }

    .metric-value {
        color: #111827;
        font-size: 30px;
        font-weight: 800;
        margin-top: 8px;
    }

    .section-title {
        font-size: 26px;
        font-weight: 800;
        color: #111827;
        margin-top: 20px;
        margin-bottom: 12px;
    }

    .insight-card {
        background: white;
        padding: 18px 20px;
        border-radius: 16px;
        border-left: 5px solid #7c3aed;
        margin-bottom: 12px;
        box-shadow:
            0 5px 16px rgba(15, 23, 42, 0.06);
    }

    .video-card {
        background: white;
        padding: 18px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        margin-bottom: 12px;
        box-shadow:
            0 5px 15px rgba(15, 23, 42, 0.05);
    }

    .small-muted {
        color: #64748b;
        font-size: 13px;
    }

    .badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 20px;
        background: #ede9fe;
        color: #6d28d9;
        font-size: 12px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# COLORS
# ============================================================

SENTIMENT_COLORS = {
    "positive": "#22c55e",
    "neutral": "#94a3b8",
    "negative": "#ef4444",
}


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "content_info" not in st.session_state:
    st.session_state.content_info = None

if "content_type" not in st.session_state:
    st.session_state.content_type = None

if "channel_videos" not in st.session_state:
    st.session_state.channel_videos = []


# ============================================================
# HELPERS
# ============================================================

def format_number(value):

    if value is None:
        return "0"

    try:
        value = float(value)
    except:
        return str(value)

    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"

    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"

    if value >= 1_000:
        return f"{value / 1_000:.1f}K"

    return f"{int(value):,}"


def metric_card(label, value):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                {label}
            </div>
            <div class="metric-value">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title):

    st.markdown(
        f"""
        <div class="section-title">
            {title}
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_percentage(summary, sentiment):

    if summary is None or summary.empty:
        return 0

    row = summary[
        summary["sentiment"] == sentiment
    ]

    if row.empty:
        return 0

    return float(
        row.iloc[0]["percentage"]
    )


def sentiment_donut(summary, title="Audience Mood"):

    if summary.empty:
        st.info("No sentiment data available.")
        return

    fig = go.Figure(
        data=[
            go.Pie(
                labels=summary["sentiment"].str.title(),
                values=summary["count"],
                hole=0.62,
                marker=dict(
                    colors=[
                        SENTIMENT_COLORS.get(
                            x,
                            "#64748b",
                        )
                        for x in summary["sentiment"]
                    ]
                ),
                textinfo="percent",
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Comments: %{value}<br>"
                    "Share: %{percent}<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        title=title,
        height=420,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
        showlegend=True,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


def sentiment_bar(summary):

    if summary.empty:
        return

    fig = px.bar(
        summary,
        x="sentiment",
        y="percentage",
        text="percentage",
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        title="Sentiment Distribution",
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )

    fig.update_layout(
        yaxis_title="Percentage",
        xaxis_title="Sentiment",
        showlegend=False,
        height=400,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# DEMO DATA
# ============================================================

@st.cache_data
def load_demo_data():

    path = (
        DATA_DIR
        / "youtube_comments_vader.csv"
    )

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


@st.cache_data
def load_csv(filename):

    path = DATA_DIR / filename

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except:
        return pd.DataFrame()


@st.cache_resource
def get_transformer():

    return load_transformer()


@st.cache_resource
def get_sarcasm_detector():

    return load_sarcasm_detector()


# ============================================================
# YOUTUBE ANALYSIS
# ============================================================

def analyze_youtube_url(
    url,
    comment_limit,
    progress=None,
):

    detected = identify_youtube_url(
        url
    )

    if not detected:
        raise ValueError(
            "The YouTube URL could not be identified."
        )

    content_type = detected.get(
        "type"
    )

    content_id = detected.get(
        "id"
    )

    if content_type not in [
        "video",
        "channel",
    ]:
        raise ValueError(
            "Please enter a valid YouTube video "
            "or channel URL."
        )

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    if content_type == "video":

        if progress:
            progress.write(
                "🎬 Getting video information..."
            )

        info = get_video_info(
            content_id
        )

        if not info:
            raise ValueError(
                "Could not retrieve video information."
            )

        if progress:
            progress.write(
                f"💬 Collecting up to "
                f"{comment_limit:,} comments..."
            )

        comments = get_video_comments(
            content_id,
            max_comments=comment_limit,
        )

        if not comments:
            raise ValueError(
                "No comments were available "
                "for this video."
            )

        if progress:
            progress.write(
                f"📝 {len(comments):,} comments collected."
            )

        return (
            content_type,
            info,
            comments,
            [],
        )

    # --------------------------------------------------------
    # CHANNEL
    # --------------------------------------------------------

    if progress:
        progress.write(
            "📺 Getting channel information..."
        )

    channel_info = get_channel_info(
        content_id
    )

    if not channel_info:
        raise ValueError(
            "Could not retrieve channel information."
        )

    if progress:
        progress.write(
            "🎥 Finding recent channel videos..."
        )

    videos = get_channel_videos(
        content_id,
        max_videos=10,
    )

    if not videos:
        raise ValueError(
            "No recent videos were found "
            "for this channel."
        )

    per_video = max(
        1,
        math.ceil(
            comment_limit / len(videos)
        ),
    )

    comments = []

    for index, video in enumerate(videos):

        if len(comments) >= comment_limit:
            break

        video_id = video.get(
            "video_id"
        )

        video_title = video.get(
            "title",
            "Untitled video",
        )

        if progress:
            progress.write(
                f"🎥 Video {index + 1}/{len(videos)}: "
                f"{video_title[:70]}"
            )

        remaining = (
            comment_limit
            - len(comments)
        )

        amount = min(
            per_video,
            remaining,
        )

        video_comments = get_video_comments(
            video_id,
            max_comments=amount,
        )

        for comment in video_comments:

            comment["video_id"] = video_id

            comment["video_title"] = (
                video_title
            )

        comments.extend(
            video_comments
        )

    comments = comments[
        :comment_limit
    ]

    if not comments:
        raise ValueError(
            "No comments were available "
            "from the recent channel videos."
        )

    if progress:
        progress.write(
            f"💬 {len(comments):,} channel comments collected."
        )

    return (
        content_type,
        channel_info,
        comments,
        videos,
    )


# ============================================================
# ANALYSIS DISPLAY
# ============================================================

def show_overview(
    result,
    info,
    content_type,
):

    comments = result["comments"]
    summary = result["summary"]
    stats = result["statistics"]

    positive_pct = safe_percentage(
        summary,
        "positive",
    )

    neutral_pct = safe_percentage(
        summary,
        "neutral",
    )

    negative_pct = safe_percentage(
        summary,
        "negative",
    )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    title = info.get(
        "title",
        info.get(
            "channel_title",
            "YouTube Analysis",
        ),
    )

    st.markdown(
        f"""
        <div class="hero">
            <div class="badge">
                {"VIDEO ANALYSIS" if content_type == "video"
                 else "CHANNEL ANALYSIS"}
            </div>
            <h1>{title}</h1>
            <p>
                Audience intelligence based on
                {len(comments):,} analysed comments.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # VIDEO / CHANNEL INFO
    # --------------------------------------------------------

    if content_type == "video":

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card(
                "👁 Views",
                format_number(
                    info.get("views", 0)
                ),
            )

        with c2:
            metric_card(
                "👍 Likes",
                format_number(
                    info.get("likes", 0)
                ),
            )

        with c3:
            metric_card(
                "💬 Comments Available",
                format_number(
                    info.get(
                        "comment_count",
                        len(comments),
                    )
                ),
            )

        with c4:
            metric_card(
                "🧠 Comments Analysed",
                format_number(
                    len(comments)
                ),
            )

    else:

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card(
                "👥 Subscribers",
                format_number(
                    info.get(
                        "subscribers",
                        0,
                    )
                ),
            )

        with c2:
            metric_card(
                "👁 Channel Views",
                format_number(
                    info.get(
                        "views",
                        0,
                    )
                ),
            )

        with c3:
            metric_card(
                "🎥 Videos",
                format_number(
                    info.get(
                        "videos",
                        0,
                    )
                ),
            )

        with c4:
            metric_card(
                "💬 Analysed Comments",
                format_number(
                    len(comments)
                ),
            )

    st.write("")

    # --------------------------------------------------------
    # SENTIMENT KPI
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "😊 Positive",
            f"{positive_pct:.1f}%",
        )

    with c2:
        metric_card(
            "😐 Neutral",
            f"{neutral_pct:.1f}%",
        )

    with c3:
        metric_card(
            "😞 Negative",
            f"{negative_pct:.1f}%",
        )

    with c4:
        metric_card(
            "❤️ Average Likes",
            format_number(
                stats.get(
                    "average_likes",
                    0,
                )
            ),
        )

    # --------------------------------------------------------
    # SENTIMENT
    # --------------------------------------------------------

    section_title(
        "📊 Audience Mood"
    )

    left, right = st.columns(
        [1, 1]
    )

    with left:
        sentiment_donut(
            summary
        )

    with right:
        sentiment_bar(
            summary
        )

    # --------------------------------------------------------
    # INSIGHTS
    # --------------------------------------------------------

    section_title(
        "💡 What the Audience Is Saying"
    )

    for insight in result["insights"]:

        st.markdown(
            f"""
            <div class="insight-card">
                💡 {insight}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# TOPICS / KEYWORDS
# ============================================================

def show_topics(result):

    section_title(
        "🔑 Topics & Language"
    )

    sentiment = st.selectbox(
        "Analyse keywords for",
        [
            "all",
            "positive",
            "neutral",
            "negative",
        ],
        format_func=lambda x: x.title(),
        key="keyword_sentiment",
    )

    keywords = result[
        "keywords"
    ].get(
        sentiment,
        pd.DataFrame(),
    )

    tfidf = result[
        "tfidf"
    ].get(
        sentiment,
        pd.DataFrame(),
    )

    left, right = st.columns(2)

    with left:

        st.subheader(
            "Most Frequent Keywords"
        )

        if not keywords.empty:

            fig = px.bar(
                keywords.sort_values(
                    "frequency"
                ),
                x="frequency",
                y="word",
                orientation="h",
                title="Keyword Frequency",
            )

            fig.update_layout(
                height=500,
                yaxis_title="",
                xaxis_title="Frequency",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.dataframe(
                keywords,
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.info(
                "No keywords available."
            )

    with right:

        st.subheader(
            "TF-IDF Terms"
        )

        if not tfidf.empty:

            fig = px.bar(
                tfidf.sort_values(
                    "tfidf"
                ),
                x="tfidf",
                y="term",
                orientation="h",
                title="TF-IDF Importance",
            )

            fig.update_layout(
                height=500,
                yaxis_title="",
                xaxis_title="TF-IDF Score",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.dataframe(
                tfidf,
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.info(
                "Not enough text for TF-IDF."
            )


# ============================================================
# ENGAGEMENT
# ============================================================

def show_engagement(result):

    st.markdown(
        """
        <div class="section-header">
            <div class="section-title">Engagement Intelligence</div>
            <div class="section-subtitle">
                Understand how audience sentiment relates to comment engagement.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    engagement = result.get(
        "engagement",
        pd.DataFrame()
    )

    statistics = result.get(
        "statistics",
        {}
    )

    if engagement.empty:
        st.info("No engagement data available.")
        return

    engagement = engagement.copy()

    # Make sure numeric columns exist
    for column in [
        "comments",
        "average_likes",
        "total_likes"
    ]:

        if column not in engagement.columns:
            engagement[column] = 0

        engagement[column] = pd.to_numeric(
            engagement[column],
            errors="coerce"
        ).fillna(0)

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    total_likes = statistics.get(
        "total_likes",
        int(engagement["total_likes"].sum())
    )

    average_likes = statistics.get(
        "average_likes",
        float(
            engagement["average_likes"].mean()
        )
    )

    total_comments = statistics.get(
        "total_comments",
        int(engagement["comments"].sum())
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Likes",
            format_number(total_likes)
        )

    with col2:
        st.metric(
            "Average Likes / Comment",
            f"{average_likes:,.1f}"
        )

    with col3:
        st.metric(
            "Comments Analyzed",
            format_number(total_comments)
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # CHART 1 — TOTAL LIKES
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            engagement,
            x="sentiment",
            y="total_likes",
            text="total_likes",
            title="Total Likes by Sentiment",
            labels={
                "sentiment": "Sentiment",
                "total_likes": "Total Likes"
            },
            color="sentiment",
            color_discrete_map={
                "positive": "#22c55e",
                "neutral": "#94a3b8",
                "negative": "#ef4444"
            }
        )

        fig.update_traces(
            texttemplate="%{text:,}",
            textposition="outside"
        )

        fig.update_layout(
            height=430,
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(
                l=20,
                r=20,
                t=70,
                b=20
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # CHART 2 — AVERAGE LIKES
    # --------------------------------------------------------

    with col2:

        fig = px.bar(
            engagement,
            x="sentiment",
            y="average_likes",
            text="average_likes",
            title="Average Likes per Comment",
            labels={
                "sentiment": "Sentiment",
                "average_likes": "Average Likes"
            },
            color="sentiment",
            color_discrete_map={
                "positive": "#22c55e",
                "neutral": "#94a3b8",
                "negative": "#ef4444"
            }
        )

        fig.update_traces(
            texttemplate="%{text:.1f}",
            textposition="outside"
        )

        fig.update_layout(
            height=430,
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(
                l=20,
                r=20,
                t=70,
                b=20
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # CHART 3 — COMMENT VOLUME
    # --------------------------------------------------------

    fig = px.bar(
        engagement,
        x="sentiment",
        y="comments",
        text="comments",
        title="Comment Volume by Sentiment",
        labels={
            "sentiment": "Sentiment",
            "comments": "Number of Comments"
        },
        color="sentiment",
        color_discrete_map={
            "positive": "#22c55e",
            "neutral": "#94a3b8",
            "negative": "#ef4444"
        }
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        height=400,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # DATA TABLE
    # --------------------------------------------------------

    st.markdown(
        "### Engagement Breakdown"
    )

    display_df = engagement.copy()

    display_df["sentiment"] = (
        display_df["sentiment"]
        .str.title()
    )

    display_df["average_likes"] = (
        display_df["average_likes"]
        .round(2)
    )

    display_df["total_likes"] = (
        display_df["total_likes"]
        .astype(int)
    )

    display_df.columns = [
        "Sentiment",
        "Comments",
        "Average Likes",
        "Total Likes"
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# COMMENTS
# ============================================================

def show_comments(result):

    section_title(
        "💬 Comment Explorer"
    )

    data = result[
        "comments"
    ].copy()

    if data.empty:
        st.info(
            "No comments available."
        )
        return

    sentiment_filter = st.multiselect(
        "Sentiment",
        [
            "positive",
            "neutral",
            "negative",
        ],
        default=[
            "positive",
            "neutral",
            "negative",
        ],
    )

    search = st.text_input(
        "🔎 Search comments"
    )

    min_likes = st.slider(
        "Minimum likes",
        min_value=0,
        max_value=int(
            max(
                1,
                data["like_count"]
                .fillna(0)
                .max(),
            )
        ),
        value=0,
    )

    filtered = data[
        data["primary_sentiment"]
        .isin(sentiment_filter)
    ].copy()

    filtered["like_count"] = pd.to_numeric(
        filtered["like_count"],
        errors="coerce",
    ).fillna(0)

    filtered = filtered[
        filtered["like_count"]
        >= min_likes
    ]

    if search:

        filtered = filtered[
            filtered["comment_text"]
            .fillna("")
            .str.contains(
                search,
                case=False,
                na=False,
            )
        ]

    display_columns = [
        "comment_text",
        "primary_sentiment",
        "like_count",
    ]

    if "transformer_sentiment" in filtered.columns:
        display_columns += [
            "transformer_sentiment",
            "transformer_score",
        ]

    if "video_title" in filtered.columns:
        display_columns.insert(
            1,
            "video_title",
        )

    display_columns = [
        col
        for col in display_columns
        if col in filtered.columns
    ]

    st.write(
        f"Showing **{len(filtered):,}** comments"
    )

    st.dataframe(
        filtered[
            display_columns
        ],
        use_container_width=True,
        hide_index=True,
        height=550,
    )

    csv = filtered.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Filtered Comments",
        data=csv,
        file_name="youtube_pulse_comments.csv",
        mime="text/csv",
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

def show_models(result):

    section_title(
        "🧠 VADER vs Transformer"
    )

    data = result[
        "comments"
    ]

    if (
        "transformer_sentiment"
        not in data.columns
    ):
        st.info(
            "Transformer results are unavailable."
        )
        return

    agreement = result[
        "model_agreement"
    ]

    c1, c2 = st.columns(2)

    with c1:
        metric_card(
            "Model Agreement",
            f"{agreement:.1f}%"
            if agreement is not None
            else "N/A",
        )

    with c2:
        metric_card(
            "Comments Compared",
            format_number(
                len(data)
            ),
        )

    comparison = pd.DataFrame(
        {
            "VADER": data[
                "vader_sentiment"
            ].value_counts(),

            "Transformer": data[
                "transformer_sentiment"
            ].value_counts(),
        }
    ).fillna(0)

    comparison = comparison.reindex(
        [
            "positive",
            "neutral",
            "negative",
        ]
    ).fillna(0)

    comparison = (
        comparison
        .reset_index()
        .rename(
            columns={
                "index": "sentiment"
            }
        )
    )

    melted = comparison.melt(
        id_vars="sentiment",
        var_name="model",
        value_name="count",
    )

    fig = px.bar(
        melted,
        x="sentiment",
        y="count",
        color="model",
        barmode="group",
        title="Sentiment Predictions by Model",
    )

    fig.update_layout(
        height=420,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader(
        "Prediction Comparison"
    )

    st.dataframe(
        data[
            [
                "comment_text",
                "vader_sentiment",
                "transformer_sentiment",
                "transformer_score",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        height=500,
    )


# ============================================================
# CHANNEL BREAKDOWN
# ============================================================

def show_channel_breakdown(
    result,
    videos,
):

    section_title(
        "📺 Channel Video Breakdown"
    )

    comments = result[
        "comments"
    ].copy()

    if (
        not videos
        or "video_id" not in comments.columns
    ):
        st.info(
            "Video-level channel information "
            "is not available."
        )
        return

    rows = []

    for video in videos:

        video_id = video.get(
            "video_id"
        )

        subset = comments[
            comments["video_id"]
            == video_id
        ]

        if subset.empty:
            continue

        counts = (
            subset["primary_sentiment"]
            .value_counts()
        )

        total = len(subset)

        rows.append(
            {
                "video_id": video_id,
                "video_title": video.get(
                    "title",
                    "Untitled",
                ),
                "comments": total,
                "positive": round(
                    counts.get(
                        "positive",
                        0,
                    )
                    / total
                    * 100,
                    1,
                ),
                "neutral": round(
                    counts.get(
                        "neutral",
                        0,
                    )
                    / total
                    * 100,
                    1,
                ),
                "negative": round(
                    counts.get(
                        "negative",
                        0,
                    )
                    / total
                    * 100,
                    1,
                ),
            }
        )

    video_df = pd.DataFrame(
        rows
    )

    if video_df.empty:
        st.info(
            "No video-level data available."
        )
        return

    chart_data = video_df.melt(
        id_vars=[
            "video_title",
            "comments",
        ],
        value_vars=[
            "positive",
            "neutral",
            "negative",
        ],
        var_name="sentiment",
        value_name="percentage",
    )

    fig = px.bar(
        chart_data,
        x="percentage",
        y="video_title",
        color="sentiment",
        orientation="h",
        barmode="stack",
        color_discrete_map=SENTIMENT_COLORS,
        title="Sentiment Across Recent Videos",
    )

    fig.update_layout(
        height=max(
            450,
            len(video_df) * 55,
        ),
        xaxis_title="Percentage",
        yaxis_title="",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    display = video_df.drop(
        columns=["video_id"]
    )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# COMPLETE RESULTS
# ============================================================

def show_sarcasm(result):

    section_title("🎭 Sarcasm Analysis")

    summary = result.get("sarcasm_summary", {})
    comments = result.get("comments", pd.DataFrame()).copy()

    if not summary.get("available", False):
        st.warning(
            "Sarcasm detection is unavailable for this analysis. "
            "The rest of the analysis is still available."
        )
        return

    total = int(summary.get("total_comments", 0))
    sarcastic = int(summary.get("sarcastic_comments", 0))
    percentage = float(summary.get("sarcasm_percentage", 0.0))
    confidence = float(summary.get("average_sarcasm_confidence", 0.0))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("🎭 Potentially Sarcastic", f"{percentage:.1f}%")
    with c2:
        metric_card("🎭 Sarcastic Comments", f"{sarcastic:,}")
    with c3:
        metric_card("💬 Comments Checked", f"{total:,}")
    with c4:
        metric_card("🎯 Avg. Detector Confidence", f"{confidence:.1f}%")

    st.info(
        "Sarcasm detection is probabilistic. A detected comment should be treated "
        "as potentially sarcastic, especially because the model was trained on "
        "social/news text rather than YouTube-specific conversations."
    )

    by_sentiment = result.get("sarcasm_by_sentiment", pd.DataFrame())
    left, right = st.columns(2)

    with left:
        if not by_sentiment.empty:
            fig = px.bar(
                by_sentiment,
                x="vader_sentiment",
                y="sarcasm_percentage",
                text="sarcasm_percentage",
                title="Potential Sarcasm by VADER Sentiment",
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(
                yaxis_title="Potentially Sarcastic (%)",
                xaxis_title="VADER Sentiment",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

    with right:
        if not by_sentiment.empty:
            fig = px.bar(
                by_sentiment,
                x="vader_sentiment",
                y="sarcastic_comments",
                text="sarcastic_comments",
                title="Sarcastic Comment Count by Sentiment",
            )
            fig.update_layout(
                yaxis_title="Sarcastic Comments",
                xaxis_title="VADER Sentiment",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

    if "sarcasm_label" in comments.columns:
        examples = sarcastic_comment_examples(comments, n=15)
        st.subheader("Highest-Confidence Potentially Sarcastic Comments")
        if examples.empty:
            st.info("No potentially sarcastic comments were detected.")
        else:
            display_cols = [
                col for col in [
                    "comment_text",
                    "vader_sentiment",
                    "transformer_sentiment",
                    "sarcasm_score",
                    "like_count",
                ] if col in examples.columns
            ]
            table = examples[display_cols].copy()
            if "sarcasm_score" in table.columns:
                table["sarcasm_score"] = (table["sarcasm_score"] * 100).round(1)
                table = table.rename(columns={"sarcasm_score": "sarcasm_confidence_%"})
            st.dataframe(table, use_container_width=True, hide_index=True, height=500)


def show_analysis_results():

    result = (
        st.session_state
        .analysis_result
    )

    info = (
        st.session_state
        .content_info
    )

    content_type = (
        st.session_state
        .content_type
    )

    if result is None:
        return

    show_overview(
        result,
        info,
        content_type,
    )

    tabs = st.tabs(
        [
            "📊 Overview",
            "🔑 Topics",
            "❤️ Engagement",
            "💬 Comments",
            "🎭 Sarcasm",
            "🧠 Models",
        ]
    )

    with tabs[0]:

        if content_type == "channel":

            show_channel_breakdown(
                result,
                st.session_state.channel_videos,
            )

    with tabs[1]:

        show_topics(
            result
        )

    with tabs[2]:

        show_engagement(
            result
        )

    with tabs[3]:

        show_comments(
            result
        )

    with tabs[4]:

        show_sarcasm(
            result
        )

    with tabs[5]:

        show_models(
            result
        )


# ============================================================
# DEMO DATASET DASHBOARD
# ============================================================

def show_demo_dashboard():

    st.markdown(
        """
        <div class="hero">
            <div class="badge">
                RESEARCH DATASET
            </div>
            <h1>Demo Dataset Intelligence</h1>
            <p>
                Explore the project's 750-comment
                sentiment-analysis dataset,
                domains, creators, videos,
                keywords and model validation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    data = load_demo_data()

    if data.empty:
        st.error(
            "Demo dataset not found."
        )
        return

    # --------------------------------------------------------
    # DATA PREPARATION
    # --------------------------------------------------------

    if "like_count" in data.columns:

        data["like_count"] = pd.to_numeric(
            data["like_count"],
            errors="coerce",
        ).fillna(0)

    total_comments = len(data)

    creators = (
        data["creator"].nunique()
        if "creator" in data.columns
        else 0
    )

    videos = (
        data["video_id"].nunique()
        if "video_id" in data.columns
        else 0
    )

    domains = (
        data["domain"].nunique()
        if "domain" in data.columns
        else 0
    )

    avg_likes = (
        data["like_count"].mean()
        if "like_count" in data.columns
        else 0
    )

    summary = (
        data["vader_sentiment"]
        .value_counts()
        .reindex(
            [
                "positive",
                "neutral",
                "negative",
            ],
            fill_value=0,
        )
        .reset_index()
    )

    summary.columns = [
        "sentiment",
        "count",
    ]

    summary["percentage"] = (
        summary["count"]
        / total_comments
        * 100
    ).round(2)

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        metric_card(
            "💬 Comments",
            format_number(
                total_comments
            ),
        )

    with c2:
        metric_card(
            "👤 Creators",
            format_number(
                creators
            ),
        )

    with c3:
        metric_card(
            "🎥 Videos",
            format_number(
                videos
            ),
        )

    with c4:
        metric_card(
            "🌐 Domains",
            format_number(
                domains
            ),
        )

    with c5:
        metric_card(
            "❤️ Avg Likes",
            format_number(
                avg_likes
            ),
        )

    st.write("")

    # --------------------------------------------------------
    # DEMO TABS
    # --------------------------------------------------------

    tabs = st.tabs(
        [
            "📊 Overview",
            "🌐 Domains",
            "👤 Creators",
            "🎥 Videos",
            "🔑 Keywords",
            "💬 Comments",
            "🧠 Model Validation",
            "⚠️ Error Analysis",
        ]
    )

    # --------------------------------------------------------
    # OVERVIEW
    # --------------------------------------------------------

    with tabs[0]:

        left, right = st.columns(2)

        with left:

            sentiment_donut(
                summary,
                "Overall Dataset Sentiment",
            )

        with right:

            sentiment_bar(
                summary
            )

        section_title(
            "Dataset Insights"
        )

        positive = safe_percentage(
            summary,
            "positive",
        )

        neutral = safe_percentage(
            summary,
            "neutral",
        )

        negative = safe_percentage(
            summary,
            "negative",
        )

        insights = [
            f"The dataset contains {total_comments:,} analysed YouTube comments.",
            f"{positive:.1f}% of comments are classified as positive.",
            f"{neutral:.1f}% of comments are classified as neutral.",
            f"{negative:.1f}% of comments are classified as negative.",
            f"The dataset covers {domains} content domains, {creators} creators and {videos} videos.",
        ]

        for insight in insights:

            st.markdown(
                f"""
                <div class="insight-card">
                    📌 {insight}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # DOMAINS
    # --------------------------------------------------------

    with tabs[1]:

        if "domain" not in data.columns:

            st.info(
                "Domain information unavailable."
            )

        else:

            domain_summary = (
                data
                .groupby(
                    [
                        "domain",
                        "vader_sentiment",
                    ]
                )
                .size()
                .reset_index(
                    name="count"
                )
            )

            domain_total = (
                domain_summary
                .groupby("domain")
                ["count"]
                .transform("sum")
            )

            domain_summary[
                "percentage"
            ] = (
                domain_summary["count"]
                / domain_total
                * 100
            ).round(2)

            fig = px.bar(
                domain_summary,
                x="domain",
                y="percentage",
                color="vader_sentiment",
                barmode="stack",
                color_discrete_map=SENTIMENT_COLORS,
                title="Sentiment Distribution by Domain",
            )

            fig.update_layout(
                height=500,
                yaxis_title="Percentage",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.dataframe(
                domain_summary,
                use_container_width=True,
                hide_index=True,
            )

    # --------------------------------------------------------
    # CREATORS
    # --------------------------------------------------------

    with tabs[2]:

        if "creator" not in data.columns:

            st.info(
                "Creator information unavailable."
            )

        else:

            creator_summary = (
                data
                .groupby(
                    [
                        "creator",
                        "vader_sentiment",
                    ]
                )
                .size()
                .reset_index(
                    name="comments"
                )
            )

            fig = px.bar(
                creator_summary,
                x="creator",
                y="comments",
                color="vader_sentiment",
                barmode="group",
                color_discrete_map=SENTIMENT_COLORS,
                title="Creator Sentiment Distribution",
            )

            fig.update_layout(
                height=550,
                xaxis_tickangle=-35,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.dataframe(
                creator_summary,
                use_container_width=True,
                hide_index=True,
            )

    # --------------------------------------------------------
    # VIDEOS
    # --------------------------------------------------------

    with tabs[3]:

        if "video_title" not in data.columns:

            st.info(
                "Video information unavailable."
            )

        else:

            video_summary = (
                data
                .groupby(
                    [
                        "video_title",
                        "vader_sentiment",
                    ]
                )
                .size()
                .reset_index(
                    name="comments"
                )
            )

            fig = px.bar(
                video_summary,
                x="comments",
                y="video_title",
                color="vader_sentiment",
                orientation="h",
                barmode="stack",
                color_discrete_map=SENTIMENT_COLORS,
                title="Sentiment Across Videos",
            )

            fig.update_layout(
                height=max(
                    500,
                    len(video_summary)
                    * 30,
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.dataframe(
                video_summary,
                use_container_width=True,
                hide_index=True,
            )

    # --------------------------------------------------------
    # KEYWORDS
    # --------------------------------------------------------

    with tabs[4]:

        from src.sentiment_analyzer import (
            extract_keywords,
            extract_tfidf,
        )

        sentiment_choice = st.selectbox(
            "Sentiment",
            [
                "All",
                "Positive",
                "Neutral",
                "Negative",
            ],
            key="demo_keyword_sentiment",
        )

        selected = (
            None
            if sentiment_choice == "All"
            else sentiment_choice.lower()
        )

        keywords = extract_keywords(
            data,
            sentiment=selected,
            top_n=20,
        )

        tfidf = extract_tfidf(
            data,
            sentiment=selected,
            top_n=20,
        )

        left, right = st.columns(2)

        with left:

            st.subheader(
                "Top Keywords"
            )

            if not keywords.empty:

                fig = px.bar(
                    keywords.sort_values(
                        "frequency"
                    ),
                    x="frequency",
                    y="word",
                    orientation="h",
                    title="Most Frequent Terms",
                )

                fig.update_layout(
                    height=600
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

        with right:

            st.subheader(
                "TF-IDF"
            )

            if not tfidf.empty:

                fig = px.bar(
                    tfidf.sort_values(
                        "tfidf"
                    ),
                    x="tfidf",
                    y="term",
                    orientation="h",
                    title="Most Important Terms",
                )

                fig.update_layout(
                    height=600
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

    # --------------------------------------------------------
    # COMMENTS
    # --------------------------------------------------------

    with tabs[5]:

        sentiment_filter = st.multiselect(
            "Sentiment",
            [
                "positive",
                "neutral",
                "negative",
            ],
            default=[
                "positive",
                "neutral",
                "negative",
            ],
            key="demo_sentiment_filter",
        )

        search = st.text_input(
            "Search dataset comments",
            key="demo_search",
        )

        filtered = data[
            data["vader_sentiment"]
            .isin(sentiment_filter)
        ].copy()

        if search:

            filtered = filtered[
                filtered["comment_text"]
                .fillna("")
                .str.contains(
                    search,
                    case=False,
                    na=False,
                )
            ]

        st.write(
            f"Showing {len(filtered):,} comments"
        )

        columns = [
            "domain",
            "creator",
            "video_title",
            "comment_text",
            "like_count",
            "vader_sentiment",
        ]

        columns = [
            col
            for col in columns
            if col in filtered.columns
        ]

        st.dataframe(
            filtered[columns],
            use_container_width=True,
            hide_index=True,
            height=600,
        )

        csv = filtered.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Dataset Selection",
            data=csv,
            file_name="youtube_pulse_demo_filtered.csv",
            mime="text/csv",
        )

    # --------------------------------------------------------
    # MODEL VALIDATION
    # --------------------------------------------------------

    with tabs[6]:

        model_comparison = load_csv(
            "model_comparison.csv"
        )

        vader_validation = load_csv(
            "vader_validation_results.csv"
        )

        transformer_validation = load_csv(
            "transformer_validation_results.csv"
        )

        if not model_comparison.empty:

            st.subheader(
                "Model Performance"
            )

            st.dataframe(
                model_comparison,
                use_container_width=True,
                hide_index=True,
            )

            metric_columns = [
                col
                for col in [
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "macro_precision",
                    "macro_recall",
                    "macro_f1",
                ]
                if col in model_comparison.columns
            ]

            if metric_columns:

                numeric = model_comparison[
                    metric_columns
                ].copy()

                if (
                    numeric.max().max()
                    <= 1.0
                ):
                    numeric = numeric * 100

                numeric["model"] = (
                    model_comparison
                    .iloc[:, 0]
                    .astype(str)
                )

                melted = numeric.melt(
                    id_vars="model",
                    var_name="metric",
                    value_name="score",
                )

                fig = px.bar(
                    melted,
                    x="metric",
                    y="score",
                    color="model",
                    barmode="group",
                    title="Model Performance Comparison",
                )

                fig.update_layout(
                    height=500,
                    yaxis_title="Score (%)",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

        # VADER confusion matrix
        if not vader_validation.empty:

            human_col = (
                "human-sentiment"
                if "human-sentiment"
                in vader_validation.columns
                else "human"
            )

            if (
                human_col in vader_validation.columns
                and "vader_sentiment"
                in vader_validation.columns
            ):

                cm = confusion_matrix(
                    vader_validation[
                        human_col
                    ],
                    vader_validation[
                        "vader_sentiment"
                    ],
                    labels=[
                        "positive",
                        "neutral",
                        "negative",
                    ],
                )

                fig = px.imshow(
                    cm,
                    x=[
                        "positive",
                        "neutral",
                        "negative",
                    ],
                    y=[
                        "positive",
                        "neutral",
                        "negative",
                    ],
                    text_auto=True,
                    title="VADER Confusion Matrix",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

        # Transformer confusion matrix
        if not transformer_validation.empty:

            human_col = (
                "human-sentiment"
                if "human-sentiment"
                in transformer_validation.columns
                else "human"
            )

            if (
                human_col
                in transformer_validation.columns
                and "transformer_sentiment"
                in transformer_validation.columns
            ):

                cm = confusion_matrix(
                    transformer_validation[
                        human_col
                    ],
                    transformer_validation[
                        "transformer_sentiment"
                    ],
                    labels=[
                        "positive",
                        "neutral",
                        "negative",
                    ],
                )

                fig = px.imshow(
                    cm,
                    x=[
                        "positive",
                        "neutral",
                        "negative",
                    ],
                    y=[
                        "positive",
                        "neutral",
                        "negative",
                    ],
                    text_auto=True,
                    title="Transformer Confusion Matrix",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

    # --------------------------------------------------------
    # ERROR ANALYSIS
    # --------------------------------------------------------

    with tabs[7]:

        error_data = load_csv(
            "error_analysis.csv"
        )

        if error_data.empty:

            st.info(
                "Error analysis file not found."
            )

        else:

            if "error_category" in error_data.columns:

                counts = (
                    error_data[
                        "error_category"
                    ]
                    .value_counts()
                    .reset_index()
                )

                counts.columns = [
                    "category",
                    "count",
                ]

                fig = px.bar(
                    counts,
                    x="count",
                    y="category",
                    orientation="h",
                    title="Model Error Analysis",
                )

                fig.update_layout(
                    height=500
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

                fig = px.pie(
                    counts,
                    names="category",
                    values="count",
                    hole=0.55,
                    title="Error Category Distribution",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            st.dataframe(
                error_data,
                use_container_width=True,
                hide_index=True,
                height=600,
            )


# ============================================================
# COMMENT ANALYZER HELPERS
# ============================================================

def explain_sarcasm_context(comment: str, sentiment: str, is_sarcastic: bool) -> dict:
    """Create a human-friendly explanation of a sarcasm prediction.

    This is an interpretation layer, not a second ML prediction.
    It uses visible language cues plus the detected sentiment.
    """
    text = str(comment).strip()
    lower = text.lower()

    if not is_sarcastic:
        return {
            "context": "Straightforward tone",
            "why": "The sarcasm model did not flag the comment as sarcastic.",
            "meaning": "The detected sentiment can be read relatively directly from the wording.",
        }

    if sentiment == "positive":
        context = "Praise that may be deliberately exaggerated"
        meaning = "The wording sounds positive, but the commenter may be using exaggerated praise to make a critical or playful point."
    elif sentiment == "negative":
        context = "Mocking or ironic criticism"
        meaning = "The comment carries negative language and may be using irony to make the criticism sharper or more humorous."
    else:
        context = "Playful irony / mixed tone"
        meaning = "The wording is not strongly positive or negative, so the sarcasm may come mainly from the contrast between the words and the intended tone."

    cues = []
    if "!" in text:
        cues.append("strong punctuation")
    if "?" in text:
        cues.append("rhetorical-question style")
    if any(word in lower for word in ["yeah right", "sure", "totally", "obviously", "of course", "love that", "great job", "amazing"]):
        cues.append("exaggerated or ironic wording")
    if text.isupper() or any(word.isupper() and len(word) > 2 for word in text.split()):
        cues.append("emphasis/capitalisation")

    cue_text = ", ".join(cues[:2]) if cues else "a mismatch between literal wording and detected tone"

    return {
        "context": context,
        "why": f"The detector flagged possible sarcasm, with {cue_text} providing additional context for the interpretation.",
        "meaning": meaning,
    }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        # 🎬 YouTube Pulse

        **YouTube Audience Intelligence**
        """
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🔎 Analyze",
            "📊 Demo Dataset",
            "💬 Comment Analyzer",
            "ℹ️ About",
        ],
    )


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.markdown(
        """<div class="hero">
    <div class="badge">AI-POWERED YOUTUBE ANALYTICS</div>

    <h1>YouTube Pulse</h1>
            
    <p>
        Understand what audiences really think about YouTube content using
        sentiment, engagement, keywords and AI-powered analysis.
    </p>
</div>""",
        unsafe_allow_html=True,
    )

    st.subheader(
        "Analyze any YouTube video or channel"
    )

    st.write(
        "Paste a YouTube URL and discover audience "
        "sentiment, engagement, topics, keywords "
        "and model-based insights."
    )

    url = st.text_input(
        "YouTube URL",
        placeholder=(
            "https://www.youtube.com/watch?v=..."
            " or https://www.youtube.com/@channel"
        ),
        key="home_url",
    )

    comment_limit = st.slider(
        "Comments to analyze",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        help=(
            "Maximum number of comments collected "
            "for this analysis."
        ),
    )

    if st.button(
        "✨ Analyze Now",
        type="primary",
        use_container_width=True,
    ):

        if not url.strip():

            st.warning(
                "Please enter a YouTube URL."
            )

        else:

            try:

                with st.status(
                    "🔄 Analyzing...",
                    expanded=True,
                ) as status:

                    progress = st

                    content_type, info, comments, videos = (
                        analyze_youtube_url(
                            url,
                            comment_limit,
                            progress,
                        )
                    )

                    progress.write(
                        "🧠 Loading sentiment model..."
                    )

                    transformer = get_transformer()

                    progress.write(
                        "📊 Running VADER and Transformer sentiment analysis..."
                    )

                    result = run_complete_analysis(
                        comments,
                        transformer_classifier=transformer,
                        use_transformer=True,
                    )

                    st.session_state.analysis_result = (
                        result
                    )

                    st.session_state.content_info = (
                        info
                    )

                    st.session_state.content_type = (
                        content_type
                    )

                    st.session_state.channel_videos = (
                        videos
                    )

                    status.update(
                        label="✅ Analysis complete!",
                        state="complete",
                        expanded=False,
                    )

                st.rerun()

            except Exception as e:

                st.error(
                    f"Analysis failed: {str(e)}"
                )


    st.divider()

    section_title(
        "📊 Explore the Research Dataset"
    )

    demo = load_demo_data()

    if not demo.empty:

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card(
                "Comments",
                format_number(
                    len(demo)
                ),
            )

        with c2:
            metric_card(
                "Creators",
                format_number(
                    demo["creator"].nunique()
                )
                if "creator" in demo.columns
                else "0",
            )

        with c3:
            metric_card(
                "Videos",
                format_number(
                    demo["video_id"].nunique()
                )
                if "video_id" in demo.columns
                else "0",
            )

        with c4:
            metric_card(
                "Domains",
                format_number(
                    demo["domain"].nunique()
                )
                if "domain" in demo.columns
                else "0",
            )


# ============================================================
# ANALYZE
# ============================================================

elif page == "🔎 Analyze":

    st.markdown(
        """<div class="hero">
    <div class="badge">LIVE YOUTUBE ANALYSIS</div>
            
    <h1>Analyze Content</h1>
    <p>
        Analyze videos or channels using
        real YouTube comments.
    </p>
</div>""",
        unsafe_allow_html=True,
    )

    url = st.text_input(
        "YouTube URL",
        placeholder=(
            "Video or channel URL"
        ),
        key="analysis_url",
    )

    comment_limit = st.slider(
        "Maximum comments",
        100,
        5000,
        1000,
        100,
        key="analysis_limit",
    )

    if st.button(
        "🚀 Start Analysis",
        type="primary",
        use_container_width=True,
    ):

        if not url.strip():

            st.warning(
                "Enter a YouTube URL first."
            )

        else:

            try:

                with st.status(
                    "🔄 Processing YouTube content...",
                    expanded=True,
                ) as status:

                    content_type, info, comments, videos = (
                        analyze_youtube_url(
                            url,
                            comment_limit,
                            st,
                        )
                    )

                    st.write(
                        f"📝 Processing {len(comments):,} comments..."
                    )

                    transformer = get_transformer()

                    st.write(
                        "🧠 Running VADER and Transformer sentiment analysis..."
                    )

                    result = run_complete_analysis(
                        comments,
                        transformer_classifier=transformer,
                        use_transformer=True,
                    )

                    st.session_state.analysis_result = (
                        result
                    )

                    st.session_state.content_info = (
                        info
                    )

                    st.session_state.content_type = (
                        content_type
                    )

                    st.session_state.channel_videos = (
                        videos
                    )

                    status.update(
                        label="✅ Analysis complete",
                        state="complete",
                        expanded=False,
                    )

                st.rerun()

            except Exception as e:

                st.error(
                    f"Analysis failed: {str(e)}"
                )


# ============================================================
# DISPLAY RESULTS
# ============================================================

if (
    page in [
        "🏠 Home",
        "🔎 Analyze",
    ]
    and st.session_state.analysis_result
    is not None
):

    st.divider()

    show_analysis_results()


# ============================================================
# DEMO DATASET
# ============================================================

elif page == "📊 Demo Dataset":

    show_demo_dashboard()


# ============================================================
# COMMENT ANALYZER
# ============================================================

elif page == "💬 Comment Analyzer":

    st.markdown(
        """<div class="hero">
    <div class="badge">SINGLE COMMENT AI</div>

    <h1>Comment Analyzer</h1>

    <p>
        See the sentiment behind a comment — and
        explore whether its tone may be sarcastic.
    </p>
</div>""",
        unsafe_allow_html=True,
    )

    comment = st.text_area(
        "Enter a YouTube comment",
        height=180,
        placeholder=(
            "Example: This video was absolutely amazing!"
        ),
    )

    if st.button(
        "🔍 Analyze Comment",
        type="primary",
    ):

        if not comment.strip():

            st.warning(
                "Enter a comment first."
            )

        else:

            result = get_vader_sentiment(
                comment
            )

            sentiment = result[
                "sentiment"
            ]

            st.subheader(
                f"Detected sentiment: {sentiment.title()}"
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                metric_card(
                    "Positive",
                    f"{result['positive_score']:.2%}",
                )

            with c2:
                metric_card(
                    "Neutral",
                    f"{result['neutral_score']:.2%}",
                )

            with c3:
                metric_card(
                    "Negative",
                    f"{result['negative_score']:.2%}",
                )

            with c4:
                metric_card(
                    "Compound",
                    f"{result['compound_score']:.3f}",
                )

            # --------------------------------------------------------
            # SARCASM LENS
            # --------------------------------------------------------

            st.divider()

            sarcasm_detector = get_sarcasm_detector()
            sarcasm_result = detect_sarcasm(
                comment,
                detector=sarcasm_detector,
            )

            st.subheader("🎭 Sarcasm Lens")
            st.caption(
                "A separate AI check for possible sarcasm. Sarcasm is context-dependent, so treat this as an interpretation rather than a certainty."
            )

            if sarcasm_result["sarcasm_label"] == "unavailable":
                st.warning(
                    "The sarcasm detector is unavailable right now. Sentiment analysis still worked normally."
                )
            else:
                is_sarcastic = sarcasm_result["is_sarcastic"]
                confidence = sarcasm_result["sarcasm_score"]

                s1, s2, s3 = st.columns(3)

                with s1:
                    metric_card(
                        "Sarcasm",
                        "Potentially yes" if is_sarcastic else "Not detected",
                    )

                with s2:
                    metric_card(
                        "Confidence",
                        f"{confidence:.1%}",
                    )

                with s3:
                    metric_card(
                        "Tone",
                        sentiment.title(),
                    )

                explanation = explain_sarcasm_context(
                    comment,
                    sentiment,
                    is_sarcastic,
                )

                if is_sarcastic:
                    st.markdown(
                        f"""
                        <div class="insight-card">
                            <div class="insight-title">🎭 What might be happening?</div>
                            <div class="insight-body">
                                <strong>{explanation['context']}</strong><br>
                                {explanation['why']}<br><br>
                                <strong>How to read it:</strong> {explanation['meaning']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.info(
                        "💡 Think of sarcasm as a second layer of meaning: the sentiment model reads the words, while the sarcasm detector asks whether the commenter may mean something different from the literal wording."
                    )
                else:
                    st.success(
                        "No potential sarcasm was detected. The comment's sentiment appears relatively straightforward from the available text."
                    )

            st.divider()
            st.subheader("🎭 Sarcasm Detection")

            if sarcasm_result["sarcasm_label"] == "unavailable":
                st.warning("Sarcasm detector is unavailable, but sentiment analysis worked normally.")
            else:
                s1, s2 = st.columns(2)
                with s1:
                    metric_card(
                        "Potential Sarcasm",
                        "Yes" if sarcasm_result["is_sarcastic"] else "No",
                    )
                with s2:
                    metric_card(
                        "Detector Confidence",
                        f"{sarcasm_result['sarcasm_score']:.1%}",
                    )
                if sarcasm_result["is_sarcastic"]:
                    st.warning(
                        "This comment is potentially sarcastic. The result is a model prediction, not a certainty."
                    )
                else:
                    st.success(
                        "The sarcasm detector did not identify this comment as sarcastic."
                    )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.markdown(
        """
        <div class="hero">
            <div class="badge">
                ABOUT THE PROJECT
            </div>

            <h1>
                YouTube Pulse
            </h1>

            <p>
                An interactive YouTube audience
                sentiment intelligence platform.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_title(
        "What does YouTube Pulse do?"
    )

    st.write(
        """
        YouTube Pulse analyses YouTube comments to
        understand audience sentiment, engagement,
        language and discussion topics.
        """
    )

    c1, c2 = st.columns(2)

    with c1:

        st.subheader(
            "🎯 Analysis Features"
        )

        st.markdown(
            """
            - Video sentiment analysis
            - Channel sentiment analysis
            - VADER sentiment
            - Transformer sentiment
            - Keyword extraction
            - TF-IDF analysis
            - Engagement analysis
            - Comment exploration
            - Automatic insights
            - Sarcasm detection
            - Sarcasm vs sentiment analysis
            - Video-level channel comparison
            """
        )

    with c2:

        st.subheader(
            "🧠 NLP Models"
        )

        st.markdown(
            """
            **VADER**

            A lexicon and rule-based sentiment
            analysis approach.

            **Transformer**

            CardiffNLP's Twitter-RoBERTa sentiment
            model is used to provide contextual
            sentiment predictions.

            The application also compares the
            predictions produced by both models.
            """
        )

    st.divider()

    st.subheader(
        "📊 Research Dataset"
    )

    st.write(
        """
        The project also contains a 750-comment
        research dataset used for keyword analysis,
        TF-IDF, sentiment analysis, model validation,
        model comparison and error analysis.
        """
    )