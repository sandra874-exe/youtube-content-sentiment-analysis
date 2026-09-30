import plotly.express as px
import plotly.graph_objects as go

SENTIMENT_ORDER = ["positive", "neutral", "negative"]
SENTIMENT_COLORS = {
    "positive": "#7CBF9E",
    "neutral": "#D7B96E",
    "negative": "#D9859B",
}


def polish(fig, height=430):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=25, r=25, t=70, b=35),
        font=dict(family="Arial", size=13, color="#403B49"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
        ),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(100,100,100,0.12)", zeroline=False)
    return fig


def sentiment_bar(df, column="transformer_sentiment"):
    counts = (
        df[column]
        .value_counts()
        .reindex(SENTIMENT_ORDER, fill_value=0)
        .reset_index()
    )
    counts.columns = ["sentiment", "comments"]

    fig = px.bar(
        counts,
        x="sentiment",
        y="comments",
        text="comments",
        category_orders={"sentiment": SENTIMENT_ORDER},
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        title="Overall audience sentiment",
    )
    fig.update_traces(textposition="outside")
    return polish(fig)


def sentiment_donut(df, column="transformer_sentiment"):
    counts = (
        df[column]
        .value_counts()
        .reindex(SENTIMENT_ORDER, fill_value=0)
        .reset_index()
    )
    counts.columns = ["sentiment", "comments"]

    fig = px.pie(
        counts,
        names="sentiment",
        values="comments",
        hole=0.58,
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        title="Sentiment share",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return polish(fig, 430)


def creator_sentiment(df):
    grouped = (
        df.groupby(["creator", "transformer_sentiment"])
        .size()
        .reset_index(name="comments")
    )

    fig = px.bar(
        grouped,
        x="creator",
        y="comments",
        color="transformer_sentiment",
        barmode="stack",
        category_orders={"transformer_sentiment": SENTIMENT_ORDER},
        color_discrete_map=SENTIMENT_COLORS,
        title="Sentiment composition by creator",
    )
    return polish(fig, 470)


def domain_sentiment(df):
    grouped = (
        df.groupby(["domain", "transformer_sentiment"])
        .size()
        .reset_index(name="comments")
    )

    fig = px.bar(
        grouped,
        x="domain",
        y="comments",
        color="transformer_sentiment",
        barmode="relative",
        category_orders={"transformer_sentiment": SENTIMENT_ORDER},
        color_discrete_map=SENTIMENT_COLORS,
        title="Sentiment composition by content domain",
    )
    return polish(fig, 450)


def video_sentiment(df, top_n=10):
    top_titles = df["video_title"].value_counts().head(top_n).index
    filtered = df[df["video_title"].isin(top_titles)]

    grouped = (
        filtered.groupby(["video_title", "transformer_sentiment"])
        .size()
        .reset_index(name="comments")
    )

    fig = px.bar(
        grouped,
        x="comments",
        y="video_title",
        color="transformer_sentiment",
        barmode="stack",
        orientation="h",
        category_orders={"transformer_sentiment": SENTIMENT_ORDER},
        color_discrete_map=SENTIMENT_COLORS,
        title=f"Sentiment across top {top_n} commented videos",
    )
    fig.update_yaxes(categoryorder="total ascending")
    return polish(fig, 560)


def vader_roberta_distribution(df):
    rows = []
    for method, column in (
        ("VADER", "vader_sentiment"),
        ("RoBERTa", "transformer_sentiment"),
    ):
        counts = (
            df[column]
            .value_counts()
            .reindex(SENTIMENT_ORDER, fill_value=0)
        )
        for sentiment, count in counts.items():
            rows.append({
                "method": method,
                "sentiment": sentiment,
                "comments": int(count),
            })

    comparison = __import__("pandas").DataFrame(rows)
    fig = px.bar(
        comparison,
        x="sentiment",
        y="comments",
        color="method",
        barmode="group",
        category_orders={"sentiment": SENTIMENT_ORDER},
        title="VADER vs RoBERTa sentiment distribution",
    )
    return polish(fig)


def engagement_box(df):
    if "like_count" not in df.columns:
        return None

    fig = px.box(
        df,
        x="transformer_sentiment",
        y="like_count",
        color="transformer_sentiment",
        category_orders={"transformer_sentiment": SENTIMENT_ORDER},
        color_discrete_map=SENTIMENT_COLORS,
        points=False,
        title="Distribution of comment likes by sentiment",
    )
    fig.update_yaxes(type="log", title="Comment likes (log scale)")
    return polish(fig, 470)


def engagement_scatter(df):
    if not {"video_views", "like_count"}.issubset(df.columns):
        return None

    fig = px.scatter(
        df,
        x="video_views",
        y="like_count",
        color="transformer_sentiment",
        hover_name="video_title",
        hover_data=["creator", "domain", "comment"],
        color_discrete_map=SENTIMENT_COLORS,
        opacity=0.65,
        title="Video reach vs comment engagement",
    )
    fig.update_xaxes(type="log", title="Video views (log scale)")
    fig.update_yaxes(type="log", title="Comment likes (log scale)")
    return polish(fig, 520)


def model_metrics(results_df):
    long = results_df.melt(
        id_vars="model",
        value_vars=["accuracy", "precision", "recall", "f1"],
        var_name="metric",
        value_name="score",
    )

    fig = px.bar(
        long,
        x="model",
        y="score",
        color="metric",
        barmode="group",
        text_auto=".2f",
        title="Model performance on the same holdout test set",
    )
    fig.update_yaxes(range=[0, 1], title="Score")
    return polish(fig, 520)


def model_f1(results_df):
    fig = px.bar(
        results_df,
        x="model",
        y="f1",
        text="f1",
        title="Weighted F1-score comparison",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_yaxes(range=[0, 1], title="Weighted F1")
    fig.update_xaxes(tickangle=-15)
    return polish(fig, 430)


def cross_validation_chart(results_df):
    data = results_df.dropna(subset=["cv_mean_f1"]).copy()
    if data.empty:
        return None

    fig = px.bar(
        data,
        x="model",
        y="cv_mean_f1",
        error_y="cv_std_f1",
        text="cv_mean_f1",
        title="5-fold cross-validation weighted F1",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_yaxes(range=[0, 1], title="Mean weighted F1")
    return polish(fig, 430)


def confusion_matrix_figure(matrix, title):
    labels = ["negative", "neutral", "positive"]
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale="Purples",
            text=matrix,
            texttemplate="%{text}",
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            showscale=True,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted label",
        yaxis_title="Actual label",
    )
    return polish(fig, 430)


def topic_keywords_figure(topic_df):
    if topic_df.empty:
        return None

    fig = px.bar(
        topic_df,
        x="weight",
        y="keyword",
        color="topic",
        facet_col="topic",
        facet_col_wrap=2,
        orientation="h",
        title="Top keywords discovered by NMF topic modelling",
    )
    fig.update_layout(showlegend=False)
    return polish(fig, 760)
