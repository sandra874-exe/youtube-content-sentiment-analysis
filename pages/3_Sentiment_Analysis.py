import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import COMMENTS_FILE, FINAL_FILE
from src.sentiment import run_sentiment_analysis
from src.sentiment_models import load_transformer
from src.charts import (
    sentiment_bar,
    sentiment_donut,
    creator_sentiment,
    domain_sentiment,
    video_sentiment,
    vader_roberta_distribution,
    engagement_box,
    engagement_scatter,
)
from src.analysis import model_agreement

st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="wide")

st.markdown(
    """
    <div style="
        padding:35px 40px;
        border-radius:26px;
        background:linear-gradient(135deg,#eee5ff,#ffe8ef,#e3f7ed);
        border:1px solid #eadff4;
        margin-bottom:28px;
    ">
        <div style="font-size:12px;font-weight:800;letter-spacing:1.5px;color:#82758d;">
            AUDIENCE SENTIMENT
        </div>
        <div style="font-size:38px;font-weight:800;color:#292638;margin-top:8px;">
            💬 Sentiment Analysis
        </div>
        <div style="font-size:15px;line-height:1.7;color:#706a77;max-width:900px;margin-top:10px;">
            VADER and RoBERTa power the large-scale YouTube analytics shown on this page.
            The supervised TF-IDF models are evaluated separately using the internal labelled dataset.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not COMMENTS_FILE.exists():
    st.warning("No YouTube comments found. Run Creator Setup first.")
    st.stop()

comments_df = pd.read_csv(COMMENTS_FILE)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Comments", f"{len(comments_df):,}")
c2.metric("Creators", comments_df["creator"].nunique() if "creator" in comments_df.columns else 0)
c3.metric("Videos", comments_df["video_id"].nunique() if "video_id" in comments_df.columns else 0)
c4.metric("Domains", comments_df["domain"].nunique() if "domain" in comments_df.columns else 0)

st.divider()

if st.button("🚀 Run VADER + RoBERTa Analysis", type="primary", use_container_width=True):
    try:
        with st.spinner("Loading RoBERTa sentiment model..."):
            classifier = load_transformer()

        with st.spinner("Analysing YouTube comments..."):
            df = run_sentiment_analysis(classifier)

        st.session_state["sentiment_df"] = df
        st.success(f"Analysed {len(df):,} comments.")
    except Exception as error:
        st.error(f"Sentiment analysis failed: {error}")

if "sentiment_df" in st.session_state:
    df = st.session_state["sentiment_df"]
elif FINAL_FILE.exists():
    df = pd.read_csv(FINAL_FILE)
else:
    st.info("Run the analysis above to generate the dashboard results.")
    st.stop()

required = {"vader_sentiment", "transformer_sentiment"}
missing = required - set(df.columns)
if missing:
    st.error(f"Missing model outputs: {sorted(missing)}")
    st.stop()

positive = int((df["transformer_sentiment"] == "positive").sum())
neutral = int((df["transformer_sentiment"] == "neutral").sum())
negative = int((df["transformer_sentiment"] == "negative").sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total comments", f"{len(df):,}")
c2.metric("Positive", f"{positive:,}")
c3.metric("Neutral", f"{neutral:,}")
c4.metric("Negative", f"{negative:,}")

st.markdown("### Overall audience sentiment")
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(sentiment_bar(df), use_container_width=True)
with col2:
    st.plotly_chart(sentiment_donut(df), use_container_width=True)

if "creator" in df.columns:
    st.markdown("### Creator comparison")
    st.plotly_chart(creator_sentiment(df), use_container_width=True)

if "domain" in df.columns:
    st.markdown("### Content-domain comparison")
    st.plotly_chart(domain_sentiment(df), use_container_width=True)

if "video_title" in df.columns:
    st.markdown("### Video-level sentiment")
    st.plotly_chart(video_sentiment(df), use_container_width=True)

st.markdown("### VADER vs RoBERTa")
st.plotly_chart(vader_roberta_distribution(df), use_container_width=True)

agreement = model_agreement(df)
if not agreement.empty:
    st.dataframe(agreement.round(2), use_container_width=True, hide_index=True)

engagement = engagement_box(df)
if engagement is not None:
    st.markdown("### Sentiment and comment engagement")
    st.plotly_chart(engagement, use_container_width=True)

scatter = engagement_scatter(df)
if scatter is not None:
    st.plotly_chart(scatter, use_container_width=True)

st.markdown("### Comment explorer")
filters = st.columns(3)
with filters[0]:
    creator_options = ["All"] + sorted(df["creator"].dropna().unique().tolist()) if "creator" in df.columns else ["All"]
    selected_creator = st.selectbox("Creator", creator_options)
with filters[1]:
    domain_options = ["All"] + sorted(df["domain"].dropna().unique().tolist()) if "domain" in df.columns else ["All"]
    selected_domain = st.selectbox("Domain", domain_options)
with filters[2]:
    selected_sentiment = st.selectbox("RoBERTa sentiment", ["All", "positive", "neutral", "negative"])

filtered = df.copy()
if selected_creator != "All":
    filtered = filtered[filtered["creator"] == selected_creator]
if selected_domain != "All":
    filtered = filtered[filtered["domain"] == selected_domain]
if selected_sentiment != "All":
    filtered = filtered[filtered["transformer_sentiment"] == selected_sentiment]

display_columns = [
    c for c in [
        "creator", "domain", "video_title", "comment",
        "vader_sentiment", "vader_score",
        "transformer_sentiment", "transformer_score", "like_count"
    ] if c in filtered.columns
]
st.dataframe(filtered[display_columns].head(200), use_container_width=True, hide_index=True)
