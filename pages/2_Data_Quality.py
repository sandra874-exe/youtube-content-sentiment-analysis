import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import COMMENTS_FILE
from src.preprocessing import prepare_dataframe

st.set_page_config(page_title="Data Quality", page_icon="🧹", layout="wide")

st.title("🧹 Data Quality & Preprocessing")
st.caption("Inspect the data before applying sentiment models.")

if not COMMENTS_FILE.exists():
    st.warning("No YouTube comments found. Run Creator Setup first.")
    st.stop()

raw = pd.read_csv(COMMENTS_FILE)
clean = prepare_dataframe(raw)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Raw comments", f"{len(raw):,}")
c2.metric("Usable comments", f"{len(clean):,}")
c3.metric("Unique comments", f"{clean['sentiment_text'].nunique():,}")
c4.metric("Creators", clean["creator"].nunique() if "creator" in clean.columns else 0)

st.markdown("### Preprocessing")
st.markdown(
    """
    **Sentiment text** keeps important sentiment cues such as punctuation and emojis,
    while removing URLs, HTML and repeated whitespace.

    **Processed text** is the cleaned text used by TF-IDF. User mentions are removed,
    while the original comment is retained for interpretation in the dashboard.
    """
)

col1, col2 = st.columns(2)
with col1:
    fig = px.histogram(
        clean,
        x="word_count",
        nbins=30,
        title="Comment length distribution",
        labels={"word_count": "Words per comment"},
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    counts = clean["creator"].value_counts().reset_index()
    counts.columns = ["creator", "comments"]
    fig = px.bar(
        counts,
        x="creator",
        y="comments",
        text="comments",
        title="Comments collected by creator",
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("### Dataset preview")
columns = [
    c for c in [
        "creator", "domain", "video_title", "comment",
        "sentiment_text", "processed_text", "word_count", "character_count"
    ] if c in clean.columns
]
st.dataframe(clean[columns].head(150), use_container_width=True, hide_index=True)
