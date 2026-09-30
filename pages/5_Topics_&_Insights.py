import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import FINAL_FILE
from src.preprocessing import prepare_dataframe
from src.analysis import engagement_summary, topic_model
from src.charts import topic_keywords_figure

st.set_page_config(page_title="Topics & Insights", page_icon="🔎", layout="wide")

st.title("🔎 Topics & Audience Insights")
st.caption("Explore what audiences discuss, not only whether they are positive or negative.")

if not FINAL_FILE.exists():
    st.warning("Run Sentiment Analysis first.")
    st.stop()

df = pd.read_csv(FINAL_FILE)
df = prepare_dataframe(df)

if "transformer_sentiment" not in df.columns:
    st.warning("RoBERTa sentiment is required. Run Sentiment Analysis first.")
    st.stop()

if "like_count" in df.columns:
    engagement = engagement_summary(df)
    st.markdown("### Engagement by sentiment")
    st.dataframe(engagement.round(2), use_container_width=True, hide_index=True)

st.markdown("### Topic discovery")
n_topics = st.slider("Number of topics", 3, 8, 5)
topic_df, assignments = topic_model(df, n_topics=n_topics, top_words=8)

if topic_df.empty:
    st.warning("Not enough usable text for topic modelling.")
    st.stop()

st.plotly_chart(topic_keywords_figure(topic_df), use_container_width=True)

st.markdown("### Discovered keywords")
st.dataframe(topic_df, use_container_width=True, hide_index=True)

if "transformer_sentiment" in assignments.columns:
    topic_sentiment = (
        assignments.groupby(["topic", "transformer_sentiment"])
        .size()
        .reset_index(name="comments")
    )

    st.markdown("### Topic-level sentiment")
    st.dataframe(topic_sentiment, use_container_width=True, hide_index=True)

st.info(
    "Topic modelling discovers recurring word patterns. A topic should be interpreted "
    "using its top keywords and the underlying comments rather than treating the generated "
    "topic number as a semantic label by itself."
)
