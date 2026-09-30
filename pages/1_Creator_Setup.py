import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import COMMENTS_FILE, VIDEOS_FILE, YOUTUBE_API_KEY
from src.collector import collect_creator

st.set_page_config(page_title="Creator Setup", page_icon="🎥", layout="wide")

st.title("🎥 Creator Setup")
st.caption("Collect YouTube creators, top videos, comments and engagement metadata.")

if not YOUTUBE_API_KEY:
    st.error("YOUTUBE_API_KEY is missing. Add it to .env in the project root.")
    st.stop()

with st.sidebar:
    st.markdown("### Collection controls")
    max_videos = st.slider("Top videos per creator", 5, 30, 20)
    comments_per_video = st.slider("Comments per video", 25, 100, 50, step=25)

DOMAIN_OPTIONS = [
    "Technology",
    "Travel",
    "Animals",
    "Gaming",
    "Vlogs",
    "Politics",
    "Other",
]

if "creator_rows" not in st.session_state:
    st.session_state.creator_rows = [
        {"creator": "", "domain": "Technology"}
    ]

for i, row in enumerate(st.session_state.creator_rows):
    c1, c2, c3 = st.columns([5, 3, 1])

    with c1:
        row["creator"] = st.text_input(
            f"Creator {i + 1}",
            value=row["creator"],
            placeholder="@Mrwhosetheboss",
            key=f"creator_{i}",
        )

    with c2:
        row["domain"] = st.selectbox(
            "Content domain",
            DOMAIN_OPTIONS,
            index=DOMAIN_OPTIONS.index(row["domain"]),
            key=f"domain_{i}",
        )

    with c3:
        if st.button("✕", key=f"remove_{i}"):
            if len(st.session_state.creator_rows) > 1:
                st.session_state.creator_rows.pop(i)
                st.rerun()

if st.button("＋ Add another creator"):
    st.session_state.creator_rows.append(
        {"creator": "", "domain": "Technology"}
    )
    st.rerun()

st.divider()

if st.button("🚀 Fetch YouTube Data", type="primary", use_container_width=True):
    creators = [
        row for row in st.session_state.creator_rows
        if row["creator"].strip()
    ]

    if not creators:
        st.warning("Enter at least one creator.")
        st.stop()

    all_videos = []
    all_comments = []
    overall_progress = st.progress(0)

    for creator_index, item in enumerate(creators, start=1):
        st.subheader(f"🎥 {item['creator']}")
        progress = st.progress(0)
        status = st.empty()

        try:
            videos, comments = collect_creator(
                item["creator"],
                item["domain"],
                YOUTUBE_API_KEY,
                max_videos=max_videos,
                comments_per_video=comments_per_video,
                progress_callback=progress.progress,
            )

            if not videos.empty:
                all_videos.append(videos)
            if not comments.empty:
                all_comments.append(comments)

            status.success(
                f"Collected {len(videos)} videos and {len(comments)} comments."
            )
        except Exception as error:
            status.error(str(error))

        overall_progress.progress(creator_index / len(creators))

    if not all_videos:
        st.error("No videos were collected.")
        st.stop()

    videos_df = pd.concat(all_videos, ignore_index=True).drop_duplicates("video_id")
    comments_df = (
        pd.concat(all_comments, ignore_index=True)
        if all_comments
        else pd.DataFrame()
    )

    if not comments_df.empty:
        comments_df = comments_df.drop_duplicates("comment_id")

    videos_df.to_csv(VIDEOS_FILE, index=False)
    comments_df.to_csv(COMMENTS_FILE, index=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Creators", len(creators))
    c2.metric("Videos", f"{len(videos_df):,}")
    c3.metric("Comments", f"{len(comments_df):,}")

    st.success("YouTube collection completed. Continue to Data Quality.")

    with st.expander("Collected videos", expanded=False):
        st.dataframe(videos_df, use_container_width=True, hide_index=True)
