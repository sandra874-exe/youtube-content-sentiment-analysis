from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from .youtube_api import (
    get_channel_info,
    get_channel_videos,
    get_video_comments,
    get_video_info,
    identify_youtube_url,
)
from .sentiment_analyzer import (
    extract_keywords,
    model_agreement,
    run_complete_analysis,
)


def _resolve_target(url: str) -> Tuple[dict, List[dict]]:
    detected = identify_youtube_url(url)

    if not detected or detected.get("type") == "unknown":
        raise ValueError("Could not identify this YouTube URL.")

    if detected["type"] == "channel":
        channel = get_channel_info(detected["id"])
        if not channel:
            raise ValueError("Could not retrieve the YouTube channel.")

        videos = get_channel_videos(detected["id"], max_videos=20)
        if not videos:
            raise ValueError("No videos were found for this channel.")

        return channel, videos

    video = get_video_info(detected["id"])
    if not video:
        raise ValueError("Could not retrieve the YouTube video.")

    channel_id = video.get("channel_id")
    channel = get_channel_info(channel_id) if channel_id else None

    if not channel:
        channel = {
            "channel_id": channel_id or "",
            "title": video.get("channel_title", "Unknown creator"),
            "subscribers": 0,
            "views": 0,
            "videos": 1,
        }

    return channel, [{
        "video_id": video["video_id"],
        "title": video.get("title", ""),
        "published_at": video.get("published_at", ""),
        "thumbnail": video.get("thumbnail", ""),
        "_single_video_info": video,
    }]


def collect_creator_comparison(
    url: str,
    max_videos: int = 5,
    comments_per_video: int = 50,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Collect a balanced sample for one creator.

    The same max_videos and comments_per_video settings are applied to
    every creator so the resulting comparison is descriptive and balanced.
    """

    channel, candidate_videos = _resolve_target(url)
    selected_videos = candidate_videos[:max_videos]

    creator_name = channel.get("title", "Unknown creator")
    channel_id = channel.get("channel_id", "")

    video_rows = []
    comment_rows = []

    total_videos = max(len(selected_videos), 1)

    for index, video in enumerate(selected_videos, start=1):
        video_id = video["video_id"]

        info = video.get("_single_video_info")
        if info is None:
            info = get_video_info(video_id) or {}

        title = info.get("title", video.get("title", "Untitled video"))

        comments = get_video_comments(
            video_id,
            max_comments=comments_per_video,
        )

        video_views = int(info.get("views", 0) or 0)
        video_likes = int(info.get("likes", 0) or 0)
        video_comment_count = int(info.get("comment_count", 0) or 0)

        video_rows.append({
            "creator": creator_name,
            "channel_id": channel_id,
            "video_id": video_id,
            "video_title": title,
            "published_at": info.get(
                "published_at",
                video.get("published_at", ""),
            ),
            "video_url": f"https://www.youtube.com/watch?v={video_id}",
            "video_views": video_views,
            "video_likes": video_likes,
            "video_comment_count": video_comment_count,
        })

        for comment in comments:
            comment_rows.append({
                **comment,
                "creator": creator_name,
                "channel_id": channel_id,
                "video_title": title,
                "video_views": video_views,
                "video_likes": video_likes,
                "video_comment_count": video_comment_count,
            })

        if progress_callback:
            progress_callback(index / total_videos)

    videos_df = pd.DataFrame(video_rows)
    comments_df = pd.DataFrame(comment_rows)

    if not videos_df.empty:
        videos_df = videos_df.drop_duplicates("video_id")

    if not comments_df.empty:
        comments_df = comments_df.drop_duplicates("comment_id")

    return channel, videos_df, comments_df


def _primary_column(df: pd.DataFrame) -> str:
    if "primary_sentiment" in df.columns:
        return "primary_sentiment"
    if "transformer_sentiment" in df.columns:
        return "transformer_sentiment"
    return "vader_sentiment"


def _creator_metrics(df: pd.DataFrame, creator: str, videos_df: pd.DataFrame) -> dict:
    column = _primary_column(df)
    counts = (
        df[column]
        .value_counts()
        .reindex(["positive", "neutral", "negative"], fill_value=0)
    )

    likes = pd.to_numeric(df.get("like_count", 0), errors="coerce").fillna(0)

    return {
        "creator": creator,
        "comments_analyzed": int(len(df)),
        "videos_analyzed": int(videos_df["video_id"].nunique()),
        "positive_pct": round(float(counts["positive"] / len(df) * 100), 2) if len(df) else 0,
        "neutral_pct": round(float(counts["neutral"] / len(df) * 100), 2) if len(df) else 0,
        "negative_pct": round(float(counts["negative"] / len(df) * 100), 2) if len(df) else 0,
        "average_comment_likes": round(float(likes.mean()), 2) if len(df) else 0,
        "median_comment_likes": round(float(likes.median()), 2) if len(df) else 0,
        "total_comment_likes": int(likes.sum()),
        "average_video_views": round(float(videos_df["video_views"].mean()), 2) if not videos_df.empty else 0,
        "average_video_likes": round(float(videos_df["video_likes"].mean()), 2) if not videos_df.empty else 0,
        "model_agreement_pct": model_agreement(df),
    }


def _video_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    column = _primary_column(df)

    grouped = (
        df.groupby(["creator", "video_id", "video_title", column])
        .size()
        .reset_index(name="comments")
    )

    if grouped.empty:
        return pd.DataFrame()

    pivot = (
        grouped.pivot_table(
            index=["creator", "video_id", "video_title"],
            columns=column,
            values="comments",
            fill_value=0,
        )
        .reset_index()
    )

    for sentiment in ["positive", "neutral", "negative"]:
        if sentiment not in pivot.columns:
            pivot[sentiment] = 0

    pivot["total"] = (
        pivot["positive"] + pivot["neutral"] + pivot["negative"]
    )

    for sentiment in ["positive", "neutral", "negative"]:
        pivot[f"{sentiment}_pct"] = (
            pivot[sentiment] / pivot["total"].replace(0, pd.NA) * 100
        ).fillna(0).round(2)

    return pivot


def _keyword_table(df: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    rows = []

    for creator, creator_df in df.groupby("creator"):
        column = _primary_column(creator_df)
        keywords = extract_keywords(
            creator_df,
            sentiment=None,
            top_n=top_n,
            sentiment_column=column,
        )

        for rank, row in keywords.iterrows():
            rows.append({
                "creator": creator,
                "rank": int(rank + 1),
                "keyword": row["word"],
                "frequency": int(row["frequency"]),
            })

    return pd.DataFrame(rows)


def compare_creators(
    urls: List[str],
    max_videos: int = 5,
    comments_per_video: int = 50,
    transformer_classifier=None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, pd.DataFrame]:
    """Collect, analyze and compare 2–3 YouTube creators."""

    cleaned_urls = [url.strip() for url in urls if url and url.strip()]

    if not 2 <= len(cleaned_urls) <= 3:
        raise ValueError("Enter exactly 2 or 3 YouTube creator links.")

    all_comments = []
    all_videos = []
    creator_info = []

    for index, url in enumerate(cleaned_urls, start=1):
        if progress_callback:
            progress_callback(
                index,
                f"Collecting creator {index} of {len(cleaned_urls)}",
            )

        channel, videos_df, comments_df = collect_creator_comparison(
            url,
            max_videos=max_videos,
            comments_per_video=comments_per_video,
        )

        if comments_df.empty:
            raise ValueError(
                f"No comments were collected for {channel.get('title', 'this creator')}."
            )

        analyzed = run_complete_analysis(
            comments_df,
            transformer_classifier=transformer_classifier,
            use_transformer=True,
        )

        analyzed_comments = analyzed["comments"].copy()

        if not analyzed_comments.empty:
            all_comments.append(analyzed_comments)

        if not videos_df.empty:
            all_videos.append(videos_df)

        creator_info.append({
            "creator": channel.get("title", "Unknown creator"),
            "channel_id": channel.get("channel_id", ""),
            "subscribers": int(channel.get("subscribers", 0) or 0),
            "channel_views": int(channel.get("views", 0) or 0),
            "channel_videos": int(channel.get("videos", 0) or 0),
        })

    comments = pd.concat(all_comments, ignore_index=True)
    videos = pd.concat(all_videos, ignore_index=True)

    metrics_rows = []
    for creator in comments["creator"].dropna().unique():
        creator_comments = comments[comments["creator"] == creator]
        creator_videos = videos[videos["creator"] == creator]
        metrics_rows.append(
            _creator_metrics(
                creator_comments,
                creator,
                creator_videos,
            )
        )

    metrics = pd.DataFrame(metrics_rows)

    creator_info_df = pd.DataFrame(creator_info)
    metrics = metrics.merge(
        creator_info_df,
        on="creator",
        how="left",
    )

    return {
        "metrics": metrics,
        "comments": comments,
        "videos": videos,
        "video_sentiment": _video_sentiment(comments),
        "keywords": _keyword_table(comments),
        "creator_info": creator_info_df,
    }
