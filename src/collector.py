import pandas as pd

from .youtube_api import find_channel, get_top_videos, get_comments


def collect_creator(
    creator_input,
    domain,
    api_key,
    max_videos=20,
    comments_per_video=50,
    progress_callback=None,
):
    channel = find_channel(creator_input, api_key)
    if channel is None:
        raise ValueError(f"Could not find channel: {creator_input}")

    channel_id = channel["id"]
    channel_name = channel["snippet"]["title"]

    videos = get_top_videos(
        channel_id,
        api_key,
        max_videos=max_videos,
    )

    video_rows = []
    comment_rows = []

    total_videos = max(len(videos), 1)

    for index, video in enumerate(videos, start=1):
        video_id = video["id"]
        snippet = video.get("snippet", {})
        statistics = video.get("statistics", {})

        title = snippet.get("title", "")
        views = int(statistics.get("viewCount", 0))
        likes = int(statistics.get("likeCount", 0))
        comment_count = int(statistics.get("commentCount", 0))

        video_rows.append({
            "creator": channel_name,
            "channel_id": channel_id,
            "domain": domain,
            "video_id": video_id,
            "video_title": title,
            "published_at": snippet.get("publishedAt", ""),
            "video_url": f"https://www.youtube.com/watch?v={video_id}",
            "views": views,
            "likes": likes,
            "comment_count": comment_count,
        })

        comments = get_comments(
            video_id,
            api_key,
            max_comments=comments_per_video,
        )

        for comment in comments:
            comment.update({
                "creator": channel_name,
                "domain": domain,
                "video_title": title,
                "video_views": views,
                "video_likes": likes,
                "video_comment_count": comment_count,
            })
            comment_rows.append(comment)

        if progress_callback:
            progress_callback(index / total_videos)

    videos_df = pd.DataFrame(video_rows)
    comments_df = pd.DataFrame(comment_rows)

    if not videos_df.empty:
        videos_df = videos_df.drop_duplicates(subset=["video_id"])
    if not comments_df.empty:
        comments_df = comments_df.drop_duplicates(subset=["comment_id"])

    return videos_df, comments_df
