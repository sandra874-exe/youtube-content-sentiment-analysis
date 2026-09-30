import os
import re
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("YOUTUBE_API_KEY")

if not API_KEY:
    raise ValueError(
        "YOUTUBE_API_KEY was not found. "
        "Please add it to the .env file."
    )

youtube = build(
    "youtube",
    "v3",
    developerKey=API_KEY
)


# ============================================================
# VIDEO URL
# ============================================================

def extract_video_id(url):

    if not url:
        return None

    patterns = [
        r"(?:youtube\.com/watch\?v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([A-Za-z0-9_-]{11})"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            url
        )

        if match:
            return match.group(1)

    return None


# ============================================================
# CHANNEL URL
# ============================================================

def resolve_channel_url(url):

    if not url:
        return None

    url = url.strip()

    # --------------------------------------------------------
    # Direct channel ID
    # --------------------------------------------------------

    match = re.search(
        r"youtube\.com/channel/([A-Za-z0-9_-]+)",
        url
    )

    if match:
        return match.group(1)

    # --------------------------------------------------------
    # @handle
    # --------------------------------------------------------

    match = re.search(
        r"youtube\.com/@([^/?]+)",
        url
    )

    if match:

        handle = match.group(1).strip()

        # Remove accidental @
        handle = handle.lstrip("@")

        # First attempt: direct handle lookup
        try:

            response = youtube.channels().list(
                part="id,snippet,contentDetails,statistics",
                forHandle=handle
            ).execute()

            items = response.get(
                "items",
                []
            )

            if items:
                return items[0]["id"]

        except HttpError:
            pass

        # Second attempt: search
        try:

            response = youtube.search().list(
                part="snippet",
                q=f"@{handle}",
                type="channel",
                maxResults=10
            ).execute()

            items = response.get(
                "items",
                []
            )

            if items:

                # Prefer an exact-looking match
                for item in items:

                    snippet = item.get(
                        "snippet",
                        {}
                    )

                    channel_id = snippet.get(
                        "channelId"
                    )

                    channel_title = snippet.get(
                        "channelTitle",
                        ""
                    )

                    if channel_id:
                        return channel_id

        except HttpError:
            pass

    # --------------------------------------------------------
    # /c/customname
    # --------------------------------------------------------

    match = re.search(
        r"youtube\.com/c/([^/?]+)",
        url
    )

    if match:

        custom_name = match.group(1)

        try:

            response = youtube.search().list(
                part="snippet",
                q=custom_name,
                type="channel",
                maxResults=10
            ).execute()

            items = response.get(
                "items",
                []
            )

            if items:

                return items[0]["snippet"]["channelId"]

        except HttpError:
            pass

    # --------------------------------------------------------
    # /user/username
    # --------------------------------------------------------

    match = re.search(
        r"youtube\.com/user/([^/?]+)",
        url
    )

    if match:

        username = match.group(1)

        try:

            response = youtube.channels().list(
                part="id",
                forUsername=username
            ).execute()

            items = response.get(
                "items",
                []
            )

            if items:
                return items[0]["id"]

        except HttpError:
            pass

    return None


# ============================================================
# IDENTIFY URL
# ============================================================

def identify_youtube_url(url):

    if not url:
        return {
            "type": "unknown",
            "id": None,
            "url": url
        }

    url = url.strip()

    # Video first
    video_id = extract_video_id(url)

    if video_id:

        return {
            "type": "video",
            "id": video_id,
            "url": url
        }

    # Channel second
    channel_id = resolve_channel_url(url)

    if channel_id:

        return {
            "type": "channel",
            "id": channel_id,
            "url": url
        }

    return {
        "type": "unknown",
        "id": None,
        "url": url
    }


# ============================================================
# VIDEO INFORMATION
# ============================================================

def get_video_info(video_id):

    try:

        response = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=video_id
        ).execute()

        if not response.get("items"):
            return None

        item = response["items"][0]

        snippet = item.get(
            "snippet",
            {}
        )

        statistics = item.get(
            "statistics",
            {}
        )

        content_details = item.get(
            "contentDetails",
            {}
        )

        return {
            "video_id": video_id,
            "title": snippet.get(
                "title",
                ""
            ),
            "description": snippet.get(
                "description",
                ""
            ),
            "channel_id": snippet.get(
                "channelId",
                ""
            ),
            "channel_title": snippet.get(
                "channelTitle",
                ""
            ),
            "published_at": snippet.get(
                "publishedAt",
                ""
            ),
            "thumbnail": (
                snippet
                .get("thumbnails", {})
                .get("high", {})
                .get("url", "")
            ),
            "views": int(
                statistics.get(
                    "viewCount",
                    0
                )
            ),
            "likes": int(
                statistics.get(
                    "likeCount",
                    0
                )
            ),
            "comment_count": int(
                statistics.get(
                    "commentCount",
                    0
                )
            ),
            "duration": content_details.get(
                "duration",
                ""
            )
        }

    except HttpError as error:

        raise RuntimeError(
            f"YouTube API error while retrieving video: {error}"
        )


# ============================================================
# CHANNEL INFORMATION
# ============================================================

def get_channel_info(channel_id):

    try:

        response = youtube.channels().list(
            part="snippet,statistics,contentDetails",
            id=channel_id
        ).execute()

        if not response.get("items"):
            return None

        item = response["items"][0]

        snippet = item.get(
            "snippet",
            {}
        )

        statistics = item.get(
            "statistics",
            {}
        )

        content_details = item.get(
            "contentDetails",
            {}
        )

        return {
            "channel_id": channel_id,
            "title": snippet.get(
                "title",
                ""
            ),
            "description": snippet.get(
                "description",
                ""
            ),
            "published_at": snippet.get(
                "publishedAt",
                ""
            ),
            "thumbnail": (
                snippet
                .get("thumbnails", {})
                .get("high", {})
                .get("url", "")
            ),
            "subscribers": int(
                statistics.get(
                    "subscriberCount",
                    0
                )
            ),
            "views": int(
                statistics.get(
                    "viewCount",
                    0
                )
            ),
            "videos": int(
                statistics.get(
                    "videoCount",
                    0
                )
            ),
            "uploads_playlist_id": (
                content_details
                .get("relatedPlaylists", {})
                .get("uploads", "")
            )
        }

    except HttpError as error:

        raise RuntimeError(
            f"YouTube API error while retrieving channel: {error}"
        )


# ============================================================
# VIDEO COMMENTS
# ============================================================

def get_video_comments(
    video_id,
    max_comments=5000
):

    comments = []

    next_page_token = None

    try:

        while len(comments) < max_comments:

            remaining = (
                max_comments -
                len(comments)
            )

            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(
                    100,
                    remaining
                ),
                pageToken=next_page_token,
                textFormat="plainText"
            ).execute()

            for item in response.get(
                "items",
                []
            ):

                top_comment = (
                    item["snippet"]
                    ["topLevelComment"]
                    ["snippet"]
                )

                comment_id = (
                    item["snippet"]
                    ["topLevelComment"]
                    ["id"]
                )

                comments.append(
                    {
                        "comment_id": comment_id,
                        "video_id": video_id,
                        "author": top_comment.get(
                            "authorDisplayName",
                            ""
                        ),
                        "comment_text": top_comment.get(
                            "textDisplay",
                            ""
                        ),
                        "like_count": int(
                            top_comment.get(
                                "likeCount",
                                0
                            )
                        ),
                        "published_at": top_comment.get(
                            "publishedAt",
                            ""
                        ),
                        "updated_at": top_comment.get(
                            "updatedAt",
                            ""
                        )
                    }
                )

                if len(comments) >= max_comments:
                    break

            next_page_token = response.get(
                "nextPageToken"
            )

            if not next_page_token:
                break

        return comments

    except HttpError as error:

        error_text = str(error)

        if "commentsDisabled" in error_text:
            return []

        raise RuntimeError(
            f"YouTube API error while retrieving comments: {error}"
        )


# ============================================================
# CHANNEL VIDEOS
# ============================================================

def get_channel_videos(
    channel_id,
    max_videos=40
):

    channel_info = get_channel_info(
        channel_id
    )

    if not channel_info:
        return []

    playlist_id = channel_info.get(
        "uploads_playlist_id"
    )

    if not playlist_id:
        return []

    videos = []

    next_page_token = None

    try:

        while len(videos) < max_videos:

            remaining = (
                max_videos -
                len(videos)
            )

            response = youtube.playlistItems().list(
                part="snippet,contentDetails",
                playlistId=playlist_id,
                maxResults=min(
                    50,
                    remaining
                ),
                pageToken=next_page_token
            ).execute()

            for item in response.get(
                "items",
                []
            ):

                snippet = item.get(
                    "snippet",
                    {}
                )

                content_details = item.get(
                    "contentDetails",
                    {}
                )

                video_id = content_details.get(
                    "videoId"
                )

                if not video_id:
                    continue

                videos.append(
                    {
                        "video_id": video_id,
                        "title": snippet.get(
                            "title",
                            ""
                        ),
                        "published_at": snippet.get(
                            "publishedAt",
                            ""
                        ),
                        "thumbnail": (
                            snippet
                            .get("thumbnails", {})
                            .get("medium", {})
                            .get("url", "")
                        )
                    }
                )

                if len(videos) >= max_videos:
                    break

            next_page_token = response.get(
                "nextPageToken"
            )

            if not next_page_token:
                break

        return videos

    except HttpError as error:

        raise RuntimeError(
            f"YouTube API error while retrieving channel videos: {error}"
        )