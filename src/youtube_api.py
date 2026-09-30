import requests

BASE_URL = "https://www.googleapis.com/youtube/v3"


class YouTubeAPIError(Exception):
    pass


def youtube_get(endpoint, params, api_key):
    if not api_key:
        raise YouTubeAPIError("YOUTUBE_API_KEY is missing.")

    request_params = dict(params)
    request_params["key"] = api_key

    response = requests.get(
        f"{BASE_URL}/{endpoint}",
        params=request_params,
        timeout=30,
    )

    if response.status_code != 200:
        try:
            message = response.json().get("error", {}).get("message", "Unknown API error")
        except Exception:
            message = response.text
        raise YouTubeAPIError(message)

    return response.json()


def find_channel(creator_input, api_key):
    value = creator_input.strip()

    for prefix in (
        "https://www.youtube.com/",
        "http://www.youtube.com/",
        "www.youtube.com/",
    ):
        value = value.replace(prefix, "")

    if value.startswith("@"):
        data = youtube_get(
            "channels",
            {"part": "snippet,statistics", "forHandle": value},
            api_key,
        )
        if data.get("items"):
            return data["items"][0]

    data = youtube_get(
        "search",
        {"part": "snippet", "q": value, "type": "channel", "maxResults": 1},
        api_key,
    )

    if not data.get("items"):
        return None

    channel_id = data["items"][0]["snippet"]["channelId"]
    channel_data = youtube_get(
        "channels",
        {"part": "snippet,statistics", "id": channel_id},
        api_key,
    )

    return channel_data["items"][0] if channel_data.get("items") else None


def get_top_videos(channel_id, api_key, max_videos=20, scan_limit=100):
    channel_data = youtube_get(
        "channels",
        {"part": "contentDetails", "id": channel_id},
        api_key,
    )

    if not channel_data.get("items"):
        return []

    uploads_playlist_id = (
        channel_data["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
    )

    video_ids = []
    next_page_token = None

    while len(video_ids) < scan_limit:
        params = {
            "part": "contentDetails",
            "playlistId": uploads_playlist_id,
            "maxResults": 50,
        }
        if next_page_token:
            params["pageToken"] = next_page_token

        data = youtube_get("playlistItems", params, api_key)

        for item in data.get("items", []):
            video_id = item.get("contentDetails", {}).get("videoId")
            if video_id:
                video_ids.append(video_id)

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

    video_ids = list(dict.fromkeys(video_ids))[:scan_limit]
    if not video_ids:
        return []

    videos = []
    for start in range(0, len(video_ids), 50):
        batch = video_ids[start:start + 50]
        data = youtube_get(
            "videos",
            {
                "part": "snippet,statistics,contentDetails",
                "id": ",".join(batch),
            },
            api_key,
        )
        videos.extend(data.get("items", []))

    videos.sort(
        key=lambda video: int(video.get("statistics", {}).get("viewCount", 0)),
        reverse=True,
    )

    return videos[:max_videos]


def get_comments(video_id, api_key, max_comments=100):
    try:
        data = youtube_get(
            "commentThreads",
            {
                "part": "snippet",
                "videoId": video_id,
                "maxResults": min(max_comments, 100),
                "order": "relevance",
                "textFormat": "plainText",
            },
            api_key,
        )
    except YouTubeAPIError:
        return []

    comments = []
    for item in data.get("items", []):
        try:
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "comment_id": item["snippet"]["topLevelComment"]["id"],
                "video_id": video_id,
                "author": snippet.get("authorDisplayName", ""),
                "comment": snippet.get("textDisplay", ""),
                "like_count": int(snippet.get("likeCount", 0)),
                "published_at": snippet.get("publishedAt", ""),
            })
        except (KeyError, TypeError, ValueError):
            continue

    return comments[:max_comments]
