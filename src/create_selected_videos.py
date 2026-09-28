import os
import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")

if not API_KEY:
    raise ValueError("YOUTUBE_API_KEY was not found in .env")

youtube = build(
    "youtube",
    "v3",
    developerKey=API_KEY
)

channels = [
    {
        "channel_id": "UCMiJRAwDNSNzuYeN2uWa0pA",
        "creator": "Mrwhosetheboss",
        "domain": "Technology"
    },
    {
        "channel_id": "UCnmGIkw-KdI0W5siakKPKog",
        "creator": "Ryan Trahan",
        "domain": "Travel"
    },
    {
        "channel_id": "UCvsTLkf9SiHt-3HPqq_2TFw",
        "creator": "Abram Engle",
        "domain": "Lifestyle"
    }
]

selected_videos = []

for channel in channels:

    print(f"\nFinding videos for {channel['creator']}...")

    channel_request = youtube.channels().list(
        part="contentDetails",
        id=channel["channel_id"]
    )

    channel_response = channel_request.execute()

    if not channel_response["items"]:
        print("Channel not found.")
        continue

    uploads_playlist = channel_response["items"][0][
        "contentDetails"
    ]["relatedPlaylists"]["uploads"]

    videos_request = youtube.playlistItems().list(
        part="snippet",
        playlistId=uploads_playlist,
        maxResults=5
    )

    videos_response = videos_request.execute()

    for item in videos_response["items"]:

        video_id = item["snippet"]["resourceId"]["videoId"]

        selected_videos.append({
            "domain": channel["domain"],
            "creator": channel["creator"],
            "video_id": video_id
        })

        print(
            item["snippet"]["title"],
            "->",
            video_id
        )

df = pd.DataFrame(selected_videos)

os.makedirs("config", exist_ok=True)

df.to_csv(
    "config/selected_videos.csv",
    index=False
)

print("\n===================================")
print("Selected videos created")
print("===================================")

print(df)

print("\nTotal videos:", len(df))