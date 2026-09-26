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


# Load our selected videos
input_path = "config/selected_videos.csv"
videos_df = pd.read_csv(input_path)

video_ids = videos_df["video_id"].tolist()


# Request video information
request = youtube.videos().list(
    part="snippet,statistics,contentDetails",
    id=",".join(video_ids)
)

response = request.execute()


video_info = []

for item in response["items"]:

    snippet = item["snippet"]
    statistics = item["statistics"]
    content = item["contentDetails"]

    video_info.append({
        "video_id": item["id"],
        "title": snippet["title"],
        "channel_id": snippet["channelId"],
        "channel_title": snippet["channelTitle"],
        "published_at": snippet["publishedAt"],
        "views": statistics.get("viewCount", 0),
        "likes": statistics.get("likeCount", 0),
        "comment_count": statistics.get("commentCount", 0),
        "duration": content["duration"]
    })


metadata_df = pd.DataFrame(video_info)


# Add our manually assigned domain and creator
metadata_df = metadata_df.merge(
    videos_df,
    on="video_id",
    how="left"
)


# Reorder columns
metadata_df = metadata_df[
    [
        "domain",
        "creator",
        "video_id",
        "title",
        "channel_title",
        "channel_id",
        "published_at",
        "duration",
        "views",
        "likes",
        "comment_count"
    ]
]


# Save metadata
output_path = "data/raw/selected_video_metadata.csv"
metadata_df.to_csv(output_path, index=False)


print("\nSelected videos:")
print(metadata_df.to_string(index=False))

print(f"\nSaved metadata to: {output_path}")
