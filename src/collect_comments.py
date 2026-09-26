import os

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build


# Load environment variables
load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")

if not API_KEY:
    raise ValueError("YOUTUBE_API_KEY was not found in .env")


# Create YouTube API client
youtube = build(
    "youtube",
    "v3",
    developerKey=API_KEY
)


# Test video
VIDEO_ID = "dQw4w9WgXcQ"


# Get comments
request = youtube.commentThreads().list(
    part="snippet",
    videoId=VIDEO_ID,
    maxResults=50,
    textFormat="plainText"
)

response = request.execute()


# Store comments
comments = []

for item in response["items"]:
    comment = item["snippet"]["topLevelComment"]["snippet"]

    comments.append({
        "video_id": VIDEO_ID,
        "comment_id": item["snippet"]["topLevelComment"]["id"],
        "comment_text": comment["textDisplay"],
        "like_count": comment["likeCount"],
        "published_at": comment["publishedAt"]
    })


# Convert to DataFrame
df = pd.DataFrame(comments)


# Save dataset
output_path = "data/raw/test_comments.csv"
df.to_csv(output_path, index=False)


print(f"Collected {len(df)} comments")
print(f"Saved dataset to: {output_path}")