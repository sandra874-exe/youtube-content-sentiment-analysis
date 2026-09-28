import os

from dotenv import load_dotenv
from googleapiclient.discovery import build

# Load environment variables from .env
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

# Request video information
request = youtube.videos().list(
    part="snippet,statistics",
    id=VIDEO_ID
)

response = request.execute()

if not response["items"]:
    print("Video not found.")
else:
    video = response["items"][0]

    snippet = video["snippet"]
    statistics = video["statistics"]

    print("\n--- YouTube API Test ---")
    print("Title:", snippet["title"])
    print("Channel:", snippet["channelTitle"])
    print("Published:", snippet["publishedAt"])
    print("Views:", statistics.get("viewCount", "N/A"))
    print("Likes:", statistics.get("likeCount", "N/A"))
    print("Comments:", statistics.get("commentCount", "N/A"))
