import os

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


CHANNEL_HANDLES = [
    "@Mrwhosetheboss",
    "@ryan",
    "@abrameng"
]


for handle in CHANNEL_HANDLES:

    request = youtube.channels().list(
        part="snippet,statistics,contentDetails",
        forHandle=handle
    )

    response = request.execute()

    if not response["items"]:
        print(f"\nChannel not found: {handle}")
        continue

    channel = response["items"][0]

    print("\n------------------------------")
    print("Handle:", handle)
    print("Channel:", channel["snippet"]["title"])
    print("Channel ID:", channel["id"])
    print("Subscribers:", channel["statistics"].get("subscriberCount"))
    print("Videos:", channel["statistics"].get("videoCount"))
    print("Views:", channel["statistics"].get("viewCount"))

