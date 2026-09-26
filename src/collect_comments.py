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

input_path = "data/raw/selected_video_metadata.csv"

videos_df = pd.read_csv(input_path)

all_comments = []

for _, video in videos_df.iterrows():

    video_id = video["video_id"]

    print(f"\nCollecting comments for: {video['title']}")

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=50,
        textFormat="plainText"
    )

    response = request.execute()

    video_comments = 0

    for item in response.get("items", []):

        comment = item["snippet"]["topLevelComment"]["snippet"]

        all_comments.append({
            "domain": video["domain"],
            "creator": video["creator"],
            "video_id": video_id,
            "video_title": video["title"],
            "comment_id": item["snippet"]["topLevelComment"]["id"],
            "comment_text": comment["textDisplay"],
            "like_count": comment["likeCount"],
            "published_at": comment["publishedAt"]
        })

        video_comments += 1

    print(f"Collected: {video_comments} comments")

comments_df = pd.DataFrame(all_comments)

output_path = "data/raw/youtube_comments_750.csv"

comments_df.to_csv(
    output_path,
    index=False
)

print("\n===================================")
print("Collection complete")

print(f"Total comments: {len(comments_df)}")
print(f"Videos processed: {comments_df['video_id'].nunique()}")

print("\nComments by domain:")
print(comments_df["domain"].value_counts())

print(f"\nSaved to: {output_path}")