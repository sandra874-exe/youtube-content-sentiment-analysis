from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models" / "saved"

for directory in (RAW_DIR, PROCESSED_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env")

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")

VIDEOS_FILE = PROCESSED_DIR / "youtube_videos.csv"
COMMENTS_FILE = PROCESSED_DIR / "youtube_comments.csv"
FINAL_FILE = PROCESSED_DIR / "final_analysis_dataset.csv"

# Prefer the user's 150-row file when it exists.
LABELLED_DATASET_CANDIDATES = [
    RAW_DIR / "human_labelled_150.csv",
    RAW_DIR / "human_labelled_120.csv",
    RAW_DIR / "vader_validation_sample.csv",
]

TRANSFORMER_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
VADER_POSITIVE_THRESHOLD = 0.05
VADER_NEGATIVE_THRESHOLD = -0.05
