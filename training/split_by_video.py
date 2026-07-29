from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================================
# LOAD METADATA
# ==========================================================

metadata = pd.read_csv("processed_dataset/metadata.csv")

# ==========================================================
# GET UNIQUE VIDEOS
# ==========================================================

videos = metadata["video_id"].unique()

print("Total Videos:", len(videos))

# ==========================================================
# TRAIN / TEMP
# ==========================================================

train_videos, temp_videos = train_test_split(
    videos,
    test_size=0.30,
    random_state=42
)

# ==========================================================
# VALIDATION / TEST
# ==========================================================

val_videos, test_videos = train_test_split(
    temp_videos,
    test_size=0.50,
    random_state=42
)

print()

print("Train Videos:", len(train_videos))
print("Validation Videos:", len(val_videos))
print("Test Videos:", len(test_videos))