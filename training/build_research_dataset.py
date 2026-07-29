import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================================
# PATHS
# ==========================================================

SOURCE = Path("processed_faces")
OUTPUT = Path("processed_dataset")

METADATA = OUTPUT / "metadata.csv"

# ==========================================================
# LOAD METADATA
# ==========================================================

df = pd.read_csv(METADATA)

print("=" * 60)
print("Loaded Metadata")
print("=" * 60)

print(df.head())
print()

# ==========================================================
# UNIQUE VIDEOS
# ==========================================================

videos = df["video_id"].unique()

train_videos, temp_videos = train_test_split(
    videos,
    test_size=0.30,
    random_state=42
)

val_videos, test_videos = train_test_split(
    temp_videos,
    test_size=0.50,
    random_state=42
)

print(f"Train Videos      : {len(train_videos)}")
print(f"Validation Videos : {len(val_videos)}")
print(f"Test Videos       : {len(test_videos)}")
print()

# ==========================================================
# CREATE OUTPUT FOLDERS
# ==========================================================

for split in ["train", "val", "test"]:
    for label in ["real", "fake"]:
        (OUTPUT / split / label).mkdir(parents=True, exist_ok=True)

# ==========================================================
# ASSIGN SPLIT
# ==========================================================

def get_split(video):

    if video in train_videos:
        return "train"

    if video in val_videos:
        return "val"

    return "test"

# ==========================================================
# COPY FILES
# ==========================================================

counts = {
    "train": {"real": 0, "fake": 0},
    "val": {"real": 0, "fake": 0},
    "test": {"real": 0, "fake": 0},
}

print("=" * 60)
print("Copying Images...")
print("=" * 60)

for _, row in df.iterrows():

    split = get_split(row["video_id"])

    label = row["label"]

    src = SOURCE / label / row["image_name"]

    dst = OUTPUT / split / label / row["image_name"]

    shutil.copy2(src, dst)

    counts[split][label] += 1

# ==========================================================
# FINAL REPORT
# ==========================================================

print()
print("=" * 60)
print("DATASET CREATED")
print("=" * 60)

for split in ["train", "val", "test"]:

    real = counts[split]["real"]
    fake = counts[split]["fake"]

    print(f"\n{split.upper()}")

    print(f"Real : {real}")
    print(f"Fake : {fake}")
    print(f"Total: {real + fake}")

print()

print("Finished Successfully!")