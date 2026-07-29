from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

audio_metadata = pd.read_csv(
    PROJECT_ROOT / "processed_audio" / "metadata.csv"
)

# ==========================================================
# CREATE UNIQUE VIDEO ID
# ==========================================================

def make_unique_video_id(row):

    audio_name = Path(row["audio_name"]).stem

    parts = audio_name.split("_")

    if "wavtolip" in audio_name or "faceswap" in audio_name:

        return "_".join(parts[:4])

    return "_".join(parts[:4])


audio_metadata["unique_video_id"] = audio_metadata.apply(
    make_unique_video_id,
    axis=1,
)

# ==========================================================
# VIDEO LEVEL SPLIT
# ==========================================================

videos = audio_metadata["unique_video_id"].unique()

train_videos, temp_videos = train_test_split(
    videos,
    test_size=0.30,
    random_state=42,
)

val_videos, test_videos = train_test_split(
    temp_videos,
    test_size=0.50,
    random_state=42,
)

audio_metadata["split"] = ""

audio_metadata.loc[
    audio_metadata.unique_video_id.isin(train_videos),
    "split",
] = "train"

audio_metadata.loc[
    audio_metadata.unique_video_id.isin(val_videos),
    "split",
] = "val"

audio_metadata.loc[
    audio_metadata.unique_video_id.isin(test_videos),
    "split",
] = "test"

# ==========================================================
# SAVE FULL DATASET
# ==========================================================

output_dir = PROJECT_ROOT / "processed_dataset"
output_dir.mkdir(exist_ok=True)

audio_metadata.to_csv(
    output_dir / "audio_dataset.csv",
    index=False,
)

# ==========================================================
# BALANCED TRAINING SET
# ==========================================================

train_df = audio_metadata[
    audio_metadata["split"] == "train"
]

real_train = train_df[
    train_df.label == "real"
]

fake_train = train_df[
    train_df.label == "fake"
]

fake_train = fake_train.sample(
    n=len(real_train),
    random_state=42,
)

balanced_train = pd.concat(
    [
        real_train,
        fake_train,
    ]
)

val_df = audio_metadata[
    audio_metadata["split"] == "val"
]

test_df = audio_metadata[
    audio_metadata["split"] == "test"
]

balanced = pd.concat(
    [
        balanced_train,
        val_df,
        test_df,
    ],
    ignore_index=True,
)

balanced.to_csv(
    output_dir / "audio_dataset_balanced.csv",
    index=False,
)

# ==========================================================
# SUMMARY
# ==========================================================

print("=" * 60)
print("AUDIO DATASET CREATED")
print("=" * 60)

print()

print("FULL DATASET")

print(audio_metadata.groupby(["split", "label"]).size())

print()

print("BALANCED DATASET")

print(balanced.groupby(["split", "label"]).size())

print()

print("Saved:")

print(output_dir / "audio_dataset.csv")

print(output_dir / "audio_dataset_balanced.csv")