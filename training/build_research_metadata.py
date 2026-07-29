from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_FACES = PROJECT_ROOT / "processed_faces"
PROCESSED_AUDIO = PROJECT_ROOT / "processed_audio"
OUTPUT_DIR = PROJECT_ROOT / "processed_dataset"

OUTPUT_DIR.mkdir(exist_ok=True)

# ==========================================================
# LOAD AUDIO METADATA
# ==========================================================

audio_metadata = pd.read_csv(PROCESSED_AUDIO / "metadata.csv")

# ==========================================================
# BUILD AUDIO LOOKUP
# ==========================================================

audio_lookup = {}

for _, row in audio_metadata.iterrows():

    key = row["video_name"]

    audio_lookup[key] = {
        "audio_path": str(
            PROCESSED_AUDIO
            / row["label"]
            / row["audio_name"]
        ),
        "audio_exists": row["audio_exists"],
        "label": row["label"],
    }

# ==========================================================
# COLLECT FACE IMAGES
# ==========================================================

rows = []

sample_id = 0

for label in ["real", "fake"]:

    folder = PROCESSED_FACES / label

    images = sorted(folder.glob("*.jpg"))

    for image in images:

        sample_id += 1

        filename = image.stem

        # -------------------------------------
        # VIDEO ID
        #
        # 00001_30
        # ->
        # 00001
        #
        # 00109_id00476_wavtolip_60
        # ->
        # 00109
        # -------------------------------------

        video_id = filename.split("_")[0]

        audio_info = audio_lookup.get(
            video_id,
            {
                "audio_path": "",
                "audio_exists": False,
            },
        )

        rows.append(
            {
                "sample_id": sample_id,
                "video_id": video_id,
                "label": label,
                "image_path": str(image),
                "audio_path": audio_info["audio_path"],
                "audio_exists": audio_info["audio_exists"],
            }
        )

# ==========================================================
# CREATE DATAFRAME
# ==========================================================

df = pd.DataFrame(rows)

print("=" * 60)
print("DATASET CREATED")
print("=" * 60)

print(df.head())
print()

print("Total Samples :", len(df))

# ==========================================================
# TRAIN / VAL / TEST SPLIT
# ==========================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label"],
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"],
)

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

final_df = pd.concat(
    [
        train_df,
        val_df,
        test_df,
    ],
    ignore_index=True,
)

# ==========================================================
# SAVE
# ==========================================================

output_csv = OUTPUT_DIR / "research_dataset.csv"

final_df.to_csv(
    output_csv,
    index=False,
)

print()

print("=" * 60)
print("RESEARCH DATASET SAVED")
print("=" * 60)

print(output_csv)

print()

print(final_df["split"].value_counts())

print()

print(final_df["label"].value_counts())