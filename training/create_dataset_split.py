from pathlib import Path
import pandas as pd

# ==========================================================
# PATHS
# ==========================================================

SOURCE_DATASET = Path("processed_faces")
OUTPUT_DATASET = Path("processed_dataset")

real_images = list((SOURCE_DATASET / "real").glob("*.jpg"))
fake_images = list((SOURCE_DATASET / "fake").glob("*.jpg"))

rows = []

# ==========================================================
# PROCESS REAL IMAGES
# ==========================================================

for image in real_images:

    video_id = "_".join(image.stem.split("_")[:-1])

    rows.append({
        "image_name": image.name,
        "label": "real",
        "video_id": video_id
    })

# ==========================================================
# PROCESS FAKE IMAGES
# ==========================================================

for image in fake_images:

    video_id = "_".join(image.stem.split("_")[:-1])

    rows.append({
        "image_name": image.name,
        "label": "fake",
        "video_id": video_id
    })

# ==========================================================
# SAVE CSV
# ==========================================================

df = pd.DataFrame(rows)

csv_path = OUTPUT_DATASET / "metadata.csv"

df.to_csv(csv_path, index=False)

print("=" * 60)
print("Metadata Created Successfully!")
print("=" * 60)

print(df.head())

print()

print("Total Images :", len(df))
print("CSV Saved To :", csv_path)