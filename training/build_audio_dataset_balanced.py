"""
==============================================================
PrismShieldAI

Balanced Audio Dataset Builder

Creates a balanced train / validation / test split
for audio classification.

Author:
Shivaaya
==============================================================
"""

from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = PROJECT_ROOT / "processed_dataset" / "audio_dataset.csv"

OUTPUT_CSV = PROJECT_ROOT / "processed_dataset" / "audio_dataset_balanced.csv"

print("=" * 60)
print("BUILDING BALANCED AUDIO DATASET")
print("=" * 60)

df = pd.read_csv(INPUT_CSV)

print()
print("Original Dataset")
print(df["label"].value_counts())

# ==========================================================
# Split by class
# ==========================================================

real_df = df[df["label"] == "real"].copy()
fake_df = df[df["label"] == "fake"].copy()

print()
print(f"Real Samples : {len(real_df)}")
print(f"Fake Samples : {len(fake_df)}")

# ==========================================================
# Shuffle
# ==========================================================

real_df = real_df.sample(
    frac=1,
    random_state=SEED,
).reset_index(drop=True)

fake_df = fake_df.sample(
    frac=1,
    random_state=SEED,
).reset_index(drop=True)

# ==========================================================
# REAL SPLIT
# ==========================================================

real_train, real_temp = train_test_split(
    real_df,
    test_size=150,
    random_state=SEED,
)

real_val, real_test = train_test_split(
    real_temp,
    test_size=76,
    random_state=SEED,
)

# ==========================================================
# FAKE SPLIT
# ==========================================================

fake_train = fake_df.iloc[:350].copy()
fake_val = fake_df.iloc[350:424].copy()
fake_test = fake_df.iloc[424:500].copy()

# ==========================================================
# Assign split labels
# ==========================================================

real_train["split"] = "train"
real_val["split"] = "val"
real_test["split"] = "test"

fake_train["split"] = "train"
fake_val["split"] = "val"
fake_test["split"] = "test"

# ==========================================================
# Merge
# ==========================================================

balanced_df = pd.concat(

    [

        real_train,
        real_val,
        real_test,

        fake_train,
        fake_val,
        fake_test,

    ],

    ignore_index=True,

)

balanced_df = balanced_df.sample(

    frac=1,

    random_state=SEED,

).reset_index(drop=True)

# ==========================================================
# Save
# ==========================================================

balanced_df.to_csv(

    OUTPUT_CSV,

    index=False,

)

print()
print("=" * 60)
print("BALANCED DATASET CREATED")
print("=" * 60)

print()
print("Split Distribution")
print(
    balanced_df.groupby(
        ["split", "label"]
    ).size()
)

print()

print("Total")
print(
    balanced_df["split"].value_counts()
)

print()

print("Saved To")
print(OUTPUT_CSV)