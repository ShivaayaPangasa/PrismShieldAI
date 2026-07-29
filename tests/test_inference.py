"""
==============================================================
PrismShieldAI

Inference Test

Verifies

✓ Preprocessor
✓ Predictor
✓ Fusion Model
✓ Inference Engine

==============================================================
"""

from pathlib import Path

from inference.inference import PrismShieldInference

import pandas as pd

DATASET = pd.read_csv(
    "processed_dataset/research_dataset.csv"
)

TEST_SAMPLE = DATASET[
    DATASET["split"] == "test"
].iloc[0]

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = PROJECT_ROOT / Path(
    TEST_SAMPLE["image_path"].replace("\\", "/")
)

AUDIO_PATH = PROJECT_ROOT / Path(
    TEST_SAMPLE["audio_path"].replace("\\", "/")
)

LABEL = TEST_SAMPLE["label"]

# ==========================================================
# MAIN
# ==========================================================

def main():

    print()

    print("=" * 60)
    print("PRISMSHIELDAI INFERENCE TEST")
    print("=" * 60)

    if not IMAGE_PATH.exists():

        raise FileNotFoundError(

            f"\nImage not found:\n{IMAGE_PATH}"

        )

    if not AUDIO_PATH.exists():

        raise FileNotFoundError(

            f"\nAudio not found:\n{AUDIO_PATH}"

        )

    print()

    print("Loading inference engine...")
    
    print()

    print("Ground Truth :", LABEL)
    
    print("Image Exists :", IMAGE_PATH.exists())
    
    print("Audio Exists :", AUDIO_PATH.exists())

    print("Image :", IMAGE_PATH)

    print("Audio :", AUDIO_PATH)

    engine = PrismShieldInference()

    print()

    print("Running inference...")
    
    result = engine.predict_from_image_audio(

    image_path=IMAGE_PATH,

    audio_path=AUDIO_PATH,
    
    )

    engine.print_result(result)

    print()

    print("=" * 60)
    print("RAW OUTPUT")
    print("=" * 60)

    print(result)

    print()

    print("=" * 60)
    print("TEST FINISHED")
    print("=" * 60)


if __name__ == "__main__":

    main()