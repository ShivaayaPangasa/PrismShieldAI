"""
==============================================================
PrismShieldAI

Audio Extraction

Purpose
-------
1. Build the processed audio dataset.
2. Extract audio from uploaded videos for inference.

Used by

• Dataset Builder
• Streamlit
• Video Pipeline
• Webcam (future)
• Live Monitoring (future)

==============================================================
"""

from pathlib import Path
import subprocess
import tempfile

import pandas as pd
from tqdm import tqdm

from configs.settings import (
    DATASET_PATH,
    PROCESSED_AUDIO,
    FFMPEG_PATH,
)

# ==========================================================
# CONFIG
# ==========================================================

DEBUG = False

REAL_LIMIT = 10 if DEBUG else None
FAKE_LIMIT = 10 if DEBUG else None

# ==========================================================
# OUTPUT FOLDERS
# ==========================================================

REAL_OUTPUT = PROCESSED_AUDIO / "real"
FAKE_OUTPUT = PROCESSED_AUDIO / "fake"

REAL_OUTPUT.mkdir(parents=True, exist_ok=True)
FAKE_OUTPUT.mkdir(parents=True, exist_ok=True)

# ==========================================================
# FIND VIDEOS
# ==========================================================

real_videos = sorted(
    (DATASET_PATH / "RealVideo-RealAudio").rglob("*.mp4")
)

fake_videos = sorted(
    (DATASET_PATH / "FakeVideo-FakeAudio").rglob("*.mp4")
)

if REAL_LIMIT:
    real_videos = real_videos[:REAL_LIMIT]

if FAKE_LIMIT:
    fake_videos = fake_videos[:FAKE_LIMIT]

# ==========================================================
# METADATA
# ==========================================================

metadata = []

extracted = 0
failed = 0

# ==========================================================
# UNIQUE FILE NAME
# ==========================================================


def build_filename(video_path: Path):

    relative = video_path.relative_to(DATASET_PATH)

    parts = relative.parts

    filename = "_".join(parts[1:-1]) + "_" + video_path.stem + ".wav"

    return filename.replace("\\", "_").replace("/", "_")


# ==========================================================
# DATASET AUDIO EXTRACTION
# ==========================================================


def extract_audio(video_path: Path, label: str):

    global extracted
    global failed

    output_folder = REAL_OUTPUT if label == "real" else FAKE_OUTPUT

    output_name = build_filename(video_path)

    output_path = output_folder / output_name

    command = [
        str(FFMPEG_PATH),
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]

    subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    success = output_path.exists()

    if success:
        extracted += 1
    else:
        failed += 1

    metadata.append(
        {
            "audio_name": output_name,
            "label": label,
            "video_name": video_path.stem,
            "source_video": str(video_path),
            "audio_exists": success,
        }
    )


# ==========================================================
# REUSABLE VIDEO AUDIO EXTRACTION
# ==========================================================


def extract_audio_from_video(
    video_path,
    sample_rate=16000,
):
    """
    Extract audio from any uploaded video.

    Parameters
    ----------
    video_path : str | Path

    sample_rate : int

    Returns
    -------
    Path
        Temporary WAV file.
    """

    video_path = Path(video_path)

    if not video_path.exists():

        raise FileNotFoundError(
            f"Video not found:\n{video_path}"
        )

    temp_dir = Path(tempfile.gettempdir())

    output_audio = temp_dir / f"{video_path.stem}.wav"

    command = [
        str(FFMPEG_PATH),
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(output_audio),
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:

        raise RuntimeError(
            result.stderr.decode()
        )

    if not output_audio.exists():

        raise RuntimeError(
            "Audio extraction failed."
        )

    return output_audio


# ==========================================================
# DATASET BUILDER
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("AUDIO EXTRACTION")
    print("=" * 60)
    print(f"Real Videos : {len(real_videos)}")
    print(f"Fake Videos : {len(fake_videos)}")
    print()

    print("Processing REAL videos...")

    for video in tqdm(real_videos):
        extract_audio(video, "real")

    print("\nProcessing FAKE videos...")

    for video in tqdm(fake_videos):
        extract_audio(video, "fake")

    metadata_df = pd.DataFrame(metadata)

    csv_path = PROCESSED_AUDIO / "metadata.csv"

    metadata_df.to_csv(
        csv_path,
        index=False,
    )

    print("\n" + "=" * 60)
    print("AUDIO EXTRACTION COMPLETE")
    print("=" * 60)

    print(f"Extracted : {extracted}")
    print(f"Failed    : {failed}")
    print(f"Metadata  : {csv_path}")

    print("\nFirst 5 rows:\n")

    print(metadata_df.head())