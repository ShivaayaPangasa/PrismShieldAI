"""
PrismShieldAI Dataset Builder - Chunk 1
"""

from pathlib import Path
import subprocess
import hashlib
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "datasets" / "FakeAVCeleb_v1.2" / "FakeAVCeleb_v1.2"
OUTPUT_FACES = PROJECT_ROOT / "processed_faces"
OUTPUT_AUDIO = PROJECT_ROOT / "processed_audio"
OUTPUT_DATASET = PROJECT_ROOT / "processed_dataset"

for p in (OUTPUT_FACES, OUTPUT_AUDIO, OUTPUT_DATASET):
    p.mkdir(parents=True, exist_ok=True)

DATASET_NAME = "FakeAVCeleb"
FACES_PER_VIDEO = 5
MAX_AUDIO_SEGMENTS = 3
FRAME_INTERVAL = 30
IMAGE_SIZE = 224
AUDIO_SAMPLE_RATE = 16000
AUDIO_SEGMENT_LENGTH = 5.0
BLUR_THRESHOLD = 80.0

mp_face = mp.solutions.face_detection
FACE_DETECTOR = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.6)

def variance_of_laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def image_hash(image):
    return hashlib.md5(cv2.resize(image, (32,32)).tobytes()).hexdigest()

def is_duplicate(face_hash, seen_hashes):
    if face_hash in seen_hashes:
        return True
    seen_hashes.add(face_hash)
    return False

def get_video_duration(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return 0.0 if fps <= 0 else frames / fps

def extract_audio(video_path: Path, output_path: Path, start_time: float, duration: float, ffmpeg="ffmpeg") -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,"-y","-ss",str(start_time),"-i",str(video_path),
        "-t",str(duration),"-vn","-ac","1","-ar",str(AUDIO_SAMPLE_RATE),
        str(output_path)
    ]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return r.returncode == 0 and output_path.exists()

def create_metadata_record(**kwargs):
    return kwargs

if __name__ == "__main__":
    print("Chunk 1 ready")

"""
PrismShieldAI Dataset Builder - Chunk 2

Complete process_video() implementation.
Requires Chunk 1.
"""

import cv2
from pathlib import Path

# This file is intended to be appended after Chunk 1.

def process_video(
    video_path: Path,
    label: str,
    video_id: int,
    sample_id: int,
    metadata: list,
):

    output_face_dir = OUTPUT_FACES / label
    output_audio_dir = OUTPUT_AUDIO / label

    output_face_dir.mkdir(parents=True, exist_ok=True)
    output_audio_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    seen_hashes = set()
    candidates = []

    frame_number = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        if frame_number % FRAME_INTERVAL != 0:
            frame_number += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = FACE_DETECTOR.process(rgb)

        if results.detections:

            h, w = frame.shape[:2]

            for detection in results.detections:

                box = detection.location_data.relative_bounding_box

                x = max(0, int(box.xmin * w))
                y = max(0, int(box.ymin * h))
                bw = int(box.width * w)
                bh = int(box.height * h)

                face = frame[y:y+bh, x:x+bw]

                if face.size == 0:
                    continue

                score = variance_of_laplacian(face)

                if score < BLUR_THRESHOLD:
                    continue

                hsh = image_hash(face)

                if is_duplicate(hsh, seen_hashes):
                    continue

                face = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))

                candidates.append({
                    "face": face,
                    "blur": score,
                    "frame": frame_number,
                    "timestamp": frame_number / fps
                })

        frame_number += 1

    cap.release()

    if len(candidates) == 0:
        return sample_id

    candidates = sorted(
        candidates,
        key=lambda x: x["blur"],
        reverse=True
    )[:FACES_PER_VIDEO]

    saved_faces = []

    for idx, c in enumerate(candidates):

        image_name = f"{video_id:05d}_{c['frame']}.jpg"

        image_path = output_face_dir / image_name

        cv2.imwrite(str(image_path), c["face"])

        saved_faces.append({
            "face_index": idx,
            "image_name": image_name,
            "image_path": image_path,
            "frame": c["frame"],
            "timestamp": c["timestamp"],
            "blur_score": c["blur"]
        })

    duration = get_video_duration(video_path)

    if duration <= 0:
        return sample_id

    timestamps = [0.0]

    if MAX_AUDIO_SEGMENTS > 1 and duration > AUDIO_SEGMENT_LENGTH:
        usable = max(0.0, duration - AUDIO_SEGMENT_LENGTH)
        step = usable / (MAX_AUDIO_SEGMENTS - 1)
        timestamps = [round(i * step, 2) for i in range(MAX_AUDIO_SEGMENTS)]

    saved_audio = []

    for clip_idx, start in enumerate(timestamps):

        audio_name = f"{video_path.stem}_clip{clip_idx}.wav"

        audio_path = output_audio_dir / audio_name

        if extract_audio(
            video_path,
            audio_path,
            start,
            AUDIO_SEGMENT_LENGTH,
        ):

            saved_audio.append({
                "audio_clip_index": clip_idx,
                "audio_name": audio_name,
                "audio_path": audio_path,
                "start_time": start,
            })

    for face in saved_faces:
        for audio in saved_audio:

            metadata.append(create_metadata_record(

                sample_id=sample_id,
                video_id=video_id,
                dataset=DATASET_NAME,
                label=label,
                source_video=str(video_path.relative_to(DATASET_PATH)),

                face_index=face["face_index"],
                audio_clip_index=audio["audio_clip_index"],

                image_name=face["image_name"],
                image_path=str(Path("processed_faces")/label/face["image_name"]),

                frame=face["frame"],
                timestamp=face["timestamp"],
                face_blur_score=face["blur_score"],

                audio_name=audio["audio_name"],
                audio_path=str(Path("processed_audio")/label/audio["audio_name"]),
                audio_timestamp=audio["start_time"],

            ))

            sample_id += 1

    return sample_id

"""
PrismShieldAI Dataset Builder - Chunk 3

Main execution, dataset split, CSV generation,
verification and reporting.

Append after Chunk 2.
"""

from pathlib import Path

if __name__ == "__main__":

    print("=" * 60)
    print("BUILDING RESEARCH DATASET")
    print("=" * 60)

    metadata = []
    failed_videos = []

    sample_id = 0
    video_id = 0

    real_dirs = ["RealVideo-RealAudio"]
    fake_dirs = [
        "FakeVideo-FakeAudio",
        "FakeVideo-RealAudio",
        "RealVideo-FakeAudio",
    ]

    # -------------------------
    # REAL
    # -------------------------

    for folder in real_dirs:

        folder_path = DATASET_PATH / folder

        if not folder_path.exists():
            continue

        for video_path in sorted(folder_path.rglob("*.mp4")):

            try:

                sample_id = process_video(
                    video_path=video_path,
                    label="real",
                    video_id=video_id,
                    sample_id=sample_id,
                    metadata=metadata,
                )

                video_id += 1

            except Exception as e:

                failed_videos.append(
                    {
                        "video": str(video_path),
                        "error": str(e),
                    }
                )

    # -------------------------
    # FAKE
    # -------------------------

    for folder in fake_dirs:

        folder_path = DATASET_PATH / folder

        if not folder_path.exists():
            continue

        for video_path in sorted(folder_path.rglob("*.mp4")):

            try:

                sample_id = process_video(
                    video_path=video_path,
                    label="fake",
                    video_id=video_id,
                    sample_id=sample_id,
                    metadata=metadata,
                )

                video_id += 1

            except Exception as e:

                failed_videos.append(
                    {
                        "video": str(video_path),
                        "error": str(e),
                    }
                )

    # -------------------------
    # Build dataframe
    # -------------------------

    df = pd.DataFrame(metadata)

    if len(df) == 0:
        raise RuntimeError("No samples were generated.")

    unique_videos = df["video_id"].unique()

    train_videos, temp_videos = train_test_split(
        unique_videos,
        test_size=0.30,
        random_state=42,
        shuffle=True,
    )

    val_videos, test_videos = train_test_split(
        temp_videos,
        test_size=0.50,
        random_state=42,
        shuffle=True,
    )

    df["split"] = "train"

    df.loc[df.video_id.isin(val_videos), "split"] = "val"
    df.loc[df.video_id.isin(test_videos), "split"] = "test"
    
    # ==========================================================
    # BALANCE TRAINING SET (VIDEO LEVEL)
    # ==========================================================
    
    print()
    print("=" * 60)
    print("BALANCING TRAINING SET")
    print("=" * 60)

    # -----------------------------------
    # Separate training samples
    # -----------------------------------

    train_df = df[df["split"] == "train"].copy()

    val_df = df[df["split"] == "val"].copy()

    test_df = df[df["split"] == "test"].copy()

    # -----------------------------------
    # Unique training videos
    # -----------------------------------

    real_video_ids = (
        train_df[train_df["label"] == "real"]["video_id"]
        .unique()
    )

    fake_video_ids = (
        train_df[train_df["label"] == "fake"]["video_id"]
        .unique()
    )

    print(f"Original Real Videos : {len(real_video_ids)}")
    print(f"Original Fake Videos : {len(fake_video_ids)}")

    # -----------------------------------
    # Keep ALL real videos
    # -----------------------------------

    num_real = len(real_video_ids)

    # Keep only 2x fake videos

    num_fake_keep = min(
        len(fake_video_ids),
        num_real * 2,
    )

    np.random.seed(42)

    selected_fake = np.random.choice(

        fake_video_ids,

        size=num_fake_keep,

        replace=False,

    )

    selected_videos = np.concatenate([

        real_video_ids,

        selected_fake,

    ])

    balanced_train = train_df[
        train_df["video_id"].isin(selected_videos)
    ]

    # -----------------------------------
    # Merge back
    # -----------------------------------

    df = pd.concat(
        
        [

            balanced_train,

            val_df,

            test_df,

        ],

        ignore_index=True,

    )

    print()
    print("Balanced Training Samples")

    print(
    balanced_train["label"].value_counts()
    )
    
    print()

    print("Total Balanced Train Samples :")

    print(len(balanced_train))

    csv_path = OUTPUT_DATASET / "research_dataset.csv"
    df.to_csv(csv_path, index=False)

    if failed_videos:

        failed_path = OUTPUT_DATASET / "failed_videos.csv"

        pd.DataFrame(failed_videos).to_csv(
            failed_path,
            index=False,
        )

    # -------------------------
    # Verify paths
    # -------------------------

    missing_images = 0
    missing_audio = 0

    for _, row in df.iterrows():

        img = PROJECT_ROOT / row["image_path"]
        aud = PROJECT_ROOT / row["audio_path"]

        if not img.exists():
            missing_images += 1

        if not aud.exists():
            missing_audio += 1

    # -------------------------
    # Summary
    # -------------------------

    print()
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"Dataset           : {DATASET_NAME}")
    print(f"Videos Processed  : {video_id}")
    print(f"Samples Generated : {len(df)}")
    print(f"Train Samples     : {(df.split=='train').sum()}")
    print(f"Validation        : {(df.split=='val').sum()}")
    print(f"Test              : {(df.split=='test').sum()}")
    print(f"Missing Images    : {missing_images}")
    print(f"Missing Audio     : {missing_audio}")
    print(f"Failed Videos     : {len(failed_videos)}")

    print()
    print("CSV:")
    print(csv_path)

    print()
    print("Dataset builder finished successfully.")
