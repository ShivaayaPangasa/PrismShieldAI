import cv2
import mediapipe as mp
from pathlib import Path

# ==================================================
# CONFIG
# ==================================================

DATASET_PATH = Path(
    "datasets/FakeAVCeleb_v1.2/FakeAVCeleb_v1.2"
)

OUTPUT_REAL = Path("processed_faces/real")
OUTPUT_FAKE = Path("processed_faces/fake")

OUTPUT_REAL.mkdir(parents=True, exist_ok=True)
OUTPUT_FAKE.mkdir(parents=True, exist_ok=True)

MAX_REAL_VIDEOS = 100
MAX_FAKE_VIDEOS = 100

FRAME_SKIP = 30

# ==================================================
# MEDIAPIPE
# ==================================================

mp_face = mp.solutions.face_detection

face_detector = mp_face.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)

# ==================================================
# GET VIDEOS
# ==================================================

real_videos = list(
    (DATASET_PATH / "RealVideo-RealAudio").rglob("*.mp4")
)

fake_videos = list(
    (DATASET_PATH / "FakeVideo-FakeAudio").rglob("*.mp4")
)

real_videos = real_videos[:MAX_REAL_VIDEOS]
fake_videos = fake_videos[:MAX_FAKE_VIDEOS]

print(f"Real Videos Selected: {len(real_videos)}")
print(f"Fake Videos Selected: {len(fake_videos)}")

# ==================================================
# PROCESS FUNCTION
# ==================================================

def process_videos(video_list, output_dir, label):

    saved_faces = 0

    for video_idx, video_path in enumerate(video_list):

        print(
            f"[{label}] "
            f"{video_idx+1}/{len(video_list)} "
            f"{video_path.name}"
        )

        cap = cv2.VideoCapture(str(video_path))

        frame_number = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            if frame_number % FRAME_SKIP == 0:

                h, w, _ = frame.shape

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                results = face_detector.process(rgb)

                if results.detections:

                    detection = results.detections[0]

                    bbox = (
                        detection
                        .location_data
                        .relative_bounding_box
                    )

                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)

                    bw = int(bbox.width * w)
                    bh = int(bbox.height * h)

                    x = max(0, x)
                    y = max(0, y)

                    face = frame[
                        y:y+bh,
                        x:x+bw
                    ]

                    if face.size > 0:

                        save_path = (
                            output_dir /
                            f"{video_path.stem}_{frame_number}.jpg"
                        )

                        cv2.imwrite(
                            str(save_path),
                            face
                        )

                        saved_faces += 1

            frame_number += 1

        cap.release()

    return saved_faces

# ==================================================
# RUN
# ==================================================

print("\nProcessing REAL videos...\n")

real_faces = process_videos(
    real_videos,
    OUTPUT_REAL,
    "REAL"
)

print("\nProcessing FAKE videos...\n")

fake_faces = process_videos(
    fake_videos,
    OUTPUT_FAKE,
    "FAKE"
)

print("\n====================")
print("DATASET COMPLETE")
print("====================")

print("Real Faces:", real_faces)
print("Fake Faces:", fake_faces)