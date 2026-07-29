import cv2
import mediapipe as mp
from pathlib import Path

# -------------------------
# CONFIG
# -------------------------

DATASET_PATH = Path(
    "datasets/FakeAVCeleb_v1.2/FakeAVCeleb_v1.2"
)

OUTPUT_REAL = Path("processed_faces/real")
OUTPUT_FAKE = Path("processed_faces/fake")

OUTPUT_REAL.mkdir(parents=True, exist_ok=True)
OUTPUT_FAKE.mkdir(parents=True, exist_ok=True)

MAX_VIDEOS = 20  # Start small

# -------------------------
# MEDIAPIPE
# -------------------------

mp_face = mp.solutions.face_detection

face_detector = mp_face.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)

# -------------------------
# GET VIDEOS
# -------------------------

videos = list(DATASET_PATH.rglob("*.mp4"))

print(f"Found {len(videos)} videos")

saved_faces = 0

# -------------------------
# PROCESS
# -------------------------

for video_idx, video_path in enumerate(videos[:MAX_VIDEOS]):

    print(f"\nProcessing {video_idx+1}/{MAX_VIDEOS}")
    print(video_path.name)

    cap = cv2.VideoCapture(str(video_path))

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # every 30th frame
        if frame_number % 30 == 0:

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

                face = frame[
                    max(0, y):y+bh,
                    max(0, x):x+bw
                ]

                if face.size > 0:

                    # Label
                    if "RealVideo-RealAudio" in str(video_path):
                        output_dir = OUTPUT_REAL
                    else:
                        output_dir = OUTPUT_FAKE

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

print("\nDone!")
print("Faces Saved:", saved_faces)