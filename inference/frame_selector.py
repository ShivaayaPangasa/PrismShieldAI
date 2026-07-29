"""
==============================================================
PrismShieldAI

frame_selector.py

Universal Video Frame Selection

Pipeline

Video
    ↓
Frame Sampling
    ↓
Face Detection
    ↓
Quality Scoring
    ↓
Best Representative Frame

Used by

• Streamlit
• Video Inference
• Webcam
• Live Monitoring
• OS Daemon

==============================================================
"""

# ==========================================================
# IMPORTS
# ==========================================================

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


class FrameSelector:
    """
    Selects the best representative frame from a video.

    Strategy

    1. Read video
    2. Sample every N frames
    3. Detect largest face
    4. Compute quality score
    5. Return highest scoring frame
    """

    def __init__(

        self,

        frame_skip: int = 30,

        face_confidence: float = 0.50,

        padding: float = 0.15,

    ):

        self.frame_skip = frame_skip

        self.padding = padding

        self.face_detector = (

            mp.solutions.face_detection.FaceDetection(

                model_selection=1,

                min_detection_confidence=face_confidence,

            )

        )

    # ======================================================
    # FACE EXTRACTION
    # ======================================================

    def detect_face(

        self,

        frame,

    ):

        """
        Returns

        face
        bbox
        confidence

        or

        None
        """

        height, width = frame.shape[:2]

        rgb = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2RGB,

        )

        results = self.face_detector.process(rgb)

        if not results.detections:

            return None

        largest_face = None

        largest_area = 0

        for detection in results.detections:

            bbox = detection.location_data.relative_bounding_box

            score = detection.score[0]

            x = int(bbox.xmin * width)
            y = int(bbox.ymin * height)
            w = int(bbox.width * width)
            h = int(bbox.height * height)

            area = w * h

            if area > largest_area:

                largest_area = area

                largest_face = (

                    x,
                    y,
                    w,
                    h,
                    score,

                )

        if largest_face is None:

            return None

        x, y, w, h, score = largest_face

        pad_x = int(w * self.padding)
        pad_y = int(h * self.padding)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)

        x2 = min(width, x + w + pad_x)
        y2 = min(height, y + h + pad_y)

        face = frame[

            y1:y2,

            x1:x2,

        ]

        if face.size == 0:

            return None

        return {

            "face": face,

            "bbox": (x1, y1, x2, y2),

            "confidence": score,

            "area": largest_area,

        }

    # ======================================================
    # QUALITY SCORE
    # ======================================================
    
    def score_frame(

        self,

        face_info,

    ):

        """
        Improved quality score.

        Uses:
            • Face detector confidence
            • Face size
            • Image sharpness
        """

        face = face_info["face"]

        gray = cv2.cvtColor(

            face,

            cv2.COLOR_BGR2GRAY,

        )

        sharpness = cv2.Laplacian(

            gray,

            cv2.CV_64F,

        ).var()

        area_score = face_info["area"] / 100000

        confidence_score = face_info["confidence"]

        sharpness_score = min(

            sharpness / 500,

            1.0,

        )

        final_score = (

            0.5 * confidence_score +

            0.3 * area_score +

            0.2 * sharpness_score

        )

        return final_score

    # ======================================================
    # MAIN FUNCTION
    # ======================================================

    def select_best_frame(

        self,

        video_path,

    ):

        """
        Returns

        {

            "frame",
            "face",
            "bbox",
            "confidence",
            "frame_index",
            "score"

        }

        or

        None
        """

        video_path = Path(video_path)

        if not video_path.exists():

            raise FileNotFoundError(

                f"Video not found:\n{video_path}"

            )

        cap = cv2.VideoCapture(

            str(video_path)

        )

        best_result = None

        best_score = -1

        frame_index = 0

        while True:

            success, frame = cap.read()

            if not success:

                break

            if frame_index % self.frame_skip != 0:

                frame_index += 1

                continue

            face_info = self.detect_face(

                frame

            )

            if face_info is None:

                frame_index += 1

                continue

            score = self.score_frame(

                face_info

            )

            if score > best_score:

                best_score = score

                best_result = {

                    "frame": frame.copy(),

                    "face": face_info["face"],

                    "bbox": face_info["bbox"],

                    "confidence": face_info["confidence"],

                    "frame_index": frame_index,

                    "score": score,

                }

            frame_index += 1

        cap.release()

        return best_result

    # ======================================================
    # TOP FRAME SELECTION
    # ======================================================

    def select_top_frames(

        self,

        video_path,

        top_k=8,

    ):

        """
        Returns the Top-K highest quality frames.
        """

        video_path = Path(video_path)

        if not video_path.exists():

            raise FileNotFoundError(

                f"Video not found:\n{video_path}"

            )

        cap = cv2.VideoCapture(

            str(video_path)

        )

        frame_index = 0

        candidates = []

        while True:

            success, frame = cap.read()

            if not success:

                break

            if frame_index % self.frame_skip != 0:

                frame_index += 1

                continue

            face_info = self.detect_face(

                frame

            )

            if face_info is not None:

                score = self.score_frame(

                    face_info

                )

                candidates.append(

                    {

                        "frame": frame.copy(),

                        "face": face_info["face"],

                        "bbox": face_info["bbox"],

                        "confidence": face_info["confidence"],

                        "frame_index": frame_index,

                        "score": score,

                    }

                )

            frame_index += 1

        cap.release()

        candidates.sort(

            key=lambda x: x["score"],

            reverse=True,

        )

        return candidates[:top_k]

# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    VIDEO = "sample.mp4"

    selector = FrameSelector()

    result = selector.select_best_frame(VIDEO)

    if result is None:

        print()

        print("=" * 60)
        print("NO FACE DETECTED")
        print("=" * 60)

    else:

        print()

        print("=" * 60)
        print("BEST FRAME FOUND")
        print("=" * 60)

        print("Frame Index :", result["frame_index"])
        print("Confidence  :", round(result["confidence"], 3))
        print("Score       :", round(result["score"], 2))

        cv2.imwrite(

            "selected_face.jpg",

            result["face"],

        )

        print()

        print("Saved : selected_face.jpg")