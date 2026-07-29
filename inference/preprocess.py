"""
==============================================================
PrismShieldAI

preprocess.py

Production Preprocessing Pipeline

Supports

• Images
• Audio
• Videos
• Webcam Frames

Used by

• Streamlit
• Predictor
• OS Daemon
• Video Inference

==============================================================
"""

# ==========================================================
# IMPORTS
# ==========================================================

from pathlib import Path
import tempfile

import cv2
import librosa
import mediapipe as mp
import numpy as np
import torch

from PIL import Image

from torchvision import transforms

from transformers import AutoProcessor

from configs.settings import *

# ==========================================================
# PREPROCESSOR
# ==========================================================


class PrismShieldPreprocessor:

    """
    Universal preprocessing engine for PrismShieldAI.

    Handles

        Image
            ↓
        Face Detection
            ↓
        Crop
            ↓
        Resize
            ↓
        Normalize

    and

        Audio
            ↓
        Resample
            ↓
        Crop / Pad
            ↓
        Wav2Vec2 Processor

    Future

        Video
            ↓
        Uniform Frame Sampling
            ↓
        Face Detection
            ↓
        Predictor
    """

    # ======================================================
    # INITIALIZATION
    # ======================================================

    def __init__(

        self,

        image_size=IMAGE_SIZE,

        sample_rate=SAMPLE_RATE,

        max_audio_length=MAX_AUDIO_LENGTH,

        face_confidence=0.50,

    ):

        self.image_size = image_size

        self.sample_rate = sample_rate

        self.max_audio_length = max_audio_length

        self.face_confidence = face_confidence
        

        # --------------------------------------------------
        # Wav2Vec2 Processor
        # --------------------------------------------------

        self.processor = AutoProcessor.from_pretrained(

            "facebook/wav2vec2-base"

        )

        # --------------------------------------------------
        # MediaPipe Face Detector
        # --------------------------------------------------

        self.face_detector = (

            mp.solutions.face_detection.FaceDetection(

                model_selection=1,

                min_detection_confidence=face_confidence,

            )

        )

        # --------------------------------------------------
        # Image Transform
        # --------------------------------------------------

        self.image_transform = transforms.Compose([

            transforms.Resize(

                (

                    image_size,

                    image_size,

                )

            ),

            transforms.ToTensor(),

            transforms.Normalize(

                mean=[0.485, 0.456, 0.406],

                std=[0.229, 0.224, 0.225],

            ),

        ])

        print()

        print(PRINT_SEPARATOR)

        print("PREPROCESSOR INITIALIZED")

        print(PRINT_SEPARATOR)

        print("Image Size      :", image_size)

        print("Sample Rate     :", sample_rate)

        print("Audio Length    :", max_audio_length)

        print("Face Confidence :", face_confidence)
    
    # ======================================================
    # FACE EXTRACTION
    # ======================================================

    def extract_face(

        self,

        image,

        padding=0.15,

    ):

        """
        Detect the largest face using MediaPipe.

        Parameters
        ----------
        image : numpy.ndarray (BGR)

        Returns
        -------
        Cropped face (BGR)

        or

        None
        """

        if image is None:

            return None

        height, width = image.shape[:2]

        rgb = cv2.cvtColor(

            image,

            cv2.COLOR_BGR2RGB,

        )

        results = self.face_detector.process(rgb)

        if not results.detections:

            return None

        largest_face = None

        largest_area = 0

        for detection in results.detections:
            
            score = detection.score[0]

            bbox = detection.location_data.relative_bounding_box

            x = int(bbox.xmin * width)

            y = int(bbox.ymin * height)

            w = int(bbox.width * width)

            h = int(bbox.height * height)

            area = w * h

            if area > largest_area:

                largest_area = area

                largest_face = (x, y, w, h, score, )
                
        if largest_face is None:

            return None
        
        x, y, w, h, score = largest_face

        pad_x = int(w * padding)

        pad_y = int(h * padding)

        x1 = max(0, x - pad_x)

        y1 = max(0, y - pad_y)

        x2 = min(width, x + w + pad_x)

        y2 = min(height, y + h + pad_y)

        face = image[

            y1:y2,

            x1:x2,

        ]

        if face.size == 0:

            return None
        
        return {
            
            "face":face,
            
            "bbox":(x1,y1,x2,y2),
            
            "confidence":score
        }

    # ======================================================
    # IMAGE PREPROCESSING
    # ======================================================

    def preprocess_image(

        self,

        image,

        detect_face=True,

    ):

        """
        Converts an image into the tensor expected
        by EfficientNet.

        Parameters
        ----------
        image

            Path
            PIL.Image
            numpy.ndarray

        detect_face

            True -> MediaPipe crop

            False -> use image directly

        Returns
        -------
        torch.Tensor

        Shape

        (3,224,224)
        """

        # --------------------------------------------------
        # Load image
        # --------------------------------------------------

        if isinstance(image, (str, Path)):

            image = cv2.imread(

                str(image)

            )

            if image is None:

                raise FileNotFoundError(

                    f"Could not load image:\n{image}"

                )

        elif isinstance(image, Image.Image):

            image = cv2.cvtColor(

                np.array(image),

                cv2.COLOR_RGB2BGR,

            )

        elif not isinstance(image, np.ndarray):

            raise TypeError(

                "Unsupported image type."

            )

        # --------------------------------------------------
        # Face Detection
        # --------------------------------------------------

        if detect_face:
            
            face_data = self.extract_face(image)
            
            if face_data is not None:
                
                image = face_data["face"]
            
        # --------------------------------------------------
        # Convert BGR → RGB
        # --------------------------------------------------

        image = cv2.cvtColor(

            image,

            cv2.COLOR_BGR2RGB,

        )

        image = Image.fromarray(

            image

        )

        image_tensor = self.image_transform(

            image

        )

        return image_tensor

    # ======================================================
    # AUDIO PREPROCESSING
    # ======================================================

    def preprocess_audio(

        self,

        audio_path,

    ):

        """
        Preprocess audio for Wav2Vec2.

        Parameters
        ----------
        audio_path

            Path to audio file.

        Returns
        -------
        dict

        {

            "input_values",

            "attention_mask"

        }
        """

        # --------------------------------------------------
        # Verify file
        # --------------------------------------------------

        audio_path = Path(audio_path)

        if not audio_path.exists():

            raise FileNotFoundError(

                f"Audio file not found:\n{audio_path}"

            )

        # --------------------------------------------------
        # Load Audio
        # --------------------------------------------------

        waveform, sample_rate = librosa.load(

            str(audio_path),

            sr=self.sample_rate,

            mono=True,

        )

        # --------------------------------------------------
        # Crop
        # --------------------------------------------------

        waveform = waveform[: self.max_audio_length]

        # --------------------------------------------------
        # Pad
        # --------------------------------------------------

        if len(waveform) < self.max_audio_length:

            waveform = np.pad(

                waveform,

                (

                    0,

                    self.max_audio_length - len(waveform),

                ),

            )

        waveform = waveform.astype(

            np.float32

        )

        # --------------------------------------------------
        # Wav2Vec2 Processor
        # --------------------------------------------------

        audio = self.processor(

            waveform,

            sampling_rate=self.sample_rate,

            return_tensors="pt",

            padding="max_length",

            truncation=True,

            max_length=self.max_audio_length,

            return_attention_mask=True,

        )

        # --------------------------------------------------
        # Attention Mask
        # --------------------------------------------------

        if hasattr(audio, "attention_mask"):

            attention_mask = (

                audio.attention_mask.squeeze(0)

            )

        else:

            attention_mask = torch.ones_like(

                audio.input_values.squeeze(0)

            )

        # --------------------------------------------------
        # Return
        # --------------------------------------------------

        return {

            "input_values": audio.input_values.squeeze(0),

            "attention_mask": attention_mask,

        }
        
    # ======================================================
    # VIDEO PREPROCESSING
    # ======================================================

    def preprocess_video(

        self,

        video_path,

    ):

        """
        Complete video preprocessing pipeline.

        Video
            ↓
        Best Frame Selection
            ↓
        Audio Extraction
            ↓
        Image Preprocessing
            ↓
        Audio Preprocessing

        Returns
        -------

        {

            "image",

            "input_values",

            "attention_mask",

            "selected_frame"

        }
        """

        from inference.frame_selector import FrameSelector
        from audio.extract_audio import extract_audio_from_video

        selector = FrameSelector()

        frame_result = selector.select_best_frame(

            video_path

        )

        if frame_result is None:

            raise RuntimeError(

                "No face detected in uploaded video."

            )

        image_tensor = self.preprocess_image(

            frame_result["face"],

            detect_face=False,

        )

        audio_path = extract_audio_from_video(

            video_path

        )

        audio = self.preprocess_audio(

            audio_path

        )
        
        return {
            
            "image": image_tensor,

            "input_values": audio["input_values"],

            "attention_mask": audio["attention_mask"],

            "selected_frame": frame_result["face"],

            "audio_path": audio_path,

        }
        
# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    preprocessor = PrismShieldPreprocessor()

    print()

    print(PRINT_SEPARATOR)

    print("PREPROCESSOR READY")

    print(PRINT_SEPARATOR)

    print()

    print("Image preprocessing : READY")

    print("Audio preprocessing : READY")

    print()

    print("Waiting for inference...")
    