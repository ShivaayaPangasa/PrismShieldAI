"""
==============================================================
PrismShieldAI

Inference Engine

Complete Deployment Pipeline

Image
+
Audio
      ↓
Preprocessor
      ↓
Fusion Predictor
      ↓
Prediction

Used by

• Streamlit
• Windows OS Daemon
• REST API
• Video Inference
• Webcam

==============================================================
"""

# ==========================================================
# IMPORTS
# ==========================================================

from pathlib import Path

import torch

from configs.settings import *

from inference.predictor import PrismShieldPredictor
from inference.preprocess import PrismShieldPreprocessor


# ==========================================================
# INFERENCE ENGINE
# ==========================================================

class PrismShieldInference:
    """
    Complete end-to-end inference engine.

    Current
    -------

        Image
            +
        Audio
        
        Video

    Future
    ------

        Webcam

        Screen Capture

        Live Meetings

    """

    # ======================================================
    # INITIALIZATION
    # ======================================================

    def __init__(

        self,

        checkpoint=None,

        device=DEVICE,

    ):

        self.device = device

        # --------------------------------------------------
        # Checkpoint
        # --------------------------------------------------

        if checkpoint is None:

            checkpoint = WEIGHTS_DIR / BEST_FUSION_MODEL

        else:

            checkpoint = Path(checkpoint)

        self.checkpoint = checkpoint

        # --------------------------------------------------
        # Load Predictor
        # --------------------------------------------------

        print()

        print(PRINT_SEPARATOR)

        print("LOADING PREDICTOR")

        print(PRINT_SEPARATOR)
        
        self.predictor = PrismShieldPredictor(
            device=device,
        )
        
        # --------------------------------------------------
        # Load Preprocessor
        # --------------------------------------------------

        print()

        print(PRINT_SEPARATOR)

        print("LOADING PREPROCESSOR")

        print(PRINT_SEPARATOR)

        self.preprocessor = PrismShieldPreprocessor()

        # --------------------------------------------------
        # Ready
        # --------------------------------------------------

        print()

        print(PRINT_SEPARATOR)

        print("INFERENCE ENGINE READY")

        print(PRINT_SEPARATOR)

        print("Checkpoint :", checkpoint)

        print("Device     :", device)

        print()

        print("Supported Inputs")

        print("----------------")

        print("✓ Image + Audio")

        print("✓ Video Analysis")

        print("✓ Webcam (Coming Soon)")

        print("✓ Screen Capture (Coming Soon)")

    # ======================================================
    # IMAGE + AUDIO INFERENCE
    # ======================================================

    def predict_from_image_audio(

        self,

        image_path,

        audio_path,

    ):

        """
        Complete inference pipeline.

            Image
                +
            Audio
                ↓
            Preprocessor
                ↓
            Predictor
                ↓
            Result

        Parameters
        ----------
        image_path : str | Path

        audio_path : str | Path

        Returns
        -------
        dict
        """

        try:

            # --------------------------------------------------
            # Verify files
            # --------------------------------------------------

            image_path = Path(image_path)
            audio_path = Path(audio_path)

            if not image_path.exists():

                raise FileNotFoundError(

                    f"Image not found:\n{image_path}"

                )

            if not audio_path.exists():

                raise FileNotFoundError(

                    f"Audio not found:\n{audio_path}"

                )

            # --------------------------------------------------
            # Image
            # --------------------------------------------------

            image = self.preprocessor.preprocess_image(

                image_path

            )

            # --------------------------------------------------
            # Audio
            # --------------------------------------------------

            audio = self.preprocessor.preprocess_audio(

                audio_path

            )

            # --------------------------------------------------
            # Prediction
            # --------------------------------------------------

            result = self.predictor.predict(

                image=image,

                input_values=audio["input_values"],

                attention_mask=audio["attention_mask"],

            )

            # --------------------------------------------------
            # Formatting
            # --------------------------------------------------
            
            class_id = int(result["class_id"])
            
            prediction = result["prediction"]

            output = {

                "status": "success",

                "prediction": prediction,

                "class_id": class_id,

                "probability": float(

                    result["probability"]

                ),

                "confidence": float(

                    result["confidence"]

                ),

            }

            return output

        except Exception as e:

            return {

                "status": "error",

                "prediction": None,

                "class_id": None,

                "probability": None,

                "confidence": None,

                "error": str(e),

            }
        
    # ======================================================
    # VIDEO INFERENCE
    # ======================================================

    def predict_from_video(
        self,
        video_path,
    ):
        """
        Complete video inference pipeline.

        Video
            ↓
        Frame Selection
            ↓
        Audio Extraction
            ↓
         Multimodal Fusion
            ↓
        Prediction
        """
        
        try:

            video_path = Path(video_path)

            if not video_path.exists():

                raise FileNotFoundError(

                    f"Video not found:\n{video_path}"

                )

            result = self.predictor.predict_video(

                video_path

            )

            return {

                "status": "success",

                "prediction": result["prediction"],

                "class_id": int(result["class_id"]),

                "probability": float(result["probability"]),

                "confidence": float(result["confidence"]),

                "selected_frame": result["selected_frame"],

                "audio_path": result["audio_path"],

            }

        except Exception as e:

            return {

                "status": "error",

                "prediction": None,

                "class_id": None,

                "probability": None,

                "confidence": None,

                "selected_frame": None,

                "audio_path": None,

                "error": str(e),

            }      

    # ======================================================
    # PRETTY PRINT RESULT
    # ======================================================

    def print_result(

        self,

        result,

    ):

        """
        Pretty console output.
        """

        print()

        print(PRINT_SEPARATOR)

        print("PRISMSHIELDAI RESULT")

        print(PRINT_SEPARATOR)

        if result["status"] != "success":

            print("Status :", result["status"])

            print("Error  :", result["error"])

            return

        print("Prediction  :", result["prediction"])

        print("Class ID    :", result["class_id"])

        print(f"Probability : {result['probability']:.4f}")

        print(f"Confidence  : {result['confidence']:.4f}")

        print()

        if result["prediction"] == "FAKE":

            print("⚠ Deepfake Detected")

        else:

            print("✓ Appears Authentic")

    # ======================================================
    # STREAMLIT OUTPUT
    # ======================================================

    def streamlit_output(

        self,

        result,

    ):

        """
        Standardized output for Streamlit,
        REST API and OS Daemon.
        """

        return {

            "prediction": result["prediction"],

            "probability": result["probability"],

            "confidence": result["confidence"],

            "class_id": result["class_id"],

            "status": result["status"],

        }

# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print()

    print(PRINT_SEPARATOR)

    print("PRISMSHIELDAI INFERENCE ENGINE")

    print(PRINT_SEPARATOR)

    engine = PrismShieldInference()

    print()

    print("Inference engine loaded successfully.")

    print()

    print("Ready for")

    print("✓ Streamlit")

    print("✓ Windows OS Daemon")

    print("✓ REST API")

    print("✓ Webcam")

    print("✓ Video Inference")

    print()

    print("Example")

    print("-" * 60)

    print(

        'result = engine.predict_from_image_audio('

    )

    print(

        '    image_path="face.jpg",'

    )

    print(

        '    audio_path="voice.wav"'

    )

    print(

        ')'

    )