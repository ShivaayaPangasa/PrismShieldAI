"""
==============================================================
PrismShieldAI

Reusable Predictor

Loads the trained fusion model once and performs
multimodal inference.

This class is shared by:

• Streamlit
• Webcam
• Video inference
• OS Daemon
• REST API

==============================================================
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from configs.settings import *
from models.fusion_model import FusionModel
from inference.preprocess import PrismShieldPreprocessor

class PrismShieldPredictor:

    def __init__(
        self,
        checkpoint_path=None,
        device=None,
    ):

        # --------------------------------------------------
        # Device
        # --------------------------------------------------

        self.device = device or DEVICE

        # --------------------------------------------------
        # Model
        # --------------------------------------------------

        self.model = FusionModel().to(self.device)

        # --------------------------------------------------
        # Preprocessor
        # --------------------------------------------------

        self.preprocessor = PrismShieldPreprocessor()

        # --------------------------------------------------
        # Checkpoint
        # --------------------------------------------------

        if checkpoint_path is None:

            checkpoint_path = WEIGHTS_DIR / BEST_FUSION_MODEL

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():

            raise FileNotFoundError(

                f"\nCheckpoint not found:\n{checkpoint_path}"

            )

        checkpoint = torch.load(

            checkpoint_path,

            map_location=self.device,

        )

        self.model.load_state_dict(

            checkpoint["model_state_dict"]

        )

        self.model.eval()

        print()

        print("=" * 60)
        print("PREDICTOR READY")
        print("=" * 60)

        print("Checkpoint :", checkpoint_path)
        print("Device     :", self.device)

    # ======================================================
    # Prediction
    # ======================================================

    @torch.no_grad()
    def predict(

        self,

        image,

        input_values,

        attention_mask,

    ):

        # ---------------------------------------------
        # Move to device
        # ---------------------------------------------

        image = image.to(self.device)

        input_values = input_values.to(self.device)

        attention_mask = attention_mask.to(self.device)

        # ---------------------------------------------
        # Batch dimension
        # ---------------------------------------------

        if image.dim() == 3:

            image = image.unsqueeze(0)

        if input_values.dim() == 1:

            input_values = input_values.unsqueeze(0)

        if attention_mask.dim() == 1:

            attention_mask = attention_mask.unsqueeze(0)

        # ---------------------------------------------
        # Forward
        # ---------------------------------------------

        outputs = self.model(

            input_values=input_values,

            attention_mask=attention_mask,

            images=image,

        )

        logits = outputs["logits"]
        
        print("\n" + "=" * 60)
        print("RAW LOGITS")
        print(logits.detach().cpu())

        probabilities = F.softmax(logits, dim=1)

        print("\nSOFTMAX")
        print(probabilities.detach().cpu())

        print("\nPREDICTED CLASS")
        print(probabilities.argmax(dim=1).item())

        print("=" * 60)
        
     

        probabilities = F.softmax(

            logits,

            dim=1,

        )

        class_id = probabilities.argmax(

            dim=1

        ).item()

        probability = probabilities.max().item()

        confidence = outputs["confidence"].item()

        label_map = {

            0: "FAKE",

            1: "REAL",

        }

        prediction = label_map[class_id]

        return {

            "prediction": prediction,

            "class_id": class_id,

            "probability": probability,

            "confidence": confidence,

            "logits": logits.cpu(),

            "reasoning_embedding":
                outputs["reasoning_embedding"].cpu(),

            "audio_embedding":
                outputs["audio_embedding"].cpu(),

            "visual_embedding":
                outputs["visual_embedding"].cpu(),

            "similarity":
                outputs["similarity"].cpu(),

            "audio_attention":
                outputs["audio_attention"].cpu(),

            "visual_attention":
                outputs["visual_attention"].cpu(),

        }
        
    # ======================================================
    # VIDEO PREDICTION
    # ======================================================

    @torch.no_grad()
    def predict_video(

        self,

        video_path,

    ):

        """
        Complete video inference pipeline.

        Video
            ↓
        Preprocessing
            ↓
        Fusion Model
            ↓
        Prediction
        """
        
        video_data = self.preprocessor.preprocess_video(
            video_path
        )

        result = self.predict(
            
            image=video_data["image"],

            input_values=video_data["input_values"],

            attention_mask=video_data["attention_mask"],

        )   

        result["selected_frame"] = video_data["selected_frame"]

        result["audio_path"] = video_data["audio_path"]

        return result

if __name__ == "__main__":

    print()

    print("=" * 60)
    print("PrismShield Predictor")
    print("=" * 60)

    predictor = PrismShieldPredictor()

    print()

    print("Predictor loaded successfully.")

    print("Ready for inference.")