"""
==============================================================
PrismShieldAI

Fusion Model

Audio
↓

Visual
↓

Cross Modal Attention
↓

Contrastive Alignment
↓

Causal Reasoning
↓

Final Classification

==============================================================
"""

import torch
import torch.nn as nn

from models.audio_encoder import AudioEncoder
from models.visual_encoder import VisualEncoder

from models.cross_modal_attention import CrossModalAttention
from models.contrastive_head import ContrastiveHead
from models.causal_reasoning import CausalReasoning


class FusionModel(nn.Module):

    def __init__(self):

        super().__init__()

        # --------------------------------------------------
        # Encoders
        # --------------------------------------------------

        self.audio_encoder = AudioEncoder()

        self.visual_encoder = VisualEncoder()

        # --------------------------------------------------
        # Freeze encoders initially
        # --------------------------------------------------

        self.freeze_encoders()

        # --------------------------------------------------
        # Multimodal Modules
        # --------------------------------------------------

        self.cross_attention = CrossModalAttention()

        self.contrastive = ContrastiveHead()

        self.reasoning = CausalReasoning()

        # --------------------------------------------------
        # Final Classifier
        # --------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(256, 128),

            nn.GELU(),

            nn.Dropout(0.30),

            nn.Linear(128, 64),

            nn.GELU(),

            nn.Dropout(0.20),

            nn.Linear(64, 2),

        )
    # ==================================================
    # Freeze Encoders
    # ==================================================
    
    def freeze_encoders(self):
        
        for param in self.audio_encoder.parameters():
            param.requires_grad = False

        for param in self.visual_encoder.parameters():
            param.requires_grad = False

    # ==================================================
    # Unfreeze Visual Encoder
    # ==================================================

    def unfreeze_visual(self):

        print("\nUnfreezing Visual Encoder...")

        for param in self.visual_encoder.backbone.parameters():
            param.requires_grad = True

    # ==================================================
    # Unfreeze Audio Encoder
    # ==================================================

    def unfreeze_audio(self):

        print("\nUnfreezing Audio Encoder...")

        for param in self.audio_encoder.backbone.encoder.layers[-2:].parameters():
            param.requires_grad = True
    
    # ==================================================
    # Forward
    # ==================================================

    def forward(

        self,

        input_values,

        attention_mask,

        images,

        gaze_embedding=None,

        blink_embedding=None,

        lipsync_embedding=None,

        emotion_embedding=None,

        headpose_embedding=None,

    ):

        # ==================================================
        # Audio / Visual Embeddings
        # ==================================================

        audio_embedding = self.audio_encoder.extract_embedding(

            input_values,

            attention_mask,

        )
        
        print("\nAudio Embedding")
        print("Mean :", audio_embedding.mean().item())
        print("Std  :", audio_embedding.std().item())
        print("Min  :", audio_embedding.min().item())
        print("Max  :", audio_embedding.max().item())

        visual_embedding = self.visual_encoder.extract_embedding(

            images,

        )
        
        print("\nVisual Embedding")
        print("Mean :", visual_embedding.mean().item())
        print("Std  :", visual_embedding.std().item())
        print("Min  :", visual_embedding.min().item())
        print("Max  :", visual_embedding.max().item())

        # ==================================================
        # Cross Modal Attention
        # ==================================================

        (

            audio_embedding,

            visual_embedding,

            audio_weights,

            visual_weights,

        ) = self.cross_attention(

            audio_embedding,

            visual_embedding,

        )

        # ==================================================
        # Contrastive Alignment
        # ==================================================

        (

            audio_projection,

            visual_projection,

            similarity,

        ) = self.contrastive(

            audio_embedding,

            visual_embedding,

        )

        # ==================================================
        # Causal Reasoning
        # ==================================================

        reasoning = self.reasoning(

            audio_projection,

            visual_projection,

            gaze_embedding,

            blink_embedding,

            lipsync_embedding,

            emotion_embedding,

            headpose_embedding,

        )
        
        assert reasoning["reasoning_embedding"].shape[1] == 256
        assert reasoning["confidence"].shape[1] == 1

        reasoning_embedding = reasoning["reasoning_embedding"]

        confidence = reasoning["confidence"]
        
        print("\nReasoning Embedding")
        print("Mean :", reasoning_embedding.mean().item())
        print("Std  :", reasoning_embedding.std().item())
        print("Min  :", reasoning_embedding.min().item())
        print("Max  :", reasoning_embedding.max().item())

        print("\nConfidence")
        print(confidence.detach().cpu())

        # ==================================================
        # Classification
        # ==================================================

        logits = self.classifier(

            reasoning_embedding

        )

        return {

            "logits": logits,

            "confidence": confidence,

            "reasoning_embedding": reasoning_embedding,

            "audio_embedding": audio_embedding,

            "visual_embedding": visual_embedding,

            "audio_projection": audio_projection,

            "visual_projection": visual_projection,

            "similarity": similarity,

            "audio_attention": audio_weights,

            "visual_attention": visual_weights,

        }


if __name__ == "__main__":

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print()
    print("=" * 60)
    print("DEVICE")
    print("=" * 60)
    print("Using:", DEVICE)

    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    model = FusionModel().to(DEVICE)

    audio = torch.randn(

        4,

        80000,

    ).to(DEVICE)

    mask = torch.ones(

        4,

        80000,

    ).to(DEVICE)

    images = torch.randn(

        4,

        3,

        224,

        224,

    ).to(DEVICE)

    outputs = model(

        input_values=audio,

        attention_mask=mask,

        images=images,

    )

    print()

    print("=" * 60)
    print("FUSION MODEL SUMMARY")
    print("=" * 60)

    total_params = sum(

        p.numel()

        for p in model.parameters()

    )

    trainable_params = sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )

    frozen_params = total_params - trainable_params

    print(f"Total Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params:,}")
    print(f"Frozen Parameters    : {frozen_params:,}")

    print()

    print("Logits               :", outputs["logits"].shape)
    print("Confidence           :", outputs["confidence"].shape)
    print("Reasoning Embedding  :", outputs["reasoning_embedding"].shape)
    print("Similarity Matrix    :", outputs["similarity"].shape)
    print("Audio Embedding      :", outputs["audio_embedding"].shape)
    print("Visual Embedding     :", outputs["visual_embedding"].shape)