import torch
import torch.nn as nn

from transformers import Wav2Vec2Model


class AudioEncoder(nn.Module):
    """
    PrismShieldAI Audio Encoder

    Wav2Vec2 Backbone
            ↓
      Mean Pooling
            ↓
     Projection Head
            ↓
   Classification Head

    Returns:
        logits
        embedding
    """

    def __init__(self):

        super().__init__()

        self.backbone = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        )
        
        # ======================================================
        # Freeze Wav2Vec2 Backbone
        # ======================================================
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        hidden_size = self.backbone.config.hidden_size

        # ======================================================
        # Projection Head
        # ======================================================

        self.projection = nn.Sequential(

            nn.Linear(hidden_size, 512),

            nn.ReLU(),

            nn.Dropout(0.30),

            nn.Linear(512, 256),

        )

        # ======================================================
        # Classification Head
        # ======================================================

        self.classifier = nn.Sequential(

            nn.Linear(256, 128),

            nn.ReLU(),

            nn.Dropout(0.20),

            nn.Linear(128, 2),

        )

    # ==========================================================

    def forward(
        self,
        input_values,
        attention_mask=None,
    ):

        outputs = self.backbone(

            input_values=input_values,

            attention_mask=attention_mask,

        )

        # ------------------------------------------------------
        # Mean Pooling
        # ------------------------------------------------------

        pooled = outputs.last_hidden_state.mean(dim=1)

        # ------------------------------------------------------
        # Projection
        # ------------------------------------------------------

        embedding = self.projection(
            pooled
        )

        # ------------------------------------------------------
        # Classification
        # ------------------------------------------------------

        logits = self.classifier(
            embedding
        )

        return logits, embedding

    # ==========================================================

    def extract_embedding(
        self,
        input_values,
        attention_mask=None,
    ):

        outputs = self.backbone(

            input_values=input_values,

            attention_mask=attention_mask,

        )

        pooled = outputs.last_hidden_state.mean(dim=1)

        embedding = self.projection(
            pooled
        )
        
        return nn.functional.normalize(
            
            embedding,
            
            p=2,
            
            dim=1,
        )
    