"""
==============================================================
PrismShieldAI

Visual Encoder

Project:
Causal and Contrastive Multimodal Reasoning
for Robust Deep Fake Detection

Author:
Shivaaya

==============================================================
"""

# ==============================================================
# IMPORTS
# ==============================================================

import torch
import torch.nn as nn
import timm

# ==============================================================
# VISUAL ENCODER
# ==============================================================

class VisualEncoder(nn.Module):
    """
    PrismShieldAI Visual Encoder

    Image
        ↓
    EfficientNet-B0
        ↓
    Global Average Pooling
        ↓
    Projection Head
        ↓
    Classification Head

    Returns
    -------
    logits
    embedding
    """

    def __init__(self):

        super().__init__()

        # ======================================================
        # BACKBONE
        # ======================================================

        self.backbone = timm.create_model(

            "efficientnet_b0",

            pretrained=True,

            num_classes=0,      # Remove classifier

            global_pool="avg",  # Global Average Pooling

        )

        # ======================================================
        # FREEZE BACKBONE
        # ======================================================

        for param in self.backbone.parameters():

            param.requires_grad = False

        hidden_size = self.backbone.num_features

        print()

        print("=" * 60)
        print("VISUAL ENCODER")
        print("=" * 60)

        print("Backbone :", "EfficientNet-B0")
        print("Feature Dimension :", hidden_size)
        
                # ======================================================
        # PROJECTION HEAD
        # ======================================================

        self.projection = nn.Sequential(

            nn.Linear(hidden_size, 512),

            nn.ReLU(),

            nn.Dropout(0.30),

            nn.Linear(512, 256),

        )

        # ======================================================
        # CLASSIFICATION HEAD
        # ======================================================

        self.classifier = nn.Sequential(

            nn.Linear(256, 128),

            nn.ReLU(),

            nn.Dropout(0.20),

            nn.Linear(128, 2),

        )

        print("Embedding Size :", 256)
    
    # ==========================================================
    # FORWARD
    # ==========================================================

    def forward(
        self,
        images,
    ):

        features = self.backbone(
            images
        )

        embedding = self.projection(
            features
        )

        logits = self.classifier(
            embedding
        )

        return logits, embedding

    # ==========================================================
    # EMBEDDING EXTRACTION
    # ==========================================================

    def extract_embedding(
        self,
        images,
    ):

        features = self.backbone(
            images
        )

        embedding = self.projection(
            features
        )
        
        return nn.functional.normalize(
            
            embedding,
            
            p=2,
            
            dim=1,
        )