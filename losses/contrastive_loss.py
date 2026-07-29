"""
==============================================================
PrismShieldAI

Contrastive Loss

Implements InfoNCE Contrastive Loss
for Audio-Visual Representation Learning.

==============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):

    def __init__(

        self,

        temperature: float = 0.07,

    ):

        super().__init__()

        self.temperature = temperature

    def forward(

        self,

        audio_embedding,

        visual_embedding,

    ):

        # --------------------------------------------------
        # Normalize embeddings
        # --------------------------------------------------

        audio_embedding = F.normalize(

            audio_embedding,

            dim=1,

        )

        visual_embedding = F.normalize(

            visual_embedding,

            dim=1,

        )

        # --------------------------------------------------
        # Similarity Matrix
        # --------------------------------------------------

        logits = torch.matmul(

            audio_embedding,

            visual_embedding.T,

        )

        logits = logits / self.temperature

        # --------------------------------------------------
        # Positive pairs
        # --------------------------------------------------

        labels = torch.arange(

            logits.size(0),

            device=logits.device,

        )

        # --------------------------------------------------
        # Audio → Visual
        # --------------------------------------------------

        loss_audio = F.cross_entropy(

            logits,

            labels,

        )

        # --------------------------------------------------
        # Visual → Audio
        # --------------------------------------------------

        loss_visual = F.cross_entropy(

            logits.T,

            labels,

        )

        # --------------------------------------------------
        # Final Loss
        # --------------------------------------------------

        loss = (

            loss_audio +

            loss_visual

        ) / 2

        return loss


if __name__ == "__main__":

    loss_fn = ContrastiveLoss()

    audio = torch.randn(

        8,

        256,

    )

    visual = torch.randn(

        8,

        256,

    )

    loss = loss_fn(

        audio,

        visual,

    )

    print()

    print("=" * 60)

    print("Contrastive Loss")

    print("=" * 60)

    print("Loss :", loss.item())