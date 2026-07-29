"""
==============================================================
PrismShieldAI

Contrastive Head

Projects audio and visual embeddings into a shared
embedding space for multimodal contrastive learning.

Used by:

• Fusion Model
• Contrastive Loss
• Retrieval
• Cross-modal Alignment

==============================================================
"""

import torch
import torch.nn as nn

from models.projection_head import ProjectionHead


class ContrastiveHead(nn.Module):

    def __init__(

        self,

        embedding_dim: int = 256,

        projection_dim: int = 256,

    ):

        super().__init__()

        self.audio_projection = ProjectionHead(

            input_dim=embedding_dim,

            hidden_dim=512,

            output_dim=projection_dim,

        )

        self.visual_projection = ProjectionHead(

            input_dim=embedding_dim,

            hidden_dim=512,

            output_dim=projection_dim,

        )

    def forward(

        self,

        audio_embedding,

        visual_embedding,

    ):

        audio_embedding = self.audio_projection(

            audio_embedding

        )

        visual_embedding = self.visual_projection(

            visual_embedding

        )

        similarity = torch.matmul(

            audio_embedding,

            visual_embedding.T,

        )

        return (

            audio_embedding,

            visual_embedding,

            similarity,

        )


if __name__ == "__main__":

    model = ContrastiveHead()

    audio = torch.randn(

        8,

        256,

    )

    visual = torch.randn(

        8,

        256,

    )

    audio_proj, visual_proj, sim = model(

        audio,

        visual,

    )

    print()

    print("=" * 60)

    print("Contrastive Head")

    print("=" * 60)

    print("Audio Projection :", audio_proj.shape)

    print("Visual Projection:", visual_proj.shape)

    print("Similarity Matrix:", sim.shape)