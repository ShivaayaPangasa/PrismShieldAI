"""
==============================================================
PrismShieldAI

Projection Head

Projects modality-specific embeddings into
a shared latent embedding space.

Used by:

• Audio Encoder
• Visual Encoder
• Contrastive Learning
• Fusion Model

==============================================================
"""

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):

    def __init__(

        self,

        input_dim: int = 256,

        hidden_dim: int = 512,

        output_dim: int = 256,

        dropout: float = 0.30,

    ):

        super().__init__()

        self.projection = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim,
                output_dim,
            ),

        )

        self.layer_norm = nn.LayerNorm(
            output_dim
        )

    def forward(

        self,

        x,

    ):

        x = self.projection(x)

        x = self.layer_norm(x)

        x = nn.functional.normalize(

            x,

            p=2,

            dim=1,

        )

        return x


if __name__ == "__main__":

    model = ProjectionHead()

    x = torch.randn(

        8,

        256,

    )

    y = model(x)

    print()

    print("=" * 60)

    print("Projection Head")

    print("=" * 60)

    print("Input :", x.shape)

    print("Output:", y.shape)