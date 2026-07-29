"""
==============================================================
PrismShieldAI

Cross Modal Attention

Audio attends to Visual
Visual attends to Audio

Produces refined multimodal embeddings.

==============================================================
"""

import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):

    def __init__(

        self,

        embedding_dim: int = 256,

        num_heads: int = 8,

        dropout: float = 0.10,

    ):

        super().__init__()

        # ---------------------------------------------
        # Audio → Visual
        # ---------------------------------------------

        self.audio_to_visual = nn.MultiheadAttention(

            embed_dim=embedding_dim,

            num_heads=num_heads,

            dropout=dropout,

            batch_first=True,

        )

        # ---------------------------------------------
        # Visual → Audio
        # ---------------------------------------------

        self.visual_to_audio = nn.MultiheadAttention(

            embed_dim=embedding_dim,

            num_heads=num_heads,

            dropout=dropout,

            batch_first=True,

        )

        self.norm_audio = nn.LayerNorm(

            embedding_dim

        )

        self.norm_visual = nn.LayerNorm(

            embedding_dim

        )

    def forward(

        self,

        audio_embedding,

        visual_embedding,

    ):

        # Convert

        audio = audio_embedding.unsqueeze(1)

        visual = visual_embedding.unsqueeze(1)

        # ---------------------------------------------
        # Audio attends to Visual
        # ---------------------------------------------

        audio_context, audio_weights = self.audio_to_visual(

            query=audio,

            key=visual,

            value=visual,

        )

        # ---------------------------------------------
        # Visual attends to Audio
        # ---------------------------------------------

        visual_context, visual_weights = self.visual_to_audio(

            query=visual,

            key=audio,

            value=audio,

        )

        audio = self.norm_audio(

            audio + audio_context

        )

        visual = self.norm_visual(

            visual + visual_context

        )

        return (

            audio.squeeze(1),

            visual.squeeze(1),

            audio_weights,

            visual_weights,

        )


if __name__ == "__main__":

    model = CrossModalAttention()

    audio = torch.randn(

        8,

        256,

    )

    visual = torch.randn(

        8,

        256,

    )

    audio_out, visual_out, aw, vw = model(

        audio,

        visual,

    )

    print()

    print("=" * 60)

    print("Cross Modal Attention")

    print("=" * 60)

    print("Audio :", audio_out.shape)

    print("Visual:", visual_out.shape)

    print("Audio Attention :", aw.shape)

    print("Visual Attention:", vw.shape)