"""
==============================================================
PrismShieldAI

Adaptive Evidence Reasoning Module

Current Inputs:
    • Audio Embedding
    • Visual Embedding

Future Inputs:
    • Lip Sync
    • Eye Gaze
    • Head Pose
    • Blink Detection
    • Voice Emotion
    • Speech Fluency
    • Temporal Consistency

The interface remains unchanged as new modalities are added.

==============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveEvidenceReasoning(nn.Module):
    def __init__(self, embedding_dim=256, hidden_dim=512):
        super().__init__()
        self.audio_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.visual_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.interaction_network = nn.Sequential(
            nn.Linear(embedding_dim*2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.importance_network = nn.Sequential(
            nn.Linear(embedding_dim*3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim,2),
        )
        self.fusion_network = nn.Sequential(
            nn.Linear(embedding_dim*3, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.reasoning_norm = nn.LayerNorm(embedding_dim)
        self.confidence_head = nn.Sequential(
            nn.Linear(embedding_dim,128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128,1),
            nn.Sigmoid(),
        )

    def forward(self,audio_embedding,visual_embedding,
                gaze_embedding=None,blink_embedding=None,
                lipsync_embedding=None,emotion_embedding=None,
                headpose_embedding=None):

        audio_evidence=self.audio_encoder(audio_embedding)
        visual_evidence=self.visual_encoder(visual_embedding)

        interaction=self.interaction_network(
            torch.cat([audio_evidence,visual_evidence],dim=1)
        )
        
        weights = F.softmax(
            
            self.importance_network(

                torch.cat(

                    [

                        audio_evidence,

                        visual_evidence,

                        interaction,

                    ],

                    dim=1,

                )

            ),

            dim=1,

        )
        
        assert torch.allclose(

            weights.sum(dim=1),

            torch.ones_like(weights[:, 0]),

            atol=1e-5,

        )

        audio_weight = weights[:, 0:1]

        visual_weight = weights[:, 1:2]

        weighted_audio = audio_evidence * audio_weight

        weighted_visual = visual_evidence * visual_weight

        fused=self.fusion_network(
            torch.cat([weighted_audio,weighted_visual,interaction],dim=1)
        )

        residual=0.5*(audio_evidence+visual_evidence)

        reasoning_embedding=self.reasoning_norm(fused+residual)

        confidence=self.confidence_head(reasoning_embedding)
        
        return {
            "reasoning_embedding": reasoning_embedding,
            "confidence": confidence,
            "audio_importance": audio_weight,
            "visual_importance": visual_weight,
        }
        
if __name__=="__main__":
    m=AdaptiveEvidenceReasoning()
    a=torch.randn(8,256)
    v=torch.randn(8,256)
    out=m(a,v)
    
    print()

    print("=" * 60)
    
    print("Adaptive Evidence Reasoning Module")

    print("=" * 60)

    print("Reasoning Embedding :", out["reasoning_embedding"].shape)

    print("Confidence          :", out["confidence"].shape)

    print("Audio Importance    :", out["audio_importance"].shape)

    print("Visual Importance   :", out["visual_importance"].shape)
    