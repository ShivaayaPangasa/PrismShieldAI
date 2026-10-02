# ============================================================
# PrismShieldAI - Paper Ablation Study Runner
# ============================================================
#
# Ablation configurations:
#   1. Visual Only
#   2. Audio Only
#   3. Audio + Visual
#   4. + Contrastive
#   5. + Adaptive Evidence Reasoning (AER)
#   6. Full PrismShieldAI
#
# IMPORTANT:
# - Train split is used ONLY for training.
# - Validation split is used ONLY for model selection.
# - Test split is kept untouched until final evaluation.
# - Audio/visual backbone features are extracted once and frozen.
# - AER uses the existing CausalReasoning implementation,
#   which is being renamed/reported as Adaptive Evidence Reasoning.
#
# ============================================================

from pathlib import Path
import sys
import argparse
import json
import random

import numpy as np
import pandas as pd
from transformers import Wav2Vec2Processor
import torch
import torch.nn as nn

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from torch.utils.data import DataLoader, Dataset

# ------------------------------------------------------------
# Make project root importable
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ------------------------------------------------------------
# Project imports
# ------------------------------------------------------------

from datasets.fusion_dataset import FusionDataset
from models.audio_encoder import AudioEncoder
from models.visual_encoder import VisualEncoder
from models.projection_head import ProjectionHead
from models.causal_reasoning import CausalReasoning
from losses.contrastive_loss import ContrastiveLoss


# ============================================================
# Configuration
# ============================================================

SEED = 42

BATCH_SIZE = 16
EPOCHS = 5
LR = 3e-4
WEIGHT_DECAY = 1e-4

TEMPERATURE = 0.07
CONTRASTIVE_WEIGHT = 0.10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

RESULTS = ROOT / "research_paper" / "results"

WEIGHTS = (
    ROOT
    / "weights"
    / "fusion_best_v3_reasoning_final.pth"
)

CSV = ROOT / "processed_dataset" / "research_dataset.csv"


# ============================================================
# Reproducibility
# ============================================================

def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


# ============================================================
# Load pretrained PrismShieldAI encoders
# ============================================================

def load_encoders():
    print("=" * 70)
    print("Loading pretrained PrismShieldAI encoders")
    print("=" * 70)

    if not WEIGHTS.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{WEIGHTS}"
        )

    checkpoint = torch.load(
        WEIGHTS,
        map_location=DEVICE,
        weights_only=False,
    )

    state = checkpoint["model_state_dict"]

    audio_encoder = AudioEncoder().to(DEVICE)
    visual_encoder = VisualEncoder().to(DEVICE)

    # --------------------------------------------------------
    # Extract audio encoder weights
    # --------------------------------------------------------

    audio_state = {
        key[len("audio_encoder.") :]: value
        for key, value in state.items()
        if key.startswith("audio_encoder.")
    }

    # --------------------------------------------------------
    # Extract visual encoder weights
    # --------------------------------------------------------

    visual_state = {
        key[len("visual_encoder.") :]: value
        for key, value in state.items()
        if key.startswith("visual_encoder.")
    }

    if not audio_state:
        raise RuntimeError(
            "No audio_encoder weights found in checkpoint."
        )

    if not visual_state:
        raise RuntimeError(
            "No visual_encoder weights found in checkpoint."
        )

    audio_encoder.load_state_dict(
        audio_state,
        strict=True,
    )

    visual_encoder.load_state_dict(
        visual_state,
        strict=True,
    )

    # Freeze pretrained encoders.
    audio_encoder.eval()
    visual_encoder.eval()

    for parameter in audio_encoder.parameters():
        parameter.requires_grad = False

    for parameter in visual_encoder.parameters():
        parameter.requires_grad = False

    print(
        f"Checkpoint epoch: {checkpoint.get('epoch')}"
    )

    print(
        f"Checkpoint best score: "
        f"{checkpoint.get('best_score')}"
    )

    print(f"Device: {DEVICE}")

    return audio_encoder, visual_encoder


# ============================================================
# Lightweight classifier head
# ============================================================

class Head(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.30),

            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.20),

            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Ablation model
# ============================================================

class Ablation(nn.Module):

    def __init__(self, mode):

        super().__init__()

        self.mode = mode

        # ----------------------------------------------------
        # 1. Visual Only
        # ----------------------------------------------------

        if mode == "visual":

            self.classifier = Head(256)

        # ----------------------------------------------------
        # 2. Audio Only
        # ----------------------------------------------------

        elif mode == "audio":

            self.classifier = Head(256)

        # ----------------------------------------------------
        # 3. Audio + Visual
        # ----------------------------------------------------

        elif mode == "av":

            self.classifier = Head(512)

        # ----------------------------------------------------
        # 4. + Contrastive
        # ----------------------------------------------------

        elif mode == "contrastive":

            self.audio_projection = ProjectionHead()
            self.visual_projection = ProjectionHead()

            self.classifier = Head(512)

        # ----------------------------------------------------
        # 5. + Adaptive Evidence Reasoning
        # ----------------------------------------------------

        elif mode == "aer":

            self.audio_projection = ProjectionHead()
            self.visual_projection = ProjectionHead()

            # Existing implementation.
            # In the paper this module is reported as AER.
            self.reasoning = CausalReasoning()

            self.classifier = Head(256)

        # ----------------------------------------------------
        # 6. Full PrismShieldAI
        # ----------------------------------------------------

        elif mode == "full":

            self.audio_projection = ProjectionHead()
            self.visual_projection = ProjectionHead()

            # Existing AER implementation.
            self.reasoning = CausalReasoning()

            self.classifier = Head(256)

        else:

            raise ValueError(
                f"Unknown ablation mode: {mode}"
            )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    def forward(self, audio, visual):

        # ----------------------------------------------------
        # Visual Only
        # ----------------------------------------------------

        if self.mode == "visual":

            logits = self.classifier(visual)

            return logits, None, None

        # ----------------------------------------------------
        # Audio Only
        # ----------------------------------------------------

        if self.mode == "audio":

            logits = self.classifier(audio)

            return logits, None, None

        # ----------------------------------------------------
        # Audio + Visual
        # ----------------------------------------------------

        if self.mode == "av":

            representation = torch.cat(
                [audio, visual],
                dim=1,
            )

            logits = self.classifier(
                representation
            )

            return logits, None, None

        # ----------------------------------------------------
        # Projection heads
        # ----------------------------------------------------

        audio_projection = self.audio_projection(audio)
        visual_projection = self.visual_projection(visual)

        # ----------------------------------------------------
        # + Contrastive
        # ----------------------------------------------------

        if self.mode == "contrastive":

            representation = torch.cat(
                [
                    audio_projection,
                    visual_projection,
                ],
                dim=1,
            )

            logits = self.classifier(
                representation
            )

            return (
                logits,
                audio_projection,
                visual_projection,
            )

        # ----------------------------------------------------
        # + AER
        # ----------------------------------------------------

        if self.mode == "aer":

            reasoning_output = self.reasoning(
                audio_projection,
                visual_projection,
            )

            logits = self.classifier(
                reasoning_output[
                    "reasoning_embedding"
                ]
            )

            return (
                logits,
                audio_projection,
                visual_projection,
            )

        # ----------------------------------------------------
        # Full PrismShieldAI
        # ----------------------------------------------------
        #
        # IMPORTANT:
        # The current production FusionModel computes
        # CrossModalAttention, but its returned contexts are
        # not passed into the projection heads.
        #
        # Therefore we do NOT artificially feed cross-attended
        # tensors here. The functional full configuration is:
        #
        #   Audio + Visual
        #       -> Projection Heads
        #       -> AER
        #       -> Classification
        #       + Contrastive Loss
        #
        # This matches the currently implemented downstream
        # computation more faithfully.
        # ----------------------------------------------------

        reasoning_output = self.reasoning(
            audio_projection,
            visual_projection,
        )

        logits = self.classifier(
            reasoning_output[
                "reasoning_embedding"
            ]
        )

        return (
            logits,
            audio_projection,
            visual_projection,
        )


# ============================================================
# Feature dataset
# ============================================================

class FeatureDataset(Dataset):

    def __init__(
        self,
        audio_features,
        visual_features,
        labels,
    ):

        self.audio = audio_features
        self.visual = visual_features
        self.labels = labels

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, index):

        return (
            self.audio[index],
            self.visual[index],
            self.labels[index],
        )

def extract_features(
    audio_encoder,
    visual_encoder,
    split,
    limit=None,
):
    print("\n" + "=" * 70)
    print(f"Extracting features: {split.upper()} split")
    print("=" * 70)

    if not CSV.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found:\n{CSV}"
        )

    # --------------------------------------------------------
    # Load complete research dataset
    # --------------------------------------------------------

    dataframe = pd.read_csv(CSV)

    # --------------------------------------------------------
    # Select requested split
    # --------------------------------------------------------

    if "split" not in dataframe.columns:
        raise KeyError(
            "The research dataset does not contain a "
            "'split' column."
        )

    dataframe = dataframe[
        dataframe["split"].astype(str).str.lower()
        == split.lower()
    ].reset_index(drop=True)

    if len(dataframe) == 0:
        raise ValueError(
            f"No samples found for split='{split}'."
        )

    # --------------------------------------------------------
    # Optional smoke-test limit
    # --------------------------------------------------------
    
    if limit is not None:
        # Balanced smoke-test subset: half real, half fake
        per_class = limit // 2
        
        real_df = dataframe[dataframe["label"].astype(str).str.lower() == "real"]
        fake_df = dataframe[dataframe["label"].astype(str).str.lower() == "fake"]

        real_sample = real_df.sample(
            n=min(per_class, len(real_df)),
            random_state=42
        )

        fake_sample = fake_df.sample(
            n=min(per_class, len(fake_df)),
            random_state=42
        )

        dataframe = (
            pd.concat([real_sample, fake_sample])
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )

        print(
            f"LIMIT MODE: balanced subset with "
            f"{(dataframe['label'].astype(str).str.lower() == 'real').sum()} real and "
            f"{(dataframe['label'].astype(str).str.lower() == 'fake').sum()} fake samples"
        )

    # --------------------------------------------------------
    # Create Wav2Vec2 processor
    # --------------------------------------------------------

    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-base"
    )

    # --------------------------------------------------------
    # Create actual local FusionDataset
    # --------------------------------------------------------

    dataset = FusionDataset(
        dataframe=dataframe,
        processor=processor,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    audio_features = []
    visual_features = []
    labels = []

    # --------------------------------------------------------
    # Extract frozen embeddings
    # --------------------------------------------------------

    with torch.no_grad():

        for batch_index, batch in enumerate(
            loader,
            start=1,
        ):

            images = batch["image"].to(
                DEVICE,
                non_blocking=True,
            )

            input_values = batch[
                "input_values"
            ].to(
                DEVICE,
                non_blocking=True,
            )

            attention_mask = batch.get(
                "attention_mask"
            )

            if attention_mask is not None:
                attention_mask = attention_mask.to(
                    DEVICE,
                    non_blocking=True,
                )

            # Audio embedding
            audio_embedding = (
                audio_encoder.extract_embedding(
                    input_values,
                    attention_mask,
                )
            )

            # Visual embedding
            visual_embedding = (
                visual_encoder.extract_embedding(
                    images
                )
            )

            audio_features.append(
                audio_embedding.cpu()
            )

            visual_features.append(
                visual_embedding.cpu()
            )

            labels.append(
                batch["label"].cpu()
            )

            if batch_index % 50 == 0:
                print(
                    f"Processed batches: "
                    f"{batch_index}"
                )

    # --------------------------------------------------------
    # Combine batches
    # --------------------------------------------------------

    audio_features = torch.cat(
        audio_features,
        dim=0,
    )

    visual_features = torch.cat(
        visual_features,
        dim=0,
    )

    labels = torch.cat(
        labels,
        dim=0,
    )

    print(
        f"{split.upper()} samples: "
        f"{len(labels)}"
    )

    print(
        f"Real: {(labels == 0).sum().item()} | "
        f"Fake: {(labels == 1).sum().item()}"
    )

    print(
        f"Audio feature shape: "
        f"{tuple(audio_features.shape)}"
    )

    print(
        f"Visual feature shape: "
        f"{tuple(visual_features.shape)}"
    )

    return (
        audio_features,
        visual_features,
        labels,
    )

# ============================================================
# Class weights
# ============================================================

def make_class_weights(labels):

    counts = torch.bincount(
        labels,
        minlength=2,
    ).float()

    total = counts.sum()

    weights = total / (
        2.0 * counts.clamp(min=1.0)
    )

    return weights.to(DEVICE)


# ============================================================
# Train one ablation configuration
# ============================================================

def train_one_mode(
    mode,
    train_audio,
    train_visual,
    train_labels,
    val_audio,
    val_visual,
    val_labels,
    epochs,
):

    print("\n" + "=" * 70)
    print(f"TRAINING: {mode}")
    print("=" * 70)

    train_dataset = FeatureDataset(
        train_audio,
        train_visual,
        train_labels,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    model = Ablation(mode).to(DEVICE)

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    class_weights = make_class_weights(
        train_labels
    )

    criterion_cls = nn.CrossEntropyLoss(
        weight=class_weights
    )

    contrastive_loss = ContrastiveLoss(
        temperature=TEMPERATURE
    )

    best_val_f1 = -1.0
    best_state = None
    best_epoch = 0

    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()

        running_loss = 0.0

        for (
            audio,
            visual,
            labels,
        ) in train_loader:

            audio = audio.to(
                DEVICE,
                non_blocking=True,
            )

            visual = visual.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits, audio_proj, visual_proj = (
                model(
                    audio,
                    visual,
                )
            )

            classification_loss = (
                criterion_cls(
                    logits,
                    labels,
                )
            )

            loss = classification_loss

            # ------------------------------------------------
            # Contrastive loss is used for:
            # + Contrastive
            # Full PrismShieldAI
            # ------------------------------------------------

            if mode in (
                "contrastive",
                "full",
            ):

                contrastive = contrastive_loss(
                    audio_proj,
                    visual_proj,
                )

                loss = (
                    classification_loss
                    + CONTRASTIVE_WEIGHT
                    * contrastive
                )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        average_loss = (
            running_loss
            / max(len(train_loader), 1)
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_metrics = evaluate(
            model,
            val_audio,
            val_visual,
            val_labels,
        )

        print(
            f"[{mode}] "
            f"Epoch {epoch}/{epochs} | "
            f"Loss={average_loss:.5f} | "
            f"Val F1={val_metrics['f1']:.5f} | "
            f"Val BA={val_metrics['balanced_accuracy']:.5f}"
        )

        # ----------------------------------------------------
        # Select best model using validation F1
        # ----------------------------------------------------

        if val_metrics["f1"] > best_val_f1:

            best_val_f1 = (
                val_metrics["f1"]
            )

            best_epoch = epoch

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:

        raise RuntimeError(
            f"No best model was selected for {mode}"
        )

    model.load_state_dict(
        best_state
    )

    print(
        f"Best validation epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best validation F1: "
        f"{best_val_f1:.5f}"
    )

    return model, best_epoch, best_val_f1


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    audio_features,
    visual_features,
    labels,
):

    model.eval()

    dataset = FeatureDataset(
        audio_features,
        visual_features,
        labels,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    y_true = []
    y_pred = []
    y_prob = []

    for (
        audio,
        visual,
        batch_labels,
    ) in loader:

        audio = audio.to(
            DEVICE,
            non_blocking=True,
        )

        visual = visual.to(
            DEVICE,
            non_blocking=True,
        )

        logits, _, _ = model(
            audio,
            visual,
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[:, 1]

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        y_true.extend(
            batch_labels.numpy().tolist()
        )

        y_pred.extend(
            predictions.cpu()
            .numpy()
            .tolist()
        )

        y_prob.extend(
            probabilities.cpu()
            .numpy()
            .tolist()
        )

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    y_prob = np.asarray(
        y_prob
    )

    metrics = {
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_true,
                y_prob,
            )
        ),

        "mcc": float(
            matthews_corrcoef(
                y_true,
                y_pred,
            )
        ),
    }

    return metrics


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run PrismShieldAI paper ablation study."
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Training epochs per configuration.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional smoke-test limit. "
            "Limits each split to the first N samples."
        ),
    )

    parser.add_argument(
        "--only",
        type=str,
        choices=[
            "visual",
            "audio",
            "av",
            "contrastive",
            "aer",
            "full",
        ],
        default=None,
        help="Run only one ablation configuration.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    seed_everything()

    RESULTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 70)
    print("PrismShieldAI Ablation Study")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(
        f"Epochs/configuration: "
        f"{args.epochs}"
    )

    print(
        f"Checkpoint: {WEIGHTS}"
    )

    print(
        f"Dataset: {CSV}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Load pretrained encoders
    # --------------------------------------------------------

    audio_encoder, visual_encoder = (
        load_encoders()
    )

    # --------------------------------------------------------
    # Extract TRAIN features
    # --------------------------------------------------------

    (
        train_audio,
        train_visual,
        train_labels,
    ) = extract_features(
        audio_encoder,
        visual_encoder,
        split="train",
        limit=args.limit,
    )

    # --------------------------------------------------------
    # Extract VALIDATION features
    # --------------------------------------------------------

    (
        val_audio,
        val_visual,
        val_labels,
    ) = extract_features(
        audio_encoder,
        visual_encoder,
        split="val",
        limit=args.limit,
    )

    # --------------------------------------------------------
    # Extract TEST features
    # --------------------------------------------------------

    (
        test_audio,
        test_visual,
        test_labels,
    ) = extract_features(
        audio_encoder,
        visual_encoder,
        split="test",
        limit=args.limit,
    )

    # --------------------------------------------------------
    # Configuration names
    # --------------------------------------------------------

    names = {
        "visual": "Visual Only",
        "audio": "Audio Only",
        "av": "Audio + Visual",
        "contrastive": "+ Contrastive",
        "aer": "+ Adaptive Evidence Reasoning",
        "full": "Full PrismShieldAI",
    }

    all_modes = [
        "visual",
        "audio",
        "av",
        "contrastive",
        "aer",
        "full",
    ]

    modes = (
        [args.only]
        if args.only
        else all_modes
    )

    # --------------------------------------------------------
    # Run ablations
    # --------------------------------------------------------

    rows = []

    for mode in modes:

        print("\n" + "#" * 70)
        print(
            f"RUNNING: {names[mode]}"
        )
        print("#" * 70)

        model, best_epoch, best_val_f1 = (
            train_one_mode(
                mode=mode,

                train_audio=train_audio,
                train_visual=train_visual,
                train_labels=train_labels,

                val_audio=val_audio,
                val_visual=val_visual,
                val_labels=val_labels,

                epochs=args.epochs,
            )
        )

        # ----------------------------------------------------
        # FINAL TEST EVALUATION
        # ----------------------------------------------------

        test_metrics = evaluate(
            model,
            test_audio,
            test_visual,
            test_labels,
        )

        print("\nFINAL TEST RESULTS")

        print(
            f"Balanced Accuracy: "
            f"{test_metrics['balanced_accuracy']:.6f}"
        )

        print(
            f"F1: "
            f"{test_metrics['f1']:.6f}"
        )

        print(
            f"ROC-AUC: "
            f"{test_metrics['roc_auc']:.6f}"
        )

        print(
            f"MCC: "
            f"{test_metrics['mcc']:.6f}"
        )

        rows.append(
            {
                "configuration": names[mode],
                "mode": mode,

                "best_epoch": best_epoch,

                "best_validation_f1": (
                    best_val_f1
                ),

                "train_samples": (
                    len(train_labels)
                ),

                "validation_samples": (
                    len(val_labels)
                ),

                "test_samples": (
                    len(test_labels)
                ),

                "balanced_accuracy": (
                    test_metrics[
                        "balanced_accuracy"
                    ]
                ),

                "f1": (
                    test_metrics["f1"]
                ),

                "roc_auc": (
                    test_metrics["roc_auc"]
                ),

                "mcc": (
                    test_metrics["mcc"]
                ),
            }
        )

    # ========================================================
    # Save results
    # ========================================================

    dataframe = pd.DataFrame(
        rows
    )

    csv_path = (
        RESULTS
        / "ablation_results.csv"
    )

    json_path = (
        RESULTS
        / "ablation_results.json"
    )

    dataframe.to_csv(
        csv_path,
        index=False,
    )

    payload = {

        "study": (
            "PrismShieldAI component-wise "
            "ablation study"
        ),

        "seed": SEED,

        "device": str(DEVICE),

        "checkpoint": str(
            WEIGHTS
        ),

        "dataset_csv": str(
            CSV
        ),

        "epochs_per_configuration": (
            args.epochs
        ),

        "batch_size": BATCH_SIZE,

        "learning_rate": LR,

        "weight_decay": WEIGHT_DECAY,

        "temperature": TEMPERATURE,

        "contrastive_weight": (
            CONTRASTIVE_WEIGHT
        ),

        "evaluation_protocol": {

            "training_split": (
                "train"
            ),

            "model_selection_split": (
                "validation"
            ),

            "final_evaluation_split": (
                "test"
            ),

            "test_used_for_training": False,

            "test_used_for_model_selection": False,

            "backbones_frozen": True,

            "feature_extraction": (
                "Pretrained audio and visual "
                "embeddings extracted from "
                "fusion_best_v3_reasoning_final.pth"
            ),
        },

        "configurations": [

            {
                "mode": "visual",
                "description": (
                    "Visual encoder features "
                    "followed by classifier."
                ),
            },

            {
                "mode": "audio",
                "description": (
                    "Audio encoder features "
                    "followed by classifier."
                ),
            },

            {
                "mode": "av",
                "description": (
                    "Concatenated audio and visual "
                    "features followed by classifier."
                ),
            },

            {
                "mode": "contrastive",
                "description": (
                    "Audio and visual projection "
                    "heads with contrastive alignment "
                    "and classification."
                ),
            },

            {
                "mode": "aer",
                "description": (
                    "Projected audio and visual "
                    "representations processed by "
                    "Adaptive Evidence Reasoning "
                    "(existing CausalReasoning "
                    "implementation)."
                ),
            },

            {
                "mode": "full",
                "description": (
                    "Projected audio and visual "
                    "representations processed by "
                    "AER with contrastive alignment."
                ),
            },
        ],

        "results": rows,
    }

    json_path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Final table
    # ========================================================

    print("\n" + "=" * 70)
    print("ABLATION STUDY COMPLETE")
    print("=" * 70)

    print(
        dataframe.to_string(
            index=False
        )
    )

    print("\nSaved files:")

    print(
        f"CSV:  {csv_path}"
    )

    print(
        f"JSON: {json_path}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()