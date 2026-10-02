"""
PrismShieldAI Research Paper
Experiment 01: Reproduce Main Test Result

This script:
1. Loads the existing trained fusion checkpoint.
2. Evaluates the existing TEST split.
3. Computes publication metrics.
4. Computes calibration metrics.
5. Saves results inside research_paper/results/.

IMPORTANT:
- Does NOT modify the trained model.
- Does NOT copy the checkpoint.
- Does NOT modify the existing evaluation pipeline.
"""

from pathlib import Path
import sys
import json
import argparse
import contextlib
import io

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from transformers import AutoProcessor

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
)

# ============================================================
# PROJECT PATH
# ============================================================

# research_paper/scripts/reproduce_main_result.py
#                    ↑
# project root = ../../

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from configs.settings import (
    DEVICE,
    IMAGE_SIZE,
    MAX_AUDIO_LENGTH,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    SEED,
)

from datasets.fusion_dataset import FusionDataset
from models.fusion_model import FusionModel


# ============================================================
# PATHS
# ============================================================

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "weights"
    / "fusion_best_v3_reasoning_final.pth"
)

CSV_PATH = (
    PROJECT_ROOT
    / "processed_dataset"
    / "research_dataset.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "research_paper"
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# RANDOM SEED
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# ECE
# ============================================================

def expected_calibration_error(
    probabilities,
    labels,
    number_of_bins=15,
):
    """
    Expected Calibration Error using
    maximum predicted class probability.
    """

    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    confidences = np.max(
        probabilities,
        axis=1,
    )

    correctness = (
        predictions == labels
    ).astype(float)

    bin_edges = np.linspace(
        0.0,
        1.0,
        number_of_bins + 1,
    )

    ece = 0.0

    for i in range(number_of_bins):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == number_of_bins - 1:
            mask = (
                (confidences >= lower)
                & (confidences <= upper)
            )
        else:
            mask = (
                (confidences >= lower)
                & (confidences < upper)
            )

        if not np.any(mask):
            continue

        bin_accuracy = np.mean(
            correctness[mask]
        )

        bin_confidence = np.mean(
            confidences[mask]
        )

        bin_fraction = np.mean(mask)

        ece += (
            bin_fraction
            * abs(
                bin_accuracy
                - bin_confidence
            )
        )

    return float(ece)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of test samples for a smoke test.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("PRISMSHIELDAI — MAIN RESULT REPRODUCTION")
    print("=" * 70)

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Device:")
    print(DEVICE)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # ========================================================
    # CHECK FILES
    # ========================================================

    print()
    print("=" * 70)
    print("CHECKING REQUIRED FILES")
    print("=" * 70)

    print(
        "Checkpoint:",
        CHECKPOINT_PATH
    )

    print(
        "Checkpoint exists:",
        CHECKPOINT_PATH.exists()
    )

    print(
        "Dataset:",
        CSV_PATH
    )

    print(
        "Dataset exists:",
        CSV_PATH.exists()
    )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found:\n{CSV_PATH}"
        )

    # ========================================================
    # LOAD DATA
    # ========================================================

    print()
    print("=" * 70)
    print("LOADING TEST DATA")
    print("=" * 70)

    dataframe = pd.read_csv(
        CSV_PATH
    )

    test_df = dataframe[
        dataframe["split"] == "test"
    ].reset_index(
        drop=True
    )

    if args.limit is not None:
        test_df = test_df.head(
            args.limit
        ).copy()

    print(
        "Test samples:",
        len(test_df)
    )

    print()
    print(
        "Class distribution:"
    )

    print(
        test_df["label"].value_counts()
    )
    
    if test_df["label"].nunique() < 2:
        raise ValueError(
            "Test split contains fewer than two classes. "
            "A full evaluation requires both real and fake samples."
        )
        
    # ========================================================
    # PROCESSOR
    # ========================================================

    print()
    print("=" * 70)
    print("LOADING WAV2VEC2 PROCESSOR")
    print("=" * 70)

    processor = AutoProcessor.from_pretrained(
        "facebook/wav2vec2-base"
    )

    # ========================================================
    # DATASET
    # ========================================================

    dataset = FusionDataset(
        dataframe=test_df,
        processor=processor,
        image_size=IMAGE_SIZE,
        max_audio_length=MAX_AUDIO_LENGTH,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    print(
        "Dataset size:",
        len(dataset)
    )

    print(
        "Number of batches:",
        len(loader)
    )

    # ========================================================
    # MODEL
    # ========================================================

    print()
    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    model = FusionModel().to(
        DEVICE
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    print(
        "Checkpoint keys:",
        list(checkpoint.keys())
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Checkpoint loaded successfully."
    )

    if "epoch" in checkpoint:
        print(
            "Checkpoint epoch:",
            checkpoint["epoch"]
        )

    if "best_score" in checkpoint:
        print(
            "Checkpoint best score:",
            checkpoint["best_score"]
        )

    # ========================================================
    # INFERENCE
    # ========================================================

    print()
    print("=" * 70)
    print("STARTING INFERENCE")
    print("=" * 70)

    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_confidences = []

    with torch.inference_mode():

        for batch_index, batch in enumerate(loader):

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

            attention_mask = batch[
                "attention_mask"
            ].to(
                DEVICE,
                non_blocking=True,
            )

            labels = batch[
                "label"
            ].to(
                DEVICE,
                non_blocking=True,
            )

            # ------------------------------------------------
            # Existing FusionModel prints many diagnostics.
            # Suppress those prints for this research run.
            # ------------------------------------------------

            with contextlib.redirect_stdout(
                io.StringIO()
            ):

                outputs = model(
                    input_values=input_values,
                    attention_mask=attention_mask,
                    images=images,
                )

            logits = outputs[
                "logits"
            ]

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predictions = probabilities.argmax(
                dim=1
            )

            confidence = outputs[
                "confidence"
            ].view(-1)

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

            all_confidences.extend(
                confidence.cpu().numpy()
            )

            if (
                batch_index + 1
            ) % 10 == 0:

                print(
                    f"Processed "
                    f"{batch_index + 1}/"
                    f"{len(loader)} batches"
                )

    # ========================================================
    # NUMPY
    # ========================================================

    labels = np.asarray(
        all_labels
    )

    predictions = np.asarray(
        all_predictions
    )

    probabilities = np.asarray(
        all_probabilities
    )

    confidences = np.asarray(
        all_confidences
    )

    # Probability of REAL class (class 1)
    real_probabilities = probabilities[:, 1]

    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            labels,
            predictions,
        )
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0,
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        labels,
        real_probabilities,
    )

    pr_auc = average_precision_score(
        labels,
        real_probabilities,
    )

    mcc = matthews_corrcoef(
        labels,
        predictions,
    )

    cm = confusion_matrix(
        labels,
        predictions,
    )

    # ========================================================
    # CALIBRATION
    # ========================================================

    ece = expected_calibration_error(
        probabilities,
        labels,
    )

    # Brier score for probability of class 1 / REAL
    brier_score = np.mean(
        (
            real_probabilities
            - labels
        ) ** 2
    )

    # Learned confidence head predicts correctness.
    correctness = (
        predictions == labels
    ).astype(float)

    confidence_brier = np.mean(
        (
            confidences
            - correctness
        ) ** 2
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(
        f"Accuracy            : {accuracy:.6f}"
    )

    print(
        f"Balanced Accuracy   : {balanced_accuracy:.6f}"
    )

    print(
        f"Precision           : {precision:.6f}"
    )

    print(
        f"Recall              : {recall:.6f}"
    )

    print(
        f"F1                  : {f1:.6f}"
    )

    print(
        f"Macro-F1            : {macro_f1:.6f}"
    )

    print(
        f"ROC-AUC             : {roc_auc:.6f}"
    )

    print(
        f"PR-AUC              : {pr_auc:.6f}"
    )

    print(
        f"MCC                 : {mcc:.6f}"
    )

    print()
    print("Confusion Matrix:")
    print(cm)

    print()
    print(
        f"ECE                 : {ece:.6f}"
    )

    print(
        f"Brier Score         : {brier_score:.6f}"
    )

    print(
        f"Confidence Brier    : {confidence_brier:.6f}"
    )

    print()
    print(
        "Learned confidence mean:",
        f"{confidences.mean():.6f}"
    )

    print(
        "Learned confidence median:",
        f"{np.median(confidences):.6f}"
    )

    print(
        "Learned confidence min:",
        f"{confidences.min():.6f}"
    )

    print(
        "Learned confidence max:",
        f"{confidences.max():.6f}"
    )

    # ========================================================
    # SAVE JSON
    # ========================================================

    results = {

        "experiment":
            "main_result_reproduction",

        "checkpoint":
            str(CHECKPOINT_PATH),

        "checkpoint_epoch":
            int(checkpoint["epoch"])
            if "epoch" in checkpoint
            else None,

        "checkpoint_best_score":
            float(checkpoint["best_score"])
            if "best_score" in checkpoint
            else None,

        "test_samples":
            int(len(labels)),

        "accuracy":
            float(accuracy),

        "balanced_accuracy":
            float(balanced_accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "macro_f1":
            float(macro_f1),

        "roc_auc":
            float(roc_auc),

        "pr_auc":
            float(pr_auc),

        "mcc":
            float(mcc),

        "confusion_matrix":
            cm.tolist(),

        "ece":
            float(ece),

        "brier_score":
            float(brier_score),

        "confidence_brier":
            float(confidence_brier),

        "confidence_mean":
            float(confidences.mean()),

        "confidence_median":
            float(np.median(confidences)),

        "confidence_min":
            float(confidences.min()),

        "confidence_max":
            float(confidences.max()),
    }

    output_path = (
        RESULTS_DIR
        / "main_result_reproduction.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
        )

    print()
    print(
        "Saved:"
    )
    print(
        output_path
    )

    # ========================================================
    # SAVE CONFUSION MATRIX
    # ========================================================

    cm_path = (
        RESULTS_DIR
        / "main_confusion_matrix.txt"
    )

    np.savetxt(
        cm_path,
        cm,
        fmt="%d",
    )

    print(
        "Saved:"
    )
    print(
        cm_path
    )

    print()
    print("=" * 70)
    print("REPRODUCTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()