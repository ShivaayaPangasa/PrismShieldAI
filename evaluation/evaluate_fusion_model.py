"""
==============================================================
PrismShieldAI

Standalone Fusion Model Evaluation

Project:
Causal and Contrastive Multimodal Reasoning
for Robust Deep Fake Detection

Loads the best trained checkpoint and evaluates only
the TEST split.

Outputs
-------
• Test Metrics
• Classification Report
• Confusion Matrix
• ROC Curve
• Precision Recall Curve
• Confidence Histogram
• predictions.csv
• metrics.json

==============================================================
"""

# ==============================================================
# IMPORTS
# ==============================================================

import json
import shutil
from datetime import datetime
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
)

from torch.utils.data import DataLoader

from transformers import AutoProcessor

from torch.optim import AdamW

from torch.amp import autocast

# ==============================================================
# PROJECT IMPORTS
# ==============================================================

from configs.settings import *

from datasets.fusion_dataset import FusionDataset

from models.fusion_model import FusionModel

from losses.contrastive_loss import ContrastiveLoss

from utils.seed import set_seed

from utils.metrics import MetricsCalculator

from utils.checkpoint import CheckpointManager

from utils.class_weights import ClassWeights

# ==============================================================
# RANDOM SEED
# ==============================================================

set_seed(SEED)

# ==============================================================
# RESULTS DIRECTORY
# ==============================================================

RUN_NAME = "fusion_v3_reasoning_final"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

RESULTS_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "evaluation_results"
    / f"{RUN_NAME}_{TIMESTAMP}"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FAILURE_DIR = RESULTS_DIR / "failure_gallery"

FAILURE_DIR.mkdir(
    exist_ok=True,
)

# ==============================================================
# DEVICE
# ==============================================================

print()

print(PRINT_SEPARATOR)
print("PRISMSHIELDAI EVALUATION")
print(PRINT_SEPARATOR)

print("Device :", DEVICE)

if DEVICE.type == "cuda":

    print(

        "GPU :", torch.cuda.get_device_name(0)

    )

# ==============================================================
# LOAD DATASET
# ==============================================================

print()

print(PRINT_SEPARATOR)
print("LOADING DATASET")
print(PRINT_SEPARATOR)

dataframe = pd.read_csv(
    CSV_FILE
)

test_df = dataframe[
    dataframe["split"] == "test"
].reset_index(
    drop=True
)

print("Test Samples :", len(test_df))

# ==============================================================
# WAV2VEC2 PROCESSOR
# ==============================================================

print()

print(PRINT_SEPARATOR)
print("LOADING PROCESSOR")
print(PRINT_SEPARATOR)

processor = AutoProcessor.from_pretrained(
    "facebook/wav2vec2-base"
)

print("facebook/wav2vec2-base")

# ==============================================================
# DATASET
# ==============================================================

print()

print(PRINT_SEPARATOR)
print("BUILDING TEST DATASET")
print(PRINT_SEPARATOR)

test_dataset = FusionDataset(

    dataframe=test_df,

    processor=processor,

    image_size=IMAGE_SIZE,

    max_audio_length=MAX_AUDIO_LENGTH,

)

print("Dataset Size :", len(test_dataset))

print("\nDataset Distribution")

print(test_df["label"].value_counts())

# ==============================================================
# DATALOADER
# ==============================================================

test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=PIN_MEMORY,

)

print("Test Batches :", len(test_loader))

# ==============================================================
# CLASS WEIGHTS
# ==============================================================

class_weights = ClassWeights().compute()

class_weights = class_weights.to(
    DEVICE
)

# ==============================================================
# MODEL
# ==============================================================

print()

print(PRINT_SEPARATOR)
print("LOADING MODEL")
print(PRINT_SEPARATOR)

model = FusionModel().to(
    DEVICE
)

total_params = sum(

    p.numel()

    for p in model.parameters()

)

trainable_params = sum(

    p.numel()

    for p in model.parameters()

    if p.requires_grad

)

print(f"Total Parameters     : {total_params:,}")

print(f"Trainable Parameters : {trainable_params:,}")

# ==============================================================
# LOSSES
# ==============================================================

criterion_cls = nn.CrossEntropyLoss(

    weight=class_weights,

    label_smoothing=0.02,

)

criterion_contrast = ContrastiveLoss(

    temperature=TEMPERATURE,

)

criterion_confidence = nn.BCELoss()

CONFIDENCE_WEIGHT = 0.20

# ==============================================================
# OPTIMIZER
#
# Required because CheckpointManager currently restores the
# optimizer state along with the model.
# ==============================================================

optimizer = AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY,

)

# ==============================================================
# UTILITIES
# ==============================================================

metrics = MetricsCalculator()

checkpoint = CheckpointManager(

    checkpoint_dir=WEIGHTS_DIR,

)

# ==============================================================
# LOAD BEST MODEL
# ==============================================================

print()

print(PRINT_SEPARATOR)
print("LOADING BEST CHECKPOINT")
print(f"Checkpoint File : {BEST_MODEL_NAME}")
print(f"Checkpoint Path : {WEIGHTS_DIR / BEST_MODEL_NAME}")
print(PRINT_SEPARATOR)

best_epoch, best_f1 = checkpoint.load(

    model=model,

    optimizer=optimizer,

    filename=BEST_MODEL_NAME,

)

print(f"Best Epoch : {best_epoch}")

print(f"Best F1    : {best_f1:.4f}")

model.eval()

# ==============================================================
# STORAGE
# ==============================================================

test_loss = 0.0

test_cls_loss = 0.0

test_contrast_loss = 0.0

test_confidence_loss = 0.0

all_predictions = []

all_labels = []

all_probabilities = []

all_confidences = []

prediction_rows = []

start_time = time.time()

print()

print(PRINT_SEPARATOR)
print("STARTING TEST EVALUATION")
print(PRINT_SEPARATOR) 

# ==============================================================
# TEST LOOP
# ==============================================================

with torch.inference_mode():

    for batch_index, batch in enumerate(test_loader):

        images = batch["image"].to(
            DEVICE,
            non_blocking=True,
        )

        input_values = batch["input_values"].to(
            DEVICE,
            non_blocking=True,
        )

        attention_mask = batch["attention_mask"].to(
            DEVICE,
            non_blocking=True,
        )

        labels = batch["label"].to(
            DEVICE,
            non_blocking=True,
        )

        with autocast(

            device_type=DEVICE.type,

            enabled=USE_AMP,

        ):

            outputs = model(

                input_values=input_values,

                attention_mask=attention_mask,

                images=images,

            )

            # --------------------------------------------------
            # Classification Loss
            # --------------------------------------------------

            cls_loss = criterion_cls(

                outputs["logits"],

                labels,

            )

            # --------------------------------------------------
            # Contrastive Loss
            # --------------------------------------------------

            contrastive_loss = criterion_contrast(

                outputs["audio_projection"],

                outputs["visual_projection"],

            )

            # --------------------------------------------------
            # Predictions
            # --------------------------------------------------

            predictions = outputs["logits"].argmax(

                dim=1

            )

            probabilities = torch.softmax(

                outputs["logits"],

                dim=1,

            )[:, 1]

            confidence_target = (

                predictions == labels

            ).float()

            total_loss = (

                CLASSIFICATION_WEIGHT * cls_loss +

                CONTRASTIVE_WEIGHT * contrastive_loss

            )

        # ------------------------------------------------------
        # Confidence Loss
        # ------------------------------------------------------

        confidence_loss = criterion_confidence(

            outputs["confidence"].float().view(-1),

            confidence_target.float().view(-1),

        )

        total_loss = total_loss + (

            CONFIDENCE_WEIGHT *

            confidence_loss

        )

        # ------------------------------------------------------
        # Running Loss
        # ------------------------------------------------------

        test_loss += total_loss.item()

        test_cls_loss += cls_loss.item()

        test_contrast_loss += contrastive_loss.item()

        test_confidence_loss += confidence_loss.item()

        # ------------------------------------------------------
        # Confidence Scores
        # ------------------------------------------------------

        confidence_scores = (

            outputs["confidence"]

            .view(-1)

            .detach()

            .cpu()

            .numpy()

        )

        # ------------------------------------------------------
        # Store Metrics
        # ------------------------------------------------------

        all_predictions.extend(

            predictions.cpu().numpy()

        )

        all_labels.extend(

            labels.cpu().numpy()

        )

        all_probabilities.extend(

            probabilities.detach()

            .cpu()

            .numpy()

        )

        all_confidences.extend(

            confidence_scores

        )

        # ------------------------------------------------------
        # Save Per-Sample Predictions
        # ------------------------------------------------------

        batch_size = labels.size(0)

        for i in range(batch_size):

            prediction_rows.append({

                "sample_id":

                    int(batch["sample_id"][i]),

                "video_id":

                    int(batch["video_id"][i]),

                "true_label":

                    int(labels[i]),

                "prediction":

                    int(predictions[i]),

                "probability":

                    float(probabilities[i]),

                "confidence":

                    float(confidence_scores[i]),

                "image_path":

                    batch["image_path"][i],

                "audio_path":

                    batch["audio_path"][i],

            })

        # ------------------------------------------------------
        # Progress
        # ------------------------------------------------------

        if (batch_index + 1) % 10 == 0:

            print(

                f"Processed "

                f"{batch_index + 1}"

                f"/{len(test_loader)} batches"

            )

# ==============================================================
# AVERAGE LOSSES
# ==============================================================

test_loss /= len(test_loader)

test_cls_loss /= len(test_loader)

test_contrast_loss /= len(test_loader)

test_confidence_loss /= len(test_loader)

elapsed_time = time.time() - start_time

print()

print(PRINT_SEPARATOR)

print("INFERENCE COMPLETE")

print(PRINT_SEPARATOR)

print(f"Evaluation Time : {elapsed_time:.2f} sec")

print(f"Samples Tested  : {len(all_labels)}")

print(f"Average Loss    : {test_loss:.4f}")

print()

# ==============================================================
# METRICS
# ==============================================================

print()

print(PRINT_SEPARATOR)
print("CALCULATING METRICS")
print(PRINT_SEPARATOR)

test_scores = metrics.calculate(

    predictions=all_predictions,

    labels=all_labels,

    probabilities=all_probabilities,

)

# ==============================================================
# PRINT RESULTS
# ==============================================================

print()

print(PRINT_SEPARATOR)
print("TEST RESULTS")
print(PRINT_SEPARATOR)

print(f"Total Loss          : {test_loss:.4f}")
print(f"Classification Loss : {test_cls_loss:.4f}")
print(f"Contrastive Loss    : {test_contrast_loss:.4f}")
print(f"Confidence Loss     : {test_confidence_loss:.4f}")

print()

print(f"Accuracy            : {test_scores['accuracy']:.4f}")
print(f"Balanced Accuracy   : {test_scores['balanced_accuracy']:.4f}")
print(f"Precision           : {test_scores['precision']:.4f}")
print(f"Recall              : {test_scores['recall']:.4f}")
print(f"F1 Score            : {test_scores['f1']:.4f}")
print(f"ROC AUC             : {test_scores['roc_auc']:.4f}")
print(f"PR AUC              : {test_scores['pr_auc']:.4f}")
print(f"MCC                 : {test_scores['mcc']:.4f}")

# ==============================================================
# CLASSIFICATION REPORT
# ==============================================================

report = classification_report(

    all_labels,

    all_predictions,

    target_names=[

        "Fake",

        "Real",

    ],

    digits=4,

)

print()

print(PRINT_SEPARATOR)
print("CLASSIFICATION REPORT")
print(PRINT_SEPARATOR)

print(report)

with open(
    
    RESULTS_DIR / "classification_report_test.txt",

    "w",

    encoding="utf-8",

) as f:

    f.write(report)

# ==============================================================
# SAVE PREDICTIONS
# ==============================================================

prediction_df = pd.DataFrame(

    prediction_rows

)

prediction_df.to_csv(
    
    RESULTS_DIR / "test_predictions.csv",

    index=False,

)

print()

print("Saved test_predictions.csv")

# ==============================================================
# MISCLASSIFIED SAMPLES
# ==============================================================

misclassified_df = prediction_df[
    prediction_df["true_label"] != prediction_df["prediction"]
]

misclassified_df.to_csv(

    RESULTS_DIR / "misclassified.csv",

    index=False,

)

import shutil

print()

print("Building Failure Gallery...")

for _, row in misclassified_df.iterrows():

    src = Path(row["image_path"])

    if src.exists():

        dst = FAILURE_DIR / (

            f"{row['sample_id']}_"

            f"GT{row['true_label']}_"

            f"PRED{row['prediction']}_"

            f"{src.name}"

        )

        shutil.copy2(

            src,

            dst,

        )

print("Failure Gallery Complete")

print(

    f"Saved {len(misclassified_df)} misclassified samples"

)

# ==============================================================
# SAVE METRICS JSON
# ==============================================================

metrics_json = {

    "loss": float(test_loss),

    "classification_loss": float(test_cls_loss),

    "contrastive_loss": float(test_contrast_loss),

    "confidence_loss": float(test_confidence_loss),

    "accuracy": float(test_scores["accuracy"]),

    "balanced_accuracy": float(test_scores["balanced_accuracy"]),

    "precision": float(test_scores["precision"]),

    "recall": float(test_scores["recall"]),

    "f1": float(test_scores["f1"]),

    "roc_auc": float(test_scores["roc_auc"]),

    "pr_auc": float(test_scores["pr_auc"]),

    "mcc": float(test_scores["mcc"]),

    "best_epoch": int(best_epoch),

    "best_validation_f1": float(best_f1),

    "evaluation_time_seconds": float(elapsed_time),

    "num_test_samples": int(len(all_labels)),

}

with open(
    RESULTS_DIR / "evaluation_metrics.json",
    "w",
) as f:
    json.dump(
        metrics_json,
        f,
        indent=4,
    )

print("Saved evaluation_metrics.json")

# ==============================================================
# CONFUSION MATRIX
# ==============================================================

print()

print(PRINT_SEPARATOR)
print("GENERATING CONFUSION MATRIX")
print(PRINT_SEPARATOR)

cm = confusion_matrix(

    all_labels,

    all_predictions,

)

disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=[

        "Fake",

        "Real",

    ],

)

fig, ax = plt.subplots(

    figsize=(6,6)

)

disp.plot(

    ax=ax,

    colorbar=False,

)

plt.title("PrismShieldAI Test Confusion Matrix")

plt.tight_layout()

plt.savefig(

    RESULTS_DIR / "confusion_matrix.png",

    dpi=300,

)

plt.close()

print("Saved confusion_matrix.png")

# ==============================================================
# ROC CURVE
# ==============================================================

print()

print(PRINT_SEPARATOR)
print("GENERATING ROC CURVE")
print(PRINT_SEPARATOR)

fpr, tpr, _ = roc_curve(

    all_labels,

    all_probabilities,

)

plt.figure(figsize=(6,6))

plt.plot(

    fpr,

    tpr,

    label=f"AUC = {test_scores['roc_auc']:.4f}",

)

plt.plot(

    [0,1],

    [0,1],

    linestyle="--",

)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(

    RESULTS_DIR / "roc_curve.png",

    dpi=300,

)

plt.close()

print("Saved roc_curve.png")

# ==============================================================
# PRECISION RECALL CURVE
# ==============================================================

precision_curve, recall_curve, _ = precision_recall_curve(

    all_labels,

    all_probabilities,

)

plt.figure(figsize=(6,6))

plt.plot(

    recall_curve,

    precision_curve,

)

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("Precision Recall Curve")

plt.grid(True)

plt.tight_layout()

plt.savefig(

    RESULTS_DIR / "precision_recall_curve.png",

    dpi=300,

)

plt.close()

print("Saved precision_recall_curve.png")

# ==============================================================
# CONFIDENCE HISTOGRAM
# ==============================================================

print()

print(PRINT_SEPARATOR)
print("GENERATING CONFIDENCE HISTOGRAM")
print(PRINT_SEPARATOR)

plt.figure(figsize=(8,6))

plt.hist(

    all_confidences,

    bins=20,

)

plt.xlabel("Confidence Score")

plt.ylabel("Number of Samples")

plt.title("Prediction Confidence Distribution")

plt.grid(True)

plt.tight_layout()

plt.savefig(

    RESULTS_DIR / "confidence_histogram.png",

    dpi=300,

)

plt.close()

print("Saved confidence_histogram.png")

# ==============================================================
# CONFIDENCE STATISTICS
# ==============================================================

confidence_stats = {

    "mean_confidence": float(np.mean(all_confidences)),

    "median_confidence": float(np.median(all_confidences)),

    "min_confidence": float(np.min(all_confidences)),

    "max_confidence": float(np.max(all_confidences)),

    "std_confidence": float(np.std(all_confidences)),

}

print()

print(PRINT_SEPARATOR)
print("CONFIDENCE STATISTICS")
print(PRINT_SEPARATOR)

for key, value in confidence_stats.items():

    print(f"{key:20s}: {value:.4f}")

# ==============================================================
# SAVE CONFIDENCE STATS
# ==============================================================

with open(

    RESULTS_DIR / "confidence_statistics.json",

    "w",

) as f:

    json.dump(

        confidence_stats,

        f,

        indent=4,

    )

# ==============================================================
# EVALUATION SUMMARY
# ==============================================================

summary_lines = [

    "PrismShieldAI Evaluation Summary",

    "=" * 60,

    "",

    f"Checkpoint          : {BEST_MODEL_NAME}",

    f"Best Epoch          : {best_epoch}",

    f"Best Validation F1  : {best_f1:.4f}",

    "",

    f"Accuracy            : {test_scores['accuracy']:.4f}",

    f"Balanced Accuracy   : {test_scores['balanced_accuracy']:.4f}",

    f"Precision           : {test_scores['precision']:.4f}",

    f"Recall              : {test_scores['recall']:.4f}",

    f"F1 Score            : {test_scores['f1']:.4f}",

    f"ROC AUC             : {test_scores['roc_auc']:.4f}",

    f"PR AUC              : {test_scores['pr_auc']:.4f}",

    f"MCC                 : {test_scores['mcc']:.4f}",

    "",

    f"Average Loss        : {test_loss:.4f}",

    f"Classification Loss : {test_cls_loss:.4f}",

    f"Contrastive Loss    : {test_contrast_loss:.4f}",

    f"Confidence Loss     : {test_confidence_loss:.4f}",

    "",

    f"Samples Evaluated   : {len(all_labels)}",

    f"Evaluation Time     : {elapsed_time:.2f} sec",
    
    f"Checkpoint          : {BEST_MODEL_NAME}",
    
    f"Total Parameters    : {total_params:,}",

    "",

    "Generated Files:",

    "----------------------------",
    
    "classification_report_test.txt",
    
    "test_predictions.csv",
    
    "evaluation_metrics.json",

    "confidence_statistics.json",

    "confusion_matrix.png",

    "roc_curve.png",

    "precision_recall_curve.png",

    "confidence_histogram.png",

]

with open(

    RESULTS_DIR / "summary.txt",

    "w",

    encoding="utf-8",

) as f:

    f.write(

        "\n".join(summary_lines)

    )

# ==============================================================
# FINAL CONSOLE REPORT
# ==============================================================

print()

print(PRINT_SEPARATOR)
print("EVALUATION COMPLETE")
print(PRINT_SEPARATOR)

print()

print(f"Checkpoint : {BEST_MODEL_NAME}")

print(f"Epoch      : {best_epoch}")

print(f"F1         : {test_scores['f1']:.4f}")

print(f"Accuracy   : {test_scores['accuracy']:.4f}")

print(f"ROC AUC    : {test_scores['roc_auc']:.4f}")

print(f"PR AUC     : {test_scores['pr_auc']:.4f}")

print()

print("Artifacts saved to:")

print(RESULTS_DIR)

print()

generated_files = sorted(

    [

        f.name

        for f in RESULTS_DIR.glob("*")

        if f.is_file()

    ]

)

for file in generated_files:

    print(f"  • {file}")

print()

print(PRINT_SEPARATOR)
print("PrismShieldAI Standalone Evaluation Finished")
print(PRINT_SEPARATOR)