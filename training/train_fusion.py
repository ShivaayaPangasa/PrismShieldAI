"""
==============================================================
PrismShieldAI

Fusion Trainer

Project:
Causal and Contrastive Multimodal Reasoning
for Robust Deep Fake Detection

==============================================================
"""

# ==============================================================
# IMPORTS
# ==============================================================

import time

from datetime import datetime

import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from transformers import AutoProcessor

from utils.class_weights import ClassWeights

from pathlib import Path

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
from utils.early_stopping import EarlyStopping

# ==============================================================
# RANDOM SEED
# ==============================================================

set_seed(SEED)

# ==============================================================
# EXPERIMENT OUTPUT DIRECTORY
# ==============================================================

RUN_NAME = "fusion_v3_reasoning_final"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

EXPERIMENT_DIR = (

    RUNS_DIR

    / f"{RUN_NAME}_{TIMESTAMP}"

)

EXPERIMENT_DIR.mkdir(

    parents=True,

    exist_ok=True,

)

print()

print(PRINT_SEPARATOR)

print("EXPERIMENT DIRECTORY")

print(PRINT_SEPARATOR)

print(EXPERIMENT_DIR)

# ==============================================================
# SAVE TRAINING CONFIGURATION
# ==============================================================

import json

config = {

    "project_name": PROJECT_NAME,

    "run_name": RUN_NAME,

    "timestamp": TIMESTAMP,

    "epochs": EPOCHS,

    "batch_size": BATCH_SIZE,

    "learning_rate": LEARNING_RATE,

    "weight_decay": WEIGHT_DECAY,

    "temperature": TEMPERATURE,

    "classification_weight": CLASSIFICATION_WEIGHT,

    "contrastive_weight": CONTRASTIVE_WEIGHT,

    "confidence_weight": 0.20,

    "image_size": IMAGE_SIZE,

    "max_audio_length": MAX_AUDIO_LENGTH,

    "embedding_dim": EMBEDDING_DIM,

    "seed": SEED,

    "device": str(DEVICE),

    "best_checkpoint": BEST_MODEL_NAME,

    "last_checkpoint": LAST_MODEL_NAME,

}

with open(

    EXPERIMENT_DIR / "training_config.json",

    "w",

) as f:

    json.dump(

        config,

        f,

        indent=4,

    )

# ==============================================================
# DEVICE
# ==============================================================

print()

print(PRINT_SEPARATOR)

print("DEVICE")

print(PRINT_SEPARATOR)

print("Using:", DEVICE)

if DEVICE.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

# ==============================================================
# LOAD CSV
# ==============================================================

print()

print(PRINT_SEPARATOR)

print("DATASET")

print(PRINT_SEPARATOR)

dataframe = pd.read_csv(
    CSV_FILE
)

train_df = dataframe[
    dataframe["split"] == "train"
].reset_index(drop=True)

# ==========================================================
# BALANCE TRAINING SET
# ==========================================================

print()
print(PRINT_SEPARATOR)
print("BALANCING TRAIN DATASET")
print(PRINT_SEPARATOR)

# Separate classes
train_real = train_df[
    train_df["label"] == "real"
]

train_fake = train_df[
    train_df["label"] == "fake"
]

print(f"Original Real : {len(train_real)}")
print(f"Original Fake : {len(train_fake)}")

# ----------------------------------------------------------
# Keep all real samples
# ----------------------------------------------------------

real_count = len(train_real)

# ----------------------------------------------------------
# Keep 2x fake samples
# ----------------------------------------------------------

fake_target = min(

    len(train_fake),

    real_count * 2,

)

train_fake = train_fake.sample(

    n=fake_target,

    random_state=SEED,

)

# ----------------------------------------------------------
# Merge
# ----------------------------------------------------------

train_df = pd.concat(

    [

        train_real,

        train_fake,

    ],

    ignore_index=True,

)

# Shuffle

train_df = train_df.sample(

    frac=1,

    random_state=SEED,

).reset_index(drop=True)

print()
print("Balanced Training Set")

print("----------------------")

print(

    train_df["label"].value_counts()

)

print()

print("Total Samples :", len(train_df))


class_weights = ClassWeights().compute()

class_weights = class_weights.to(DEVICE)

val_df = dataframe[
    dataframe["split"] == "val"
].reset_index(drop=True)

test_df = dataframe[
    dataframe["split"] == "test"
].reset_index(drop=True)

print("Train      :", len(train_df))
print("Validation :", len(val_df))
print("Test       :", len(test_df))

# ==============================================================
# PROCESSOR
# ==============================================================

print()

print(PRINT_SEPARATOR)

print("PROCESSOR")

print(PRINT_SEPARATOR)

processor = AutoProcessor.from_pretrained(
    "facebook/wav2vec2-base"
)

print("facebook/wav2vec2-base")

# ==============================================================
# BUILD DATASETS
# ==============================================================

print()

print(PRINT_SEPARATOR)

print("BUILDING DATASETS")

print(PRINT_SEPARATOR)

train_dataset = FusionDataset(
    dataframe=train_df,
    processor=processor,
    image_size=IMAGE_SIZE,
    max_audio_length=MAX_AUDIO_LENGTH,
)

val_dataset = FusionDataset(
    dataframe=val_df,
    processor=processor,
    image_size=IMAGE_SIZE,
    max_audio_length=MAX_AUDIO_LENGTH,
)

test_dataset = FusionDataset(
    dataframe=test_df,
    processor=processor,
    image_size=IMAGE_SIZE,
    max_audio_length=MAX_AUDIO_LENGTH,
)

print("Train Samples :", len(train_dataset))
print("Validation    :", len(val_dataset))
print("Test          :", len(test_dataset))

# ==============================================================
# DATALOADERS
# ==============================================================

print()

print(PRINT_SEPARATOR)

print("DATALOADERS")

print(PRINT_SEPARATOR)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

print("Train Batches      :", len(train_loader))
print("Validation Batches :", len(val_loader))
print("Test Batches       :", len(test_loader))

print()

print("Batch Size :", BATCH_SIZE)
print("Workers    :", NUM_WORKERS)
print("Pin Memory :", PIN_MEMORY)

# ==============================================================
# SANITY CHECK
# ==============================================================

print()

print(PRINT_SEPARATOR)

print("SANITY CHECK")

print(PRINT_SEPARATOR)

sample = next(iter(train_loader))

print("Image Shape       :", sample["image"].shape)
print("Audio Shape       :", sample["input_values"].shape)
print("Attention Mask    :", sample["attention_mask"].shape)
print("Labels            :", sample["label"].shape)

print()

print("Sample IDs")

print(sample["sample_id"][:5])

print()

print("Video IDs")

print(sample["video_id"][:5])

print()

print(PRINT_SEPARATOR)
print("SANITY CHECK PASSED")
print(PRINT_SEPARATOR)

"""
==============================================================
PrismShieldAI

train_fusion_part1.py

Part 1
Model, Optimizer, AMP, Scheduler, Setup

Paste after the SANITY CHECK section of train_fusion.py
==============================================================
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

# ==========================================================
# BUILD MODEL
# ==========================================================

print()
print(PRINT_SEPARATOR)
print("BUILDING FUSION MODEL")
print(PRINT_SEPARATOR)

model = FusionModel().to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total Parameters     : {total_params:,}")
print(f"Trainable Parameters : {trainable_params:,}")

# ==========================================================
# LOSSES
# ==========================================================

criterion_cls = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.02,
)

criterion_contrast = ContrastiveLoss(
    temperature=TEMPERATURE
)

# ==========================================================
# CONFIDENCE LOSS
# ==========================================================

criterion_confidence = nn.BCELoss()

print()
print("Losses")
print("-------")
print(f"Classification Weight : {CLASSIFICATION_WEIGHT}")
print(f"Contrastive Weight    : {CONTRASTIVE_WEIGHT}")

CONFIDENCE_WEIGHT = 0.20

print(f"Confidence Weight     : {CONFIDENCE_WEIGHT}")

# ==========================================================
# OPTIMIZER
# ==========================================================

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

# ==========================================================
# LR SCHEDULER
# ==========================================================

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
)

# ==========================================================
# AMP
# ==========================================================

scaler = GradScaler(
    DEVICE.type,
    enabled=USE_AMP,
)

print()
print("Automatic Mixed Precision :", USE_AMP)

# ==========================================================
# UTILITIES
# ==========================================================

metrics = MetricsCalculator()

checkpoint = CheckpointManager(
    checkpoint_dir=WEIGHTS_DIR
)

early_stopping = EarlyStopping(
    patience=PATIENCE
)

best_f1 = 0.0
best_epoch = 0

history = {

    "train_loss": [],

    "val_loss": [],

    "accuracy": [],

    "balanced_accuracy": [],

    "precision": [],

    "recall": [],

    "f1": [],

    "roc_auc": [],

    "pr_auc": [],

    "mcc": [],

}

print()
print(PRINT_SEPARATOR)
print("TRAINING SETUP COMPLETE")
print(PRINT_SEPARATOR)
print(f"Epochs      : {EPOCHS}")
print(f"Batch Size  : {BATCH_SIZE}")
print(f"Device      : {DEVICE}")
print(f"LearningRate: {LEARNING_RATE}")
print(f"WeightDecay : {WEIGHT_DECAY}")
print(f"Temperature : {TEMPERATURE}")
print("Ready for Part 2 (Training Loop)")


"""
==============================================================
PrismShieldAI

train_fusion_part2.py

Part 2
Training Loop with AMP + tqdm

Requires Part 1.
==============================================================
"""

import time

print()
print(PRINT_SEPARATOR)
print("TRAINING LOOP")
print(PRINT_SEPARATOR)

for epoch in range(EPOCHS):

    epoch_start = time.time()
    
    # ============================================
    # Progressive Unfreezing
    # ============================================
    
    if epoch == 3:
        
        print("\nUnfreezing Visual Encoder...")
        
        model.unfreeze_visual()
    
    if epoch == 6:
        
        print("\nUnfreezing Audio Encoder...")
        
        model.unfreeze_audio()
    
    model.train()
    
    
    running_loss = 0.0
    running_cls = 0.0
    running_contrast = 0.0
    running_confidence = 0.0

    train_predictions = []
    train_probabilities = []
    train_labels = []

    iterator = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{EPOCHS}",
        leave=False,
    ) if tqdm else train_loader

    for batch in iterator:

        images = batch["image"].to(DEVICE, non_blocking=True)
        input_values = batch["input_values"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        
        with autocast(
            device_type=DEVICE.type,
            enabled=USE_AMP,
        ):

            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                images=images,
            )

            cls_loss = criterion_cls(
                outputs["logits"],
                labels,
            )

            contrastive_loss = criterion_contrast(
                outputs["audio_projection"],
                outputs["visual_projection"],
            )
            
            # ==========================================================
            # CONFIDENCE TARGET
            # ==========================================================

            predictions = outputs["logits"].argmax(dim=1)

            confidence_target = (

                predictions == labels

            ).float()
            
            # ==========================================================
            # TOTAL LOSS (without confidence)
            # ==========================================================

            total_loss = (

                CLASSIFICATION_WEIGHT * cls_loss +

                CONTRASTIVE_WEIGHT * contrastive_loss

            )

        # ==========================================================
        # CONFIDENCE LOSS
        # (outside AMP)
        # ==========================================================
        
        confidence_loss = criterion_confidence(
            outputs["confidence"].float().view(-1),
            confidence_target.float().view(-1),
        )

        total_loss = total_loss + (
            CONFIDENCE_WEIGHT * confidence_loss
        )

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += total_loss.item()
        running_cls += cls_loss.item()
        running_contrast += contrastive_loss.item()

        running_confidence += confidence_loss.item()

        predictions = outputs["logits"].argmax(dim=1)
        
        probabilities = torch.softmax(
            outputs["logits"],
            dim=1,
        )[:, 1]

        train_predictions.extend(
            predictions.detach().cpu().tolist()
        )

        train_labels.extend(
            labels.detach().cpu().tolist()
        )
        
        train_probabilities.extend(
            
            probabilities.detach().cpu().tolist()

        )

        if tqdm:
            iterator.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "cls": f"{cls_loss.item():.4f}",
                "con": f"{contrastive_loss.item():.4f}",
            })

    scheduler.step()
    
    train_scores = metrics.calculate(
        
        predictions=train_predictions,
        
        labels=train_labels,
        
        probabilities=train_probabilities,

    )
    
    epoch_train_loss = running_loss / len(train_loader)
    epoch_cls_loss = running_cls / len(train_loader)
    epoch_contrast_loss = running_contrast / len(train_loader)
    epoch_confidence_loss = running_confidence / len(train_loader)

    history["train_loss"].append(epoch_train_loss)

    print()
    print("-" * 60)
    print(f"Epoch {epoch+1} Training Summary")
    print("-" * 60)
    print(f"Total Loss        : {epoch_train_loss:.4f}")
    print(f"Classification    : {epoch_cls_loss:.4f}")
    print(f"Contrastive       : {epoch_contrast_loss:.4f}")
    print(f"Confidence        : {epoch_confidence_loss:.4f}")
    print(f"Accuracy          : {train_scores['accuracy']:.4f}")
    print(f"Precision         : {train_scores['precision']:.4f}")
    print(f"Recall            : {train_scores['recall']:.4f}")
    print(f"F1                : {train_scores['f1']:.4f}")
    print(f"Learning Rate     : {optimizer.param_groups[0]['lr']:.8f}")
    print(f"Epoch Time (sec)  : {time.time()-epoch_start:.2f}")

    # Validation code (Part 3) starts here.

    """
    ==============================================================
    PrismShieldAI

    train_fusion_part3.py

    Part 3
    Validation, Checkpointing, Early Stopping

    Append immediately after Part 2.
    ==============================================================
    """
    
    print()
    print(PRINT_SEPARATOR)
    print("VALIDATION")
    print(PRINT_SEPARATOR)

    model.eval()

    val_running_loss = 0.0
    val_running_cls = 0.0
    val_running_contrast = 0.0
    val_running_confidence = 0.0

    val_predictions = []
    val_labels = []
    val_probabilities = []

    val_iterator = tqdm(
        val_loader,
        desc="Validation",
        leave=False,
    ) if tqdm else val_loader
    
    prediction_rows = []

    with torch.no_grad():

        for batch in val_iterator:

            images = batch["image"].to(DEVICE, non_blocking=True)
            input_values = batch["input_values"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, non_blocking=True)
            
            with autocast(
                device_type=DEVICE.type,
                enabled=USE_AMP,
            ):

                outputs = model(
                    input_values=input_values,
                    attention_mask=attention_mask,
                    images=images,
                )
                
                cls_loss = criterion_cls(
                    outputs["logits"],
                    labels,
                )

                contrastive_loss = criterion_contrast(
                    outputs["audio_projection"],
                    outputs["visual_projection"],
                )

                preds = outputs["logits"].argmax(dim=1)

                confidence_target = (
                    preds == labels
                ).float()

                total_loss = (
                    CLASSIFICATION_WEIGHT * cls_loss +
                    CONTRASTIVE_WEIGHT * contrastive_loss 
                )
                
            
            confidence_loss = criterion_confidence(
                outputs["confidence"].float().view(-1),
                confidence_target.float().view(-1),
            )
            
            total_loss = total_loss + (
                CONFIDENCE_WEIGHT * confidence_loss
            )

            val_running_loss += total_loss.item()
            val_running_cls += cls_loss.item()
            val_running_contrast += contrastive_loss.item()
            val_running_confidence += confidence_loss.item()
            
            preds = outputs["logits"].argmax(dim=1)

            probabilities = torch.softmax(
                outputs["logits"],
                dim=1,
            )[:, 1]

            confidences = outputs["confidence"].squeeze()

            val_predictions.extend(
                preds.cpu().tolist()
            )

            val_labels.extend(
                labels.cpu().tolist()
            )

            val_probabilities.extend(
                probabilities.cpu().tolist()
            )           

            for i in range(len(preds)):
                
                prediction_rows.append({

                    "video_id": batch["video_id"][i].item(),

                    "label": labels[i].item(),

                    "prediction": preds[i].item(),

                    "probability": probabilities[i].item(),

                    "confidence": confidences[i].item(),

        })
            
            if tqdm:
                val_iterator.set_postfix({
                    "loss": f"{total_loss.item():.4f}"
                })

    # ==========================================================
    # METRICS
    # ==========================================================
    
    val_scores = metrics.calculate(
        predictions=val_predictions,
        labels=val_labels,
        probabilities=val_probabilities,
    )

    epoch_val_loss = val_running_loss / len(val_loader)
    epoch_val_cls = val_running_cls / len(val_loader)
    epoch_val_con = val_running_contrast / len(val_loader)
    epoch_val_conf = val_running_confidence / len(val_loader)

    history["val_loss"].append(epoch_val_loss)
    history["accuracy"].append(val_scores["accuracy"])
    history["precision"].append(val_scores["precision"])
    history["recall"].append(val_scores["recall"])
    history["f1"].append(val_scores["f1"])
    history["balanced_accuracy"].append(val_scores["balanced_accuracy"])
    history["roc_auc"].append(val_scores["roc_auc"])
    history["pr_auc"].append(val_scores["pr_auc"])
    history["mcc"].append(val_scores["mcc"])

    print()
    print("-" * 60)
    print(f"Epoch {epoch+1} Validation Summary")
    print("-" * 60)
    print(f"Validation Loss : {epoch_val_loss:.4f}")
    print(f"Classification  : {epoch_val_cls:.4f}")
    print(f"Contrastive     : {epoch_val_con:.4f}")
    print(f"Confidence      : {epoch_val_conf:.4f}")
    print(f"Accuracy        : {val_scores['accuracy']:.4f}")
    print(f"Precision       : {val_scores['precision']:.4f}")
    print(f"Recall          : {val_scores['recall']:.4f}")
    print(f"F1 Score        : {val_scores['f1']:.4f}")
    print(f"Balanced Acc    : {val_scores['balanced_accuracy']:.4f}")
    print(f"ROC AUC         : {val_scores['roc_auc']:.4f}")
    print(f"PR AUC          : {val_scores['pr_auc']:.4f}")
    print(f"MCC             : {val_scores['mcc']:.4f}")
    
    prediction_df = pd.DataFrame(
        
        prediction_rows

    )
    
    prediction_df.to_csv(

        EXPERIMENT_DIR

        / f"validation_predictions_epoch_{epoch+1}.csv",

        index=False,

    )

    # ==========================================================
    # CHECKPOINTS
    # ==========================================================

    checkpoint.save(
        model=model,
        optimizer=optimizer,
        epoch=epoch + 1,
        best_score=max(best_f1, val_scores["f1"]),
        filename=LAST_MODEL_NAME,
    )

    if val_scores["f1"] > best_f1:

        best_f1 = val_scores["f1"]
        best_epoch = epoch + 1

        checkpoint.save(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            best_score=best_f1,
            filename=BEST_MODEL_NAME,
        )

        print()
        print("⭐ New Best Model Saved!")

    # ==========================================================
    # EARLY STOPPING
    # ==========================================================

    if early_stopping.step(val_scores["f1"]):

        print()
        print("=" * 60)
        print("EARLY STOPPING TRIGGERED")
        print("=" * 60)
        print(f"Stopped at Epoch : {epoch+1}")
        print(f"Best Epoch       : {best_epoch}")
        print(f"Best F1          : {best_f1:.4f}")
        break

# Part 4 starts after the epoch loop finishes.


"""
==============================================================
PrismShieldAI

train_fusion_part4.py

Part 4
Testing, Confusion Matrix, Final Report

Append AFTER the training loop (after Part 3).
==============================================================
"""

print()
print(PRINT_SEPARATOR)
print("FINAL EVALUATION")
print(PRINT_SEPARATOR)

# ----------------------------------------------------------
# Load Best Model
# ----------------------------------------------------------

best_epoch_loaded, best_score_loaded = checkpoint.load(
    model=model,
    optimizer=optimizer,
    filename=BEST_MODEL_NAME,
)

print(f"Loaded Best Epoch : {best_epoch_loaded}")
print(f"Loaded Best F1    : {best_score_loaded:.4f}")

# ----------------------------------------------------------
# TEST LOOP
# ----------------------------------------------------------

model.eval()

test_predictions = []
test_labels = []
test_probabilities = []

test_loss = 0.0
test_cls_loss = 0.0
test_contrast_loss = 0.0
test_confidence_loss = 0.0

test_iterator = tqdm(
    test_loader,
    desc="Testing",
    leave=False,
) if tqdm else test_loader

with torch.no_grad():

    for batch in test_iterator:

        images = batch["image"].to(DEVICE, non_blocking=True)
        input_values = batch["input_values"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)

        with autocast(
            device_type=DEVICE.type,
            enabled=USE_AMP,
        ):

            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                images=images,
            )
            
            cls_loss = criterion_cls(
                outputs["logits"],
                labels,
            )

            contrast_loss = criterion_contrast(
                outputs["audio_projection"],
                outputs["visual_projection"],
            )

            preds = outputs["logits"].argmax(dim=1)

            confidence_target = (
                preds == labels
            ).float()
            
            total_loss = (
                CLASSIFICATION_WEIGHT * cls_loss +
                CONTRASTIVE_WEIGHT * contrast_loss
            )
        
        confidence_loss = criterion_confidence(
            outputs["confidence"].float().view(-1),
            confidence_target.float().view(-1),
        )

        total_loss = total_loss + (
            CONFIDENCE_WEIGHT * confidence_loss
        )
    
        test_loss += total_loss.item()
        test_cls_loss += cls_loss.item()
        test_contrast_loss += contrast_loss.item()
        test_confidence_loss += confidence_loss.item()

        preds = outputs["logits"].argmax(dim=1)
        
        probabilities = torch.softmax(
            outputs["logits"],
            dim=1,
        )[:, 1]

        test_predictions.extend(preds.cpu().tolist())
        test_labels.extend(labels.cpu().tolist())
        test_probabilities.extend(
            probabilities.detach().cpu().tolist()
        )

# ----------------------------------------------------------
# METRICS
# ----------------------------------------------------------

test_scores = metrics.calculate(

    predictions=test_predictions,

    labels=test_labels,

    probabilities=test_probabilities,

)

print()
print(PRINT_SEPARATOR)
print("TEST RESULTS")
print(PRINT_SEPARATOR)

print(f"Loss              : {test_loss/len(test_loader):.4f}")
print(f"Classification    : {test_cls_loss/len(test_loader):.4f}")
print(f"Contrastive       : {test_contrast_loss/len(test_loader):.4f}")
print(f"Confidence        : {test_confidence_loss/len(test_loader):.4f}")
print(f"Accuracy          : {test_scores['accuracy']:.4f}")
print(f"Precision         : {test_scores['precision']:.4f}")
print(f"Recall            : {test_scores['recall']:.4f}")
print(f"F1 Score          : {test_scores['f1']:.4f}")
print(f"Balanced Accuracy : {test_scores['balanced_accuracy']:.4f}")
print(f"ROC AUC           : {test_scores['roc_auc']:.4f}")
print(f"PR AUC            : {test_scores['pr_auc']:.4f}")
print(f"MCC               : {test_scores['mcc']:.4f}")

metrics_json = {

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

}

with open(

    EXPERIMENT_DIR / "evaluation_metrics.json",

    "w",

) as f:

    json.dump(

        metrics_json,

        f,

        indent=4,

    )
    
from sklearn.metrics import classification_report

report = classification_report(

    test_labels,

    test_predictions,

    target_names=[

        "Fake",

        "Real",

    ],

    digits=4,

)

with open(

    EXPERIMENT_DIR / "classification_report_test.txt",

    "w",

    encoding="utf-8",

) as f:

    f.write(report)
    
prediction_df = pd.DataFrame({

    "label": test_labels,

    "prediction": test_predictions,

    "probability": test_probabilities,

})

prediction_df.to_csv(

    EXPERIMENT_DIR

    / "test_predictions.csv",

    index=False,

)

# ----------------------------------------------------------
# TRAINING HISTORY
# ----------------------------------------------------------

print()
print(PRINT_SEPARATOR)
print("TRAINING HISTORY")
print(PRINT_SEPARATOR)

print(f"Epochs Completed : {len(history['train_loss'])}")
print(f"Best Epoch       : {best_epoch}")
print(f"Best Validation F1 : {best_f1:.4f}")

print()
print("Final Learning Rate :", optimizer.param_groups[0]["lr"])

print()
print(PRINT_SEPARATOR)
print("PrismShieldAI Training Complete")
print(PRINT_SEPARATOR)

print("Best Model :", WEIGHTS_DIR / BEST_MODEL_NAME)
print("Last Model :", WEIGHTS_DIR / LAST_MODEL_NAME)
print("Ready for inference.py")

print()

print(PRINT_SEPARATOR)

print("VERIFYING SAVED CHECKPOINT")

print(PRINT_SEPARATOR)

try:

    torch.load(

        WEIGHTS_DIR / BEST_MODEL_NAME,

        map_location="cpu",

    )

    print("Checkpoint verified successfully.")

except Exception as e:

    print("Checkpoint verification FAILED!")

    print(e)