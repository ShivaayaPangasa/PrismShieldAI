# ==========================================================
# VISUAL ENCODER V1 EVALUATION
# Prism Shield AI
# ==========================================================

from pathlib import Path
import json

import torch
import timm

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ==========================================================
# CONFIG
# ==========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_DIR = Path("processed_dataset/test")

WEIGHTS = Path("weights/visual_encoder_v1.pth")

OUTPUT = Path("evaluation/outputs")
OUTPUT.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 32

# ==========================================================
# TRANSFORM
# ==========================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ==========================================================
# DATASET
# ==========================================================

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Test Images:", len(test_dataset))
print("Classes:", test_dataset.classes)

# ==========================================================
# MODEL
# ==========================================================

model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=2
)

model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))

model = model.to(DEVICE)
model.eval()

print("Model Loaded")

# ==========================================================
# PREDICTION
# ==========================================================

predictions = []
labels = []
probabilities = []

with torch.no_grad():

    for images, target in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probs = torch.softmax(outputs, dim=1)

        preds = torch.argmax(probs, dim=1)

        probabilities.extend(probs[:,1].cpu().numpy())

        predictions.extend(preds.cpu().numpy())

        labels.extend(target.numpy())

# ==========================================================
# METRICS
# ==========================================================

accuracy = accuracy_score(labels, predictions)
precision = precision_score(labels, predictions)
recall = recall_score(labels, predictions)
f1 = f1_score(labels, predictions)

print("\nRESULTS")
print("="*40)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(labels, predictions)

plt.figure(figsize=(6,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=test_dataset.classes,
    yticklabels=test_dataset.classes
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig(
    OUTPUT/"confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Confusion Matrix Saved")

# ==========================================================
# ROC CURVE
# ==========================================================

fpr, tpr, _ = roc_curve(labels, probabilities)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")

plt.plot([0,1],[0,1],"--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.savefig(
    OUTPUT/"roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("ROC Curve Saved")

# ==========================================================
# PRECISION-RECALL CURVE
# ==========================================================

from sklearn.metrics import precision_recall_curve

precision_vals, recall_vals, _ = precision_recall_curve(
    labels,
    probabilities
)

plt.figure(figsize=(6,6))

plt.plot(recall_vals, precision_vals)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title("Precision-Recall Curve")

plt.grid(True)

plt.savefig(
    OUTPUT / "precision_recall_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Precision-Recall Curve Saved")

# ==========================================================
# REPORT
# ==========================================================

report = classification_report(
    labels,
    predictions,
    target_names=test_dataset.classes
)

with open(
    OUTPUT/"classification_report.txt",
    "w"
) as f:

    f.write(report)

print(report)

# ==========================================================
# SAVE METRICS
# ==========================================================

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc)
}

with open(
    OUTPUT/"metrics.json",
    "w"
) as f:

    json.dump(metrics, f, indent=4)

print("Metrics Saved")