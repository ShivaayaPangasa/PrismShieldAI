# ==========================================================
# PrismShieldAI
# Research Evaluation Pipeline
# PART 1 / 8
#
# Purpose:
# Evaluate the trained multimodal model ONLY on the
# held-out unseen test videos.
#
# Author : Shivaaya
# Project: PrismShieldAI
# ==========================================================


# ==========================================================
# IMPORTS
# ==========================================================

import os
import sys
import json
import time
import shutil
import logging
import platform
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

import torch

# ----------------------------------------------------------
# Project Imports
# ----------------------------------------------------------

from inference.inference import PrismShieldInference
from configs.settings import *


# ==========================================================
# EVALUATION PATHS
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_CSV = ROOT_DIR / "processed_dataset" / "research_dataset.csv"

DATASET_ROOT = ROOT_DIR / DATASET_PATH

OUTPUT_DIR = ROOT_DIR / "evaluation_results"

PREDICTIONS_CSV = OUTPUT_DIR / "predictions.csv"

MISCLASSIFIED_CSV = OUTPUT_DIR / "misclassified.csv"

METRICS_JSON = OUTPUT_DIR / "metrics.json"

SUMMARY_FILE = OUTPUT_DIR / "summary.txt"

CLASSIFICATION_REPORT_FILE = OUTPUT_DIR / "classification_report.txt"

LOG_FILE = OUTPUT_DIR / "evaluation.log"

FAILURE_DIR = OUTPUT_DIR / "failure_gallery"

FAILURE_FRAME_DIR = FAILURE_DIR / "frames"

FAILURE_AUDIO_DIR = FAILURE_DIR / "audio"


# ==========================================================
# CREATE OUTPUT DIRECTORIES
# ==========================================================

OUTPUT_DIR.mkdir(exist_ok=True)

FAILURE_DIR.mkdir(exist_ok=True)

FAILURE_FRAME_DIR.mkdir(exist_ok=True)

FAILURE_AUDIO_DIR.mkdir(exist_ok=True)


# ==========================================================
# LOGGING
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("PrismShieldEvaluation")


# ==========================================================
# UTILITY FUNCTIONS
# ==========================================================

def print_separator(length=70):
    """Print separator line."""

    print("=" * length)


def log_separator(length=70):
    """Write separator to log."""

    logger.info("=" * length)


def timestamp():
    """Current timestamp."""

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_divide(a, b):
    """Avoid divide-by-zero."""

    if b == 0:
        return 0.0

    return a / b


def ensure_exists(path: Path):
    """
    Ensure a file exists.

    Raises
    ------
    FileNotFoundError
    """

    if not path.exists():
        raise FileNotFoundError(f"Missing:\n{path}")


def seconds_to_hms(seconds):
    """
    Convert seconds to HH:MM:SS
    """

    seconds = int(seconds)

    hours = seconds // 3600

    minutes = (seconds % 3600) // 60

    secs = seconds % 60

    return f"{hours:02}:{minutes:02}:{secs:02}"


print_separator()
print("PrismShieldAI Evaluation Pipeline")
print("Part 1 Loaded")
print_separator()

# ==========================================================
# PART 2 / 8
# DATASET LOADING & VALIDATION
# ==========================================================


# ==========================================================
# VERIFY REQUIRED FILES
# ==========================================================

print_separator()
print("Verifying Required Files")
print_separator()

ensure_exists(DATASET_CSV)
ensure_exists(DATASET_ROOT)

logger.info("Research dataset : %s", DATASET_CSV)
logger.info("Dataset root     : %s", DATASET_ROOT)

print("✓ research_dataset.csv found")
print("✓ Dataset directory found")


# ==========================================================
# LOAD INFERENCE ENGINE
# ==========================================================

print()
print_separator()
print("Loading PrismShieldAI Inference Engine")
print_separator()

engine = PrismShieldInference()

print()
print("✓ Inference Engine Ready")


# ==========================================================
# LOAD RESEARCH DATASET
# ==========================================================

print()
print_separator()
print("Loading Research Dataset")
print_separator()

dataset = pd.read_csv(DATASET_CSV)

logger.info("Total metadata rows : %d", len(dataset))

print(f"Total Samples : {len(dataset):,}")


# ==========================================================
# KEEP ONLY HELD-OUT TEST SPLIT
# ==========================================================

test_df = dataset[
    dataset["split"].str.lower() == "test"
].copy()

print(f"Test Samples : {len(test_df):,}")

logger.info("Test samples : %d", len(test_df))


# ==========================================================
# KEEP ONLY ONE ENTRY PER ORIGINAL VIDEO
# ==========================================================

test_df = (
    test_df
    .sort_values("video_id")
    .drop_duplicates(subset="video_id")
    .reset_index(drop=True)
)

LABEL_MAP = {
    "real": 1,
    "fake": 0,
}

test_df["label_id"] = (
    test_df["label"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map(LABEL_MAP)
)

print(f"Unique Test Videos : {len(test_df):,}")

logger.info(
    "Unique evaluation videos : %d",
    len(test_df)
)


# ==========================================================
# VERIFY DATASET INTEGRITY
# ==========================================================

print()
print_separator()
print("Verifying Video Paths")
print_separator()

missing_videos = []

video_paths = []

for _, row in test_df.iterrows():

    video_path = DATASET_ROOT / row["source_video"]

    video_paths.append(video_path)

    if not video_path.exists():
        missing_videos.append(video_path)

if len(missing_videos) > 0:

    print()

    print(f"Missing Videos : {len(missing_videos)}")

    logger.error(
        "Missing %d videos",
        len(missing_videos)
    )

    for path in missing_videos[:10]:
        print(path)

    raise FileNotFoundError(
        "Dataset integrity check failed."
    )

print("✓ All test videos located successfully")

logger.info("Dataset integrity verified")


# ==========================================================
# DATASET SUMMARY
# ==========================================================

labels = test_df["label"].astype(str).str.lower()

real_videos = labels.eq("real").sum()
fake_videos = labels.eq("fake").sum()

print()
print_separator()
print("Held-Out Test Dataset Summary")
print_separator()

print(f"Total Videos : {len(test_df):,}")
print(f"Real Videos  : {real_videos:,}")
print(f"Fake Videos  : {fake_videos:,}")

logger.info("Real videos : %d", real_videos)
logger.info("Fake videos : %d", fake_videos)


# ==========================================================
# STORAGE FOR RESULTS
# ==========================================================

predictions = []

ground_truth = []

prediction_scores = []

prediction_labels = []

confidence_scores = []

inference_times = []

failed_videos = []

print()
print("✓ Dataset Ready For Evaluation")
print_separator()

# ==========================================================
# PART 3 / 8
# VIDEO EVALUATION LOOP
# ==========================================================

print()
print_separator()
print("Starting Evaluation")
print_separator()

evaluation_start = time.perf_counter()

LABEL_MAP = {
    "real": 1,
    "fake": 0,
}

for index, row in tqdm(
    test_df.iterrows(),
    total=len(test_df),
    desc="Evaluating Videos",
):

    video_path = DATASET_ROOT / row["source_video"]
    
    true_label = row["label_id"]

    video_id = row["video_id"]

    # ------------------------------------------------------
    # Measure Inference Time
    # ------------------------------------------------------

    start_time = time.perf_counter()

    try:
        
        result = engine.predict_from_video(video_path)

        print()
        print("=" * 70)
        print("DEBUG VIDEO")
        print("=" * 70)
        print("Video:", video_path)
        print("Ground Truth:", true_label)
        print("Prediction:", result["prediction"])
        print("Probability:", result["probability"])
        print("Confidence:", result["confidence"])
        print("=" * 70)

        end_time = time.perf_counter()

        inference_time = end_time - start_time

        inference_times.append(inference_time)

        # --------------------------------------------------
        # Check inference status
        # --------------------------------------------------

        if result["status"] != "success":

            raise RuntimeError(result["error"])

        predicted_label = int(result["class_id"])

        probability = float(result["probability"])

        confidence = float(result["confidence"])

        prediction = result["prediction"]
        
        frame = result.get("selected_frame")

        if isinstance(frame, np.ndarray):
            cv2.imwrite("debug_selected_frame.jpg", frame)
            print("Saved: debug_selected_frame.jpg")

        # --------------------------------------------------
        # Store metrics
        # --------------------------------------------------

        ground_truth.append(true_label)

        prediction_labels.append(predicted_label)

        prediction_scores.append(probability)

        confidence_scores.append(confidence)

        predictions.append({

            "video_id": video_id,

            "video_path": str(video_path),

            "true_label": true_label,

            "predicted_label": predicted_label,

            "prediction": prediction,

            "probability": probability,

            "confidence": confidence,

            "correct": predicted_label == true_label,

            "inference_time_sec": round(inference_time, 4),

        })

        # --------------------------------------------------
        # Save Misclassified Samples
        # --------------------------------------------------
        
        if predicted_label != true_label:
            
            import numpy as np
            import cv2

            frame = result.get("selected_frame")
            audio_path = result.get("audio_path")

            # -----------------------------
            # Save frame
            # -----------------------------
            if isinstance(frame, np.ndarray):

                output_path = FAILURE_FRAME_DIR / f"{video_id}_frame.jpg"
                cv2.imwrite(str(output_path), frame)

            elif isinstance(frame, (str, Path)):

                frame_path = Path(frame)

                if frame_path.exists():

                    shutil.copy2(
                        frame_path,
                        FAILURE_FRAME_DIR / frame_path.name,
                )   

            # -----------------------------
            # Save audio
            # -----------------------------
            if audio_path is not None:

                audio_path = Path(audio_path)

            if audio_path.exists():

                shutil.copy2(
                    audio_path,
                    FAILURE_AUDIO_DIR / audio_path.name,
                )

        logger.info(

            "Video %s | GT=%d | Pred=%d | Conf=%.4f | Time=%.3fs",

            video_id,

            true_label,

            predicted_label,

            confidence,

            inference_time,

        )

    # ------------------------------------------------------
    # Error Handling
    # ------------------------------------------------------

    except Exception as e:

        logger.exception(

            "Failed Video : %s",

            video_path,

        )

        failed_videos.append({

            "video_id": video_id,

            "video_path": str(video_path),

            "error": str(e),

        })

        continue


evaluation_end = time.perf_counter()

total_evaluation_time = evaluation_end - evaluation_start

print()
print_separator()
print("Evaluation Finished")
print_separator()

print(f"Videos Evaluated : {len(predictions):,}")

print(f"Videos Failed    : {len(failed_videos):,}")

print(f"Total Time       : {seconds_to_hms(total_evaluation_time)}")

print(f"Average Time     : {safe_divide(total_evaluation_time, len(predictions)):.3f} sec/video")

logger.info("Evaluation complete")
logger.info("Successful videos : %d", len(predictions))
logger.info("Failed videos : %d", len(failed_videos))
logger.info("Evaluation time : %.2f sec", total_evaluation_time)

# ==========================================================
# PART 4 / 8
# SAVE PREDICTIONS & PER-VIDEO STATISTICS
# ==========================================================

print()
print_separator()
print("Saving Evaluation Results")
print_separator()

# ----------------------------------------------------------
# Create Predictions DataFrame
# ----------------------------------------------------------

predictions_df = pd.DataFrame(predictions)

if predictions_df.empty:
    raise RuntimeError(
        "No successful predictions were generated."
    )

# ----------------------------------------------------------
# Save All Predictions
# ----------------------------------------------------------

predictions_df.to_csv(
    PREDICTIONS_CSV,
    index=False,
)

logger.info(
    "Predictions saved -> %s",
    PREDICTIONS_CSV,
)

print(f"✓ Predictions CSV : {PREDICTIONS_CSV}")


# ----------------------------------------------------------
# Misclassified Videos
# ----------------------------------------------------------

misclassified_df = predictions_df[
    predictions_df["correct"] == False
].copy()

misclassified_df.to_csv(
    MISCLASSIFIED_CSV,
    index=False,
)

logger.info(
    "Misclassified samples : %d",
    len(misclassified_df),
)

print(
    f"✓ Misclassified CSV : {MISCLASSIFIED_CSV}"
)


# ----------------------------------------------------------
# Timing Statistics
# ----------------------------------------------------------

average_time = float(np.mean(inference_times))

median_time = float(np.median(inference_times))

minimum_time = float(np.min(inference_times))

maximum_time = float(np.max(inference_times))

std_time = float(np.std(inference_times))


# ----------------------------------------------------------
# Confidence Statistics
# ----------------------------------------------------------

average_confidence = float(
    np.mean(confidence_scores)
)

minimum_confidence = float(
    np.min(confidence_scores)
)

maximum_confidence = float(
    np.max(confidence_scores)
)

std_confidence = float(
    np.std(confidence_scores)
)


# ----------------------------------------------------------
# Correct / Incorrect Counts
# ----------------------------------------------------------

correct_predictions = int(
    predictions_df["correct"].sum()
)

incorrect_predictions = (
    len(predictions_df)
    - correct_predictions
)

success_rate = safe_divide(
    correct_predictions,
    len(predictions_df),
)


# ----------------------------------------------------------
# Console Summary
# ----------------------------------------------------------

print()
print_separator()
print("Per-Video Evaluation Summary")
print_separator()

print(f"Videos Evaluated     : {len(predictions_df):,}")
print(f"Correct Predictions  : {correct_predictions:,}")
print(f"Incorrect Predictions: {incorrect_predictions:,}")
print(f"Failed Videos        : {len(failed_videos):,}")

print()

print(f"Average Confidence : {average_confidence:.4f}")
print(f"Minimum Confidence : {minimum_confidence:.4f}")
print(f"Maximum Confidence : {maximum_confidence:.4f}")

print()

print(f"Average Time : {average_time:.4f} sec")
print(f"Median Time  : {median_time:.4f} sec")
print(f"Fastest      : {minimum_time:.4f} sec")
print(f"Slowest      : {maximum_time:.4f} sec")

logger.info("Average confidence : %.4f", average_confidence)
logger.info("Average inference time : %.4f sec", average_time)
logger.info("Prediction success rate : %.4f", success_rate)

print()
print("✓ Prediction statistics generated")
print_separator()

# ==========================================================
# PART 5 / 8
# MODEL EVALUATION METRICS
# ==========================================================

print()
print_separator()
print("Computing Evaluation Metrics")
print_separator()

# ----------------------------------------------------------
# Basic Classification Metrics
# ----------------------------------------------------------

accuracy = accuracy_score(
    ground_truth,
    prediction_labels,
)

precision = precision_score(
    ground_truth,
    prediction_labels,
    zero_division=0,
)

recall = recall_score(
    ground_truth,
    prediction_labels,
    zero_division=0,
)

f1 = f1_score(
    ground_truth,
    prediction_labels,
    zero_division=0,
)

balanced_accuracy = balanced_accuracy_score(
    ground_truth,
    prediction_labels,
)


# ----------------------------------------------------------
# ROC-AUC
# ----------------------------------------------------------

try:

    roc_auc = roc_auc_score(
        ground_truth,
        prediction_scores,
    )

except ValueError:

    roc_auc = float("nan")

    logger.warning(
        "ROC-AUC could not be computed."
    )


# ----------------------------------------------------------
# Confusion Matrix
# ----------------------------------------------------------

cm = confusion_matrix(
    ground_truth,
    prediction_labels,
)

tn, fp, fn, tp = cm.ravel()


# ----------------------------------------------------------
# Per-Class Accuracy
# ----------------------------------------------------------

real_accuracy = safe_divide(
    tn,
    tn + fp,
)

fake_accuracy = safe_divide(
    tp,
    tp + fn,
)


# ----------------------------------------------------------
# Additional Metrics
# ----------------------------------------------------------

specificity = safe_divide(
    tn,
    tn + fp,
)

false_positive_rate = safe_divide(
    fp,
    fp + tn,
)

false_negative_rate = safe_divide(
    fn,
    fn + tp,
)


# ----------------------------------------------------------
# Classification Report
# ----------------------------------------------------------

classification_text = classification_report(

    ground_truth,

    prediction_labels,

    target_names=[

        "REAL",

        "FAKE",

    ],

    digits=4,

    zero_division=0,

)

with open(

    CLASSIFICATION_REPORT_FILE,

    "w",

    encoding="utf-8",

) as file:

    file.write(classification_text)


# ----------------------------------------------------------
# Console Output
# ----------------------------------------------------------

print()
print_separator()
print("Classification Metrics")
print_separator()

print(f"Accuracy           : {accuracy:.4f}")
print(f"Precision          : {precision:.4f}")
print(f"Recall             : {recall:.4f}")
print(f"F1 Score           : {f1:.4f}")
print(f"Balanced Accuracy  : {balanced_accuracy:.4f}")
print(f"ROC-AUC            : {roc_auc:.4f}")

print()

print(f"REAL Accuracy      : {real_accuracy:.4f}")
print(f"FAKE Accuracy      : {fake_accuracy:.4f}")

print()

print(f"Specificity        : {specificity:.4f}")
print(f"False Positive Rate: {false_positive_rate:.4f}")
print(f"False Negative Rate: {false_negative_rate:.4f}")

logger.info("Accuracy           : %.4f", accuracy)
logger.info("Precision          : %.4f", precision)
logger.info("Recall             : %.4f", recall)
logger.info("F1 Score           : %.4f", f1)
logger.info("Balanced Accuracy  : %.4f", balanced_accuracy)
logger.info("ROC-AUC            : %.4f", roc_auc)
logger.info("REAL Accuracy      : %.4f", real_accuracy)
logger.info("FAKE Accuracy      : %.4f", fake_accuracy)

print()
print("✓ Evaluation metrics computed")
print_separator()

# ==========================================================
# PART 6 / 8
# VISUALIZATION
# ==========================================================

print()
print_separator()
print("Generating Visualizations")
print_separator()

CONFUSION_MATRIX_PNG = OUTPUT_DIR / "confusion_matrix.png"
ROC_CURVE_PNG = OUTPUT_DIR / "roc_curve.png"
CONFIDENCE_HISTOGRAM_PNG = OUTPUT_DIR / "confidence_histogram.png"

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

plt.figure(figsize=(6, 6))

plt.imshow(
    cm,
    interpolation="nearest",
    cmap="Blues",
)

plt.title("Confusion Matrix")

plt.colorbar()

classes = ["REAL", "FAKE"]

tick_marks = np.arange(len(classes))

plt.xticks(tick_marks, classes)

plt.yticks(tick_marks, classes)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=12,
        )

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PNG,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("✓ Confusion Matrix Saved")

logger.info(
    "Confusion matrix saved."
)


# ==========================================================
# ROC CURVE
# ==========================================================

if not np.isnan(roc_auc):

    fpr, tpr, _ = roc_curve(
        ground_truth,
        prediction_scores,
    )

    plt.figure(figsize=(7, 6))

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"AUC = {roc_auc:.4f}",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend(loc="lower right")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        ROC_CURVE_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("✓ ROC Curve Saved")

    logger.info(
        "ROC curve saved."
    )

else:

    logger.warning(
        "ROC curve skipped."
    )


# ==========================================================
# CONFIDENCE HISTOGRAM
# ==========================================================

plt.figure(figsize=(8, 5))

plt.hist(
    confidence_scores,
    bins=20,
)

plt.xlabel("Prediction Confidence")

plt.ylabel("Number of Videos")

plt.title("Confidence Distribution")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    CONFIDENCE_HISTOGRAM_PNG,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("✓ Confidence Histogram Saved")

logger.info(
    "Confidence histogram saved."
)

print()
print_separator()
print("Visualization Complete")
print_separator()

# ==========================================================
# PART 7 / 8
# REPORTS & METADATA
# ==========================================================

print()
print_separator()
print("Saving Reports")
print_separator()

# ----------------------------------------------------------
# Environment Information
# ----------------------------------------------------------

environment_info = {

    "timestamp": timestamp(),

    "platform": platform.platform(),

    "python_version": platform.python_version(),

    "pytorch_version": torch.__version__,

    "cuda_available": torch.cuda.is_available(),

    "cuda_version": torch.version.cuda,

    "device": (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    ),

}

# ----------------------------------------------------------
# Metrics Dictionary
# ----------------------------------------------------------

metrics = {

    "accuracy": float(accuracy),

    "precision": float(precision),

    "recall": float(recall),

    "f1_score": float(f1),

    "balanced_accuracy": float(balanced_accuracy),

    "roc_auc": (
        None
        if np.isnan(roc_auc)
        else float(roc_auc)
    ),

    "real_accuracy": float(real_accuracy),

    "fake_accuracy": float(fake_accuracy),

    "specificity": float(specificity),

    "false_positive_rate": float(false_positive_rate),

    "false_negative_rate": float(false_negative_rate),

    "videos_evaluated": int(len(predictions_df)),

    "failed_videos": int(len(failed_videos)),

    "correct_predictions": int(correct_predictions),

    "incorrect_predictions": int(incorrect_predictions),

    "average_confidence": float(average_confidence),

    "average_inference_time": float(average_time),

    "median_inference_time": float(median_time),

    "fastest_inference_time": float(minimum_time),

    "slowest_inference_time": float(maximum_time),

    "confusion_matrix": {

        "true_negative": int(tn),

        "false_positive": int(fp),

        "false_negative": int(fn),

        "true_positive": int(tp),

    },

    "environment": environment_info,

}

# ----------------------------------------------------------
# Save JSON Metrics
# ----------------------------------------------------------

with open(

    METRICS_JSON,

    "w",

    encoding="utf-8",

) as file:

    json.dump(

        metrics,

        file,

        indent=4,

    )

print("✓ metrics.json saved")

logger.info("Metrics JSON saved.")


# ----------------------------------------------------------
# Save Summary Report
# ----------------------------------------------------------

with open(

    SUMMARY_FILE,

    "w",

    encoding="utf-8",

) as file:

    file.write("PrismShieldAI Evaluation Summary\n")

    file.write("=" * 60 + "\n\n")

    file.write(f"Timestamp                : {timestamp()}\n")

    file.write(f"Videos Evaluated         : {len(predictions_df)}\n")

    file.write(f"Correct Predictions      : {correct_predictions}\n")

    file.write(f"Incorrect Predictions    : {incorrect_predictions}\n")

    file.write(f"Failed Videos            : {len(failed_videos)}\n\n")

    file.write(f"Accuracy                 : {accuracy:.4f}\n")

    file.write(f"Precision                : {precision:.4f}\n")

    file.write(f"Recall                   : {recall:.4f}\n")

    file.write(f"F1 Score                 : {f1:.4f}\n")

    file.write(f"Balanced Accuracy        : {balanced_accuracy:.4f}\n")

    file.write(f"ROC-AUC                  : {roc_auc:.4f}\n\n")

    file.write(f"REAL Accuracy            : {real_accuracy:.4f}\n")

    file.write(f"FAKE Accuracy            : {fake_accuracy:.4f}\n\n")

    file.write(f"Average Confidence       : {average_confidence:.4f}\n")

    file.write(f"Average Inference Time   : {average_time:.4f} sec\n")

    file.write(f"Median Inference Time    : {median_time:.4f} sec\n")

    file.write(f"Fastest Inference        : {minimum_time:.4f} sec\n")

    file.write(f"Slowest Inference        : {maximum_time:.4f} sec\n\n")

    file.write("Environment\n")

    file.write("-" * 40 + "\n")

    for key, value in environment_info.items():

        file.write(f"{key:20}: {value}\n")

print("✓ summary.txt saved")

logger.info("Summary report saved.")

print("✓ classification_report.txt saved")

logger.info("Classification report already saved.")

print()
print_separator()
print("Reports Saved Successfully")
print_separator()

# ==========================================================
# PART 8 / 8
# FINAL SUMMARY & COMPLETION
# ==========================================================

print()
print_separator()
print("PrismShieldAI Evaluation Complete")
print_separator()

total_runtime = time.perf_counter() - evaluation_start

print(f"Evaluation Finished : {timestamp()}")
print(f"Total Runtime       : {seconds_to_hms(total_runtime)}")
print(f"Videos Evaluated    : {len(predictions_df):,}")
print(f"Failed Videos       : {len(failed_videos):,}")

print()

print("Final Performance")
print("-----------------")
print(f"Accuracy            : {accuracy:.4f}")
print(f"Precision           : {precision:.4f}")
print(f"Recall              : {recall:.4f}")
print(f"F1 Score            : {f1:.4f}")
print(f"Balanced Accuracy   : {balanced_accuracy:.4f}")
print(f"ROC-AUC             : {roc_auc:.4f}")

print()

print("Generated Artifacts")
print("-------------------")
print(f"Predictions CSV         : {PREDICTIONS_CSV}")
print(f"Misclassified CSV       : {MISCLASSIFIED_CSV}")
print(f"Metrics JSON            : {METRICS_JSON}")
print(f"Classification Report   : {CLASSIFICATION_REPORT_FILE}")
print(f"Summary Report          : {SUMMARY_FILE}")
print(f"Confusion Matrix        : {CONFUSION_MATRIX_PNG}")
print(f"ROC Curve              : {ROC_CURVE_PNG}")
print(f"Confidence Histogram    : {CONFIDENCE_HISTOGRAM_PNG}")
print(f"Evaluation Log          : {LOG_FILE}")
print(f"Failure Gallery         : {FAILURE_DIR}")

print()
print("Research Evaluation Successfully Completed.")
print("PrismShieldAI evaluation pipeline finished.")

logger.info("=" * 70)
logger.info("PrismShieldAI Evaluation Finished")
logger.info("Runtime : %s", seconds_to_hms(total_runtime))
logger.info("Accuracy : %.4f", accuracy)
logger.info("Precision : %.4f", precision)
logger.info("Recall : %.4f", recall)
logger.info("F1 Score : %.4f", f1)
logger.info("Balanced Accuracy : %.4f", balanced_accuracy)
logger.info("ROC-AUC : %.4f", roc_auc)
logger.info("Reports saved to: %s", OUTPUT_DIR)
logger.info("=" * 70)


# ==========================================================
# END OF FILE
# ==========================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("PrismShieldAI Research Evaluation Pipeline Finished")
    print("=" * 70)