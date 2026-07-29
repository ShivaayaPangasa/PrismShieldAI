"""
==============================================================
PrismShieldAI

Video Pipeline Evaluation
==============================================================
"""
import random
import shutil
import cv2
import pandas as pd
import time
import sys
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
) 

import matplotlib.pyplot as plt
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


from inference.inference import PrismShieldInference

print("=" * 70)
print("PrismShieldAI Video Pipeline Evaluation")
print("=" * 70)

# ----------------------------------------------------------
# DATASET PATH
# ----------------------------------------------------------

DATASET_ROOT = Path(
    "datasets/FakeAVCeleb_v1.2/FakeAVCeleb_v1.2"
)

print(f"Dataset Root : {DATASET_ROOT}")

# ----------------------------------------------------------
# LOAD ENGINE
# ----------------------------------------------------------

engine = PrismShieldInference()

print("\nInference engine loaded successfully.\n")

# ==========================================================
# SETTINGS
# ==========================================================

VIDEOS_PER_CLASS = None

DEBUG_DIR = Path("evaluation/video_pipeline")

FRAME_DIR = DEBUG_DIR / "frames"

AUDIO_DIR = DEBUG_DIR / "audio"

DEBUG_DIR.mkdir(parents=True, exist_ok=True)
FRAME_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

random.seed(42)

REAL_FOLDERS = [

    "RealVideo-RealAudio",

]

FAKE_FOLDERS = [

    "FakeVideo-FakeAudio",

    "FakeVideo-RealAudio",

    "RealVideo-FakeAudio",

]

def collect(folder_list, total=None):

    videos = []

    for folder in folder_list:

        path = DATASET_ROOT / folder

        videos.extend(
            list(path.rglob("*.mp4"))
        )

    random.shuffle(videos)

    if total is None:
        return videos

    return videos[:total]

# ==========================================================
# LOAD TEST SPLIT FROM RESEARCH CSV
# ==========================================================

csv_path = Path("processed_dataset/research_dataset.csv")

df_split = pd.read_csv(csv_path)

# Keep only unseen test samples
df_test = df_split[df_split["split"] == "test"].copy()

# One entry per original video
df_test = df_test.drop_duplicates(subset=["video_id"])

evaluation_set = []

for _, row in df_test.iterrows():

    video_path = DATASET_ROOT / row["source_video"]

    expected = row["label"].upper()

    evaluation_set.append(

        (video_path, expected)

    )

print(f"Test Videos : {len(evaluation_set)}")

print()

print("REAL Videos :",

      len(df_test[df_test["label"] == "real"]))

print("FAKE Videos :",

      len(df_test[df_test["label"] == "fake"]))

# ==========================================================
# EVALUATION
# ==========================================================

results = []

print("\n")
print("=" * 70)
print("STARTING VIDEO EVALUATION")
print("=" * 70)
start_time = time.time()

for index, (video, expected) in enumerate(evaluation_set, start=1):
    
    print(f"\n[{index}/{len(evaluation_set)}] {video.relative_to(DATASET_ROOT)}")

    result = engine.predict_from_video(video)
    
    if result["status"] != "success":

        print("FAILED")

        print(result["error"])

        results.append({

            "video": str(video.relative_to(DATASET_ROOT)),

            "folder": str(video.parent),

            "expected": expected,

            "prediction": "ERROR",

            "probability": None,

            "confidence": None,

            "correct": False,

            "error": result["error"],

        })

        continue

    prediction = result["prediction"]

    probability = result["probability"]

    confidence = result["confidence"]

    correct = prediction == expected
    
    relative = video.relative_to(DATASET_ROOT)

    safe_name = "_".join(relative.parts).replace(".mp4", "")
    
    
    
    
    # ==========================================================
    # SAVE ONLY MISCLASSIFIED SAMPLES
    # ==========================================================

    if not correct:

        frame = result["selected_frame"]

        if frame is not None:

            frame_path = FRAME_DIR / f"{safe_name}.jpg"

            cv2.imwrite(

                str(frame_path),

                frame,

            )

        audio = result["audio_path"]

        if audio is not None:

            shutil.copy(

                audio,

                AUDIO_DIR / f"{safe_name}.wav",

            )
    
    results.append({

        "video": str(video.relative_to(DATASET_ROOT)),

        "folder": str(video.parent),

        "expected": expected,

        "prediction": prediction,

        "probability": probability,

        "confidence": confidence,

        "correct": correct,

        "error": None,

    })
    
    print(
    f"[{index}/{len(evaluation_set)}] "
    f"{prediction} "
    f"(Expected: {expected}) "
    f"Prob={probability:.4f}"
    )
    
elapsed = time.time() - start_time

print(f"Evaluation completed in {elapsed:.2f} seconds")
    
# ==========================================================
# SAVE RESULTS
# ==========================================================

df = pd.DataFrame(results)

csv_path = DEBUG_DIR / "predictions.csv"

df.to_csv(csv_path, index=False)

print(f"\nPredictions saved to:\n{csv_path}")

# ==========================================================
# FILTER VALID PREDICTIONS
# ==========================================================

valid = df[df["prediction"] != "ERROR"].copy()

# ==========================================================
# CLASS COUNTS
# ==========================================================

total_real = (valid["expected"] == "REAL").sum()

total_fake = (valid["expected"] == "FAKE").sum()

correct_real = (

    (valid["expected"] == "REAL")

    &

    (valid["prediction"] == "REAL")

).sum()

correct_fake = (

    (valid["expected"] == "FAKE")

    &

    (valid["prediction"] == "FAKE")

).sum()

# ==========================================================
# LABEL MAPPING
# ==========================================================

label_map = {

    "REAL": 1,

    "FAKE": 0,

}

y_true = valid["expected"].map(label_map)

y_pred = valid["prediction"].map(label_map)

# ==========================================================
# METRICS
# ==========================================================

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(y_true, y_pred)

recall = recall_score(y_true, y_pred)

f1 = f1_score(y_true, y_pred)

# ROC-AUC
#
# Since your model outputs probabilities,
# use the probability of the REAL class.

roc_auc = roc_auc_score(

    y_true,

    valid["probability"],

)
# ==========================================================
# SAVE METRICS JSON
# ==========================================================

metrics = {

    "accuracy": float(accuracy),

    "precision": float(precision),

    "recall": float(recall),

    "f1_score": float(f1),

    "roc_auc": float(roc_auc),

    "videos_processed": int(len(valid)),

    "videos_failed": int(len(df) - len(valid)),

}

with open(

    DEBUG_DIR / "metrics.json",

    "w",

) as f:

    json.dump(metrics, f, indent=4)

# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

report = classification_report(

    y_true,

    y_pred,

    target_names=["FAKE", "REAL"],

)

with open(

    DEBUG_DIR / "classification_report.txt",

    "w",

) as f:

    f.write(report)

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 6))

plt.imshow(cm)

plt.title("Confusion Matrix")

plt.colorbar()

plt.xticks([0, 1], ["FAKE", "REAL"])

plt.yticks([0, 1], ["FAKE", "REAL"])

plt.xlabel("Predicted")

plt.ylabel("Actual")

for i in range(2):

    for j in range(2):

        plt.text(

            j,

            i,

            cm[i, j],

            ha="center",

            va="center",

            fontsize=14,

        )

plt.tight_layout()

plt.savefig(

    DEBUG_DIR / "confusion_matrix.png",

    dpi=300,

    bbox_inches="tight",

)

plt.close()

# ==========================================================
# SUMMARY
# ==========================================================

summary = f"""
====================================================
PrismShieldAI Video Pipeline Evaluation
====================================================

Videos Tested      : {len(df)}

Videos Processed   : {len(valid)}

Failed Videos      : {len(df) - len(valid)}

----------------------------------------------------

REAL Videos        : {total_real}

FAKE Videos        : {total_fake}

Correct REAL       : {correct_real}

Correct FAKE       : {correct_fake}

----------------------------------------------------

Accuracy           : {accuracy:.4f}

Precision          : {precision:.4f}

Recall             : {recall:.4f}

F1 Score           : {f1:.4f}

ROC-AUC            : {roc_auc:.4f}

----------------------------------------------------

Evaluation Time    : {elapsed:.2f} seconds

====================================================
"""