"""
==============================================================
PrismShieldAI

Global Configuration

Single source of truth for the entire project.

==============================================================
"""

from pathlib import Path
import shutil
import torch

# ==============================================================
# PROJECT
# ==============================================================

PROJECT_NAME = "PrismShieldAI"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==============================================================
# RAW DATASET
# ==============================================================

DATASET_PATH = (

    PROJECT_ROOT

    / "datasets"

    / "FakeAVCeleb_v1.2"

    / "FakeAVCeleb_v1.2"

)

# ==============================================================
# PROCESSED DATA
# ==============================================================

PROCESSED_DATASET = PROJECT_ROOT / "processed_dataset"

PROCESSED_FACES = PROJECT_ROOT / "processed_faces"

PROCESSED_AUDIO = PROJECT_ROOT / "processed_audio"

CSV_FILE = PROCESSED_DATASET / "research_dataset.csv"

# ==============================================================
# OUTPUTS
# ==============================================================

WEIGHTS_DIR = PROJECT_ROOT / "weights"

RUNS_DIR = PROJECT_ROOT / "runs"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

PRESENTATIONS_DIR = PROJECT_ROOT / "presentations"

WEIGHTS_DIR.mkdir(exist_ok=True)

RUNS_DIR.mkdir(exist_ok=True)

# ==============================================================
# FFMPEG
# ==============================================================

# Find FFmpeg from the system PATH instead of a fixed location
FFMPEG_PATH = Path(shutil.which("ffmpeg") or "ffmpeg")

# ==============================================================
# DATASET
# ==============================================================

NUM_CLASSES = 2

IMAGE_SIZE = 224

SAMPLE_RATE = 16000

MAX_AUDIO_LENGTH = 80000

EMBEDDING_DIM = 256

# ==============================================================
# TRAINING
# ==============================================================

BATCH_SIZE = 16

EPOCHS = 30

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 1e-4

PATIENCE = 8

NUM_WORKERS = 0

PIN_MEMORY = True

# ==============================================================
# CONTRASTIVE LEARNING
# ==============================================================

TEMPERATURE = 0.07

CLASSIFICATION_WEIGHT = 1.0

CONTRASTIVE_WEIGHT = 0.10

# ==============================================================
# DEVICE
# ==============================================================

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)

USE_AMP = torch.cuda.is_available()

# ==============================================================
# RANDOMNESS
# ==============================================================

SEED = 42

# ==============================================================
# CHECKPOINTS
# ==============================================================

# --------------------------------------------------------------
# Original Models (V1)
# --------------------------------------------------------------

BEST_AUDIO_MODEL = "audio_best.pth"

BEST_VISUAL_MODEL = "visual_best.pth"

# --------------------------------------------------------------
# Fusion Model V2 (Confidence Learning)
# --------------------------------------------------------------

BEST_FUSION_MODEL = "fusion_best_v3_reasoning_final.pth"

LAST_FUSION_MODEL = "fusion_last_v3_reasoning_final.pth"

# Aliases used by the training pipeline
BEST_MODEL_NAME = BEST_FUSION_MODEL
LAST_MODEL_NAME = LAST_FUSION_MODEL

# ==============================================================
# STREAMLIT
# ==============================================================

SUPPORTED_IMAGE_FORMATS = [

    ".jpg",

    ".jpeg",

    ".png",

]

SUPPORTED_AUDIO_FORMATS = [

    ".wav",

    ".mp3",

    ".flac",

]

SUPPORTED_VIDEO_FORMATS = [

    ".mp4",

    ".avi",

    ".mov",

    ".mkv",

]

# ==============================================================
# LOGGING
# ==============================================================

PRINT_SEPARATOR = "=" * 60

# ==============================================================
# TEST
# ==============================================================

if __name__ == "__main__":

    print()

    print(PRINT_SEPARATOR)

    print(PROJECT_NAME)

    print(PRINT_SEPARATOR)

    print("Project Root :", PROJECT_ROOT)

    print("Raw Dataset  :", DATASET_PATH)

    print("CSV File     :", CSV_FILE)

    print("Faces        :", PROCESSED_FACES)

    print("Audio        :", PROCESSED_AUDIO)

    print("Weights      :", WEIGHTS_DIR)

    print()

    print("Device       :", DEVICE)

    print("AMP          :", USE_AMP)

    print()

    print("Image Size   :", IMAGE_SIZE)

    print("Audio Length :", MAX_AUDIO_LENGTH)

    print("Embedding    :", EMBEDDING_DIM)

    print()

    print("Batch Size   :", BATCH_SIZE)

    print("Epochs       :", EPOCHS)

    print("LearningRate :", LEARNING_RATE)

    print("Temperature  :", TEMPERATURE)

    print()

    print("Best Fusion  :", BEST_FUSION_MODEL)