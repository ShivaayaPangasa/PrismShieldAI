from pathlib import Path

# ==========================
# PROJECT ROOT
# ==========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==========================
# DATASET
# ==========================
DATASET_PATH = PROJECT_ROOT / "datasets" / "FakeAVCeleb_v1.2" / "FakeAVCeleb_v1.2"

# ==========================
# OUTPUT FOLDERS
# ==========================
PROCESSED_FACES = PROJECT_ROOT / "processed_faces"
PROCESSED_AUDIO = PROJECT_ROOT / "processed_audio"
WEIGHTS = PROJECT_ROOT / "weights"

# ==========================
# FFMPEG
# ==========================
FFMPEG_PATH = Path(
    r"C:\Users\Shivaaya\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe"
)