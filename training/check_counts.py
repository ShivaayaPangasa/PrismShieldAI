from pathlib import Path

dataset_path = Path(
    "datasets/FakeAVCeleb_v1.2/FakeAVCeleb_v1.2"
)

real_videos = list(
    (dataset_path / "RealVideo-RealAudio").rglob("*.mp4")
)

fake_videos = list(
    (dataset_path / "FakeVideo-FakeAudio").rglob("*.mp4")
)

print("Dataset Exists:", dataset_path.exists())
print("Real Videos:", len(real_videos))
print("Fake Videos:", len(fake_videos))