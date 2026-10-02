"""
PrismShieldAI - leading-silence robustness evaluation

No retraining. The best checkpoint is evaluated twice on the same TEST
split, but this script only reports the trimmed condition. Compare it with
paper_eval_v3/paper_metrics_v3.json for the original condition.

The transform removes only leading low-energy audio and then restores the
same 80,000-sample 16-kHz input size expected by the model.
"""

from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
)

from configs.settings import *
from datasets.fusion_dataset import FusionDataset
from models.fusion_model import FusionModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = PROJECT_ROOT / "research_paper" / "results" / "leading_silence_v3"
CHECKPOINT = WEIGHTS_DIR / BEST_MODEL_NAME
TARGET_SR = 16000
TARGET_LEN = 80000
FRAME = 320          # 20 ms
HOP = 160            # 10 ms
TRIM_DB = 40.0
MAX_TRIM_SECONDS = 1.0


def resolve_path(raw):
    p = Path(str(raw).replace("\\", "/"))
    return p if p.is_absolute() else PROJECT_ROOT / p


def trim_leading_silence(audio, sr):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = resample_poly(audio, TARGET_SR, sr).astype(np.float32)
        sr = TARGET_SR

    if len(audio) == 0:
        return audio, 0.0

    # Remove DC offset before onset measurement.
    audio = audio - float(audio.mean())
    if len(audio) <= FRAME:
        return audio, 0.0

    starts = np.arange(0, len(audio) - FRAME + 1, HOP)
    rms = np.sqrt(
        np.mean(
            np.square(
                np.lib.stride_tricks.sliding_window_view(audio, FRAME)[::HOP]
            ),
            axis=1,
        ) + 1e-12
    )
    peak = float(rms.max())
    if peak <= 1e-8:
        return audio, 0.0

    threshold = peak * (10.0 ** (-TRIM_DB / 20.0))
    valid = np.where(rms >= threshold)[0]
    if len(valid) == 0:
        return audio, 0.0

    first = int(valid[0] * HOP)
    first = min(first, int(MAX_TRIM_SECONDS * TARGET_SR))
    first = max(0, first)

    # Keep one frame before the detected onset when possible.
    first = max(0, first - HOP)
    return audio[first:], first / TARGET_SR


class TrimmedDataset(Dataset):
    def __init__(self, base, processor):
        self.base = base
        self.processor = processor

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        audio, sr = sf.read(resolve_path(item["audio_path"]), always_2d=False)
        trimmed, removed = trim_leading_silence(audio, sr)

        processed = self.processor(
            trimmed,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=False,
        )
        values = processed.input_values.squeeze(0)
        mask = processed.get("attention_mask")
        if mask is not None:
            mask = mask.squeeze(0)

        if values.numel() > TARGET_LEN:
            values = values[:TARGET_LEN]
            if mask is not None:
                mask = mask[:TARGET_LEN]
        else:
            pad = TARGET_LEN - values.numel()
            values = torch.nn.functional.pad(values, (0, pad))
            if mask is not None:
                mask = torch.nn.functional.pad(mask, (0, pad))

        item["input_values"] = values.float()
        item["attention_mask"] = (
            mask.long() if mask is not None else torch.ones(TARGET_LEN, dtype=torch.long)
        )
        item["trimmed_seconds"] = float(removed)
        return item


def load_model():
    model = FusionModel().to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, ckpt


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    model, ckpt = load_model()
    df = pd.read_csv(CSV_FILE)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
    base = FusionDataset(
        dataframe=test_df,
        processor=processor,
        image_size=IMAGE_SIZE,
        max_audio_length=MAX_AUDIO_LENGTH,
    )
    ds = TrimmedDataset(base, processor)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=PIN_MEMORY,
    )

    y, p, prob = [], [], []
    trimmed = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(DEVICE, non_blocking=True)
            values = batch["input_values"].to(DEVICE, non_blocking=True)
            mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, non_blocking=True)

            outputs = model(
                input_values=values,
                attention_mask=mask,
                images=images,
            )
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=1)[:, 1]
            pred = logits.argmax(dim=1)

            y.extend(labels.cpu().numpy().tolist())
            p.extend(pred.cpu().numpy().tolist())
            prob.extend(probs.cpu().numpy().tolist())
            trimmed.extend(batch["trimmed_seconds"].numpy().tolist())

    metrics = {
        "checkpoint": CHECKPOINT.name,
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "test_samples": len(y),
        "mean_trim_seconds": float(np.mean(trimmed)),
        "median_trim_seconds": float(np.median(trimmed)),
        "p95_trim_seconds": float(np.percentile(trimmed, 95)),
        "accuracy": float(accuracy_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "f1_fake": float(f1_score(y, p, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
        "mcc": float(matthews_corrcoef(y, p)),
    }

    baseline_path = PROJECT_ROOT / "research_paper" / "results" / "paper_eval_v3" / "paper_metrics_v3.json"
    if baseline_path.exists():
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline = json.load(f)
        deltas = {}
        for key in ["accuracy", "balanced_accuracy", "f1_fake", "macro_f1", "roc_auc", "pr_auc", "mcc"]:
            if key in baseline and key in metrics:
                deltas[key] = float(metrics[key] - baseline[key])
        metrics["delta_vs_original"] = deltas

    with open(RESULTS / "leading_silence_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("=" * 76)
    print("LEADING-SILENCE TRIM ROBUSTNESS")
    print("=" * 76)
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print(f"Saved: {RESULTS / 'leading_silence_metrics.json'}")


if __name__ == "__main__":
    main()