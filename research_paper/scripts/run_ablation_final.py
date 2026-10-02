"""
PrismShieldAI - Final paper ablation study

Cumulative configurations required by the mentor:
1. Visual only
2. Audio only
3. Audio + Visual concatenation
4. + Contrastive alignment
5. + Adaptive Evidence Reasoning (AER)
6. Full PrismShieldAI (adds the current embedding-level cross-modal block)

The pretrained AudioEncoder and VisualEncoder are frozen. Only the
ablation heads/modules are trained, so this is a controlled component
study rather than six end-to-end backbone retrainings.
"""

from pathlib import Path
import argparse
import json
import random
import sys

# ------------------------------------------------------------
# Project root
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ------------------------------------------------------------
# Third-party imports
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2Processor

# ------------------------------------------------------------
# Project imports
# ------------------------------------------------------------

from datasets.fusion_dataset import FusionDataset
from models.audio_encoder import AudioEncoder
from models.visual_encoder import VisualEncoder
from models.projection_head import ProjectionHead
from models.cross_modal_attention import CrossModalAttention
from models.adaptive_evidence_reasoning import AdaptiveEvidenceReasoning
from losses.contrastive_loss import ContrastiveLoss

SEED = 42
BATCH_SIZE = 16
EPOCHS = 30
LR = 3e-4
WEIGHT_DECAY = 1e-4
TEMPERATURE = 0.07
CONTRASTIVE_WEIGHT = 0.10
PATIENCE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "research_paper" / "results"
WEIGHTS = ROOT / "weights" / "fusion_best_v3_reasoning_final.pth"
CSV = ROOT / "processed_dataset" / "research_dataset.csv"

def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def load_frozen_encoders():
    ckpt = torch.load(WEIGHTS, map_location=DEVICE, weights_only=False)
    state = ckpt["model_state_dict"]

    audio = AudioEncoder().to(DEVICE)
    visual = VisualEncoder().to(DEVICE)

    audio.load_state_dict(
        {k[len("audio_encoder."):]: v for k, v in state.items()
         if k.startswith("audio_encoder.")},
        strict=True,
    )
    visual.load_state_dict(
        {k[len("visual_encoder."):]: v for k, v in state.items()
         if k.startswith("visual_encoder.")},
        strict=True,
    )

    audio.eval()
    visual.eval()
    for p in audio.parameters():
        p.requires_grad = False
    for p in visual.parameters():
        p.requires_grad = False

    print(f"Checkpoint: {WEIGHTS.name}")
    print(f"Checkpoint epoch: {ckpt.get('epoch')} | best score: {ckpt.get('best_score')}")
    return audio, visual


class Head(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


class Ablation(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

        if mode == "visual":
            self.cls = Head(256)
        elif mode == "audio":
            self.cls = Head(256)
        elif mode == "av":
            self.cls = Head(512)
        else:
            self.ap = ProjectionHead()
            self.vp = ProjectionHead()

            if mode == "contrastive":
                self.cls = Head(512)
            elif mode == "aer":
                self.reasoning = AdaptiveEvidenceReasoning()
                self.cls = Head(256)
            elif mode == "full":
                self.cross = CrossModalAttention()
                self.reasoning = AdaptiveEvidenceReasoning()
                self.cls = Head(256)
            else:
                raise ValueError(mode)

    def forward(self, a, v):
        if self.mode == "visual":
            return self.cls(v), None, None
        if self.mode == "audio":
            return self.cls(a), None, None
        if self.mode == "av":
            return self.cls(torch.cat([a, v], dim=1)), None, None

        if self.mode == "contrastive":
            ap, vp = self.ap(a), self.vp(v)
            return self.cls(torch.cat([ap, vp], dim=1)), ap, vp

        if self.mode == "aer":
            ap, vp = self.ap(a), self.vp(v)
            r = self.reasoning(ap, vp)
            return self.cls(r["reasoning_embedding"]), ap, vp

        # Full = embedding-level cross-modal interaction + contrastive + AER.
        
        ac, _ = self.cross.audio_to_visual(
            a.unsqueeze(1),
            v.unsqueeze(1),
            v.unsqueeze(1),
        )

        vc, _ = self.cross.visual_to_audio(
            v.unsqueeze(1),
            a.unsqueeze(1),
            a.unsqueeze(1),
        )

        ac = self.cross.norm_audio(
            a.unsqueeze(1) + ac
        ).squeeze(1)

        vc = self.cross.norm_visual(
            v.unsqueeze(1) + vc
        ).squeeze(1)
        
        ap, vp = self.ap(ac), self.vp(vc)
        r = self.reasoning(ap, vp)
        return self.cls(r["reasoning_embedding"]), ap, vp

class FeatureDataset(Dataset):
    def __init__(self, audio, visual, labels, sample_ids, video_ids):
        self.audio = audio
        self.visual = visual
        self.labels = labels
        self.sample_ids = sample_ids
        self.video_ids = video_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "audio": self.audio[i],
            "visual": self.visual[i],
            "label": self.labels[i],
            "sample_id": self.sample_ids[i],
            "video_id": self.video_ids[i],
        }


def extract_features(split, processor, audio_encoder, visual_encoder):
    df = pd.read_csv(CSV)
    df = df[df["split"] == split].reset_index(drop=True)

    ds = FusionDataset(
        dataframe=df,
        processor=processor,
        image_size=224,
        max_audio_length=80000,
    )
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    aa, vv, yy, ss, vids = [], [], [], [], []
    with torch.inference_mode():
        for batch in dl:
            images = batch["image"].to(DEVICE, non_blocking=True)
            wav = batch["input_values"].to(DEVICE, non_blocking=True)
            mask = batch.get("attention_mask")
            if mask is not None:
                mask = mask.to(DEVICE, non_blocking=True)

            aa.append(audio_encoder.extract_embedding(wav, mask).cpu())
            vv.append(visual_encoder.extract_embedding(images).cpu())
            yy.append(batch["label"].cpu())
            ss.extend([int(x) for x in batch["sample_id"]])
            vids.extend([int(x) for x in batch["video_id"]])

    out = {
        "audio": torch.cat(aa),
        "visual": torch.cat(vv),
        "labels": torch.cat(yy),
        "sample_ids": np.asarray(ss),
        "video_ids": np.asarray(vids),
    }
    print(
        f"{split}: {len(out['labels'])} samples | "
        f"real={(out['labels'] == 0).sum().item()} | "
        f"fake={(out['labels'] == 1).sum().item()}"
    )
    return out


def validate(model, bundle):
    model.eval()
    dl = DataLoader(
        FeatureDataset(
            bundle["audio"], bundle["visual"], bundle["labels"],
            bundle["sample_ids"], bundle["video_ids"]
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    y, p, prob = [], [], []
    with torch.no_grad():
        for batch in dl:
            logits, _, _ = model(
                batch["audio"].to(DEVICE), batch["visual"].to(DEVICE)
            )
            pp = torch.softmax(logits, dim=1)[:, 1]
            pred = logits.argmax(dim=1)
            y.extend(batch["label"].numpy().tolist())
            p.extend(pred.cpu().numpy().tolist())
            prob.extend(pp.cpu().numpy().tolist())

    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "f1": float(f1_score(y, p, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "mcc": float(matthews_corrcoef(y, p)),
    }


def train_one(mode, train_bundle, val_bundle, epochs):
    model = Ablation(mode).to(DEVICE)

    train_labels = train_bundle["labels"].numpy()
    counts = np.bincount(train_labels, minlength=2).astype(np.float32)
    weights = counts.sum() / (2.0 * np.maximum(counts, 1.0))
    class_weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.02)
    contrastive = ContrastiveLoss(TEMPERATURE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

    dl = DataLoader(
        FeatureDataset(
            train_bundle["audio"], train_bundle["visual"], train_bundle["labels"],
            train_bundle["sample_ids"], train_bundle["video_ids"]
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    best_f1 = -1.0
    best_epoch = 0
    best_state = None
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in dl:
            a = batch["audio"].to(DEVICE)
            v = batch["visual"].to(DEVICE)
            y = batch["label"].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits, ap, vp = model(a, v)
            loss = criterion(logits, y)
            if mode in {"contrastive", "aer", "full"}:
                loss = loss + CONTRASTIVE_WEIGHT * contrastive(ap, vp)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        val = validate(model, val_bundle)
        print(
            f"[{mode}] epoch {epoch:02d}/{epochs} "
            f"loss={total_loss / max(len(dl), 1):.5f} "
            f"val_F1={val['f1']:.6f}"
        )

        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"[{mode}] early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state, strict=True)
    return model, best_epoch, best_f1


def test_predictions(model, bundle):
    model.eval()
    ds = FeatureDataset(
        bundle["audio"], bundle["visual"], bundle["labels"],
        bundle["sample_ids"], bundle["video_ids"]
    )
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    y, p, prob, ids, vids = [], [], [], [], []
    with torch.no_grad():
        for batch in dl:
            logits, _, _ = model(
                batch["audio"].to(DEVICE), batch["visual"].to(DEVICE)
            )
            pp = torch.softmax(logits, dim=1)[:, 1]
            pred = logits.argmax(dim=1)
            y.extend(batch["label"].numpy().tolist())
            p.extend(pred.cpu().numpy().tolist())
            prob.extend(pp.cpu().numpy().tolist())
                        
            sample_ids = batch["sample_id"]
            video_ids = batch["video_id"]

            if torch.is_tensor(sample_ids):
                ids.extend(sample_ids.cpu().tolist())
            else:
                ids.extend(list(sample_ids))

            if torch.is_tensor(video_ids):
                vids.extend(video_ids.cpu().tolist())
            else:
                vids.extend(list(video_ids))
            
            
            

    return y, p, prob, ids, vids


def compute_metrics(y, p, prob):
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "f1_fake": float(f1_score(y, p, pos_label=1, zero_division=0)),
        "f1_real": float(f1_score(y, p, pos_label=0, zero_division=0)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "mcc": float(matthews_corrcoef(y, p)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--only", choices=["visual", "audio", "av", "contrastive", "aer", "full"])
    args = parser.parse_args()

    seed_everything()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "ablation_models").mkdir(parents=True, exist_ok=True)
    (RESULTS / "ablation_predictions").mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("PrismShieldAI FINAL CUMULATIVE ABLATION")
    print(f"Device: {DEVICE} | max epochs/config: {args.epochs}")
    print("=" * 76)

    processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base"
)
    audio_encoder, visual_encoder = load_frozen_encoders()

    train_bundle = extract_features("train", processor, audio_encoder, visual_encoder)
    val_bundle = extract_features("val", processor, audio_encoder, visual_encoder)
    test_bundle = extract_features("test", processor, audio_encoder, visual_encoder)

    modes = [args.only] if args.only else ["visual", "audio", "av", "contrastive", "aer", "full"]
    names = {
        "visual": "Visual Only",
        "audio": "Audio Only",
        "av": "Audio + Visual",
        "contrastive": "+ Contrastive",
        "aer": "+ AER",
        "full": "Full PrismShieldAI",
    }

    rows = []
    for mode in modes:
        print("\n" + "=" * 76)
        print("RUNNING:", names[mode])
        print("=" * 76)

        model, best_epoch, best_val_f1 = train_one(
            mode, train_bundle, val_bundle, args.epochs
        )
        y, p, prob, ids, vids = test_predictions(model, test_bundle)
        metrics = compute_metrics(y, p, prob)

        row = {
            "configuration": names[mode],
            "mode": mode,
            "epochs_max": args.epochs,
            "best_epoch": best_epoch,
            "best_validation_f1": best_val_f1,
            "test_samples": len(y),
            **metrics,
        }
        rows.append(row)
        print(row)

        torch.save(
            {
                "mode": mode,
                "best_epoch": best_epoch,
                "best_validation_f1": best_val_f1,
                "state_dict": model.state_dict(),
            },
            RESULTS / "ablation_models" / f"{mode}_best.pth",
        )

        pred_df = pd.DataFrame(
            {
                "sample_id": ids,
                "video_id": vids,
                "true_label": y,
                "prediction": p,
                "fake_probability": prob,
            }
        )
        pred_df.to_csv(
            RESULTS / "ablation_predictions" / f"{mode}_test_predictions.csv",
            index=False,
        )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS / "ablation_results_final.csv", index=False)
    with open(RESULTS / "ablation_results_final.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": SEED,
                "batch_size": BATCH_SIZE,
                "max_epochs": args.epochs,
                "patience": PATIENCE,
                "device": str(DEVICE),
                "checkpoint": str(WEIGHTS),
                "encoders_frozen": True,
                "cumulative_design": True,
                "results": rows,
            },
            f,
            indent=2,
        )

    print("\nSaved:")
    print(RESULTS / "ablation_results_final.csv")
    print(RESULTS / "ablation_results_final.json")


if __name__ == "__main__":
    main()
