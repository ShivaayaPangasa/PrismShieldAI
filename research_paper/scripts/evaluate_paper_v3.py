"""
PrismShieldAI - Final paper evaluation / artifact generator

One held-out TEST pass generates:
- headline metrics + class-wise metrics + Macro-F1
- test_predictions_final.csv
- ROC + PR combined figure
- confidence-head + class-probability calibration (ECE/Brier)
- reliability diagram
- selective-risk curve
- 2x2 t-SNE before/after contrastive projection
- 16x16 contrastive similarity heatmap
- adaptive modality-weight distributions by manipulation type
- manipulation-type breakdown
- four-category failure gallery with face + spectrogram + scores

The confidence head is evaluated as a correctness estimator because its
training target is prediction correctness, not the Real/Fake softmax.
"""

from pathlib import Path
import json
import math
import random
import sys

# ------------------------------------------------------------
# Add project root before importing project modules
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import soundfile as sf
from scipy import signal

import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.manifold import TSNE

from configs.settings import *
from datasets.fusion_dataset import FusionDataset
from models.fusion_model import FusionModel

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = PROJECT_ROOT / "research_paper" / "results" / "paper_eval_v3"
CHECKPOINT = WEIGHTS_DIR / BEST_MODEL_NAME


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def resolve_path(raw):
    p = Path(str(raw).replace("\\", "/"))
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_model():
    model = FusionModel().to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, ckpt


def ece_from_confidence(confidence, correctness, n_bins=10):
    confidence = np.asarray(confidence, dtype=float)
    correctness = np.asarray(correctness, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(confidence)
    ece = 0.0
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        count = int(mask.sum())
        if count == 0:
            rows.append({
                "bin": i + 1, "lower": lo, "upper": hi,
                "count": 0, "mean_confidence": np.nan,
                "empirical_accuracy": np.nan,
            })
            continue
        mean_c = float(confidence[mask].mean())
        acc = float(correctness[mask].mean())
        ece += (count / total) * abs(acc - mean_c)
        rows.append({
            "bin": i + 1, "lower": lo, "upper": hi,
            "count": count, "mean_confidence": mean_c,
            "empirical_accuracy": acc,
        })
    return float(ece), pd.DataFrame(rows)


def load_audio_for_plot(path):
    audio, sr = sf.read(resolve_path(path), always_2d=False)
    if np.ndim(audio) > 1:
        audio = np.mean(audio, axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    return audio, sr


def spectrogram(ax, path):
    try:
        audio, sr = load_audio_for_plot(path)
        if len(audio) == 0:
            ax.text(0.5, 0.5, "empty audio", ha="center", va="center")
            ax.axis("off")
            return
        
        nperseg = min(512, len(audio))
        noverlap = min(256, nperseg // 2)

        f , t, sxx = signal.spectrogram(
        audio, fs=sr, nperseg=nperseg, noverlap=noverlap)
        sxx_db = 10 * np.log10(sxx + 1e-10)
        ax.pcolormesh(t, f, sxx_db, shading="auto")
        ax.set_ylim(0, min(8000, sr / 2))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title("Audio spectrogram")
    except Exception as exc:
        ax.text(0.5, 0.5, f"spectrogram unavailable\n{exc}", ha="center", va="center")
        ax.axis("off")


def build_failure_gallery(pred_df):
    gallery_dir = RESULTS / "failure_gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    correct = pred_df[pred_df["correct"]].copy()
    
    # Real samples incorrectly predicted as Fake
    real_predicted_fake = pred_df[
        (pred_df["true_label"] == 1) &
        (pred_df["prediction"] == 0)
    ].copy()

    # Fake samples incorrectly predicted as Real
    fake_predicted_real = pred_df[
        (pred_df["true_label"] == 0) &
        (pred_df["prediction"] == 1)
    ].copy()

    selections = {}
    if not correct.empty:
        selections["correct_high_confidence"] = correct.sort_values("confidence", ascending=False).head(1)
        selections["correct_low_confidence"] = correct.sort_values("confidence", ascending=True).head(1)
    selections["real_predicted_fake"] = real_predicted_fake.head(1)
    selections["fake_predicted_real"] = fake_predicted_real.head(1)

    for name, group in selections.items():
        if group.empty:
            continue
        row = group.iloc[0]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        try:
            img = Image.open(resolve_path(row["image_path"])).convert("RGB")
            axes[0].imshow(img)
            axes[0].axis("off")
            axes[0].set_title("Face crop")
        except Exception as exc:
            axes[0].text(0.5, 0.5, f"image unavailable\n{exc}", ha="center", va="center")
            axes[0].axis("off")

        spectrogram(axes[1], row["audio_path"])

        text = (
            f"Category: {name}\n"
            f"sample_id={row['sample_id']} | video_id={row['video_id']}\n"
            f"GT={'Fake' if row['true_label'] == 0 else 'Real'} | "
            f"Pred={'Fake' if row['prediction'] == 0 else 'Real'}\n"
            f"Fake probability={row['fake_probability']:.4f}\n"
            f"Learned confidence={row['confidence']:.4f}\n"
            f"α_audio={row['alpha_audio']:.4f} | α_visual={row['alpha_visual']:.4f}"
        )
        fig.text(0.5, 0.01, text, ha="center", va="bottom", fontsize=10)
        plt.tight_layout(rect=[0, 0.13, 1, 1])
        fig.savefig(gallery_dir / f"{name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main():
    seed_everything()
    RESULTS.mkdir(parents=True, exist_ok=True)
    print("=" * 76)
    print("PrismShieldAI FINAL PAPER EVALUATION")
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT}")
    print("=" * 76)

    model, ckpt = load_model()
    df = pd.read_csv(CSV_FILE)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    dataset = FusionDataset(
        dataframe=test_df,
        processor=processor,
        image_size=IMAGE_SIZE,
        max_audio_length=MAX_AUDIO_LENGTH,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    labels, preds, fake_probs, confidences = [], [], [], []
    rows = []
    audio_embs, visual_embs = [], []
    audio_proj, visual_proj = [], []
    alpha_audio, alpha_visual = [], []
    first_similarity = None

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(DEVICE, non_blocking=True)
            inputs = batch["input_values"].to(DEVICE, non_blocking=True)
            mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            y = batch["label"].to(DEVICE, non_blocking=True)

            with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                out = model(
                    input_values=inputs,
                    attention_mask=mask,
                    images=images,
                )
                logits = out["logits"]
                p_fake = torch.softmax(logits, dim=1)[:, 0]
                pred = logits.argmax(dim=1)

            conf = out["confidence"].view(-1)
            aa = out["audio_importance"].view(-1)
            av = out["visual_importance"].view(-1)

            labels.extend(y.cpu().numpy().tolist())
            preds.extend(pred.cpu().numpy().tolist())
            fake_probs.extend(p_fake.float().cpu().numpy().tolist())
            confidences.extend(conf.float().cpu().numpy().tolist())
            alpha_audio.extend(aa.float().cpu().numpy().tolist())
            alpha_visual.extend(av.float().cpu().numpy().tolist())
            audio_embs.append(out["audio_embedding"].float().cpu().numpy())
            visual_embs.append(out["visual_embedding"].float().cpu().numpy())
            audio_proj.append(out["audio_projection"].float().cpu().numpy())
            visual_proj.append(out["visual_projection"].float().cpu().numpy())
            if first_similarity is None:
                first_similarity = out["similarity"].float().cpu().numpy()

            for i in range(len(y)):
                rows.append({
                    "sample_id": int(batch["sample_id"][i]),
                    "video_id": int(batch["video_id"][i]),
                    "true_label": int(y[i]),
                    "prediction": int(pred[i]),
                    "fake_probability": float(p_fake[i]),
                    "confidence": float(conf[i]),
                    "alpha_audio": float(aa[i]),
                    "alpha_visual": float(av[i]),
                    "image_path": batch["image_path"][i],
                    "audio_path": batch["audio_path"][i],
                })

    pred_df = pd.DataFrame(rows)
    pred_df["correct"] = pred_df["true_label"] == pred_df["prediction"]

    meta_cols = ["sample_id", "source_video", "label", "dataset", "face_index", "audio_clip_index", "timestamp", "audio_timestamp"]
    meta = test_df[meta_cols].copy()
    meta["manipulation_type"] = meta["source_video"].astype(str).str.replace("/", "\\", regex=False).str.split("\\").str[0]
    pred_df = pred_df.merge(meta, on="sample_id", how="left", suffixes=("", "_meta"))
    pred_df.to_csv(RESULTS / "test_predictions_final.csv", index=False)

    y = pred_df["true_label"].to_numpy()
    p = pred_df["prediction"].to_numpy()
    prob = pred_df["fake_probability"].to_numpy()
    
    label_text = pred_df["label"].astype(str).str.strip().str.lower()

    if not label_text.isin(["fake", "real"]).all():
        raise ValueError("Unexpected text labels found in test metadata.")

    expected_labels = label_text.map({"fake": 0, "real": 1}).to_numpy()

    if not np.array_equal(y, expected_labels):
        raise ValueError(
            "Numeric labels do not match metadata labels. "
            "Expected 0=Fake and 1=Real."
    )
     
    conf = pred_df["confidence"].to_numpy()
    correct = pred_df["correct"].to_numpy().astype(float)
    
    # Numeric class mapping: 0 = Fake, 1 = Real
    fake_true = (y == 0).astype(int)

    cm = confusion_matrix(y, p, labels=[0, 1])

    metrics = {
        "accuracy": float(accuracy_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),

        "precision_fake": float(
            precision_score(y, p, pos_label=0, zero_division=0)
        ),
        "recall_fake": float(
            recall_score(y, p, pos_label=0, zero_division=0)
        ),
        "f1_fake": float(
            f1_score(y, p, pos_label=0, zero_division=0)
        ),

        "precision_real": float(
            precision_score(y, p, pos_label=1, zero_division=0)
        ),
        "recall_real": float(
            recall_score(y, p, pos_label=1, zero_division=0)
        ),
        "f1_real": float(
            f1_score(y, p, pos_label=1, zero_division=0)
        ),

        "macro_f1": float(
            f1_score(y, p, average="macro", zero_division=0)
        ),

        "roc_auc": float(roc_auc_score(fake_true, prob)),
        "pr_auc": float(average_precision_score(fake_true, prob)),
        "mcc": float(matthews_corrcoef(y, p)),

        "num_test_samples": int(len(y)),
        "num_fake": int((y == 0).sum()),
        "num_real": int((y == 1).sum()),
        "best_checkpoint_epoch": int(ckpt.get("epoch", -1)),
    }    

    report = classification_report(
        y,
        p,
        labels=[0, 1],
        target_names=["Fake", "Real"],
        digits=4,
        zero_division=0
    )

    (RESULTS / "classification_report.txt").write_text(
        report,
        encoding="utf-8"
    )    

    # Calibration: confidence head against correctness target.
    confidence_ece, conf_bins = ece_from_confidence(
        conf, correct, n_bins=10
    )
    confidence_brier = float(np.mean((conf - correct) ** 2))

    # Classification calibration:
    # prob is P(Fake), and fake_true is 1 for Fake, 0 for Real.
    max_prob = np.maximum(prob, 1.0 - prob)

    class_ece, class_bins = ece_from_confidence(
        max_prob, correct, n_bins=10
    )

    class_brier = float(np.mean((prob - fake_true) ** 2))

    calibration = {
        "confidence_head": {
            "interpretation": "learned correctness estimator",
            "ece": confidence_ece,
            "brier": confidence_brier,
        },
        "classification_probability": {
            "interpretation": "softmax probability for Fake (class 0)",
            "ece": class_ece,
            "brier": class_brier,
        },
    }
    with open(RESULTS / "calibration_metrics.json", "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)
    conf_bins.to_csv(RESULTS / "confidence_reliability_bins.csv", index=False)
    class_bins.to_csv(RESULTS / "classification_reliability_bins.csv", index=False)

    # Reliability diagram.
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    cb = conf_bins.dropna(subset=["mean_confidence", "empirical_accuracy"])
    pb = class_bins.dropna(subset=["mean_confidence", "empirical_accuracy"])
    ax.plot(cb["mean_confidence"], cb["empirical_accuracy"], marker="o", label="Confidence head")
    ax.plot(pb["mean_confidence"], pb["empirical_accuracy"], marker="s", label="Softmax max probability")
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Empirical correctness")
    ax.set_title("Reliability Diagram")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(RESULTS / "reliability_diagram.png", dpi=300)
    plt.close(fig)

    # Selective risk.
    order = np.argsort(-conf)
    sorted_correct = correct[order]
    coverages = np.linspace(0.1, 1.0, 10)
    risk_rows = []
    for cov in coverages:
        n = max(1, int(round(cov * len(sorted_correct))))
        risk = float(1.0 - sorted_correct[:n].mean())
        risk_rows.append({"coverage": float(n / len(sorted_correct)), "risk": risk})
    risk_df = pd.DataFrame(risk_rows)
    risk_df.to_csv(RESULTS / "selective_risk.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(risk_df["coverage"], risk_df["risk"], marker="o")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Error rate (risk)")
    ax.set_title("Selective Risk vs Coverage")
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(RESULTS / "selective_risk.png", dpi=300)
    plt.close(fig)
    
    # ROC + PR curves with Fake as the positive class.
    fpr, tpr, _ = roc_curve(fake_true, prob)
    prec, rec, _ = precision_recall_curve(fake_true, prob)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, label=f"AUC={metrics['roc_auc']:.4f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(rec, prec, label=f"AP={metrics['pr_auc']:.4f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    fig.savefig(RESULTS / "roc_pr_combined.png", dpi=300)
    plt.close(fig)

    # Confusion matrix.
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm)
    
    ax.set_xticks([0, 1], ["Fake", "Real"])
    ax.set_yticks([0, 1], ["Fake", "Real"])
    
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("PrismShieldAI Test Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(RESULTS / "confusion_matrix.png", dpi=300)
    plt.close(fig)

    # Embedding export + t-SNE before/after contrastive projection.
    aemb = np.concatenate(audio_embs, axis=0)
    vemb = np.concatenate(visual_embs, axis=0)
    aproj = np.concatenate(audio_proj, axis=0)
    vproj = np.concatenate(visual_proj, axis=0)
    np.savez_compressed(
        RESULTS / "test_embeddings.npz",
        audio_before=aemb,
        visual_before=vemb,
        audio_after=aproj,
        visual_after=vproj,
        labels=y,
    )

    rng = np.random.default_rng(SEED)
    n_tsne = min(len(y), 1500)
    idx = rng.choice(len(y), size=n_tsne, replace=False)
    blocks = [aemb[idx], vemb[idx], aproj[idx], vproj[idx]]
    combined = np.concatenate(blocks, axis=0)
    pca_n = min(50, combined.shape[1])
    from sklearn.decomposition import PCA
    combined50 = PCA(n_components=pca_n, random_state=SEED).fit_transform(combined)
    perplexity = min(30, max(5, (len(combined50) - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=SEED,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(combined50)
    chunks = np.split(coords, 4)
    labels_tsne = y[idx]
    titles = [
        "Before contrastive: audio embedding",
        "Before contrastive: visual embedding",
        "After contrastive: audio projection",
        "After contrastive: visual projection",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, xy, title in zip(axes.ravel(), chunks, titles):
        for cls, name in [(0, "Fake"), (1, "Real")]:
            mask = labels_tsne == cls
            ax.scatter(xy[mask, 0], xy[mask, 1], s=8, alpha=0.55, label=name)
        ax.set_title(title)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(True)
    axes[0, 0].legend()
    plt.tight_layout()
    fig.savefig(RESULTS / "tsne_before_after_contrastive.png", dpi=300)
    plt.close(fig)

    # 16x16 similarity heatmap from the first test batch.
    if first_similarity is not None:
        sim = np.asarray(first_similarity)[:16, :16]
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(sim, aspect="auto")
        ax.set_xlabel("Visual sample index")
        ax.set_ylabel("Audio sample index")
        ax.set_title("16x16 Audio-Visual Contrastive Similarity")
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        fig.savefig(RESULTS / "contrastive_similarity_16x16.png", dpi=300)
        plt.close(fig)

    # Modality-weight distributions and summary.
    weight_df = pred_df[["manipulation_type", "alpha_audio", "alpha_visual"]].copy()
    weight_df.to_csv(RESULTS / "adaptive_modality_weights.csv", index=False)
    weight_stats = weight_df.groupby("manipulation_type").agg(
        audio_mean=("alpha_audio", "mean"),
        audio_std=("alpha_audio", "std"),
        visual_mean=("alpha_visual", "mean"),
        visual_std=("alpha_visual", "std"),
        count=("alpha_audio", "size"),
    ).reset_index()
    weight_stats.to_csv(RESULTS / "adaptive_modality_weights_summary.csv", index=False)

    types = list(weight_df["manipulation_type"].dropna().unique())
    if types:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        groups_a = [weight_df.loc[weight_df["manipulation_type"] == t, "alpha_audio"] for t in types]
        groups_v = [weight_df.loc[weight_df["manipulation_type"] == t, "alpha_visual"] for t in types]
        axes[0].boxplot(groups_a, tick_labels=types, showfliers=False)
        axes[1].boxplot(groups_v, tick_labels=types, showfliers=False)
        axes[0].set_title("Adaptive audio importance")
        axes[1].set_title("Adaptive visual importance")
        for ax in axes:
            ax.set_ylabel("Importance weight")
            ax.tick_params(axis="x", rotation=25)
            ax.grid(True)
        plt.tight_layout()
        fig.savefig(RESULTS / "adaptive_modality_weight_distributions.png", dpi=300)
        plt.close(fig)

    # Manipulation-type performance.
    # Numeric class mapping: 0 = Fake, 1 = Real.
    mt_rows = []

    for mt, g in pred_df.groupby("manipulation_type"):
        yy = g["true_label"].to_numpy()
        pp = g["prediction"].to_numpy()
        fake_prob = g["fake_probability"].to_numpy()
        fake_target = (yy == 0).astype(int)

        row = {
            "manipulation_type": mt,
            "samples": int(len(g)),
            "accuracy": float(accuracy_score(yy, pp)),
        }
        if len(np.unique(yy)) == 2:
            row.update({
                "balanced_accuracy": float(balanced_accuracy_score(yy, pp)),
                "macro_f1": float(f1_score(yy, pp, average="macro", zero_division=0)),
                "roc_auc": float(roc_auc_score(fake_target, fake_prob)),
                "mcc": float(matthews_corrcoef(yy, pp)),
            })
        else:
            row.update({
                "balanced_accuracy": np.nan,
                "macro_f1": np.nan,
                "roc_auc": np.nan,
                "mcc": np.nan,
            })
        mt_rows.append(row)
    pd.DataFrame(mt_rows).to_csv(RESULTS / "manipulation_type_performance.csv", index=False)

    build_failure_gallery(pred_df)

    metrics["confusion_matrix"] = cm.tolist()
    metrics["calibration"] = calibration
    with open(RESULTS / "paper_metrics_v3.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nFINAL METRICS")
    for k, v in metrics.items():
        if k not in {"confusion_matrix", "calibration"}:
            print(f"{k}: {v}")
    print("\nCalibration:")
    print(json.dumps(calibration, indent=2))
    print("\nSaved all paper artifacts to:")
    print(RESULTS)


if __name__ == "__main__":
    main()
