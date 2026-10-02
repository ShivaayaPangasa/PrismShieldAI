from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    ROOT
    / "research_paper"
    / "results"
    / "paper_eval_v3"
    / "test_predictions_final.csv"
)

FIG_DIR = ROOT / "research_paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# ============================================================
# 1. MANIPULATION-TYPE ANALYSIS
# ============================================================

order = [
    "RealVideo-RealAudio",
    "FakeVideo-RealAudio",
    "RealVideo-FakeAudio",
    "FakeVideo-FakeAudio",
]

rows = []

for manipulation_type in order:

    g = df[df["manipulation_type"] == manipulation_type]

    y_true = g["true_label"]
    y_pred = g["prediction"]

    # Fake = 0 in this evaluation CSV.
    fake_f1 = None

    if (y_true == 0).any():
        fake_f1 = f1_score(
            y_true,
            y_pred,
            pos_label=0,
            zero_division=0,
        )

    rows.append({
        "Manipulation Type": manipulation_type,
        "N": len(g),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Fake F1": fake_f1,
    })

manipulation_df = pd.DataFrame(rows)

print("\n" + "=" * 70)
print("MANIPULATION-TYPE RESULTS")
print("=" * 70)
print(manipulation_df.to_string(index=False))

manipulation_df.to_csv(
    ROOT
    / "research_paper"
    / "results"
    / "paper_eval_v3"
    / "manipulation_type_results.csv",
    index=False,
)

# ============================================================
# 2. MANIPULATION-TYPE FIGURE
# ============================================================

labels = [
    "Real–Real",
    "FakeV–RealA",
    "RealV–FakeA",
    "Fake–Fake",
]

accuracy = manipulation_df["Accuracy"].values * 100

fake_f1 = manipulation_df["Fake F1"].fillna(float("nan")).values * 100

x = range(len(labels))
width = 0.36

fig, ax = plt.subplots(figsize=(9, 5.5))

bars1 = ax.bar(
    [i - width / 2 for i in x],
    accuracy,
    width,
    label="Accuracy",
)

bars2 = ax.bar(
    [i + width / 2 for i in x],
    fake_f1,
    width,
    label="Fake F1",
)

for bars in [bars1, bars2]:

    for bar in bars:

        height = bar.get_height()

        if pd.notna(height):

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.7,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

ax.set_ylabel("Score (%)")
ax.set_xlabel("Manipulation condition")
ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.set_ylim(0, 108)
ax.set_title(
    "Performance Across Audio-Visual Manipulation Conditions"
)
ax.legend()
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()

manipulation_fig = (
    FIG_DIR / "manipulation_type_performance.png"
)

plt.savefig(
    manipulation_fig,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("\nSaved:")
print(manipulation_fig)

# ============================================================
# 3. FINAL CONTROLLED ABLATION
# ============================================================

# Previously completed experiment.
# DO NOT rerun training.

ablation_df = pd.DataFrame({

    "Configuration": [
        "Visual",
        "Audio",
        "Audio-Visual",
        "Contrastive",
        "AER",
        "Full",
    ],

    "Balanced Accuracy": [
        96.02206,
        66.90385,
        98.58364,
        98.52981,
        98.51187,
        98.49053,
    ],

    "Fake F1": [
        66.4151,
        18.3940,
        94.4245,
        93.6664,
        93.4164,
        94.3294,
    ],
})

print("\n" + "=" * 70)
print("ABLATION RESULTS")
print("=" * 70)
print(ablation_df.to_string(index=False))

ablation_df.to_csv(
    ROOT
    / "research_paper"
    / "results"
    / "ablation_results_final_reconstructed.csv",
    index=False,
)

# ============================================================
# 4. ABLATION FIGURE
# ============================================================

x = range(len(ablation_df))
width = 0.36

fig, ax = plt.subplots(figsize=(9, 5.5))

bars1 = ax.bar(
    [i - width / 2 for i in x],
    ablation_df["Balanced Accuracy"],
    width,
    label="Balanced Accuracy",
)

bars2 = ax.bar(
    [i + width / 2 for i in x],
    ablation_df["Fake F1"],
    width,
    label="Fake F1",
)

for bars in [bars1, bars2]:

    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1.0,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

ax.set_ylabel("Score (%)")
ax.set_xlabel("Model configuration")
ax.set_xticks(list(x))
ax.set_xticklabels(
    ablation_df["Configuration"],
    rotation=20,
    ha="right",
)

ax.set_ylim(0, 108)

ax.set_title(
    "Controlled Component Ablation on the Held-Out Test Set"
)

ax.legend()
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()

ablation_fig = (
    FIG_DIR / "ablation_results.png"
)

plt.savefig(
    ablation_fig,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("\nSaved:")
print(ablation_fig)

# ============================================================
# 5. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL PAPER FIGURES GENERATED")
print("=" * 70)

print("1.", manipulation_fig)
print("2.", ablation_fig)

print("\nNo training or inference was rerun.")
print("Only existing final evaluation CSV + completed ablation results were used.")