# PrismShieldAI

### Contrastive and Adaptive Multimodal Evidence Reasoning for Audio-Visual Deepfake Detection

PrismShieldAI is a research prototype for audio-visual deepfake detection. It combines visual and speech representations to investigate how multimodal evidence can support the classification of authentic and manipulated media.

The framework brings together pretrained feature encoders, embedding-level cross-modal interaction, contrastive representation learning, adaptive evidence reasoning, and a learned correctness-estimation head. It also includes a video inference pipeline and an interactive Streamlit dashboard.

**Research status:** The current model has been evaluated on a processed FakeAVCeleb dataset partition. Cross-dataset generalization, identity-disjoint evaluation, and fine-grained temporal audio-visual alignment have not been established.

---

## Overview

Synthetic media may contain manipulation cues in the visual stream, the audio stream, or both. PrismShieldAI processes audio and visual inputs through separate encoders, interacts their global representations, and combines them for binary classification.

### Key components

* **Visual encoder:** EfficientNet-B0 extracts facial image representations.
* **Audio encoder:** Wav2Vec2 extracts speech/audio representations.
* **Cross-modal interaction:** Embedding-level interaction between global audio and visual representations, implemented using bidirectional attention modules.
* **Contrastive learning:** Projection heads and a symmetric InfoNCE objective encourage corresponding audio and visual representations to occupy a shared latent space.
* **Adaptive Evidence Reasoning (AER):** Learns modality-specific evidence transformations, interaction features, and sample-dependent audio/visual importance weights.
* **Binary classifier:** Predicts whether the input is Fake or Real from the fused reasoning representation.
* **Correctness-estimation head:** Produces a learned score trained in relation to whether a prediction is correct. This score is not a guarantee of calibrated uncertainty.
* **Inference dashboard:** A Streamlit interface for uploading and analyzing media using the trained model.

> **Attention scope:** The current attention modules operate on one global token per modality. With only one token in each attention sequence, the attention distribution is trivial. This is embedding-level cross-modal interaction, not frame-to-frame, phoneme-to-frame, or fine-grained temporal alignment.

> **Terminology:** Adaptive Evidence Reasoning describes learned neural transformations and modality weighting. The current implementation does not establish structural causal modeling, causal interventions, or counterfactual inference.

---

## Architecture

```text
                   Input Video
                       |
               Frame / Face Processing
                       |
                 EfficientNet-B0
                       |
                Visual Embedding
                       |
                       +--------------------------+
                                                  |
                                                  v
Audio Input                                  Cross-Modal
    |                                        Interaction
Audio Processing                                  |
    |                                             |
  Wav2Vec2                                        |
    |                                             |
Audio Embedding                                   |
    |                                             |
    +---------------------------------------------+
                                                  |
                                   Contrastive Projection Heads
                                                  |
                                      Symmetric InfoNCE Loss
                                                  |
                                    Adaptive Evidence Reasoning
                                                  |
                                Weighted Multimodal Representation
                                                  |
                                        Binary Classifier
                                                  |
                                     Fake / Real Prediction
                                                  |
                                   Correctness-Estimation Score
```

This diagram presents the high-level processing flow. The contrastive objective contributes to training, while the classifier produces the final binary prediction. The correctness-estimation head provides an additional learned score.

The architecture does not imply causal inference or explicit temporal synchronization between audio and video.

---

## Dataset and evaluation

The current experiments use **FakeAVCeleb**. The processed dataset is partitioned by video ID so that samples derived from a video are assigned to the same split.

### Dataset split

| Split      |    Samples |
| ---------- | ---------: |
| Training   |      5,027 |
| Validation |      8,755 |
| Test       |      8,897 |
| **Total**  | **22,679** |

### Test-set label distribution

| Class     | Label ID | Test samples |
| --------- | -------: | -----------: |
| Fake      |        0 |        8,360 |
| Real      |        1 |          537 |
| **Total** |          |    **8,897** |

The test set is substantially imbalanced. For that reason, accuracy is reported alongside balanced accuracy, class-specific F1, Macro-F1, ROC-AUC, PR-AUC, and Matthews Correlation Coefficient (MCC).

The split is video-disjoint according to the dataset-building protocol. Identity-disjoint separation has not been established. Results should therefore be interpreted as **within-dataset evaluation on the processed FakeAVCeleb partition**, not as evidence of generalization to unseen identities, generators, datasets, or real-world media.

---

## Main evaluation results

The following metrics are from the committed evaluation results for the current checkpoint, selected at epoch 21.

| Metric                           | Result |
| -------------------------------- | -----: |
| Test samples                     |  8,897 |
| Accuracy                         | 99.04% |
| Balanced Accuracy                | 98.71% |
| Fake Precision                   | 99.89% |
| Fake Recall                      | 99.09% |
| Fake F1-score                    | 99.49% |
| Real Precision                   | 87.42% |
| Real Recall                      | 98.32% |
| Real F1-score                    | 92.55% |
| Macro-F1                         | 96.02% |
| ROC-AUC (Real positive class)    | 99.83% |
| PR-AUC (Real positive class)     | 94.93% |
| Matthews Correlation Coefficient | 0.9222 |

### Confusion matrix

Class order: **Fake (0), Real (1)**. Rows represent actual labels and columns represent predicted labels.

| Actual \ Predicted |  Fake | Real |
| ------------------ | ----: | ---: |
| Fake               | 8,284 |   76 |
| Real               |     9 |  528 |

The test metrics describe this specific processed FakeAVCeleb test partition. They should not be interpreted as expected performance on other datasets or as proof of reliable detection in every real-world setting.

The reported precision, recall, and F1 values for Fake and Real are class-specific. ROC-AUC and PR-AUC use **Real (label 1)** as the positive class.

---

## Additional analyses

The repository includes supporting experiments and analysis artifacts, including:

* **Component ablation:** Comparisons of visual-only, audio-only, multimodal, contrastive, adaptive-reasoning, and full configurations.
* **Leading-silence robustness check:** A limited sensitivity analysis involving leading-silence trimming.
* **Calibration analysis:** Evaluation using metrics such as Expected Calibration Error (ECE) and Brier score, alongside calibration visualizations.
* **Selective-risk analysis:** Analysis of model performance under confidence-based selection.
* **Manipulation-type analysis:** Evaluation across RealVideo–RealAudio, FakeVideo–RealAudio, RealVideo–FakeAudio, and FakeVideo–FakeAudio conditions.
* **Representation visualizations:** Contrastive similarity and embedding visualizations.
* **Modality-weight analysis:** Analysis of learned adaptive audio and visual importance weights.

Ablation results should be interpreted as comparisons within the documented experimental setup. They do not establish that every added component independently improves performance, nor should they be treated as independently retrained end-to-end results unless the corresponding experiment confirms that procedure.

The leading-silence experiment is a limited sensitivity check; it does not prove that all dataset shortcuts have been eliminated. Learned modality weights are model signals and should not automatically be interpreted as causal explanations.

---

## Repository structure

```text
PrismShieldAI/
├── app/
│   └── app2.py
├── audio/
├── configs/
├── datasets/
├── evaluation/
├── inference/
├── losses/
├── models/
├── notebooks/
├── research_paper/
│   ├── figures/
│   ├── historical/
│   ├── results/
│   └── scripts/
├── tests/
├── training/
├── utils/
├── weights/
├── .gitignore
└── README.md
```

The `research_paper` directory contains research evaluation scripts, aggregate result files, figures, and archived historical artifacts. Large intermediate files and per-sample outputs are kept separately from the canonical aggregate summaries where applicable.

Exact contents may change as the repository evolves; consult the repository tree for the current file-level structure.

---

## Technology stack

* Python
* PyTorch
* EfficientNet-B0
* Wav2Vec2
* Hugging Face Transformers
* OpenCV
* Streamlit
* NumPy
* Pandas
* Scikit-learn
* Matplotlib / Seaborn (where used in analysis scripts)

---

## Running the dashboard

### 1. Clone the repository

```powershell
git clone https://github.com/ShivaayaPangasa/PrismShieldAI.git
cd PrismShieldAI
```

### 2. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Install the dependencies specified by the repository's dependency file, if available. Otherwise, consult the project configuration and scripts to install the required compatible versions of PyTorch, Transformers, Streamlit, and the other libraries.

### 4. Prepare required assets

Ensure the required model checkpoint, dataset-derived inputs, and media-processing dependencies are available. Dataset files and model weights may not be included in the repository.

### 5. Launch the Streamlit interface

From the repository root:

```powershell
streamlit run app/app2.py
```

The dashboard is a research prototype. Its output is a model prediction and a learned score, not definitive proof that media is authentic or manipulated.

---

## Reproducibility and artifacts

The repository contains scripts and artifacts for the main evaluation, component ablation, robustness checks, and research-figure generation.

Relevant aggregate results and figures are stored under:

* `research_paper/results/`
* `research_paper/figures/`

For reproduction, use the same dataset version, preprocessing, split definitions, checkpoint, dependencies, and evaluation settings as the reported experiment. Consult the individual scripts and configuration files for exact implementation details.

The main evaluation metrics are recorded in the evaluation JSON artifact. Ablation summaries are separate from the main-model evaluation and should not be substituted for the main results.

Model weights and datasets may not be included in the repository. Ensure that you have legitimate access to the dataset and the required checkpoint before attempting reproduction.

---

## Limitations

The current work has several limitations:

* Evaluation is restricted to a processed FakeAVCeleb dataset partition.
* The split is video-disjoint, but identity-disjoint separation has not been established.
* Cross-dataset and unseen-generator generalization have not been evaluated.
* Cross-modal interaction uses global embeddings rather than fine-grained temporal tokens.
* One-token attention does not provide a meaningful learned distribution over multiple temporal elements.
* The leading-silence experiment is a limited shortcut-sensitivity check.
* Adaptive modality weights are learned model signals, not causal explanations.
* The correctness-estimation head does not guarantee calibrated uncertainty or reliable explanations for individual predictions.
* Benchmark performance does not establish suitability for high-stakes forensic, legal, identity, or other consequential decisions.
* Further independent validation is required before making claims about performance outside the evaluated setting.

---

## Future work

Potential directions for future development include:

* Cross-dataset and unseen-generator evaluation
* Identity-disjoint evaluation
* Fine-grained temporal audio-visual interaction
* Additional shortcut and distribution-shift testing
* Improved uncertainty estimation and calibration
* More extensive interpretability and failure analysis
* Evaluation on broader real-world media conditions

These are future research directions and are not claimed as completed capabilities of the current version.

---

## Research manuscript

**Working title:**

*PrismShieldAI: Contrastive and Adaptive Multimodal Evidence Reasoning for Audio-Visual Deepfake Detection*

The manuscript is being prepared around the current implementation and evaluation. It will document the model design, experimental protocol, results, limitations, and related work.

---

## Disclaimer

PrismShieldAI is an experimental research prototype for audio-visual deepfake detection. It is not a forensic certification tool and should not be used as the sole basis for consequential decisions about the authenticity of media.

## Author

**Shivaaya Pangasa**
B.Tech Artificial Intelligence
Amity University, Noida, India

**Repository:** https://github.com/ShivaayaPangasa/PrismShieldAI
