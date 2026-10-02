# PrismShieldAI

### Contrastive and Adaptive Multimodal Evidence Reasoning for Audio-Visual Deepfake Detection

PrismShieldAI is a research prototype for audio-visual deepfake detection. It combines visual and speech representations to investigate whether joint audiovisual evidence can support the classification of authentic and manipulated media.

The framework brings together pretrained feature encoders, embedding-level cross-modal interaction, contrastive representation learning, adaptive evidence reasoning, and a learned correctness-estimation head. It also includes an inference pipeline and an interactive Streamlit dashboard.

**Research status:** Experimental evaluation on FakeAVCeleb is complete for the current model version. The research manuscript is being prepared. Cross-dataset generalization and fine-grained temporal audio-visual alignment have not been evaluated.

---

## Overview

Synthetic media can contain manipulation cues in the visual stream, audio stream, or both. PrismShieldAI explores a multimodal approach that processes visual and audio inputs separately, interacts their global representations, and combines them for final classification.

### Key components

* **Visual encoder:** EfficientNet-B0 for facial image representation.
* **Audio encoder:** Wav2Vec2 for speech/audio representation.
* **Cross-modal interaction:** Bidirectional embedding-level interaction between global audio and visual representations.
* **Contrastive learning:** Separate projection heads and a symmetric InfoNCE objective to encourage corresponding audio and visual representations to share a latent space.
* **Adaptive Evidence Reasoning (AER):** Learns modality-specific evidence transformations, interaction features, and sample-dependent modality importance weights.
* **Classifier:** Predicts the final binary class from the fused reasoning representation.
* **Correctness-estimation head:** Produces a learned estimate related to prediction correctness; it should not be interpreted as a guarantee of calibrated confidence.
* **Inference dashboard:** Streamlit interface for uploading and analyzing media through the trained model.

> **Attention scope:** The current cross-modal attention operates on one global token per modality. It is therefore embedding-level interaction, not frame-to-frame, phoneme-to-frame, or fine-grained temporal alignment.

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
                 +-----------------------------+
                                               |
                                               v
Audio Input                              Cross-Modal
    |                                     Interaction
Audio Processing                               |
    |                                          |
  Wav2Vec2                                     |
    |                                          |
Audio Embedding                                |
    |                                          |
    +------------------------------------------+
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
                                  Correctness Estimation
```

The diagram is a high-level representation of the processing flow. It does not imply causal inference or explicit temporal synchronization.

---

## Dataset and evaluation

The current experiments use **FakeAVCeleb**. The reported evaluation uses a video-disjoint partitioning protocol, meaning videos are separated across the train, validation, and test splits.

The current processed dataset contains 22,679 samples:

| Split      |    Samples |
| ---------- | ---------: |
| Training   |      5,027 |
| Validation |      8,755 |
| Test       |      8,897 |
| **Total**  | **22,679** |

The test distribution is imbalanced, with 8,360 Real and 537 Fake samples as recorded in the committed classification report. Balanced accuracy, Macro-F1, and MCC are therefore reported alongside accuracy.

The split is video-disjoint; identity-disjoint separation has not been established. The current results should be interpreted as **within-dataset evaluation on FakeAVCeleb**, not evidence of generalization to unseen datasets, identities, or generators.

---

## Reported results

The following aggregate metrics are from the committed evaluation results for the current checkpoint, selected at epoch 21.

| Metric                           | Result |
| -------------------------------- | -----: |
| Accuracy                         | 99.04% |
| Balanced Accuracy                | 98.71% |
| Macro-F1                         | 96.02% |
| ROC-AUC                          | 99.83% |
| PR-AUC                           | 94.93% |
| Matthews Correlation Coefficient | 0.9222 |

These values describe the evaluated FakeAVCeleb test partition. They should not be interpreted as expected performance on other datasets or real-world media.

Class-specific metrics and the confusion-matrix class ordering should be checked against the final dataset-label mapping before publication.

---

## Additional analyses

The repository includes supporting experiments and visualizations:

* **Component ablation:** compares visual-only, audio-only, multimodal, contrastive, adaptive-reasoning, and full configurations.
* **Leading-silence robustness check:** evaluates the same checkpoint after leading-silence trimming.
* **Calibration analysis:** includes ECE, Brier score, reliability diagrams, and classification-probability calibration results.
* **Selective-risk analysis:** examines performance under confidence-based selection.
* **Manipulation-type analysis:** reports results across RealVideo–RealAudio, FakeVideo–RealAudio, RealVideo–FakeAudio, and FakeVideo–FakeAudio conditions.
* **Representation visualizations:** include contrastive similarity and embedding visualizations.
* **Modality-weight analysis:** summarizes learned adaptive audio and visual importance weights.

The ablation is a controlled component study using the completed evaluation setup, not a claim that every component was independently retrained end-to-end. The measured results should be interpreted directly; they do not establish that every added module produces a monotonic improvement.

The leading-silence test is a limited sensitivity check. It does not prove that all dataset shortcuts have been eliminated.

---

## Repository structure

```text
PrismShieldAI/
├── app/
│   └── app2.py
├── audio/
├── configs/
├── evaluation/
├── inference/
├── losses/
├── models/
├── notebooks/
├── research_paper/
│   ├── figures/
│   ├── results/
│   └── scripts/
├── tests/
├── training/
├── utils/
├── weights/
├── .gitignore
└── README.md
```

The `research_paper` directory contains evaluation scripts, aggregate result files, and figures used to prepare the manuscript. Large intermediate artifacts, such as embeddings and per-sample predictions, are retained separately from the aggregate research summaries.

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
* Matplotlib / Seaborn (where used in analysis scripts)

---

## Running the dashboard

Create and activate a Python virtual environment, install the project's required dependencies, and ensure the required model checkpoint and media-processing dependencies are available.

From the repository root, launch the Streamlit interface:

```powershell
streamlit run app/app2.py
```

The dashboard is a research prototype. Its outputs are model predictions and learned estimates, not definitive proof that media is authentic or manipulated.

---

## Reproducibility and artifacts

The repository includes scripts for:

* Main evaluation
* Leading-silence robustness evaluation
* Main-result reproduction
* Component ablation
* Final paper-figure generation

Relevant aggregate metrics and figures are stored under `research_paper/results/` and `research_paper/figures/`.

To reproduce results, use the same dataset preprocessing, split definitions, model checkpoint, dependencies, and evaluation settings as the reported experiment. The repository's scripts should be consulted for their exact arguments and configuration.

Model weights and datasets may not be included in the repository. Ensure that you have legitimate access to the dataset and the required checkpoint before attempting reproduction.

---

## Limitations

The current work has several limitations:

* Evaluation is restricted to FakeAVCeleb.
* The split is video-disjoint, but identity-disjoint separation has not been verified.
* Cross-dataset and unseen-generator generalization have not been evaluated.
* Cross-modal interaction is performed on global embeddings rather than fine-grained temporal tokens.
* The leading-silence experiment is a limited shortcut-sensitivity check.
* Adaptive modality weights are learned model signals and should not automatically be interpreted as causal explanations.
* Correctness estimation does not guarantee calibrated uncertainty or reliable explanations for individual predictions.
* Performance on a benchmark does not establish suitability for high-stakes forensic or identity decisions.

---

## Future work

Potential directions include:

* Cross-dataset and unseen-generator evaluation
* Identity-disjoint evaluation
* Fine-grained temporal audio-visual interaction
* Further shortcut and distribution-shift testing
* Improved uncertainty estimation and calibration
* More extensive interpretability and failure analysis
* Evaluation on broader real-world media conditions

These are future directions, not capabilities claimed as completed in the current version.

---

## Research manuscript

**Working title:**

*PrismShieldAI: Contrastive and Adaptive Multimodal Evidence Reasoning for Audio-Visual Deepfake Detection*

The manuscript is being prepared around the current implementation and evaluation. The final paper will document the model design, experimental protocol, results, limitations, and related work.

---

## Disclaimer

PrismShieldAI is an experimental research prototype. It is not a forensic certification tool and should not be used as the sole basis for consequential decisions about the authenticity of media.

## Author

**Shivaaya Pangasa**
B.Tech Artificial Intelligence
Amity University, Noida, India

**Repository:** https://github.com/ShivaayaPangasa/PrismShieldAI