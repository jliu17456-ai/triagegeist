# The Risk-Controlled Second Reader

**Triagegeist Kaggle Hackathon 2026** — AI-Powered Undertriage Safety Net with Conformal Guarantees

## Overview

Emergency department undertriage — assigning lower-than-appropriate acuity to a high-risk patient — is a patient safety problem. This project ships a nurse-AI disagreement safety net with finite-sample, distribution-free False Negative Rate (FNR) guarantees, validated on two independent real-world datasets.

**Core idea**: Instead of competing with the triage nurse, the model predicts what an expert panel would decide. The gap between the nurse's judgment and the model's prediction is the alert signal.

## Key Results

| Metric | Value |
|--------|-------|
| Disagreement score AUC | **0.824** (95% CI: 0.789–0.859) |
| Primary operating point (α=0.15) | Alarm 57%, Recall 95%, FNR 5.1% |
| CRC theoretical bound | ≤ 16.1% |
| TSTR gap vs. HALO baseline | **87×** larger (Δ_acc = 0.433) |
| Yale ESI 3 AUROC | **0.743** (n=150k) |

## Structure

```
├── triagegeist_notebook.ipynb   # Main submission — runs end-to-end on Kaggle CPU (~8 min)
├── pipeline.py                  # Extended validation pipeline (Act 1–3, runs on Modal)
├── results/                     # Result JSONs backing all quantitative claims
└── WRITEUP.md                   # Competition writeup (≤2000 words)
```

## Three-Act Approach

**Act 1 — Synthetic Benchmark Audit (SBAP)**: Formally characterize the competition's synthetic data generator. A depth-16 decision tree recovers the NEWS2 backbone at R²=0.9964. Human accuracy ceiling: 77.3% vs. benchmark 97.75% — saturation reflects the generative process, not model strength.

**Act 2 — Risk-Controlled Safety Net**: Nurse-model disagreement score (AUC 0.824) with split-conformal CRC providing finite-sample FNR guarantees. Validated on KTAS expert-adjudicated data (n=1,267) and Yale outcome-anchored data (n=560,486).

**Act 3 — Sim-to-Real Formalization**: TSTR gap Δ_acc=0.433 — roughly 87× the HALO synthetic EHR benchmark gap. Metamorphic testing confirms the reversal is structural, not a sample-size artifact.

## Datasets

- **KTAS**: [moonssop/ktas](https://www.kaggle.com/datasets/moonssop/ktas) — Moon et al., PLOS One 2019
- **Yale ED**: Hong et al., PLOS One 2018
- **Synthetic data**: Triagegeist competition dataset

## Reproducibility

All experiments run on [Modal](https://modal.com) (CPU containers). Notebook runs end-to-end on Kaggle CPU (~8 min). Random seed: 42.

```
lightgbm==4.5.0
scikit-learn==1.5.2
pandas==2.2.3
numpy==2.1.3
```
