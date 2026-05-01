# Building-Lightweight-CNN-Models-on-CIFAR-10-and-Fashion-MNIST
Reproduction and extension of arXiv:2501.15547. Implements a dual-input-output lightweight CNN baseline on CIFAR-10, then proposes an improved model with residual+SE attention, depthwise separable convolutions, MixUp/CutMix augmentation, and AdamW with warmup-cosine scheduling. Achieves 72% on CIFAR-10 and 91.6% on Fashion MNIST.

## Project Overview

This project has two parts:

1. **Reproduction** — Faithfully re-implements the base paper's two-stage pipeline: a dual-input-output CNN trained simultaneously on original and augmented CIFAR-10 data, followed by feature concatenation and progressive layer unfreezing using SGD.

2. **Improvement** — Proposes an enhanced lightweight architecture using residual blocks with Squeeze-and-Excitation (SE) attention, depthwise separable convolutions, MixUp/CutMix augmentation, AdamW with warmup-cosine scheduling, label smoothing, and adaptive dropout scheduling.

## Results

| Model | CIFAR-10 Accuracy | Params | Size |
|---|---|---|---|
| Base Paper (Reproduced) | 65% | 14,862 | 0.06 MB |
| Improved Baseline CNN | 68% | 120,000 | 0.46 MB |
| Improved (Residual + SE) | **72%** | 85,000 | 0.32 MB |

**Fashion MNIST (cross-dataset evaluation):**

| Model | Accuracy |
|---|---|
| Improved Baseline CNN | 90.0% |
| Improved (Residual + SE) | **91.6%** |

McNemar's statistical significance test confirms improvements are genuine (p < 0.05) on both datasets.

## Key Techniques

- Dual-input-output CNN with progressive unfreezing (base paper reproduction)
- Residual blocks with Squeeze-and-Excitation channel attention
- Depthwise separable convolutions for parameter efficiency
- MixUp (alpha=0.2) and CutMix (alpha=1.0) augmentation
- AdamW optimizer with warmup-cosine learning rate schedule
- Label smoothing (epsilon=0.1) and adaptive dropout scheduling (0.4 to 0.2)

## Repository Structure
├── Implementation.ipynb   # Full implementation: reproduction + improved model
├── report_final.tex       # LaTeX source of the report
└── README.md

## How to Run

1. Clone the repository
2. Install dependencies:
```bash
   pip install tensorflow numpy pandas matplotlib seaborn scikit-learn scipy statsmodels
```
3. Open `Implementation.ipynb` in Jupyter or Google Colab and run all cells

> The notebook runs on CPU. For faster training, upload to Google Colab and enable GPU runtime.

## Reference

Nathan, I. (2025). Building Efficient Lightweight CNN Models. arXiv:2501.15547.
