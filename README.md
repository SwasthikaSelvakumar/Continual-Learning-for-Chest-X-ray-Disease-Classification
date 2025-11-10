## Continual Learning for Chest X-Ray Disease Classification

## Overview

This repository contains an implementation of a **continual learning framework** for medical image classification using chest X-ray datasets. The model integrates **Elastic Weight Consolidation (EWC)** to mitigate catastrophic forgetting, enabling sequential learning across multiple disease detection tasks such as **tuberculosis (TB)** and **COVID-19**.
The project also incorporates an **autoencoder-based anomaly detection** system and **Grad-CAM** for model interpretability.

---

## Features

* **Multi-task Learning:** Sequential training on tuberculosis and COVID-19 datasets.
* **Continual Learning (EWC):** Retains previously learned knowledge while adapting to new tasks.
* **Autoencoder for Anomaly Detection:** Identifies corrupted or unseen samples through reconstruction error analysis.
* **Model Interpretability:** Employs Grad-CAM for visualization of critical lung regions influencing predictions.
* **Comprehensive Evaluation:** Includes accuracy, loss curves, confusion matrices, and ROC analysis.
* **GPU-Optimized Implementation:** Compatible with PyTorch and TensorFlow environments (e.g., Google Colab).

---

## System Architecture

### 1. Classification Model (ResNet18)

* Backbone: ResNet18 pretrained on ImageNet.
* Task 1: Tuberculosis detection using Montgomery dataset.
* Task 2: COVID-19 classification using the COVIDx dataset.
* Fine-tuned layers for task-specific classification.

### 2. Continual Learning with EWC

Implements **Elastic Weight Consolidation**, which constrains updates to weights crucial for previously learned tasks using the Fisher Information Matrix. This ensures stable performance on earlier datasets when new ones are introduced.

### 3. Autoencoder for Anomaly Detection

* Dataset: NIH Chest X-ray normal images.
* Architecture: Convolutional encoder-decoder.
* Purpose: Detect abnormal or noisy inputs based on reconstruction loss (MSE).

### 4. Grad-CAM Visualization

Provides heatmaps for visual interpretability, highlighting the lung regions most influential to classification outcomes.

---

## Datasets

| Dataset          | Purpose                     | Classes                    | Source         |
| ---------------- | --------------------------- | -------------------------- | -------------- |
| Montgomery       | Tuberculosis classification | Normal / TB                | NIH TB Dataset |
| Shenzhen         | Auxiliary TB dataset        | Normal / TB                | NIH            |
| COVIDx           | COVID-19 detection          | COVID / Normal / Pneumonia | Kaggle         |
| NIH ChestX-ray14 | Autoencoder training        | Normal                     | NIH            |

---

## Installation

### Prerequisites

* Python 3.8 or higher
* CUDA-compatible GPU (recommended)
* Google Colab or local environment with PyTorch and TensorFlow

### Setup

```bash
git clone https://github.com/<your-username>/continual-learning-ml.git
cd continual-learning-ml
pip install torch torchvision torchaudio tensorflow matplotlib pandas opencv-python tqdm seaborn scikit-learn
```

### Directory Structure

```
continual_learning_ml.py      # Main training and analysis script
datasets/                     # Datasets for TB, COVID, and NIH
models/                       # Trained model weights (.pth)
plots/                        # Training metrics and visualization outputs
README.md                     # Project documentation
```

---

## Training Procedure

1. **Task 1: Tuberculosis Classification (Montgomery)**

   * Train a ResNet18 model for binary classification.
   * Save checkpoint for later transfer.

2. **Fisher Matrix Computation**

   * Compute Fisher Information from Task 1 to identify critical parameters.

3. **Task 2: COVID-19 Classification (COVIDx)**

   * Fine-tune model with EWC regularization to retain TB knowledge.

4. **Anomaly Detection (NIH Dataset)**

   * Train a convolutional autoencoder on normal X-ray samples.
   * Use reconstruction loss to identify anomalies.

5. **Grad-CAM Visualization**

   * Generate attention heatmaps for validation samples.

---

## Results

| Task                          | Accuracy         | Validation Loss | AUC   | Dataset    |
| ----------------------------- | ---------------- | --------------- | ----- | ---------- |
| TB Detection                  | 97.8%            | 0.08            | 0.985 | Montgomery |
| COVID-19 Classification       | 97.4%            | 0.09            | 0.982 | COVIDx     |
| Autoencoder Anomaly Detection | Threshold = 0.02 | —               | —     | NIH        |

Performance metrics include confusion matrices, ROC curves, and comparative plots of accuracy and loss across tasks.

---

## Evaluation and Visualization

* Training and validation curves for loss and accuracy.
* Confusion matrices for each dataset.
* ROC curves derived from validation results.
* Grad-CAM overlays for interpretability.
* Comparative analysis of knowledge retention after each learning phase.

---

## License

This project is released under the **MIT License**.
You are free to use, modify, and distribute this code with appropriate credit.

```
MIT License
Copyright (c) 2025 Swasthika S
```

---

