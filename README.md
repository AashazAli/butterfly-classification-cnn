# 100-Species Butterfly Image Classification using Custom CNN & Grad-CAM

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Computer Vision pipeline designed to classify **100 distinct species of butterflies**. This project features an optimized **Custom Convolutional Neural Network (CNN)** with Data Augmentation and Batch Normalization, coupled with **Grad-CAM (Gradient-Weighted Class Activation Mapping)** for model interpretability and Explainable AI (XAI).

---

## Key Highlights

* **High Accuracy:** Achieved **~92.4% Test Accuracy** across 100 fine-grained butterfly species.
* **Explainable AI (XAI):** Integrated Grad-CAM visualization to verify that the model relies on biological markers (e.g., wing patterns, venation, color symmetry) rather than background noise.
* **Robust Pipeline:** Built with built-in data augmentation (`RandomFlip`, `RandomRotation`, `RandomZoom`) to prevent overfitting.
* **Handled Architectural Edge Cases:** Solved nested graph backpropagation hurdles using flat model re-architecting for gradient extraction.

---

 ##  Repository Directory Setup (CRUCIAL)

To keep this repository lightweight and adhere to GitHub's file size policies, large raw image files and trained model weights are excluded via `.gitignore`. 

However, the notebook relies on a specific folder structure to run correctly. The local `data/` and `models/` folders **must be populated manually** before executing the code.

### 1. The `data/` Directory (Dataset)
Do NOT upload image files to this folder on GitHub. This folder currently contains only a `.gitkeep` file to keep the directory structure visible.

* **Your Action:** Download the **100 Butterfly Species Dataset** from Kaggle.
* **Extraction:** Extract the download and move the `train/`, `test/`, and `valid/` folders inside your local `data/` directory so it looks like this:
  ```text
  butterfly-classification-cnn/
  ├── data/
  │   ├── .gitkeep              <-- Already here
  │   ├── train/                <-- Add this (Contains species subfolders)
  │   ├── test/                 <-- Add this (Contains species subfolders)
  │   └── valid/                <-- Add this (Contains species subfolders)

## Performance Overview

| Metric | Score |
| :--- | :--- |
| **Training Accuracy** | ~94.5% |
| **Validation Accuracy** | ~91.8% |
| **Test Accuracy** | **92.4%** |
| **Total Classes** | 100 Butterfly Species |

---

## Visualizing Model Focus with Grad-CAM

To ensure the custom CNN is learning authentic species identifiers, **Grad-CAM** was applied to the final convolutional layer (`conv_4`). 

![Grad-CAM Visualization](outputs/gradcam_sample.png)

> **Key Observation:** The activation heatmaps consistently highlight the **distal edges and distinct coloration of the wings**, demonstrating high biological validity.

---

## Architecture Design

The model utilizes a 4-block deep CNN design built using TensorFlow/Keras:

```text
Input (224, 224, 3)
   │
   ├── [Preprocessing] Random Flip, Rotation, Zoom
   ├── [Block 1] Conv2D (32 filters) ──> BatchNorm ──> MaxPool2D
   ├── [Block 2] Conv2D (64 filters) ──> BatchNorm ──> MaxPool2D
   ├── [Block 3] Conv2D (128 filters) ──> BatchNorm ──> MaxPool2D
   ├── [Block 4] Conv2D (256 filters, "conv_4") ──> BatchNorm ──> GlobalAveragePooling2D
   │
   ├── Dense (512 units, ReLU)
   ├── Dropout (0.5)
   └── Dense (100 units, Softmax Output)

  