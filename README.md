# Decoding Emotions through fNIRS: Graph-Based Semantic Analysis

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

A deep learning pipeline that classifies three emotional states — **Afraid**, **Neutral**, and **Happy** — from functional Near-Infrared Spectroscopy (fNIRS) brain signals using **Graph Convolutional Networks (GCN)**, **SVM**, and **CBAM attention mechanisms**.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline Architecture](#pipeline-architecture)
- [Methods & Models](#methods--models)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Key Findings](#key-findings)
- [Future Work](#future-work)

---

## Overview

This project investigates whether brain connectivity patterns captured by fNIRS can be used to distinguish between emotional states. Rather than treating fNIRS signals as raw time-series, the approach converts them into **brain connectivity graphs** — where nodes represent brain channels and edges represent inter-channel correlations — and applies graph neural networks to classify emotions.

The key insight is that emotional states modulate coordinated activity across multiple brain regions. By modeling these co-activation patterns as a graph, we can leverage the relational structure of brain signals that flat feature vectors would miss.

---

## Dataset

**Download:** [Google Drive — fNIRS Emotion Dataset](https://drive.google.com/drive/folders/14CI_rNmmbImserk3p-E0HJLLWaYV4Qg6)

The dataset consists of fNIRS recordings from participants experiencing three emotion conditions:

| Class | Label | Description |
|-------|-------|-------------|
| Afraid | 0 | Fear-inducing stimuli |
| Neutral | 1 | Baseline/neutral condition |
| Happy | 2 | Positive/happy stimuli |

**Signal properties:**
- **Channels:** 14 fNIRS channels (prefrontal cortex region)
- **Sampling rate:** 6 Hz
- **Window size:** 360 samples (~60 seconds) per trial
- **Wavelengths:** Two wavelengths (wl1, wl2) for HbO/HbR separation
- **Subjects:** Multiple participants; data merged and augmented to ~50 trials per class

---

## Pipeline Architecture

```
Raw fNIRS Signals (wl1, wl2)
          │
          ▼
  ┌─────────────────────┐
  │  Step 1: Windowing  │  360-sample non-overlapping windows
  └─────────────────────┘
          │
          ▼
  ┌──────────────────────────┐
  │  Step 2: Normalization   │  Min-max per channel per trial
  └──────────────────────────┘
          │
          ▼
  ┌────────────────────────────────┐
  │  Step 3: HbO - HbR Difference │  wl1 - wl2 (oxygenation signal)
  └────────────────────────────────┘
          │
          ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  Step 4: Elliptical Bandpass Filter (0.01–2 Hz, 4th order)  │
  └──────────────────────────────────────────────────────────────┘
          │
          ▼
  ┌─────────────────────────────────────────────────────┐
  │  Step 5: Pearson Correlation Matrix (14 × 14)       │  Per trial
  └─────────────────────────────────────────────────────┘
          │
          ▼
  ┌────────────────────────────────────────────────────────────┐
  │  Step 6: Graph Construction via NetworkX (threshold=0.5)   │
  │  Binary adjacency matrix + degree/betweenness centrality   │
  └────────────────────────────────────────────────────────────┘
          │
          ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Step 7: Feature Extraction                              │
  │   • Flattened adjacency matrix (14×14 = 196 features)    │
  │   • Degree centrality + Betweenness centrality (28 feat) │
  └──────────────────────────────────────────────────────────┘
          │
          ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  Step 8: Classification                                     │
  │   • SVM (RBF kernel)                                        │
  │   • GCN with various activation functions (PyTorch)         │
  │   • GCN + Max Pooling                                       │
  │   • CBAM Attention (TensorFlow/Keras)                       │
  └─────────────────────────────────────────────────────────────┘
```

---

## Methods & Models

### Signal Preprocessing

| Step | Method | Parameters |
|------|--------|------------|
| Windowing | Non-overlapping segments | 360 samples/window |
| Normalization | Min-max scaling | Per channel, per trial |
| HbO/HbR separation | Wavelength subtraction | wl1 − wl2 |
| Bandpass filtering | Elliptic IIR filter | 0.01–2 Hz, fs=6 Hz, order=4, rp=1 dB, rs=60 dB |

### Graph Construction

Each trial is represented as a **14-node undirected graph** where:
- **Nodes** = fNIRS channels
- **Edge weights** = Pearson correlation between channel time-series
- **Adjacency threshold** = 0.5 (binary edges for |correlation| > 0.5)
- **Node features** = Degree centrality + Betweenness centrality

### Classifiers

#### 1. SVM (Baseline)
- **Kernel:** RBF (C=1, gamma='scale')
- **Features:** Flattened 14×14 adjacency matrix OR interleaved centrality values
- **Split:** 70% train / 30% test

#### 2. Graph Convolutional Network (GCN) — PyTorch Geometric
Multiple GCN architectures were explored with different activation functions:

| Model | Architecture | Activation |
|-------|-------------|------------|
| GCN-ReLU | GCNConv(2,128) → GCNConv(128,256) → Linear(256,3) | ReLU |
| GCN-ReLU-4L | GCNConv×4 (2→256→256→256→128) → Linear(128,3) | ReLU |
| GCN-LeakyReLU | GCNConv(2,64) → GCNConv(64,128) → Linear(128,3) | LeakyReLU(0.2) |
| GCN-Mish | GCNConv(2,128) → GCNConv(128,64) → Linear(64,3) | Mish |
| GCN-Swish | GCNConv(2,128) → GCNConv(128,32) → Linear(32,3) | Swish |
| GCN-MaxPool | GCNConv(2,512) → MaxPool1d → MLP(128→512→256→3) | ReLU + MaxPool |

**Training configuration:**
- Optimizer: Adam (lr=0.0001)
- Loss: CrossEntropyLoss
- Epochs: up to 1000 with early stopping
- Validation: 5-fold Stratified Cross-Validation

#### 3. CBAM (Convolutional Block Attention Module) — TensorFlow/Keras
Implements dual attention on feature maps:
- **Channel Attention:** GlobalAveragePooling + GlobalMaxPooling → shared Dense layers → sigmoid gate
- **Spatial Attention:** Channel-wise mean & max → Conv2D(1, 7×7) → sigmoid gate

#### 4. Raw Signal GCN
Applies GCN directly to raw filtered fNIRS data (14 channels × 360 timesteps) without graph construction, treating each timestep as a node.

---

## Results

### Classification Performance

| Model | Feature Type | Accuracy | Notes |
|-------|-------------|----------|-------|
| **SVM (adjacency)** | 14×14 binary adjacency matrix | **72.2%** | 18 test samples |
| GCN + Max Pooling | Degree + Betweenness centrality | **63.16%** | 5-fold CV |
| SVM (centrality) | Interleaved degree + betweenness | 61.1% | 18 test samples |
| GCN on raw fNIRS | 14-channel raw signals | 53.93% | 5-fold CV, 54k samples |
| MLP-GCN variants | Degree + Betweenness | ~27% | Underfitting (2 features only) |

### Best Model: SVM on Adjacency Matrix

```
              precision  recall  f1-score  support
     Afraid       0.64    0.88      0.74       8
      Happy       1.00    1.00      1.00       3
    Neutral       0.75    0.43      0.55       7

   accuracy                         0.72      18
```

**Normalized Confusion Matrix:**

|          | Pred: Afraid | Pred: Neutral | Pred: Happy |
|----------|:---:|:---:|:---:|
| **Afraid**  | 87.5% | 12.5% | 0% |
| **Neutral** | 57.1% | 42.9% | 0% |
| **Happy**   | 0%    | 0%    | 100% |

- 5-fold Cross-Validation Accuracy: **63.3%**

---

## Project Structure

```
.
├── Graph_semanti.ipynb          # Main analysis notebook
└── README.md                    # This file

Data files (download separately from Google Drive):
├── sumana_wl1.xlsx              # Raw fNIRS wavelength 1 data
├── sumana_wl2.xlsx              # Raw fNIRS wavelength 2 data
├── Afraid_filtered_merged_50.xlsx   # Preprocessed afraid trials
├── Neutral_filtered_merged_50.xlsx  # Preprocessed neutral trials
├── Happy_filtered_merged_50.xlsx    # Preprocessed happy trials
├── Afraid_correlation_merged.xlsx   # 14×14 correlation matrices (afraid)
├── Neutral_correlation_merged.xlsx  # 14×14 correlation matrices (neutral)
├── Happy_correlation_merged.xlsx    # 14×14 correlation matrices (happy)
├── afraid_gnn (with new data).xlsx  # Node features for GCN (afraid)
├── neutral_gnn (with new data).xlsx # Node features for GCN (neutral)
└── happy_gnn(new data).xlsx         # Node features for GCN (happy)
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for GCN training)

### Setup

```bash
# Clone the repository
git clone https://github.com/divyajyoti9431/Decoding-Emotions-through-FNIRS.git
cd Decoding-Emotions-through-FNIRS

# Install core dependencies
pip install numpy pandas scipy openpyxl xlsxwriter
pip install scikit-learn seaborn matplotlib networkx
pip install torch torchvision torchaudio
pip install torch-geometric
pip install tensorflow>=2.15
```

### Google Colab (Recommended)

The notebook is designed to run on **Google Colab** with Google Drive mounting:

```python
from google.colab import drive
drive.mount('/content/drive')
```

All data paths in the notebook use `/content/` which maps to Colab's working directory.

---

## Usage

### Running the Full Pipeline

1. **Download the dataset** from the [Google Drive link](https://drive.google.com/drive/folders/14CI_rNmmbImserk3p-E0HJLLWaYV4Qg6) and upload to your Colab environment or `/content/` directory.

2. **Open the notebook** `Graph_semanti.ipynb` in Jupyter or Google Colab.

3. **Run cells sequentially** — the pipeline is organized as:

   | Step | Cell(s) | Description |
   |------|---------|-------------|
   | 1 | Cell 2 | Windowing: segment raw data into 360-sample windows |
   | 2 | Cell 3 | Normalization: min-max scale each trial |
   | 3 | Cell 4 | HbO−HbR difference |
   | 4 | Cell 5 | Elliptic bandpass filtering |
   | 5 | Cell 6 | Pearson correlation matrix |
   | 6 | Cell 7 | Degree matrix computation |
   | 7 | Cell 8 | Graph generation + binary adjacency |
   | 8 | Cell 10 | Centrality feature extraction |
   | 9 | Cells 13–14 | SVM classification (adjacency) |
   | 10 | Cells 16–17 | SVM classification (centrality) |
   | 11 | Cells 24–33 | GCN variants (PyTorch Geometric) |
   | 12 | Cells 46–48 | CBAM attention module (TensorFlow) |

### Running Only the Best Model (SVM)

```python
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load binary adjacency matrices
afraid_df  = pd.read_excel('binary_matrices_afraid.xlsx',  sheet_name=None)
neutral_df = pd.read_excel('binary_matrices_neutral.xlsx', sheet_name=None)
happy_df   = pd.read_excel('binary_matrices_happy.xlsx',   sheet_name=None)

# Flatten adjacency matrices into feature vectors
def load_class(data_dict, label):
    features, labels = [], []
    for sheet in data_dict.values():
        features.append(sheet.values.flatten())
        labels.append(label)
    return features, labels

X, y = [], []
for data, lbl in [(afraid_df, 0), (neutral_df, 1), (happy_df, 2)]:
    f, l = load_class(data, lbl)
    X.extend(f); y.extend(l)

X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

clf = SVC(kernel='rbf', C=1, gamma='scale')
clf.fit(X_train, y_train)
print(f"Accuracy: {accuracy_score(y_test, clf.predict(X_test)):.3f}")
print(classification_report(y_test, clf.predict(X_test), target_names=['Afraid','Neutral','Happy']))
```

---

## Key Findings

1. **Graph topology outperforms raw signals.** SVM on binary adjacency matrices (72.2%) significantly outperforms GCN on raw 14-channel signals (53.93%), suggesting that the **connectivity pattern** between brain regions is more discriminative than the signal amplitudes themselves.

2. **SVM beats GCN on small datasets.** With only ~50 trials per class, the classical SVM generalizes better than GCNs, which tend to overfit. GCNs are more competitive when trained on the full raw data (~18,000 samples per class).

3. **Happy emotion is most distinct.** Across all models, the "Happy" class consistently achieves the highest recall and precision, indicating that positive emotional states produce a uniquely identifiable prefrontal connectivity pattern.

4. **Max Pooling improves GCN performance.** Adding a MaxPool1d layer after graph convolution raised GCN accuracy from ~27% to 63.16%, acting as a regularizer and providing translation invariance across node features.

5. **Activation function matters.** Among GCN activations tested, Mish and Swish (smooth, non-monotonic) showed more stable training dynamics than ReLU on this small graph dataset.

---

## Future Work

- [ ] Include more subjects to increase dataset size and improve GCN generalization
- [ ] Explore dynamic graph construction (time-varying connectivity)
- [ ] Apply Graph Attention Networks (GAT) to learn edge weights
- [ ] Investigate multi-scale temporal features using temporal graph networks
- [ ] Validate on public fNIRS emotion benchmarks (e.g., DEAP, MAHNOB-HCI adapted)
- [ ] Real-time emotion recognition pipeline using sliding-window inference

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.21 | Numerical operations |
| pandas | ≥1.3 | Data I/O and manipulation |
| scipy | ≥1.7 | Signal filtering |
| scikit-learn | ≥1.0 | SVM, evaluation metrics |
| networkx | ≥2.6 | Graph construction |
| matplotlib | ≥3.4 | Visualization |
| seaborn | ≥0.11 | Confusion matrix heatmaps |
| torch | ≥2.0 | GCN training |
| torch-geometric | ≥2.5 | GCNConv layers |
| tensorflow | ≥2.15 | CBAM attention module |
| openpyxl / xlsxwriter | latest | Excel I/O |

---

## License

This project is released under the [MIT License](LICENSE).
