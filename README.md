# Enhanced HACT-Net: Visualized & Interpretable Hierarchical Graph Neural Networks for Breast Cancer Subtype Classification

An end-to-end computational pipeline for breast cancer subtype classification from Whole-Slide Images (WSIs) and Regions of Interest (ROIs) on the **BRACS dataset (7 subtypes)**.

This repository modernizes and substantially extends the original 2021 HACT-Net implementation. While the legacy framework restricted inference outputs to raw terminal text and basic metric tables, this enhanced version introduces a **comprehensive post-processing and visual analytics suite**, transforming model predictions into rich, publication-ready visual diagnostics and clinically aligned, multi-scale graph explanation maps.

---

## Repository Structure

```text
hact-net-reborn/
├── config/                 # YAML configuration files for GNN models (GIN/PNA)
├── core/                   # Core Python scripts (Preprocessing, Training, Explainability, Inference)
│   ├── generate_hact_graphs.py
│   ├── train.py
│   ├── predict_tumor_type.py
│   ├── nuclei_extraction.py
│   ├── superpixel.py
│   ├── slic.py
│   ├── pgexplainer.py
│   ├── lrp_gnn_explainer.py
│   ├── hact_model.py
│   └── ...
├── scripts/
│   └── run_me_hact_net.sh  # Master execution script orchestrating the end-to-end pipeline
├── environment_hactnet.yml # Conda environment dependencies
├── .gitignore              # Ignores large datasets, WSI files, and checkpoints
└── README.md

```

---

## Key Upgrades & Features over the Original Implementation

* **Modernized Toolchain:** Fully patched for PyTorch 1.12+/2.x, DGL 1.0+, and dynamic CUDA memory configurations.
* **Visual Output Suite:** Replaced raw console metric tables with automated visualization exports (Prediction Heatmaps, Confidence Boxplots & Scatterplots, Class Countplots).
* **Multi-Scale Graph Representation:**
* **Cell Graph (CG):** HoVer-Net nuclei extraction + ResNet34 feature encoding + $k$-NN connectivity.
* **Tissue Graph (TG):** Color-merged SLIC superpixel segmentation + Region Adjacency Graphs (RAG).
* **Assignment Matrix:** Explicit cell-to-tissue spatial assignment matrices.


* **Advanced Interpretability:** GraphLRP, PGExplainer, GradCAM/GradCAM++, GNNExplainer, and nuclei/tissue overlay visualizers.
* **Fault-Tolerant Training:** Automatic identification and clean-up of corrupt/NaN `.bin` and `.h5` graph files.

---

## Core Purpose & Subtype Classification

The primary objective of the framework is to classify histological images and WSIs into one of the **7 distinct BRACS diagnostic categories**:

1. **N** — Normal Tissue (Subtype 0)
2. **PB** — Pathological Benign (Subtype 1)
3. **UDH** — Usual Ductal Hyperplasia (Subtype 2)
4. **FEA** — Flat Epithelial Atypical (Subtype 3)
5. **ADH** — Atypical Ductal Hyperplasia (Subtype 4)
6. **DCIS** — Ductal Carcinoma In Situ (Subtype 5)
7. **IC** — Invasive Carcinoma (Subtype 6)

---

## Multi-Scale Hierarchical Architecture & Pipeline

Each tissue region is represented using a multi-level entity-graph to capture both fine-scale cellular detail and coarse tissue architecture:

```text
WSI / ROI Image ──► Nuclei Extraction (HoVer-Net) ──► Cell Graph (CG) ──┐
                └──► Tissue Segmentation (SLIC)   ──► Tissue Graph (TG) ─┼──► Assignment Matrix (A_ij) ──► HACT-Net ──► MLP Classifier ──► 7 Subtypes

```

### 1. Multi-Level Entity-Graph Representation

* **Cell Graph (CG):** Nodes represent individual cell nuclei with morphological, textural, and deep descriptors (ResNet34). Edges are constructed using $k$-nearest neighbors ($k$-NN) subject to spatial distance thresholds.
* **Tissue Graph (TG):** Nodes represent merged tissue regions (superpixels) with edges capturing spatial adjacency (Region Adjacency Graphs - RAG).
* **Binary Assignment Matrix ($A_{ij}$):** Explicitly maps each nucleus to its parent tissue region, enabling upward pooling of cellular information into the surrounding tissue context.

### 2. Dual-Stream GNN Mechanics

HACT-Net employs two graph neural network streams:

* **Cell Stream:** Processes Cell Graphs using PNA or GIN layers.
* **Tissue Stream:** Processes tissue graphs whose initial features are concatenated with pooled cell embeddings using the assignment matrix ($A_{ij}$).
* **Graph Embedding & Classification:** A global graph-level embedding derived from the tissue stream is passed to a Multi-Layer Perceptron (MLP) classifier for final subtype prediction.

---

## Performance & Comparative Results

All modules—from graph generation to inference and visualization—were implemented through custom Python scripts to ensure reproducibility.

Models were evaluated on the BRACS dataset split at the WSI level (**70% Training, 15% Validation, 15% Testing**) using the Adam optimizer ($\text{lr}=10^{-4}$, $\text{weight decay}=10^{-5}$) with weighted cross-entropy loss and early stopping based on validation loss.

### Performance Comparison on BRACS

| Model | Accuracy (%) | Weighted F1 (%) | Cross-Entropy Loss |
| --- | --- | --- | --- |
| **HACT-Net (Ours)** | **85.2 ± 0.8** | **61.5 ± 0.9** | **0.89** |
| Concat-GNN | 83.6 ± 1.2 | 59.3 ± 1.1 | 0.95 |
| CG-GNN | 81.7 ± 1.4 | 58.4 ± 1.3 | 1.01 |
| TG-GNN | 80.9 ± 1.6 | 56.6 ± 1.3 | 1.07 |
| ResNet50 (Baseline) | 78.3 ± 1.9 | 55.2 ± 1.7 | 1.13 |

> **Per-Class Observations:** HACT-Net achieves the greatest performance gains in **Invasive Carcinoma (IC)** and **Ductal Carcinoma In Situ (DCIS)**, where joint integration of cellular details and tissue context is diagnostically critical.

---

## Explainability & Clinical Interpretability

To establish clinician trust, the pipeline integrates several post-hoc graph explainers to clarify model predictions:

* **GraphLRP (Layer-wise Relevance Propagation):** Achieves the most spatially precise and clinically relevant maps, reliably highlighting clusters of malignant nuclei that correspond with known subtype features.
* **GradCAM / GradCAM++ & GNNExplainer:** Comparative analysis reveals that gradient-based activation mapping and subgraph optimization mainly highlight low-to-moderate nuclear expression, occasionally missing key features of aggressive tumors.

---

## Representative Case Study: `BRACS_1284_IC_6`

Starting from the raw Whole-Slide Image, nuclei are detected and assembled into a cell graph while tissue compartments form the tissue graph. Each nucleus is mapped via $A_{ij}$ for hierarchical pooling.

* **Diagnosis:** HACT-Net predicts **Invasive Carcinoma (IC)** with high confidence.
* **LRP Node-Level Explanation:** Node colors represent predicted cancer expression (red = low, yellow = high). Dense clusters of yellow nodes highlight nuclei with strong malignant features, matching the invasive carcinoma diagnosis.
* **Comparison:** While LRP accurately pinpoints aggressive malignant cells, GradCAM, GradCAM++, and GNNExplainer mostly highlight red/orange nodes, demonstrating the superior diagnostic fidelity of GraphLRP.

---

## Discussion

* **Clinical Alignment:** The hierarchical entity-centric design of HACT-Net directly parallels the diagnostic reasoning of pathologists, combining microscopic cellular detail with architectural tissue patterns.
* **Strengths:** Multi-scale integration, robust interpretability, and resilience to real-world variability.
* **Limitations:** Dependency on segmentation quality, computational overhead, and domain adaptation requirements.

---

## Acknowledgments & Citation

This project builds upon, refactors, and extends the open-source **HACT-Net** framework developed by HistoCartography:

* **Original Repository:** [histocartography/hact-net](https://github.com/histocartography/hact-net)

```
