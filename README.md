# DL-Pipelines-for-Medical-Imaging-Sensor-Stream-Analytics

> Two end-to-end machine learning pipelines: a **CNN-based diabetic retinopathy classifier** built on the 88 GB Kaggle Diabetic Retinopathy dataset, and a **K-Means temporal clustering analysis** of UCI hourly air-quality sensor data with formal drift detection.

---
## Overview

This repository contains two production-style data pipelines built to demonstrate the full machine-learning workflow — from raw data ingestion through preprocessing, modelling, evaluation, and result interpretation.

| Question | Domain | Task | Technique |
| --- | --- | --- | --- |
| **Q1** | Medical imaging | 5-class image classification | Transfer learning (EfficientNetB0, ResNet50) with two-phase fine-tuning |
| **Q2** | Environmental sensor data | Temporal clustering & drift analysis | K-Means with multi-metric model selection and KL/Jensen–Shannon drift quantification |

---
## Tech Stack

**Languages & frameworks**
Python 3.10, TensorFlow / Keras 2.15, scikit-learn, OpenCV, pandas, NumPy, SciPy, Matplotlib, Seaborn, Jupyter.

**Notable techniques**
Transfer learning, two-phase fine-tuning, custom Keras callbacks, class-imbalance handling via per-class resampling, Hungarian-algorithm cluster matching, Kullback–Leibler & Jensen–Shannon divergence, PCA visualisation, Ben Graham contrast enhancement, IQR winsorisation, stratified sampling, time-series forward-fill imputation.

---


## 1 — Diabetic Retinopathy Classification
A 5-class CNN classifier (No DR / Mild / Moderate / Severe / Proliferative) trained on retinal fundus photographs from the [Kaggle Diabetic Retinopathy Detection](https://www.kaggle.com/competitions/diabetic-retinopathy-detection) dataset.

### Pipeline

1. **Data ingestion.** Multi-part 7z archive extraction (`train.zip.001`–`.005`) to `/tmp` with disk-quota-aware handling, label-to-file matching from the metadata CSV, sanity checks.
2. **EDA.** Class distribution, sample-image visual inspection, image-dimension statistics, per-channel colour analysis, eye-side distribution.
3. **Preprocessing pipeline.** Black-border cropping via contour detection, resize to 224 × 224, Ben Graham contrast normalisation (Gaussian-blur subtraction + offset) — chosen because it was developed specifically for and validated on this dataset.
4. **Stratified sampling.** 3,500 images selected proportionally per class with a floor of 150 to protect rare classes; 70 / 15 / 15 stratified train / val / test split.
5. **Augmentation.** Horizontal flip, ±10° rotation, ±5% shift, ±5% zoom (training set only). Excluded: vertical flip, shear, colour jitter, cutout — each for documented reasons.
6. **Modelling.** Identical custom head (`GAP → BatchNorm → Dense(256) → Dropout(0.5) → Dense(128) → Dropout(0.3) → Dense(5, softmax)`) on top of frozen EfficientNetB0 and ResNet50 backbones.
7. **Two-phase training.** Phase 1: head-only at LR = 5×10⁻⁴. Phase 2: unfreeze top backbone layers and fine-tune at LR = 1×10⁻⁵ to prevent catastrophic forgetting of ImageNet features.
8. **Class-imbalance handling.** Per-class resampling targets (Class 0 downsampled to 1,400; rare Classes 3 & 4 upsampled to 350) — chosen after iterating through class weights and full equalisation.
9. **Model selection.** Custom `MacroF1Checkpoint` Keras callback that monitors validation macro-F1 each epoch, saves the best weights, and early-stops on plateau — replacing default `val_loss`-based selection which collapses on imbalanced data.

### Results

| Metric | EfficientNetB0 (Optimised) | Improvement vs Baseline |
| --- | --- | --- |
| Accuracy | 0.6581 | — |
| Weighted F1 | 0.6533 | — |
| **Macro F1** | **0.4260** | **+28%** |
| **Quadratic Cohen's Kappa** | **0.5724** | **+36%** |

The accuracy trade-off (lower than a naive "predict No DR" classifier) is intentional and clinically appropriate: the optimised model actually discriminates between all five severity grades rather than collapsing onto the majority class. Quadratic kappa above 0.40 indicates moderate ordinal agreement (Landis & Koch, 1977), within range of academic baselines on this dataset.

---

## 2 — Air Quality Clustering & Drift Analysis

K-Means clustering of 9,357 hourly air-quality sensor recordings from an Italian city (UCI ML Repository, March 2004 – February 2005), with quantitative drift analysis between spring/summer and autumn/winter periods.

### Pipeline

1. **Ingestion.** European-formatted CSV (semicolon separator, comma decimal) parsed with `pd.read_csv(..., sep=';', decimal=',')`.
2. **Cleaning.** Two trailing empty columns dropped, full-NaN rows removed, `Date` + `Time` parsed into a `DateTimeIndex`. Domain-specific missing-value encoding (`-200`) converted to `NaN`; `NMHC(GT)` dropped (~90% missing). Remaining gaps (4–18% per column) filled via forward-fill (exploits temporal autocorrelation) with median fallback.
3. **EDA.** Statistical summary with range and IQR, distribution histograms, correlation heatmap, daily resampled time series with 30-day rolling means, hourly pollution patterns.
4. **Outlier treatment.** IQR-based **winsorisation** (cap at Q1 − 1.5·IQR and Q3 + 1.5·IQR) — preserves time-series continuity instead of dropping rows.
5. **Feature scaling.** `StandardScaler` — chosen over MinMax because it is more robust to remaining outliers.
6. **Period split.** Period 1: March – August 2004 (spring/summer). Period 2: September 2004 – March 2005 (autumn/winter).
7. **Optimal-K selection.** Three complementary metrics — elbow (inertia), silhouette score, Davies–Bouldin index — swept K = 2…10 on both periods. Same K used for both periods to enable direct centroid-pair comparison.
8. **K-Means fitting.** `n_init=20`, `max_iter=500`, `random_state=42` for each period independently.
9. **Cluster evaluation.** Silhouette, Calinski–Harabasz, Davies–Bouldin, inertia. Per-cluster feature profiles (in original scale) computed for human-interpretable cluster meanings.
10. **Drift analysis (three-pronged).**
    - **Centroid shifts.** Pairwise Euclidean distances between Period 1 and Period 2 centroids → optimal one-to-one matching via the **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`) — necessary because K-Means cluster IDs are arbitrary.
    - **Cluster membership distribution.** KL divergence between Period 1 and Period 2 membership proportions; symmetric Jensen–Shannon divergence reported alongside.
    - **Per-feature KL.** Histogram-based KL divergence per feature to identify which features (temperature, CO, NOx) drove the drift.
11. **PCA visualisation.** 2-component PCA fit on the full dataset (so both periods share the same projection), cluster scatter plots, centroid drift arrows between Hungarian-matched pairs.

### Results

| Component | Outcome |
| --- | --- |
| Optimal K | 4 (interpretable as clean / moderate / polluted / heavily polluted) |
| Highest-drift feature | Temperature (seasonal cycle, as expected sanity check) |
| Secondary drift drivers | CO, NOx — both elevated in winter (heating, temperature inversions) |
| Cluster-membership KL divergence | Confirms a measurable seasonal regime shift |

---
