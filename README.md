# Sleep Stage Classification from Wearable IMU and Audio Sensors

> **NTU CS Final Year Project (FYP)**  
> Automated 3-class sleep staging (Wake / NREM / REM) using a wrist/pillow-worn device combining a 6-axis IMU and PDM microphone, deployed on the Seeed XIAO nRF52840 Sense.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.x-green.svg)](https://xgboost.readthedocs.io/)
[![Arduino](https://img.shields.io/badge/Arduino-IDE_2.x-teal.svg)](https://www.arduino.cc/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Hardware Deployment](#hardware-deployment)
- [Results](#results)
- [Future Work](#future-work)

---

## Overview

Traditional sleep staging requires polysomnography (PSG) with EEG electrodes in a clinical setting — expensive, uncomfortable, and impractical for home monitoring. This project builds an automated sleep staging pipeline using only a small wearable device (PillowClip) containing a 6-axis IMU and microphone.

The pipeline covers:
- PSG–IMU data synchronisation across 44 patients
- Feature engineering: 98 IMU features + 20 audio features = **118 total**
- Patient-level cross-validated XGBoost classifier
- Real-time inference firmware for the nRF52840 microcontroller with RGB LED feedback

### Key Contributions

1. **Patient-specific normalisation** — online running-mean baseline per patient, updated each epoch, giving the single largest Kappa improvement
2. **Audio feature integration** — top-20 spectral and energy features from the PDM microphone evaluated alongside IMU; 118-feature model achieves Kappa = 0.3234
3. **Embedded deployment** — full 118-feature XGBoost model (20 trees) exported to a C header via `micromlgen` and flashed to the nRF52840 within the 1 MB flash budget
4. **RGB LED feedback** — Red = Wake, Green = NREM, Blue = REM, updated every 30-second epoch

---

## Key Results

**3-Class Classification (Wake / NREM / REM) — 5-fold patient-level CV:**

| Metric | Value |
|--------|-------|
| **Cohen's Kappa** | **0.3234 ± 0.0741** |
| **Accuracy** | **58.29 ± 5.00%** |
| **Macro F1-Score** | **0.5305 ± 0.0562** |

**Per-Class Performance (concatenated CV predictions):**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Wake | 0.611 | 0.666 | 0.637 | 2,285 |
| NREM | 0.669 | 0.578 | 0.620 | 3,166 |
| REM | 0.331 | 0.406 | 0.364 | 976 |
| **Macro avg** | **0.537** | **0.550** | **0.540** | 6,427 |

**Per-Fold Results:**

| Fold | Test Patients | n_test | Kappa | Accuracy | F1-Macro |
|------|--------------|--------|-------|----------|----------|
| 1 | 9 | 1,260 | 0.2710 | 56.98% | 0.4643 |
| 2 | 9 | 1,315 | 0.2603 | 52.62% | 0.4990 |
| 3 | 8 | 1,245 | 0.2789 | 54.86% | 0.5110 |
| 4 | 9 | 1,302 | 0.3884 | 64.06% | 0.5857 |
| 5 | 9 | 1,305 | 0.4185 | 62.91% | 0.5925 |
| **Mean** | | | **0.3234** | **58.29%** | **0.5305** |
| **Std** | | | **0.0741** | **5.00%** | **0.0562** |

**Hardware:**

| Metric | Value |
|--------|-------|
| Inference time | ~12 ms per epoch |
| Flash usage | within 1 MB budget |
| Model | XGBoost, 20 trees, 118 features |
| Platform | Seeed XIAO nRF52840 Sense |

---

## Dataset

### Sources

- **PSG data:** Expert-annotated polysomnography recordings
  - 44 patients, 6,427 epochs (30 s each)
  - 5 stages: Wake, N1, N2, N3, REM — consolidated to 3 classes

- **PillowClip sensor:** Wearable device worn on or near a pillow
  - 6-axis IMU: 3-axis accelerometer + 3-axis gyroscope at ~50 Hz (aggregated to 1 Hz per epoch)
  - PDM microphone: 16 kHz, processed as 256-sample windows

### Synchronisation

| Method | Patients |
|--------|----------|
| Timestamp-based (absolute clock match) | 31 |
| Sequential alignment (no start-time metadata) | 13 |

### Class Distribution

| Stage (5-class) | Count | % | 3-class mapping |
|-----------------|-------|---|-----------------|
| Wake | 2,285 | 35.6% | Wake |
| N1 | 718 | 11.2% | NREM |
| N2 | 1,242 | 19.3% | NREM |
| N3 | 1,206 | 18.8% | NREM |
| REM | 976 | 15.2% | REM |

---

## Methodology

### Pipeline

```
PSG annotations  ─┐
                   ├─ Synchronisation ─► Feature extraction (118 feat) ─► XGBoost CV ─► Predictions
PillowClip IMU  ──┘                          (98 IMU + 20 audio)           5-fold GroupKFold
PillowClip mic  ──┘
```

### Feature Engineering (118 features)

**IMU features (98):**

| Category | Count | Examples |
|----------|-------|---------|
| Raw axis statistics | 36 | `ax_mean`, `gz_std`, `tempC_max` |
| Movement magnitude | 21 | `acc_mag`, `gyro_mag`, `total_movement` |
| Patient normalisation | 12 | `patient_acc_mean`, `relative_activity`, `patient_wake_ratio` |
| Temporal / rolling | 6 | `acc_mag_rolling_mean_5`, `movement_diff` |
| Orientation | 7 | `ax_dominance`, `tilt_estimate` |
| Derived / variability | 17 | `stillness_duration`, `movement_bout_count` |
| Circadian | 4 | `time_since_start_min`, `sleep_cycle_sin/cos` |

**Audio features (top 20 by importance):**

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | spectral_centroid_normalized | 0.00879 |
| 2 | crest_variability | 0.00843 |
| 3 | spectral_stability | 0.00823 |
| 4 | energy_rolling_std | 0.00794 |
| 5 | activity_variability | 0.00725 |
| … | … | … |

Audio features are computed from a single 256-sample PDM window per epoch (streaming accumulator, no 30-second buffer). All spectral features are power-ratio quantities so FFT scaling cancels.

### Model

```python
XGBClassifier(
    n_estimators=20,       # reduced from 100 to fit nRF52840 1 MB flash
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softmax',
    random_state=42
)
# Class weights: compute_class_weight('balanced') applied as sample_weight per fold
```

### Evaluation

- **5-fold patient-level GroupKFold** — no patient appears in both train and test sets
- **Primary metric:** Cohen's Kappa (accounts for chance agreement)
- **Secondary:** Accuracy, Macro F1, per-class precision/recall/F1, confusion matrix

---

## Repository Structure

```
Sleep_Stage_Classifier_Clean/
├── .gitignore
├── README.md
├── requirements.txt
├── generate_figures.py              # Produces all report figures → docs/images/
│
├── src/
│   ├── preprocessing/
│   │   └── synchronize_psg_imu.py  # PSG–IMU alignment
│   ├── feature_engineering/
│   │   └── enhanced_features.py    # 98 IMU feature extraction
│   ├── modeling/
│   │   ├── train_xgboost.py        # IMU-only model training
│   │   ├── train_audio_model.py    # 118-feature audio+IMU training
│   │   ├── xgboost_audio_model.json
│   │   └── xgboost_chip_model.json
│   └── evaluation/
│       ├── feature_importance.py
│       └── audio_comparison_weighted.py
│
├── results/
│   ├── audio_feature_importance.csv         # Per-feature importance + is_audio flag
│   └── audio_model_117feat_results.txt      # Full CV results (118 features)
│
├── docs/
│   ├── appendix_source_code.tex             # LaTeX appendix for FYP report
│   └── images/                              # Generated report figures
│
├── hardware/
│   ├── firmware/
│   │   └── SleepClassifier_Audio_XGBoost/
│   │       ├── SleepClassifier_Audio_XGBoost.ino
│   │       ├── sleep_types.h
│   │       └── sleep_model_audio/
│   │           ├── features_audio.h         # FEAT_* index defines
│   │           ├── scaler_audio.h           # StandardScaler params
│   │           └── xgboost_audio.h          # Decision trees (micromlgen)
│   └── model_export/
│       ├── export_audio_to_c.py             # Model → C header via micromlgen
│       └── export_to_c.py
│
└── models/                                  # .pkl files (gitignored — too large)
```

---

## Installation

**Requirements:** Python 3.12, macOS/Linux. Arduino IDE 2.x + Seeed nRF52 board package for firmware.

```bash
git clone https://github.com/<your-username>/sleep-stage-classifier.git
cd sleep-stage-classifier

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## Usage

### 1. Synchronise PSG and IMU data

```bash
python src/preprocessing/synchronize_psg_imu.py \
    --psg_dir data/raw/PSG_Data \
    --imu_dir data/raw/PillowClip_Data \
    --output_dir data/processed/Synchronized_Data
```

### 2. Extract features and build dataset

```bash
python src/feature_engineering/enhanced_features.py \
    --input_dir data/processed/Synchronized_Data \
    --output data/processed/sleep_dataset_optimized.pkl
```

### 3. Train the 118-feature audio+IMU model

```bash
python src/modeling/train_audio_model.py
# Writes results to results/audio_model_117feat_results.txt
```

### 4. Regenerate report figures

```bash
python generate_figures.py
# Saves all figures to docs/images/
```

### 5. Export model to C headers for firmware

```bash
python hardware/model_export/export_audio_to_c.py
# Writes features_audio.h, scaler_audio.h, xgboost_audio.h
```

---

## Hardware Deployment

**Target:** Seeed XIAO nRF52840 Sense  
- ARM Cortex-M4F @ 64 MHz with hardware FPU  
- 1 MB flash, 256 KB RAM  
- LSM6DS3 6-axis IMU  
- MP34DT06JTR PDM microphone (16 kHz)  
- RGB LED (active-LOW: LEDR / LEDG / LEDB)

**Firmware** (`hardware/firmware/SleepClassifier_Audio_XGBoost/`):
- Samples IMU at 1 Hz; accumulates one 256-sample PDM window per epoch
- Extracts all 118 features in under 12 ms
- Classifies with XGBoost (20 trees) via `Eloquent::ML::Port::XGBClassifier`
- Updates RGB LED: **Red = Wake, Green = NREM, Blue = REM**
- Patient baseline updated online (running mean, O(1) per epoch)

**To flash:**
1. Open `SleepClassifier_Audio_XGBoost.ino` in Arduino IDE 2.x
2. Board: *Seeed nRF52 Boards* → *Seeed XIAO nRF52840 Sense*
3. Upload

---

## Results

### Confusion Matrix

```
                Pred:Wake  Pred:NREM  Pred:REM
  True:Wake      1,521       552       212     (66.6% recall)
  True:NREM        745     1,831       590     (57.8% recall)
  True:REM         225       355       396     (40.6% recall)
```

### Feature Category Importance

| Category | Features | Total Importance |
|----------|----------|-----------------|
| Raw IMU axes | 36 | ~30% |
| Patient normalisation | 12 | ~21% |
| Audio | 20 | ~12% |
| Movement magnitude | 21 | ~11% |
| Derived / statistical | 17 | ~11% |
| Circadian / temporal | 4 | ~9% |
| Temporal / rolling | 6 | ~4% |

---

## Future Work

- **Larger dataset** — 100+ patients for better generalisation
- **Hyperparameter search** — Bayesian optimisation on n_estimators, max_depth
- **Better microphone placement** — near-mouth positioning for breathing rate features
- **Multi-night tracking** — longitudinal sleep quality trends
- **Online personalisation** — continual learning as the patient baseline converges

---

## Contact

**Author:** F Syed Abu Thahir  
**Institution:** Nanyang Technological University (NTU)  
**Course:** Computer Science Final Year Project 2024/2025  

---

*All Python code developed under Python 3.12, macOS Sequoia. Arduino firmware compiled with Arduino IDE 2.x and Seeed nRF52 board package.*
