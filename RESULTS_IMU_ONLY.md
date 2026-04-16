# IMU-Only Model Results

**Model:** XGBoost (100 trees, max_depth=6, lr=0.1)
**Dataset:** `sleep_dataset_optimized.pkl` — 97 IMU-only features (no audio)
**Evaluation:** 5-fold patient-level cross-validation
**Dataset size:** 44 patients, 6,427 epochs

---

## Best Model Performance

**3-Class Classification (Wake, NREM, REM):**

```
Overall Metrics:
  Cohen's Kappa:  0.2938 ± 0.0931
  Accuracy:       57.90% ± 6.07%
  Macro F1-Score: 0.5002 ± 0.0713
```

---

## Per-Class Performance

```
              precision    recall  f1-score   support

        Wake      0.569     0.706     0.630      2285
        NREM      0.647     0.597     0.621      3166
         REM      0.334     0.228     0.271       976

    accuracy                          0.580      6427
   macro avg      0.517     0.510     0.507      6427
weighted avg      0.572     0.580     0.571      6427
```

---

## Confusion Matrix

```
Actual    Predicted
         Wake   NREM    REM
Wake     1613    554    118     (70.6% correct)
NREM      949   1891    326     (59.7% correct)
REM       275    478    223     (22.8% correct)
```

---

## Detailed Cross-Validation Results

| Fold | Train Patients | Test Patients | Train Epochs | Test Epochs | Kappa  | Accuracy | F1-Macro |
|------|---------------|---------------|--------------|-------------|--------|----------|----------|
| 1    | 35            | 9             | 5,167        | 1,260       | 0.1783 | 51.83%   | 0.3966   |
| 2    | 35            | 9             | 5,112        | 1,315       | 0.2563 | 54.98%   | 0.4841   |
| 3    | 36            | 8             | 5,182        | 1,245       | 0.2278 | 52.29%   | 0.4599   |
| 4    | 35            | 9             | 5,125        | 1,302       | 0.3902 | 64.75%   | 0.5767   |
| 5    | 35            | 9             | 5,122        | 1,305       | 0.4161 | 65.67%   | 0.5835   |
| **Mean** | — | — | — | — | **0.2938** | **57.90%** | **0.5002** |
| **Std**  | — | — | — | — | **0.0931** | **6.07%**  | **0.0713** |

**Observations:**
- Fold 5 best performance (Kappa 0.4161)
- Fold 1 worst performance (Kappa 0.1783)
- High variance (±0.093) indicates strong patient-dependent performance
- REM recall low (22.8%) — muscle atonia makes REM indistinguishable from NREM via IMU alone
