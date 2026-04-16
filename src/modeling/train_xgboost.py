"""
Step 1: Retrain XGBoost using only chip-computable features.

ONLY drops features that genuinely require FFT / frequency-domain analysis.
Time-domain audio features (rms_energy, peak_to_peak, crest_factor,
silence_ratio, activity_ratio, total_energy) are trivially computable on
the nRF52840 using counters already in the mic ISR — so we KEEP them.

FFT-required (dropped):
  spectral_centroid, spectral_bandwidth, spectral_rolloff, spectral_flatness
  snoring_likelihood, snoring_power, snoring_ratio  (frequency-band based)
  freq_*  (any explicit frequency-domain features)

Time-domain kept (no FFT needed):
  mic_rms        → already in ISR (g_mic_ssq, g_mic_count)
  zcr            → already in ISR (g_mic_zcr)
  rms_energy     → same accumulator as mic_rms
  total_energy   → g_mic_ssq directly
  peak_to_peak   → add max/min tracking to ISR (2 variables)
  crest_factor   → peak / rms, computed at epoch boundary
  silence_ratio  → add silence counter to ISR (samples below threshold)
  activity_ratio → 1 - silence_ratio

Outputs:
  xgboost_chip_model.json      - XGBoost model
  scaler_chip.pkl              - StandardScaler
  feature_names_chip.pkl       - Ordered feature list
  chip_feature_drop_report.txt - What was dropped and why
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD DATASET
# ============================================================================

with open('../../data/sleep_dataset_complete_v4.pkl', 'rb') as f:
    data = pickle.load(f)

X_all         = data['X_all_features']
y_raw         = data['y']
patient_ids   = data['patient_ids']
all_names     = list(data['all_feature_names'])
elapsed_times = data['elapsed_times']   # seconds since recording start per epoch

print(f"Full dataset: {X_all.shape[0]:,} epochs, {X_all.shape[1]} features")

# ============================================================================
# ADD SIN/COS SLEEP CYCLE FEATURES (Strategy 2)
# ============================================================================
# Sleep cycles are ~90 minutes. Encoding elapsed time as sin/cos lets XGBoost
# learn "REM is more likely at 90-min intervals" without assuming linearity.
# These are trivially computable on the nRF52840 (math.h sin/cos).

SLEEP_CYCLE_SEC = 90 * 60   # 5400 seconds

cycle_phase     = (elapsed_times / SLEEP_CYCLE_SEC) * 2.0 * np.pi
sleep_cycle_sin = np.sin(cycle_phase)
sleep_cycle_cos = np.cos(cycle_phase)

X_all     = np.column_stack([X_all, sleep_cycle_sin, sleep_cycle_cos])
all_names = all_names + ['sleep_cycle_sin', 'sleep_cycle_cos']

print(f"  + 2 sin/cos sleep cycle features → {X_all.shape[1]} total")

# ============================================================================
# IDENTIFY CHIP-COMPUTABLE FEATURES
# ============================================================================

# Only drop features that TRULY require FFT or frequency-domain analysis.
# Time-domain audio features (energy, peak-to-peak, silence ratio, etc.)
# are trivially computable from the raw sample stream and are KEPT.

# These require FFT → DROP
FFT_REQUIRED_KEYWORDS = [
    'spectral_',     # spectral_centroid, spectral_bandwidth, spectral_rolloff, spectral_flatness
    'snoring_',      # snoring_likelihood, snoring_power, snoring_ratio (frequency-band based)
    'freq_',         # any explicit frequency-domain features
]

# These look "audio-ish" but are pure time-domain → KEEP
TIME_DOMAIN_SAFE = [
    'mic_rms',       # RMS amplitude — already in ISR
    'zcr',           # zero-crossing rate — already in ISR
    'rms_energy',    # = mic_rms squared × count, trivial
    'total_energy',  # sum of squared samples, already have g_mic_ssq
    'peak_to_peak',  # max(samples) - min(samples), add 2 vars to ISR
    'crest_factor',  # peak / rms, trivial
    'silence_ratio', # fraction of samples below threshold, add 1 counter to ISR
    'activity_ratio',# 1 - silence_ratio
]

dropped = []
kept    = []

for name in all_names:
    # Check if any FFT keyword appears as a substring
    is_fft = any(kw in name for kw in FFT_REQUIRED_KEYWORDS)
    # Check if this is a safe time-domain feature
    is_safe = any(kw in name for kw in TIME_DOMAIN_SAFE)

    if is_fft and not is_safe:
        dropped.append(name)
    else:
        kept.append(name)

print(f"\nDropped (require FFT): {len(dropped)} features")
for n in dropped:
    print(f"  - {n}")

print(f"\nKept (chip-computable): {len(kept)} features")
print("  (includes time-domain audio: rms_energy, peak_to_peak, crest_factor,")
print("   silence_ratio, activity_ratio, total_energy — all trivially computed)")
print()

# Build reduced feature matrix
feature_idx = [all_names.index(n) for n in kept]
X = X_all[:, feature_idx]

print(f"Chip model input: {X.shape[1]} features")

# Save drop report
with open('chip_feature_drop_report.txt', 'w') as f:
    f.write(f"Full model features:  {len(all_names)}\n")
    f.write(f"Chip model features:  {len(kept)}\n")
    f.write(f"Dropped (FFT only):   {len(dropped)}\n\n")
    f.write("DROPPED — require FFT / frequency-domain analysis:\n")
    for n in dropped:
        f.write(f"  {n}\n")
    f.write("\nKEPT — chip-computable:\n")
    f.write("  [IMU features: all 30 — running stats at 50 Hz]\n")
    f.write("  [Audio time-domain: mic_rms, zcr, rms_energy, total_energy,\n")
    f.write("   peak_to_peak, crest_factor, silence_ratio, activity_ratio]\n")
    f.write("  [All 85 enhanced/critical features — derived from above]\n\n")
    f.write("FIRMWARE ISR ADDITIONS NEEDED for new time-domain features:\n")
    f.write("  peak_to_peak: track g_mic_max and g_mic_min (reset each second)\n")
    f.write("  crest_factor: peak / rms at epoch boundary\n")
    f.write("  silence_ratio: g_mic_silence_count (samples where abs(s) < THRESHOLD)\n")
    f.write("  activity_ratio: 1 - silence_ratio\n")
    f.write("  rms_energy / total_energy: g_mic_ssq already accumulated\n\n")
    f.write("Full feature list kept:\n")
    for n in kept:
        f.write(f"  {n}\n")

# ============================================================================
# 3-CLASS MAPPING
# ============================================================================

y = y_raw.copy()
y[(y == 1) | (y == 2) | (y == 3)] = 1   # NREM
y[y == 5] = 2                             # REM

print(f"Wake={np.sum(y==0):,}, NREM={np.sum(y==1):,}, REM={np.sum(y==2):,}\n")

# ============================================================================
# CROSS-VALIDATION (confirming accuracy loss from dropping spectral features)
# ============================================================================

gkf = GroupKFold(n_splits=5)
fold_results = []
all_y_true, all_y_pred = [], []

print("5-fold CV with chip-computable features only...")
print()

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=patient_ids), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    classes = np.unique(y_train)
    cw = compute_class_weight('balanced', classes=classes, y=y_train)
    sample_weights = np.array([dict(zip(classes, cw))[yi] for yi in y_train])

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss',
        verbosity=0
    )
    model.fit(X_train_sc, y_train, sample_weight=sample_weights)
    y_pred = model.predict(X_test_sc)

    kappa = cohen_kappa_score(y_test, y_pred)
    acc   = accuracy_score(y_test, y_pred)
    f1    = f1_score(y_test, y_pred, average='macro')

    fold_results.append({'fold': fold, 'kappa': kappa, 'accuracy': acc, 'f1': f1})
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

    print(f"Fold {fold}: Kappa={kappa:.4f}  Acc={acc:.4f}  F1={f1:.4f}")

df = pd.DataFrame(fold_results)
print()
print("=" * 60)
print("CHIP MODEL RESULTS (117 features, no spectral audio)")
print("=" * 60)
print(f"Mean Kappa:    {df['kappa'].mean():.4f} ± {df['kappa'].std():.4f}")
print(f"Mean Accuracy: {df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}")
print(f"Mean F1:       {df['f1'].mean():.4f} ± {df['f1'].std():.4f}")
print()
print("Per-class report:")
print(classification_report(all_y_true, all_y_pred,
                             target_names=['Wake', 'NREM', 'REM'], digits=3))
print()
print(f"Full model (138 features) kappa: ~0.3204")
print(f"Chip model ({len(kept):3d} features) kappa:  {df['kappa'].mean():.4f}")
delta = df['kappa'].mean() - 0.3204
print(f"Delta vs full model: {delta:+.4f}  ({'better' if delta > 0 else 'worse'})")

# ============================================================================
# TRAIN FINAL CHIP MODEL ON ALL DATA
# ============================================================================

print()
print("Training final chip model on all data...")

scaler_final = StandardScaler()
X_scaled = scaler_final.fit_transform(X)

classes = np.unique(y)
cw = compute_class_weight('balanced', classes=classes, y=y)
sample_weights = np.array([dict(zip(classes, cw))[yi] for yi in y])

final_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='mlogloss',
    verbosity=0
)
final_model.fit(X_scaled, y, sample_weight=sample_weights)

# ============================================================================
# SAVE ARTIFACTS
# ============================================================================

final_model.save_model('xgboost_chip_model.json')

with open('scaler_chip.pkl', 'wb') as f:
    pickle.dump(scaler_final, f)

with open('feature_names_chip.pkl', 'wb') as f:
    pickle.dump(kept, f)

print()
print("Saved:")
print("  xgboost_chip_model.json    - XGBoost model for chip")
print("  scaler_chip.pkl            - Scaler (117 features)")
print("  feature_names_chip.pkl     - Feature name list")
print("  chip_feature_drop_report.txt - Drop rationale")
print()
print("Next: run export_to_c.py to generate C code for the nRF52840.")
