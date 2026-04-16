"""
Audio Feature Comparison with Class-Balanced Weights.

Compares 5 feature configurations using 5-fold patient-level GroupKFold CV
with class-balanced sample weights (matching the chip model methodology):

  1. IMU only           (97 features)
  2. IMU + Top 10 audio (107 features)
  3. IMU + Top 20 audio (117 features)
  4. IMU + Top 30 audio (127 features)
  5. IMU + All audio    (163 features)

Top audio features are ranked by XGBoost feature importance from:
  results/audio_feature_importance.csv

Outputs printed to stdout. Update RESULTS_SUMMARY.md manually with results.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD DATASET
# ============================================================================

DATASET_PATH = '../../models/sleep_dataset_audio_enhanced_complete.pkl'
IMPORTANCE_CSV = '../../results/audio_feature_importance.csv'

with open(DATASET_PATH, 'rb') as f:
    dataset = pickle.load(f)

X            = dataset['X_all_features']
y            = dataset['y_3class']
patient_ids  = dataset['patient_ids']
feature_names = dataset['feature_names']

print(f"Dataset: {X.shape[0]:,} epochs, {X.shape[1]} features, {len(np.unique(patient_ids))} patients")

# ============================================================================
# BUILD FEATURE MASKS
# ============================================================================

# Keywords that identify audio features
AUDIO_KEYWORDS = [
    'mic', 'zcr', 'spectral', 'energy', 'breathing',
    'snoring', 'silence', 'activity', 'crest', 'peak'
]

non_audio_feats = [f for f in feature_names if not any(kw in f for kw in AUDIO_KEYWORDS)]
print(f"IMU-only features: {len(non_audio_feats)}")

# Load top audio features ranked by importance
fi = pd.read_csv(IMPORTANCE_CSV)
top_10_audio = fi[fi['is_audio'] == True].head(10)['feature'].tolist()
top_20_audio = fi[fi['is_audio'] == True].head(20)['feature'].tolist()
top_30_audio = fi[fi['is_audio'] == True].head(30)['feature'].tolist()

non_audio_mask = np.array([f in non_audio_feats for f in feature_names])
top_10_mask    = non_audio_mask | np.array([f in top_10_audio for f in feature_names])
top_20_mask    = non_audio_mask | np.array([f in top_20_audio for f in feature_names])
top_30_mask    = non_audio_mask | np.array([f in top_30_audio for f in feature_names])
all_mask       = np.ones(len(feature_names), dtype=bool)

# ============================================================================
# CROSS-VALIDATION
# ============================================================================

def run_cv(X_sub, y, patient_ids, label):
    """5-fold patient-level CV with class-balanced sample weights."""
    gkf = GroupKFold(n_splits=5)
    kappas, accs, f1s = [], [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_sub, y, groups=patient_ids), 1):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_sub[train_idx])
        X_test  = scaler.transform(X_sub[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]

        classes = np.unique(y_train)
        cw = compute_class_weight('balanced', classes=classes, y=y_train)
        sw = np.array([dict(zip(classes, cw))[yi] for yi in y_train])

        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss',
            verbosity=0
        )
        model.fit(X_train, y_train, sample_weight=sw)
        y_pred = model.predict(X_test)

        kappas.append(cohen_kappa_score(y_test, y_pred))
        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average='macro'))

    k_mean, k_std = np.mean(kappas), np.std(kappas)
    a_mean = np.mean(accs) * 100
    f_mean = np.mean(f1s)

    print(f"{label:35s} | {X_sub.shape[1]:3d} feat | Kappa: {k_mean:.4f}±{k_std:.3f} | Acc: {a_mean:.2f}% | F1: {f_mean:.4f}")
    return k_mean


print("\n5-fold CV with class weights (balanced)\n")
print(f"{'Configuration':35s} | {'Feat':>4} | {'Kappa':>14} | {'Accuracy':>10} | {'F1':>6}")
print("-" * 85)

k_imu   = run_cv(X[:, non_audio_mask], y, patient_ids, "IMU only (97)")
k_top10 = run_cv(X[:, top_10_mask],    y, patient_ids, "IMU + Top 10 audio (107)")
k_top20 = run_cv(X[:, top_20_mask],    y, patient_ids, "IMU + Top 20 audio (117)")
k_top30 = run_cv(X[:, top_30_mask],    y, patient_ids, "IMU + Top 30 audio (127)")
k_all   = run_cv(X[:, all_mask],       y, patient_ids, "IMU + All audio (163)")

print(f"\nDeltas vs IMU-only baseline ({k_imu:.4f}):")
print(f"  + Top 10 audio: {k_top10 - k_imu:+.4f}")
print(f"  + Top 20 audio: {k_top20 - k_imu:+.4f}")
print(f"  + Top 30 audio: {k_top30 - k_imu:+.4f}")
print(f"  + All audio:    {k_all   - k_imu:+.4f}")
