"""
Train XGBoost on 97 IMU + Top 20 audio features (117 features total).
30 estimators used to fit within nRF52840 1 MB flash.

Reads:
  models/sleep_dataset_audio_enhanced_complete.pkl  — 163-feature dataset
  results/audio_feature_importance.csv              — ranked audio features

Methodology:
  - Non-audio features selected by excluding keywords:
      mic, zcr, spectral, energy, breathing, snoring, silence, activity, crest, peak
  - Top 20 audio features appended (by XGBoost importance rank)
  - 5-fold patient-level GroupKFold CV with class-balanced sample weights
  - Final model trained on all data

Outputs (written to this directory: src/modeling/):
  xgboost_audio_model.json    — XGBoost model (100 trees)
  scaler_audio.pkl            — StandardScaler fitted on all data
  feature_names_audio.pkl     — Ordered list of 117 feature names

Next step:
  python hardware/model_export/export_audio_to_c.py
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (cohen_kappa_score, accuracy_score,
                              f1_score, classification_report,
                              confusion_matrix)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT           = Path(__file__).resolve().parent.parent.parent
DATASET_PATH   = ROOT / 'src' / 'evaluation' / 'sleep_dataset_audio_enhanced_complete.pkl'
IMPORTANCE_CSV = ROOT / 'results' / 'audio_feature_importance.csv'
OUT_DIR        = Path(__file__).resolve().parent   # src/modeling/
RESULTS_DIR    = ROOT / 'results'
RESULTS_FILE   = RESULTS_DIR / 'audio_model_117feat_results.txt'

# ── Load dataset ──────────────────────────────────────────────────────────────

print("Loading dataset...")
with open(DATASET_PATH, 'rb') as f:
    dataset = pickle.load(f)

X            = dataset['X_all_features']
y            = dataset['y_3class']
patient_ids  = dataset['patient_ids']
feature_names = list(dataset['feature_names'])

print(f"  {X.shape[0]:,} epochs  |  {X.shape[1]} features  |  "
      f"{len(np.unique(patient_ids))} patients")
print(f"  Wake={np.sum(y==0):,}  NREM={np.sum(y==1):,}  REM={np.sum(y==2):,}")

# ── Build 117-feature set ─────────────────────────────────────────────────────

# Use the CSV's is_audio column as the authoritative source — avoids keyword
# substring false-positives (e.g. 'activity' matching 'relative_activity').
fi = pd.read_csv(IMPORTANCE_CSV)
is_audio_set = set(fi.loc[fi['is_audio'] == True, 'feature'])

non_audio = [f for f in feature_names if f not in is_audio_set]
top_20_audio = fi[fi['is_audio'] == True].head(20)['feature'].tolist()

non_audio_mask = np.array([f in non_audio     for f in feature_names])
top_20_mask    = np.array([f in non_audio or f in top_20_audio
                           for f in feature_names])

X_117       = X[:, top_20_mask]
feat_names  = [f for f, keep in zip(feature_names, top_20_mask) if keep]

n_imu   = int(non_audio_mask.sum())
n_audio = len(top_20_audio)
print(f"\nFeature set: {n_imu} IMU + {n_audio} audio = {X_117.shape[1]} total")
print(f"\nTop 20 audio features added:")
for i, name in enumerate(top_20_audio, 1):
    row = fi[fi['feature'] == name].iloc[0]
    print(f"  {i:2d}. {name:<40s}  importance={row['importance']:.5f}")

# ── 5-fold cross-validation ───────────────────────────────────────────────────

print("\n" + "="*60)
print("5-fold patient-level GroupKFold CV")
print("="*60)

gkf        = GroupKFold(n_splits=5)
fold_results = []
all_true, all_pred = [], []
unique_patients = np.unique(patient_ids)

for fold, (train_idx, test_idx) in enumerate(
        gkf.split(X_117, y, groups=patient_ids), 1):

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_117[train_idx])
    X_test  = scaler.transform(X_117[test_idx])
    y_train, y_test = y[train_idx], y[test_idx]

    train_patients = sorted(np.unique(patient_ids[train_idx]))
    test_patients  = sorted(np.unique(patient_ids[test_idx]))

    classes = np.unique(y_train)
    cw = compute_class_weight('balanced', classes=classes, y=y_train)
    sw = np.array([dict(zip(classes, cw))[yi] for yi in y_train])

    model = xgb.XGBClassifier(
        n_estimators=20,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss',
        verbosity=0
    )
    model.fit(X_train, y_train, sample_weight=sw)
    y_pred = model.predict(X_test)

    k  = cohen_kappa_score(y_test, y_pred)
    a  = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    fold_results.append({
        'fold': fold,
        'train_patients': train_patients,
        'test_patients':  test_patients,
        'n_train': len(y_train),
        'n_test':  len(y_test),
        'kappa': k, 'accuracy': a, 'f1': f1
    })
    all_true.extend(y_test)
    all_pred.extend(y_pred)
    print(f"  Fold {fold}: Kappa={k:.4f}  Acc={a*100:.2f}%  F1={f1:.4f}  "
          f"(test n={len(y_test):,}  patients={test_patients})")

df = pd.DataFrame(fold_results)
print()
print(f"  Mean Kappa : {df['kappa'].mean():.4f} ± {df['kappa'].std():.4f}")
print(f"  Mean Acc   : {df['accuracy'].mean()*100:.2f}%")
print(f"  Mean F1    : {df['f1'].mean():.4f}")
print()
clf_report = classification_report(all_true, all_pred,
                                   target_names=['Wake', 'NREM', 'REM'], digits=3)
print(clf_report)

cm = confusion_matrix(all_true, all_pred)
print("Confusion matrix (rows=true, cols=pred):")
print(f"             Wake   NREM    REM")
labels = ['Wake', 'NREM', 'REM']
for i, row_label in enumerate(labels):
    print(f"  {row_label:>4}   " + "  ".join(f"{cm[i,j]:5d}" for j in range(3)))

# ── Train final model on all data ─────────────────────────────────────────────

print("Training final model on all data...")

scaler_final = StandardScaler()
X_scaled     = scaler_final.fit_transform(X_117)

classes = np.unique(y)
cw = compute_class_weight('balanced', classes=classes, y=y)
sw = np.array([dict(zip(classes, cw))[yi] for yi in y])

final_model = xgb.XGBClassifier(
    n_estimators=20,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='mlogloss',
    verbosity=0
)
final_model.fit(X_scaled, y, sample_weight=sw)

# ── Save artifacts ────────────────────────────────────────────────────────────

model_path   = OUT_DIR / 'xgboost_audio_model.json'
model_pkl    = OUT_DIR / 'xgboost_audio_model.pkl'   # needed by m2cgen
scaler_path  = OUT_DIR / 'scaler_audio.pkl'
names_path   = OUT_DIR / 'feature_names_audio.pkl'

final_model.save_model(str(model_path))   # XGBoost native (for reference)

with open(model_pkl, 'wb') as f:
    pickle.dump(final_model, f)           # full sklearn object (for m2cgen)

with open(scaler_path, 'wb') as f:
    pickle.dump(scaler_final, f)

with open(names_path, 'wb') as f:
    pickle.dump(feat_names, f)

print()
print("Saved:")
print(f"  {model_path}")
print(f"  {model_pkl}")
print(f"  {scaler_path}")
print(f"  {names_path}")
print()
print("Next step:")
print("  python hardware/model_export/export_audio_to_c.py")

# ── Save results file ──────────────────────────────────────────────────────────

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
with open(RESULTS_FILE, 'w') as rf:
    rf.write("=" * 70 + "\n")
    rf.write("117-Feature Audio-Enhanced XGBoost Model — Full Results\n")
    rf.write("IMU (97) + Top 20 Audio Features  |  n_estimators=20  |  max_depth=6  |  lr=0.1\n")
    rf.write("5-fold patient-level GroupKFold CV with class-balanced sample weights\n")
    rf.write("=" * 70 + "\n\n")

    rf.write("DATASET\n")
    rf.write(f"  Total epochs   : {X.shape[0]:,}\n")
    rf.write(f"  Total features : {X_117.shape[1]}  ({n_imu} IMU + {n_audio} audio)\n")
    rf.write(f"  Patients       : {len(np.unique(patient_ids))}\n")
    rf.write(f"  Wake epochs    : {np.sum(y==0):,}\n")
    rf.write(f"  NREM epochs    : {np.sum(y==1):,}\n")
    rf.write(f"  REM epochs     : {np.sum(y==2):,}\n\n")

    rf.write("TOP 20 AUDIO FEATURES USED\n")
    fi = pd.read_csv(IMPORTANCE_CSV)
    for i, name in enumerate(top_20_audio, 1):
        row = fi[fi['feature'] == name].iloc[0]
        rf.write(f"  {i:2d}. {name:<45s}  importance={row['importance']:.5f}\n")
    rf.write("\n")

    rf.write("PER-FOLD RESULTS\n")
    rf.write(f"  {'Fold':>4}  {'Train Pts':>9}  {'Test Pts':>8}  {'n_train':>7}  {'n_test':>6}  "
             f"{'Kappa':>7}  {'Accuracy':>9}  {'F1-Macro':>8}\n")
    rf.write("  " + "-" * 68 + "\n")
    for r in fold_results:
        rf.write(f"  {r['fold']:>4}  "
                 f"{str(r['train_patients']):>9}  "
                 f"{str(r['test_patients']):>8}  "
                 f"{r['n_train']:>7,}  "
                 f"{r['n_test']:>6,}  "
                 f"{r['kappa']:>7.4f}  "
                 f"{r['accuracy']*100:>8.2f}%  "
                 f"{r['f1']:>8.4f}\n")
    rf.write("  " + "-" * 68 + "\n")
    rf.write(f"  {'Mean':>4}  {'':>9}  {'':>8}  {'':>7}  {'':>6}  "
             f"{df['kappa'].mean():>7.4f}  "
             f"{df['accuracy'].mean()*100:>8.2f}%  "
             f"{df['f1'].mean():>8.4f}\n")
    rf.write(f"  {'Std':>4}  {'':>9}  {'':>8}  {'':>7}  {'':>6}  "
             f"{df['kappa'].std():>7.4f}  "
             f"{df['accuracy'].std()*100:>8.2f}%  "
             f"{df['f1'].std():>8.4f}\n\n")

    rf.write("CLASSIFICATION REPORT (concatenated CV predictions)\n")
    rf.write(clf_report + "\n")

    rf.write("CONFUSION MATRIX (rows=true label, cols=predicted label)\n")
    rf.write("                  Pred:Wake  Pred:NREM  Pred:REM\n")
    true_labels = ['True:Wake ', 'True:NREM ', 'True:REM  ']
    for i, lbl in enumerate(true_labels):
        rf.write(f"  {lbl}      {cm[i,0]:>6,}     {cm[i,1]:>6,}    {cm[i,2]:>6,}\n")

    rf.write("\n")
    rf.write("PER-CLASS TOTALS\n")
    for i, lbl in enumerate(['Wake', 'NREM', 'REM']):
        total = cm[i].sum()
        correct = cm[i, i]
        rf.write(f"  {lbl:<4}: {correct:,} / {total:,} correct  ({correct/total*100:.1f}% recall)\n")

    rf.write("\n")
    rf.write("MODEL HYPERPARAMETERS\n")
    rf.write("  n_estimators : 20   (reduced from 100 for nRF52840 1 MB flash constraint)\n")
    rf.write("  max_depth    : 6\n")
    rf.write("  learning_rate: 0.1\n")
    rf.write("  objective    : multi:softmax (3-class)\n")
    rf.write("  class_weight : balanced (computed per fold)\n")
    rf.write("  scaler       : StandardScaler (fit on train split per fold)\n")

print(f"\nResults saved to: {RESULTS_FILE}")
