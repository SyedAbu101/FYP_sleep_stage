"""
Analyze which audio features are actually useful
and create a refined dataset with only the best audio features
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import cohen_kappa_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("AUDIO FEATURE IMPORTANCE ANALYSIS")
print("="*70)

# Load audio-enhanced dataset
with open('sleep_dataset_audio_enhanced_complete.pkl', 'rb') as f:
    dataset = pickle.load(f)

X = dataset['X_all_features']
y = dataset['y_3class']
patient_ids = dataset['patient_ids']
feature_names = dataset['feature_names']

print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")

# Identify audio vs non-audio features
audio_features = [f for f in feature_names if any(x in f for x in ['mic', 'zcr', 'spectral', 'energy', 'breathing', 'snoring', 'silence', 'activity', 'crest', 'peak'])]
non_audio_features = [f for f in feature_names if f not in audio_features]

print(f"Audio features: {len(audio_features)}")
print(f"Non-audio features: {len(non_audio_features)}")

# Train model on ALL features
print("\n" + "="*70)
print("TRAINING ON ALL FEATURES")
print("="*70)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = XGBClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1, eval_metric='mlogloss')
model.fit(X_scaled, y)

# Get feature importance
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances,
    'is_audio': [f in audio_features for f in feature_names]
}).sort_values('importance', ascending=False)

print("\nTop 20 most important features:")
print(feature_importance_df.head(20).to_string(index=False))

# Analyze audio vs non-audio importance
audio_importance = feature_importance_df[feature_importance_df['is_audio']]['importance'].sum()
non_audio_importance = feature_importance_df[~feature_importance_df['is_audio']]['importance'].sum()

print(f"\n{'='*70}")
print(f"AUDIO vs NON-AUDIO IMPORTANCE")
print(f"{'='*70}")
print(f"Audio features ({len(audio_features)} features): {audio_importance:.4f} ({audio_importance/(audio_importance+non_audio_importance)*100:.1f}%)")
print(f"Non-audio features ({len(non_audio_features)} features): {non_audio_importance:.4f} ({non_audio_importance/(audio_importance+non_audio_importance)*100:.1f}%)")

# Find top audio features
top_audio_features = feature_importance_df[feature_importance_df['is_audio']].head(20)
print(f"\n{'='*70}")
print(f"TOP 20 AUDIO FEATURES")
print(f"{'='*70}")
print(top_audio_features.to_string(index=False))

# Save feature importance
feature_importance_df.to_csv('RESULTS/audio_feature_importance.csv', index=False)
print(f"\n✓ Feature importance saved to: RESULTS/audio_feature_importance.csv")

# Test different feature sets
print(f"\n{'='*70}")
print(f"TESTING DIFFERENT FEATURE COMBINATIONS")
print(f"{'='*70}")

def test_feature_set(X, y, patient_ids, feature_mask, description):
    """Test a specific set of features"""
    X_subset = X[:, feature_mask]

    gkf = GroupKFold(n_splits=5)
    kappas = []

    for train_idx, test_idx in gkf.split(X_subset, y, groups=patient_ids):
        X_train, X_test = X_subset[train_idx], X_subset[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = XGBClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1, eval_metric='mlogloss')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        kappa = cohen_kappa_score(y_test, y_pred)
        kappas.append(kappa)

    mean_kappa = np.mean(kappas)
    std_kappa = np.std(kappas)

    print(f"{description:50s} | {X_subset.shape[1]:3d} features | Kappa: {mean_kappa:.4f} ± {std_kappa:.4f}")
    return mean_kappa

# Create feature masks
all_features_mask = np.ones(len(feature_names), dtype=bool)
non_audio_mask = np.array([f in non_audio_features for f in feature_names])

# Top N audio features
top_10_audio = feature_importance_df[feature_importance_df['is_audio']].head(10)['feature'].tolist()
top_20_audio = feature_importance_df[feature_importance_df['is_audio']].head(20)['feature'].tolist()
top_30_audio = feature_importance_df[feature_importance_df['is_audio']].head(30)['feature'].tolist()

top_10_audio_mask = np.array([f in top_10_audio for f in feature_names])
top_20_audio_mask = np.array([f in top_20_audio for f in feature_names])
top_30_audio_mask = np.array([f in top_30_audio for f in feature_names])

# Combined masks (non-audio + selected audio)
top_10_combined_mask = non_audio_mask | top_10_audio_mask
top_20_combined_mask = non_audio_mask | top_20_audio_mask
top_30_combined_mask = non_audio_mask | top_30_audio_mask

results = []

print("\nTesting different feature combinations...")
print("-" * 100)

# Test various combinations
kappa_all = test_feature_set(X, y, patient_ids, all_features_mask, "All features (163)")
kappa_non_audio = test_feature_set(X, y, patient_ids, non_audio_mask, "Non-audio only")
kappa_top10 = test_feature_set(X, y, patient_ids, top_10_combined_mask, "Non-audio + Top 10 audio")
kappa_top20 = test_feature_set(X, y, patient_ids, top_20_combined_mask, "Non-audio + Top 20 audio")
kappa_top30 = test_feature_set(X, y, patient_ids, top_30_combined_mask, "Non-audio + Top 30 audio")

print(f"\n{'='*70}")
print(f"BEST CONFIGURATION")
print(f"{'='*70}")

best_kappa = max(kappa_all, kappa_non_audio, kappa_top10, kappa_top20, kappa_top30)
if best_kappa == kappa_non_audio:
    best_config = "Non-audio only (audio hurts performance)"
    best_mask = non_audio_mask
elif best_kappa == kappa_top10:
    best_config = "Non-audio + Top 10 audio features"
    best_mask = top_10_combined_mask
elif best_kappa == kappa_top20:
    best_config = "Non-audio + Top 20 audio features"
    best_mask = top_20_combined_mask
elif best_kappa == kappa_top30:
    best_config = "Non-audio + Top 30 audio features"
    best_mask = top_30_combined_mask
else:
    best_config = "All features"
    best_mask = all_features_mask

print(f"Best configuration: {best_config}")
print(f"Best Kappa: {best_kappa:.4f}")

# Create optimized dataset
print(f"\n{'='*70}")
print(f"CREATING OPTIMIZED DATASET")
print(f"{'='*70}")

selected_features = [feature_names[i] for i, mask in enumerate(best_mask) if mask]
X_optimized = X[:, best_mask]

# Save optimized dataset
dataset_optimized = {
    'X_all_features': X_optimized,
    'y_5class': dataset['y_5class'],
    'y_3class': dataset['y_3class'],
    'patient_ids': dataset['patient_ids'],
    'feature_names': selected_features,
    'n_features': len(selected_features),
    'n_samples': len(X_optimized),
    'n_patients': dataset['n_patients']
}

with open('sleep_dataset_optimized_audio.pkl', 'wb') as f:
    pickle.dump(dataset_optimized, f)

print(f"✓ Optimized dataset saved: sleep_dataset_optimized_audio.pkl")
print(f"  Features: {len(selected_features)} (reduced from {len(feature_names)})")
print(f"  Expected Kappa: {best_kappa:.4f}")

# Print which audio features were selected
audio_selected = [f for f in selected_features if f in audio_features]
print(f"\nAudio features selected: {len(audio_selected)}")
if len(audio_selected) > 0 and len(audio_selected) <= 30:
    print("\nSelected audio features:")
    for feat in audio_selected:
        importance = feature_importance_df[feature_importance_df['feature'] == feat]['importance'].values[0]
        print(f"  - {feat:50s} (importance: {importance:.6f})")

print(f"\n{'='*70}")
print(f"RECOMMENDATIONS")
print(f"{'='*70}")

if kappa_non_audio > kappa_all:
    print("⚠ Audio features are HURTING performance!")
    print("  Recommendation: Use non-audio features only")
    print(f"  Improvement: {kappa_non_audio - kappa_all:+.4f} kappa")
elif best_kappa > kappa_non_audio + 0.01:
    print("✓ Selected audio features HELP!")
    print(f"  Improvement over non-audio: {best_kappa - kappa_non_audio:+.4f} kappa")
    print(f"  Use: {best_config}")
else:
    print("○ Audio features provide marginal benefit")
    print(f"  Improvement: {best_kappa - kappa_non_audio:+.4f} kappa")
    print("  Consider using for completeness, but IMU features are more important")

print(f"\n{'='*70}")
