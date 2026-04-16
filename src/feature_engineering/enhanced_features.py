"""
Complete Feature Engineering: ALL Enhanced + Critical Features

Includes:
- Original 54 enhanced features (from add_advanced_features.py)
- NEW 17 critical features (breathing, movement bouts, rolling windows)

Total: ~130-140 features
Expected: Match or beat previous 0.343 kappa + improvement from critical features
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COMPLETE FEATURE ENGINEERING: ALL ENHANCED + CRITICAL")
print("="*80)
print()

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading data...")
files = list(Path('/Users/syed/Documents/University/Y3S2/FYP/Fresh_Start/Complete_Features_Corrected').glob('*_complete_corrected.csv'))
print(f"Found {len(files)} patient files")

all_data = []
for file in files:
    df = pd.read_csv(file)
    all_data.append(df)

df = pd.concat(all_data, ignore_index=True)
print(f"Loaded {len(df):,} epochs from {df['patient_id'].nunique()} patients\n")

# Filter to main stages
main_stages = [0, 1, 2, 3, 5]
df = df[df['sleep_stage'].isin(main_stages)].copy()
df = df.sort_values(['patient_id', 'epoch_num']).reset_index(drop=True)
print(f"After filtering: {len(df):,} epochs\n")

# ============================================================================
# PART 1: ORIGINAL ENHANCED FEATURES (54 features)
# ============================================================================

print("="*80)
print("PART 1: ADDING ORIGINAL ENHANCED FEATURES")
print("="*80)
print()

# 1. Movement Magnitude (12 features)
print("1. Movement magnitude features...")
df['acc_mag'] = np.sqrt(df['ax_mean']**2 + df['ay_mean']**2 + df['az_mean']**2)
df['acc_mag_std'] = np.sqrt(df['ax_std']**2 + df['ay_std']**2 + df['az_std']**2)
df['acc_mag_max'] = np.sqrt(df['ax_max']**2 + df['ay_max']**2 + df['az_max']**2)
df['acc_mag_min'] = np.sqrt(df['ax_min']**2 + df['ay_min']**2 + df['az_min']**2)
df['acc_mag_range'] = df['acc_mag_max'] - df['acc_mag_min']

df['gyro_mag'] = np.sqrt(df['gx_mean']**2 + df['gy_mean']**2 + df['gz_mean']**2)
df['gyro_mag_std'] = np.sqrt(df['gx_std']**2 + df['gy_std']**2 + df['gz_std']**2)
df['gyro_mag_max'] = np.sqrt(df['gx_max']**2 + df['gy_max']**2 + df['gz_max']**2)
df['gyro_mag_min'] = np.sqrt(df['gx_min']**2 + df['gy_min']**2 + df['gz_min']**2)
df['gyro_mag_range'] = df['gyro_mag_max'] - df['gyro_mag_min']

df['total_movement'] = df['acc_mag'] + df['gyro_mag']
df['total_movement_std'] = df['acc_mag_std'] + df['gyro_mag_std']
print(f"   ✓ Added 12 features")

# 2. Axis Dominance & Orientation (7 features)
print("2. Axis dominance features...")
df['ax_dominance'] = df['ax_std'] / (df['ax_std'] + df['ay_std'] + df['az_std'] + 1e-6)
df['ay_dominance'] = df['ay_std'] / (df['ax_std'] + df['ay_std'] + df['az_std'] + 1e-6)
df['az_dominance'] = df['az_std'] / (df['ax_std'] + df['ay_std'] + df['az_std'] + 1e-6)
df['gravity_x'] = df['ax_mean']
df['gravity_y'] = df['ay_mean']
df['gravity_z'] = df['az_mean']
df['tilt_estimate'] = np.arctan2(np.sqrt(df['ax_mean']**2 + df['ay_mean']**2), df['az_mean'] + 1e-6)
print(f"   ✓ Added 7 features")

# 3. Patient-Specific Normalization (13 features)
print("3. Patient-specific features...")
patient_stats = df.groupby('patient_id').agg({
    'acc_mag': ['mean', 'std'],
    'gyro_mag': ['mean', 'std'],
    'total_movement': ['mean', 'std'],
    'sleep_stage': [lambda x: (x == 0).mean(), lambda x: (x != 0).mean()]
}).reset_index()

patient_stats.columns = ['patient_id',
                          'patient_acc_mag_mean', 'patient_acc_mag_std',
                          'patient_gyro_mag_mean', 'patient_gyro_mag_std',
                          'patient_total_movement_mean', 'patient_total_movement_std',
                          'patient_wake_ratio', 'patient_sleep_ratio']

df = df.merge(patient_stats, on='patient_id', how='left')

df['acc_mag_normalized'] = (df['acc_mag'] - df['patient_acc_mag_mean']) / (df['patient_acc_mag_std'] + 1e-6)
df['gyro_mag_normalized'] = (df['gyro_mag'] - df['patient_gyro_mag_mean']) / (df['patient_gyro_mag_std'] + 1e-6)
df['total_movement_normalized'] = (df['total_movement'] - df['patient_total_movement_mean']) / (df['patient_total_movement_std'] + 1e-6)
df['acc_mag_zscore'] = df['acc_mag_normalized']
df['gyro_mag_zscore'] = df['gyro_mag_normalized']
print(f"   ✓ Added 13 features")

# 4. Temporal Features (15 features)
print("4. Temporal features (previous epoch)...")
temporal_features = ['acc_mag', 'gyro_mag', 'total_movement',
                      'acc_mag_normalized', 'gyro_mag_normalized']

for feature in temporal_features:
    df[f'{feature}_prev'] = df.groupby('patient_id')[feature].shift(1)
    df[f'{feature}_diff'] = df[feature] - df[f'{feature}_prev']
    df[f'{feature}_abs_diff'] = np.abs(df[f'{feature}_diff'])

prev_cols = [col for col in df.columns if '_prev' in col or '_diff' in col]
df[prev_cols] = df[prev_cols].fillna(0)
print(f"   ✓ Added 15 features")

# 5. Time-Based Features (3 features)
print("5. Time-based features...")
df['time_since_start_min'] = df['elapsed_time_sec'] / 60.0
df['time_since_start_hours'] = df['elapsed_time_sec'] / 3600.0
df['time_bin'] = pd.cut(df['time_since_start_hours'],
                         bins=[-np.inf, 1, 3, 5, 7, np.inf],
                         labels=[0, 1, 2, 3, 4])
df['time_bin'] = df['time_bin'].astype(float)
print(f"   ✓ Added 3 features")

# 6. Movement Variability Features (6 features)
print("6. Movement variability features...")
df['acc_cv'] = df['acc_mag_std'] / (df['acc_mag'] + 1e-6)
df['gyro_cv'] = df['gyro_mag_std'] / (df['gyro_mag'] + 1e-6)
df['movement_consistency'] = 1.0 / (df['acc_cv'] + 1.0)
df['high_acc_flag'] = (df['acc_mag_normalized'] > 1.0).astype(float)
df['high_gyro_flag'] = (df['gyro_mag_normalized'] > 1.0).astype(float)
df['high_movement_flag'] = (df['total_movement_normalized'] > 1.0).astype(float)
print(f"   ✓ Added 6 features")

enhanced_count = 12 + 7 + 13 + 15 + 3 + 6
print(f"\n✅ Total enhanced features: {enhanced_count}")

# ============================================================================
# PART 2: NEW CRITICAL FEATURES (17+ features)
# ============================================================================

print()
print("="*80)
print("PART 2: ADDING CRITICAL HIGH-ROI FEATURES")
print("="*80)
print()

# 1. Breathing Variability (7 features)
print("1. Breathing variability features...")
if 'breathing_rate_mean' in df.columns:
    df['breathing_rate_variability'] = df.groupby('patient_id')['breathing_rate_mean'].transform(
        lambda x: x.rolling(window=5, min_periods=1).std()
    )
    df['breathing_cv'] = df['breathing_rate_std'] / (df['breathing_rate_mean'] + 1e-6)
    df['breathing_rolling_std_5'] = df['breathing_rate_variability']
    df['breathing_diff_prev'] = df.groupby('patient_id')['breathing_rate_mean'].diff().fillna(0)
    df['breathing_abs_diff'] = np.abs(df['breathing_diff_prev'])
    df['breathing_stability'] = 1.0 / (df['breathing_rate_variability'] + 1.0)
    breathing_features_list = ['breathing_rate_variability', 'breathing_cv', 'breathing_rolling_std_5',
                                'breathing_diff_prev', 'breathing_abs_diff', 'breathing_stability']
else:
    df['breath_proxy'] = df['az_mean']
    df['breath_amplitude'] = df.groupby('patient_id')['az_mean'].transform(
        lambda x: x.rolling(window=5, min_periods=1).apply(lambda y: y.max() - y.min())
    )
    df['breath_depth_score'] = df.groupby('patient_id')['az_mean'].transform(
        lambda x: x.rolling(window=5, min_periods=1).std()
    )
    df['breathing_rate_variability'] = df['breath_depth_score']
    df['breathing_rolling_std_5'] = df['breath_depth_score']
    df['breathing_diff_prev'] = df.groupby('patient_id')['breath_proxy'].diff().fillna(0)
    df['breathing_abs_diff'] = np.abs(df['breathing_diff_prev'])
    df['breathing_stability'] = 1.0 / (df['breathing_rate_variability'] + 1.0)
    breathing_features_list = ['breath_amplitude', 'breath_depth_score', 'breathing_rate_variability',
                                'breathing_rolling_std_5', 'breathing_diff_prev', 'breathing_abs_diff',
                                'breathing_stability']
print(f"   ✓ Added {len(breathing_features_list)} features")

# 2. Movement Bout & Stillness (6 features)
print("2. Movement bout features...")

def compute_bout_features(patient_data):
    movement_normalized = patient_data['acc_mag_normalized'].values
    is_moving = movement_normalized > 0.5
    n = len(movement_normalized)

    bout_count = np.zeros(n)
    bout_duration = np.zeros(n)
    stillness_duration = np.zeros(n)
    fragmentation = np.zeros(n)

    window = 10
    for i in range(n):
        start = max(0, i - window + 1)
        end = i + 1
        window_moving = is_moving[start:end]

        transitions = np.diff(window_moving.astype(int))
        bout_count[i] = np.sum(transitions == 1)
        fragmentation[i] = np.sum(np.abs(transitions))

        if window_moving[-1]:
            duration = 1
            for j in range(len(window_moving)-2, -1, -1):
                if window_moving[j]:
                    duration += 1
                else:
                    break
            bout_duration[i] = duration

        still_periods = []
        current_still = 0
        for moving in window_moving:
            if not moving:
                current_still += 1
            else:
                if current_still > 0:
                    still_periods.append(current_still)
                current_still = 0
        if current_still > 0:
            still_periods.append(current_still)
        stillness_duration[i] = max(still_periods) if still_periods else 0

    return bout_count, bout_duration, stillness_duration, fragmentation

bout_results = []
for patient_id in df['patient_id'].unique():
    patient_data = df[df['patient_id'] == patient_id].copy()
    bout_count, bout_duration, stillness_duration, fragmentation = compute_bout_features(patient_data)
    patient_data['movement_bout_count'] = bout_count
    patient_data['movement_bout_duration'] = bout_duration
    patient_data['stillness_duration'] = stillness_duration
    patient_data['movement_fragmentation_index'] = fragmentation
    bout_results.append(patient_data)

df = pd.concat(bout_results, ignore_index=True)

df['tilt_diff'] = df.groupby('patient_id')['tilt_estimate'].diff().fillna(0)
df['position_change_indicator'] = (np.abs(df['tilt_diff']) > 0.5).astype(float)
df['num_position_changes'] = df.groupby('patient_id')['position_change_indicator'].transform(
    lambda x: x.rolling(window=10, min_periods=1).sum()
)

movement_bout_features_list = ['movement_bout_count', 'movement_bout_duration', 'stillness_duration',
                                'movement_fragmentation_index', 'position_change_indicator', 'num_position_changes']
print(f"   ✓ Added 6 features")

# 3. Rolling Window Temporal (4+ features)
print("3. Rolling window temporal features...")

temporal_vars = ['acc_mag']
if 'breathing_rate_mean' in df.columns:
    temporal_vars.append('breathing_rate_mean')
if 'snoring_likelihood_mean' in df.columns:
    temporal_vars.append('snoring_likelihood_mean')

rolling_features_list = []
for var in temporal_vars:
    df[f'{var}_rolling_mean_3'] = df.groupby('patient_id')[var].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    df[f'{var}_rolling_std_3'] = df.groupby('patient_id')[var].transform(
        lambda x: x.rolling(window=3, min_periods=1).std()
    )
    df[f'{var}_rolling_mean_5'] = df.groupby('patient_id')[var].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    df[f'{var}_rolling_std_5'] = df.groupby('patient_id')[var].transform(
        lambda x: x.rolling(window=5, min_periods=1).std()
    )
    rolling_features_list.extend([
        f'{var}_rolling_mean_3', f'{var}_rolling_std_3',
        f'{var}_rolling_mean_5', f'{var}_rolling_std_5'
    ])

df[rolling_features_list] = df[rolling_features_list].fillna(0)
print(f"   ✓ Added {len(rolling_features_list)} features")

critical_count = len(breathing_features_list) + len(movement_bout_features_list) + len(rolling_features_list)
print(f"\n✅ Total critical features: {critical_count}")

# ============================================================================
# FINALIZE DATASET
# ============================================================================

print()
print("="*80)
print("FINALIZING DATASET")
print("="*80)
print()

metadata_cols = ['patient_id', 'epoch_num', 'elapsed_time_sec', 'sleep_stage', 'num_samples']

# Get all feature columns
all_feature_cols = [col for col in df.columns if col not in metadata_cols]

# Categorize
original_imu = [col for col in all_feature_cols if any(x in col for x in ['ax_', 'ay_', 'az_', 'gx_', 'gy_', 'gz_', 'tempC_'])
                and not any(y in col for y in ['mag', 'normalized', 'dominance', 'gravity', 'tilt'])]

original_audio = [col for col in all_feature_cols if any(x in col for x in ['mic_rms', 'zcr', 'spectral', 'snoring', 'energy', 'silence', 'activity', 'crest', 'freq', 'peak_to_peak', 'rms_energy', 'total_energy'])
                  and not any(y in col for y in ['rolling', 'diff', 'cv', 'variability', 'stability', 'breathing_rate_mean', 'breathing_rate_std'])]

# If breathing_rate_mean is in original data
if 'breathing_rate_mean' in df.columns and 'breathing_rate_mean' not in original_audio:
    original_audio.append('breathing_rate_mean')
if 'breathing_rate_std' in df.columns and 'breathing_rate_std' not in original_audio:
    original_audio.append('breathing_rate_std')

enhanced_features = [col for col in all_feature_cols
                     if col not in original_imu and col not in original_audio]

all_features = original_imu + original_audio + enhanced_features

# Clean data
df = df.replace([np.inf, -np.inf], np.nan)
for col in all_features:
    if df[col].isna().any():
        df[col] = df[col].fillna(0)

print(f"Original IMU: {len(original_imu)}")
print(f"Original Audio: {len(original_audio)}")
print(f"Enhanced (all new): {len(enhanced_features)}")
print(f"TOTAL: {len(all_features)} features")
print()

# Create dataset
dataset = {
    'description': 'Sleep Stage Classification - ALL ENHANCED + CRITICAL FEATURES',
    'version': 'v4_complete',
    'n_epochs': len(df),
    'n_patients': df['patient_id'].nunique(),
    'n_classes': 5,
    'class_names': ['Wake', 'N1', 'N2', 'N3', 'REM'],
    'class_mapping': {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 5: 'REM'},

    'X_all_features': df[all_features].values,
    'y': df['sleep_stage'].values,
    'patient_ids': df['patient_id'].values,

    'all_feature_names': all_features,
    'original_imu_features': original_imu,
    'original_audio_features': original_audio,
    'enhanced_features': enhanced_features,

    'epoch_nums': df['epoch_num'].values,
    'elapsed_times': df['elapsed_time_sec'].values,

    'scaler': StandardScaler().fit(df[all_features]),

    'class_counts': df['sleep_stage'].value_counts().to_dict(),
}

from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced',
                                      classes=np.unique(dataset['y']),
                                      y=dataset['y'])
dataset['class_weights'] = dict(zip(np.unique(dataset['y']), class_weights))

# Save
output_file = "/Users/syed/Documents/University/Y3S2/FYP/Fresh_Start/sleep_dataset_complete_v4.pkl"

with open(output_file, 'wb') as f:
    pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)

print("="*80)
print("✅ COMPLETE DATASET SAVED!")
print("="*80)
print(f"\nFile: sleep_dataset_complete_v4.pkl")
print(f"Size: {file_size_mb:.1f} MB")
print(f"\nTotal features: {len(all_features)}")
print(f"  ├─ Original enhanced: ~56")
print(f"  └─ Critical new: ~{critical_count}")
print()
print("This should match or beat previous 0.343 kappa!")
print("="*80)
