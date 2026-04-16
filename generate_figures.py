"""
Generate all figures for FYP report

Run this script to create publication-quality figures for your report.
Figures will be saved in docs/images/

All data is sourced from:
  results/audio_model_117feat_results.txt  — verified CV results
  results/audio_feature_importance.csv     — feature importance rankings
  src/modeling/xgboost_audio_model.pkl     — trained model (for importances)
  src/modeling/feature_names_audio.pkl     — ordered feature names
"""

import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# ── Style ─────────────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

ROOT = Path(__file__).resolve().parent
output_dir = ROOT / 'docs' / 'images'
output_dir.mkdir(parents=True, exist_ok=True)

print("Generating figures for FYP report...")
print(f"Output directory: {output_dir}")

# ── Load model artefacts ───────────────────────────────────────────────────────
print("\nLoading model artefacts...")
with open(ROOT / 'src' / 'modeling' / 'xgboost_audio_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open(ROOT / 'src' / 'modeling' / 'feature_names_audio.pkl', 'rb') as f:
    feature_names = pickle.load(f)

importances_all = model.feature_importances_

fi_csv = pd.read_csv(ROOT / 'results' / 'audio_feature_importance.csv')
is_audio_set = set(fi_csv.loc[fi_csv['is_audio'] == True, 'feature'])

print(f"  {len(feature_names)} features loaded  |  "
      f"{sum(n in is_audio_set for n in feature_names)} audio  |  "
      f"{sum(n not in is_audio_set for n in feature_names)} IMU")

# ── Figure 1: Per-Fold Cross-Validation Results ────────────────────────────────
# Source: results/audio_model_117feat_results.txt
print("\n1. Per-Fold Results...")

fold_labels = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Mean']
fold_kappas = [0.2710, 0.2603, 0.2789, 0.3884, 0.4185, 0.3234]
fold_accs   = [56.98,  52.62,  54.86,  64.06,  62.91,  58.29]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

colors_k = ['#2E86AB'] * 5 + ['#E63946']
bars1 = ax1.bar(fold_labels, fold_kappas, color=colors_k, alpha=0.85,
                edgecolor='black', linewidth=0.8)
ax1.set_ylabel("Cohen's Kappa", fontsize=12, fontweight='bold')
ax1.set_title("A) Per-Fold Kappa", fontsize=13, fontweight='bold')
ax1.set_ylim(0, 0.52)
ax1.axhline(0.3234, color='#E63946', linestyle='--', linewidth=1.5,
            label='Mean = 0.3234')
ax1.legend(fontsize=9)
ax1.grid(axis='y', alpha=0.3)
for bar, k in zip(bars1, fold_kappas):
    ax1.text(bar.get_x() + bar.get_width() / 2, k + 0.006,
             f'{k:.4f}', ha='center', fontsize=8)

colors_a = ['#A23B72'] * 5 + ['#E63946']
bars2 = ax2.bar(fold_labels, fold_accs, color=colors_a, alpha=0.85,
                edgecolor='black', linewidth=0.8)
ax2.set_ylabel("Accuracy (%)", fontsize=12, fontweight='bold')
ax2.set_title("B) Per-Fold Accuracy", fontsize=13, fontweight='bold')
ax2.set_ylim(40, 70)
ax2.axhline(58.29, color='#E63946', linestyle='--', linewidth=1.5,
            label='Mean = 58.29%')
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3)
for bar, a in zip(bars2, fold_accs):
    ax2.text(bar.get_x() + bar.get_width() / 2, a + 0.4,
             f'{a:.2f}%', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / 'per_fold_results.png', bbox_inches='tight')
print("  Saved: per_fold_results.png")
plt.close()

# ── Figure 2: Confusion Matrix ─────────────────────────────────────────────────
# Source: results/audio_model_117feat_results.txt
print("\n2. Confusion Matrix...")

cm = np.array([
    [1521,  552,  212],
    [ 745, 1831,  590],
    [ 225,  355,  396],
])

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(cm, annot=False, cmap='Blues',
            cbar_kws={'label': 'Count', 'shrink': 0.8},
            xticklabels=['Wake', 'NREM', 'REM'],
            yticklabels=['Wake', 'NREM', 'REM'],
            linewidths=0.5, linecolor='gray', ax=ax)

# Annotate manually: white text on dark cells, black on light.
# Diagonal cells show count + recall to avoid right-edge clipping.
thresh = cm.max() / 2.0
recalls = [cm[i, i] / cm[i].sum() * 100 for i in range(3)]
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        color = 'white' if cm[i, j] > thresh else 'black'
        if i == j:
            label = f'{cm[i, j]:,}\n({recalls[i]:.1f}%)'
            fs = 12
        else:
            label = f'{cm[i, j]:,}'
            fs = 14
        ax.text(j + 0.5, i + 0.5, label,
                ha='center', va='center', fontsize=fs,
                fontweight='bold', color=color)

ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
ax.set_ylabel('Actual Class',    fontsize=12, fontweight='bold')
ax.set_title(
    'Confusion Matrix — 3-Class Sleep Staging\n'
    '(XGBoost, 118 features, Kappa = 0.3234)',
    fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(output_dir / 'confusion_matrix.png', bbox_inches='tight')
print("  Saved: confusion_matrix.png")
plt.close()

# ── Figure 3: Top-20 Feature Importance ───────────────────────────────────────
# Loaded from model at runtime
print("\n3. Feature Importance (top 20)...")

pairs = sorted(zip(feature_names, importances_all), key=lambda x: -x[1])
top20_names = [p[0] for p in pairs[:20]]
top20_imps  = [p[1] for p in pairs[:20]]
top20_audio = [n in is_audio_set for n in top20_names]

bar_colors = ['#E63946' if a else '#457B9D' for a in top20_audio]

fig, ax = plt.subplots(figsize=(10, 8))
y_pos = np.arange(len(top20_names))
bars = ax.barh(y_pos, top20_imps, color=bar_colors, alpha=0.85,
               edgecolor='black', linewidth=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(top20_names, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
ax.set_title(
    'Top 20 Features by XGBoost Importance\n'
    '(118-feature model — audio = red, IMU = blue)',
    fontsize=13, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)
for i, (bar, imp) in enumerate(zip(bars, top20_imps)):
    ax.text(imp + 0.0002, i, f'{imp:.4f}', va='center', fontsize=8)

legend_elements = [
    mpatches.Patch(facecolor='#457B9D', label='IMU feature',   alpha=0.85),
    mpatches.Patch(facecolor='#E63946', label='Audio feature', alpha=0.85),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / 'feature_importance.png', bbox_inches='tight')
print("  Saved: feature_importance.png")
plt.close()

# ── Figure 4: IMU vs Audio Importance Split ────────────────────────────────────
# Computed from model at runtime
print("\n4. IMU vs Audio Importance Split...")

audio_imp = sum(imp for n, imp in zip(feature_names, importances_all)
                if n in is_audio_set)
imu_imp   = sum(imp for n, imp in zip(feature_names, importances_all)
                if n not in is_audio_set)
total_imp = audio_imp + imu_imp
audio_pct = audio_imp / total_imp * 100
imu_pct   = imu_imp   / total_imp * 100

n_audio = sum(1 for n in feature_names if n in is_audio_set)
n_imu   = len(feature_names) - n_audio

fig, ax = plt.subplots(figsize=(7, 5))
groups = [f'IMU Features\n({n_imu} features)', f'Audio Features\n({n_audio} features)']
values = [imu_pct, audio_pct]
bars = ax.bar(groups, values, color=['#457B9D', '#E63946'],
              alpha=0.85, edgecolor='black', linewidth=1.2, width=0.4)
ax.set_ylabel('Total Feature Importance (%)', fontsize=12, fontweight='bold')
ax.set_title(
    'IMU vs Audio Feature Importance\n'
    '(Computed from XGBoost model, 118 features)',
    fontsize=13, fontweight='bold', pad=20)
ax.set_ylim(0, 105)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 2,
            f'{val:.1f}%', ha='center', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'imu_vs_audio_importance.png', bbox_inches='tight')
print("  Saved: imu_vs_audio_importance.png")
plt.close()

# ── Figure 5: Class Distribution ──────────────────────────────────────────────
# 3-class counts verified from results file; 5-class NREM split from dataset
print("\n5. Class Distribution...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

classes_5 = ['Wake', 'N1', 'N2', 'N3', 'REM']
counts_5  = [2285, 718, 1242, 1206, 976]
colors_5  = ['#E63946', '#F1A208', '#2A9D8F', '#264653', '#457B9D']
ax1.pie(counts_5, labels=classes_5, autopct='%1.1f%%',
        colors=colors_5, startangle=90,
        textprops={'fontsize': 11, 'fontweight': 'bold'})
ax1.set_title('A) 5-Class Distribution\n(Total: 6,427 epochs)',
              fontsize=13, fontweight='bold')

classes_3 = ['Wake', 'NREM\n(N1+N2+N3)', 'REM']
counts_3  = [2285, 3166, 976]
colors_3  = ['#E63946', '#2A9D8F', '#457B9D']
ax2.pie(counts_3, labels=classes_3, autopct='%1.1f%%',
        colors=colors_3, startangle=90,
        textprops={'fontsize': 11, 'fontweight': 'bold'})
ax2.set_title('B) 3-Class Distribution\n(Used in final model)',
              fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'class_distribution.png', bbox_inches='tight')
print("  Saved: class_distribution.png")
plt.close()

# ── Figure 6: Per-Class Performance ───────────────────────────────────────────
# Source: results/audio_model_117feat_results.txt (classification report)
print("\n6. Per-Class Performance...")

classes   = ['Wake', 'NREM', 'REM']
precision = [0.611, 0.669, 0.331]
recall    = [0.666, 0.578, 0.406]
f1_scores = [0.637, 0.620, 0.364]

fig, ax = plt.subplots(figsize=(10, 6))
x     = np.arange(len(classes))
width = 0.25

bars1 = ax.bar(x - width, precision, width, label='Precision',
               color='#E63946', alpha=0.85, edgecolor='black')
bars2 = ax.bar(x,          recall,    width, label='Recall',
               color='#2A9D8F', alpha=0.85, edgecolor='black')
bars3 = ax.bar(x + width,  f1_scores, width, label='F1-Score',
               color='#457B9D', alpha=0.85, edgecolor='black')

ax.set_xlabel('Sleep Stage', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title(
    'Per-Class Performance (XGBoost, 118 features)\n'
    'Kappa = 0.3234  |  Accuracy = 58.29%',
    fontsize=13, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 0.80)

for bar_group in [bars1, bars2, bars3]:
    for bar in bar_group:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 0.018,
                f'{h:.3f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'per_class_performance.png', bbox_inches='tight')
print("  Saved: per_class_performance.png")
plt.close()

# ── Figure 7: Feature Category Importance ─────────────────────────────────────
# Computed from model at runtime
print("\n7. Feature Category Importance...")

def categorise(name: str) -> str:
    if name in is_audio_set:
        return 'Audio'
    if name.startswith('patient_'):
        return 'Patient norm.'
    if name in ('time_since_start_min', 'sleep_cycle_phase',
                'sleep_cycle_sin', 'sleep_cycle_cos'):
        return 'Circadian/temporal'
    if (any(name.startswith(p) for p in
            ('acc_mag', 'gyro_mag', 'total_movement',
             'movement_ratio', 'movement_balance',
             'is_movement_bout', 'movement_bout_count',
             'is_stillness', 'stillness_duration', 'relative_activity'))
            or name == 'is_very_still'):
        return 'Movement magnitude'
    if (any(name.endswith(s) for s in ('_prev', '_diff', '_diff_abs'))
            or 'rolling' in name):
        return 'Temporal/rolling'
    if any(name.startswith(p) for p in
           ('ax_', 'ay_', 'az_', 'gx_', 'gy_', 'gz_', 'tempC')):
        return 'Raw IMU axes'
    return 'Derived/statistical'

cat_imp: dict = {}
for name, imp in zip(feature_names, importances_all):
    cat = categorise(name)
    cat_imp[cat] = cat_imp.get(cat, 0.0) + imp

total = sum(cat_imp.values())
cat_pct = {k: v / total * 100 for k, v in cat_imp.items()}
sorted_cats = sorted(cat_pct.items(), key=lambda x: -x[1])
cat_labels = [c[0] for c in sorted_cats]
cat_vals   = [c[1] for c in sorted_cats]

colour_map = {
    'Raw IMU axes':        '#457B9D',
    'Patient norm.':       '#E63946',
    'Audio':               '#F1A208',
    'Movement magnitude':  '#2A9D8F',
    'Derived/statistical': '#264653',
    'Circadian/temporal':  '#A23B72',
    'Temporal/rolling':    '#A8DADC',
}
bar_colors_cat = [colour_map.get(c, '#888888') for c in cat_labels]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(cat_labels, cat_vals, color=bar_colors_cat,
               alpha=0.85, edgecolor='black', linewidth=1)
ax.set_xlabel('Total Importance (%)', fontsize=12, fontweight='bold')
ax.set_title(
    'Feature Category Importance Distribution\n'
    '(XGBoost model, 118 features)',
    fontsize=13, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)
for bar, val in zip(bars, cat_vals):
    ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
            f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'feature_category_importance.png', bbox_inches='tight')
print("  Saved: feature_category_importance.png")
plt.close()

# ── Figure 8: System Architecture Diagram ─────────────────────────────────────
print("\n8. System Architecture Diagram...")

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

boxes = [
    {'text': 'PSG Data\n(EDF Files)\n44 patients',        'pos': (1, 8),   'color': '#E63946'},
    {'text': 'PillowClip Data\n(CSV Files)\n6-axis IMU',   'pos': (9, 8),   'color': '#E63946'},
    {'text': 'Data Synchronisation\nTimestamp-based (31 pt)\nSequential (13 pt)',
     'pos': (5, 6.5), 'color': '#F1A208'},
    {'text': 'Feature Engineering\n118 features:\n'
             '• IMU axes (28)\n• Patient norm. (12)\n'
             '• Movement (21)\n• Audio (20)',
     'pos': (5, 4.5), 'color': '#2A9D8F'},
    {'text': 'XGBoost Training\n5-fold Patient-level CV\n20 trees, max_depth=6',
     'pos': (5, 2.5), 'color': '#457B9D'},
    {'text': 'Embedded Deployment\nnRF52840\n20-tree inference',
     'pos': (2, 0.5), 'color': '#264653'},
    {'text': 'Sleep Stage Output\nWake / NREM / REM',
     'pos': (8, 0.5), 'color': '#264653'},
]

for box in boxes:
    rect = FancyBboxPatch(
        (box['pos'][0] - 0.8, box['pos'][1] - 0.4), 1.6, 0.8,
        boxstyle="round,pad=0.1",
        facecolor=box['color'], edgecolor='black',
        linewidth=2, alpha=0.7)
    ax.add_patch(rect)
    ax.text(box['pos'][0], box['pos'][1], box['text'],
            ha='center', va='center', fontsize=9,
            fontweight='bold', color='white')

arrows = [
    ((1, 7.6), (5, 6.9)),
    ((9, 7.6), (5, 6.9)),
    ((5, 6.1), (5, 4.9)),
    ((5, 4.1), (5, 2.9)),
    ((5, 2.1), (2, 0.9)),
    ((5, 2.1), (8, 0.9)),
]
for start, end in arrows:
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle='->',
        connectionstyle='arc3,rad=0',
        linewidth=2, color='black', alpha=0.6))

ax.set_title(
    'System Architecture — Sleep Stage Classification\nEnd-to-End Pipeline',
    fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(output_dir / 'system_architecture.png', bbox_inches='tight')
print("  Saved: system_architecture.png")
plt.close()

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("All figures generated successfully.")
print(f"{'='*60}")
print(f"\nOutput: {output_dir.resolve()}")
print("\nFiles generated:")
print("  1. per_fold_results.png           (replaces performance_progression)")
print("  2. confusion_matrix.png")
print("  3. feature_importance.png")
print("  4. imu_vs_audio_importance.png    (replaces model_comparison)")
print("  5. class_distribution.png")
print("  6. per_class_performance.png")
print("  7. feature_category_importance.png")
print("  8. system_architecture.png")
