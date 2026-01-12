"""
EAST-WEST SHRINE BOWL ANALYTICS COMPETITION
Phase 4: Final Model (Combine + Tracking Features)

Purpose: Production model with all available features
Target: >= 300 rookie snaps (HIGH-IMPACT contributor)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

PROCESSED_DIR = Path(__file__).parent / 'processed'
MODELS_DIR = Path(__file__).parent / 'models'
FIGURES_DIR = Path(__file__).parent / 'figures'
MODELS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

COMBINE_COLS = [
    'height', 'weight', 'forty_yd_dash', 'bench_reps_of_225',
    'standing_vertical', 'three_cone', 'twenty_yard_shuttle',
    'standing_broad_jump', 'hand_size', 'arm_length', 'wingspan'
]

EXCLUDE_COLS = [
    'gsis_player_id', 'gsis_id', 'football_name', 'first_name', 'last_name',
    'position', 'target', 'total_snaps', 'rookie_season',
    'shrine_bowl_year', 'cohort', 'team_name', 'team_code', 'conference'
]

# Best model configuration (Medium RF)
MODEL_PARAMS = {
    'n_estimators': 75,
    'max_depth': 8,
    'min_samples_split': 15,
    'min_samples_leaf': 7,
    'max_features': 'sqrt',
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
}

print('='*70)
print('PHASE 4: FINAL MODEL (COMBINE + TRACKING)')
print('='*70)
print(f'\nTarget: >= 300 rookie snaps (HIGH-IMPACT contributor)')

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================

print('\n' + '='*70)
print('STEP 1: LOADING DATA')
print('='*70)

# Load master dataset
master = pd.read_csv(PROCESSED_DIR / 'master_dataset_clean.csv')
master['gsis_player_id'] = master['gsis_player_id'].astype(str)
print(f'[OK] Master: {len(master)} players')

# Load tracking features if available
tracking_path = PROCESSED_DIR / 'tracking_features.csv'
if tracking_path.exists():
    tracking = pd.read_csv(tracking_path)
    # Drop rows with NaN gsis_id and convert to clean string
    tracking = tracking.dropna(subset=['gsis_id'])
    tracking['gsis_id'] = tracking['gsis_id'].apply(lambda x: str(int(float(x))))
    print(f'[OK] Tracking: {len(tracking)} players, {len(tracking.columns)-1} features')
    has_tracking = True
else:
    print('[INFO] Tracking features not found - using combine features only')
    tracking = None
    has_tracking = False

# =============================================================================
# STEP 2: MERGE DATA
# =============================================================================

print('\n' + '='*70)
print('STEP 2: MERGING DATA')
print('='*70)

# Merge tracking with master if available
if has_tracking:
    merged = master.merge(tracking, left_on='gsis_player_id', right_on='gsis_id', how='left')
    tracking_coverage = merged[tracking.columns[1]].notna().sum()
    print(f'[OK] Merged: {len(merged)} players')
    print(f'Tracking coverage: {tracking_coverage}/{len(merged)} ({tracking_coverage/len(merged)*100:.1f}%)')
else:
    merged = master.copy()
    print(f'[OK] Using master data only: {len(merged)} players')

# =============================================================================
# STEP 3: PREPARE FEATURES
# =============================================================================

print('\n' + '='*70)
print('STEP 3: PREPARING FEATURES')
print('='*70)

# Get feature columns (exclude metadata)
feature_cols = [col for col in merged.columns if col not in EXCLUDE_COLS]
print(f'Total features: {len(feature_cols)}')

# Split by cohort
train = merged[merged['cohort'] == 'TRAIN'].copy()
val = merged[merged['cohort'] == 'VALIDATE'].copy()
holdout = merged[merged['cohort'] == 'HOLDOUT'].copy()

print(f'\nCohort sizes (total):')
print(f'  TRAIN: {len(train)} players')
print(f'  VALIDATE: {len(val)} players')
print(f'  HOLDOUT: {len(holdout)} players')

# Filter to players with known outcomes
train = train[train['target'].notna()]
val = val[val['target'].notna()]

print(f'\nWith outcomes: TRAIN={len(train)}, VALIDATE={len(val)}')

X_train = train[feature_cols].copy()
y_train = train['target'].copy()

X_val = val[feature_cols].copy()
y_val = val['target'].copy()

X_holdout = holdout[feature_cols].copy()

print(f'\nTarget distribution (TRAIN):')
print(f'  Contributors: {int(y_train.sum())} ({y_train.mean()*100:.1f}%)')
print(f'  Non-contributors: {int((y_train==0).sum())} ({(y_train==0).mean()*100:.1f}%)')

# =============================================================================
# STEP 4: HANDLE MISSING VALUES
# =============================================================================

print('\n' + '='*70)
print('STEP 4: HANDLING MISSING VALUES')
print('='*70)

# Identify feature types
if has_tracking:
    tracking_cols = [col for col in feature_cols if col in tracking.columns]
else:
    tracking_cols = []
combine_features = [col for col in feature_cols if col in COMBINE_COLS]

# Calculate medians from training
combine_medians = X_train[combine_features].median()
if tracking_cols:
    tracking_medians = X_train[tracking_cols].median()

# Fill missing values
for df in [X_train, X_val, X_holdout]:
    df[combine_features] = df[combine_features].fillna(combine_medians)
    if tracking_cols:
        df[tracking_cols] = df[tracking_cols].fillna(tracking_medians)
    # Fill remaining with 0 (college features)
    df.fillna(0, inplace=True)

print(f'[OK] Missing values handled')
print(f'  TRAIN missing: {X_train.isnull().sum().sum()}')
print(f'  VALIDATE missing: {X_val.isnull().sum().sum()}')

# =============================================================================
# STEP 5: SCALE FEATURES
# =============================================================================

print('\n' + '='*70)
print('STEP 5: SCALING FEATURES')
print('='*70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_holdout_scaled = scaler.transform(X_holdout)

print('[OK] Features standardized')

# =============================================================================
# STEP 6: TRAIN MODEL
# =============================================================================

print('\n' + '='*70)
print('STEP 6: TRAINING MODEL')
print('='*70)

print(f'\nModel: Random Forest')
print(f'Parameters: {MODEL_PARAMS}')

model = RandomForestClassifier(**MODEL_PARAMS)
model.fit(X_train_scaled, y_train)

# Evaluate
train_proba = model.predict_proba(X_train_scaled)[:, 1]
val_proba = model.predict_proba(X_val_scaled)[:, 1]

train_auc = roc_auc_score(y_train, train_proba)
val_auc = roc_auc_score(y_val, val_proba)

print(f'\nPerformance:')
print(f'  Train AUC: {train_auc:.4f}')
print(f'  Val AUC: {val_auc:.4f}')
print(f'  Overfit: {train_auc - val_auc:.4f}')

# =============================================================================
# STEP 7: DETAILED EVALUATION
# =============================================================================

print('\n' + '='*70)
print('STEP 7: DETAILED EVALUATION')
print('='*70)

# Classification report
val_pred = (val_proba >= 0.5).astype(int)
print('\nClassification Report:')
print(classification_report(y_val, val_pred, target_names=['Non-contributor', 'Contributor']))

# Precision at K
print('Precision @ Top-K:')
for k in [10, 20, 30]:
    top_k_idx = np.argsort(val_proba)[-k:]
    precision = y_val.iloc[top_k_idx].mean()
    print(f'  Top {k}: {precision*100:.1f}% ({int(y_val.iloc[top_k_idx].sum())}/{k} contributors)')

# =============================================================================
# STEP 8: FEATURE IMPORTANCE
# =============================================================================

print('\n' + '='*70)
print('STEP 8: FEATURE IMPORTANCE')
print('='*70)

importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print('\nTop 15 Features:')
print(importance.head(15).to_string(index=False))

# Save
importance.to_csv(MODELS_DIR / 'feature_importance.csv', index=False)
print(f'\n[OK] Saved: {MODELS_DIR / "feature_importance.csv"}')

# Feature importance by category
tracking_importance = importance[importance['feature'].isin(tracking_cols)]['importance'].sum()
combine_importance = importance[importance['feature'].isin(COMBINE_COLS)]['importance'].sum()
other_importance = 1 - tracking_importance - combine_importance

print(f'\nImportance by Category:')
print(f'  Tracking: {tracking_importance*100:.1f}%')
print(f'  Combine: {combine_importance*100:.1f}%')
print(f'  Other: {other_importance*100:.1f}%')

# =============================================================================
# STEP 9: VISUALIZATION
# =============================================================================

print('\n' + '='*70)
print('STEP 9: VISUALIZATIONS')
print('='*70)

# Feature importance chart
fig, ax = plt.subplots(figsize=(10, 8))
top15 = importance.head(15)

colors = []
for feat in top15['feature']:
    if feat in COMBINE_COLS:
        colors.append('#1e3a5f')  # Dark blue for combine
    elif feat in tracking_cols:
        colors.append('#4a90d9')  # Light blue for tracking
    else:
        colors.append('#2ecc71')  # Green for other

y_pos = np.arange(len(top15))
ax.barh(y_pos, top15['importance'].values * 100, color=colors[::-1])
ax.set_yticks(y_pos)
ax.set_yticklabels(top15['feature'].values[::-1])
ax.set_xlabel('Importance (%)')
ax.set_title('Top 15 Predictive Features')

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#1e3a5f', label=f'Combine ({combine_importance*100:.0f}%)'),
    Patch(facecolor='#4a90d9', label=f'Tracking ({tracking_importance*100:.0f}%)')
]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
print(f'[OK] Saved: {FIGURES_DIR / "feature_importance.png"}')
plt.close()

# =============================================================================
# STEP 10: FINAL PREDICTIONS
# =============================================================================

print('\n' + '='*70)
print('STEP 10: FINAL PREDICTIONS')
print('='*70)

holdout_proba = model.predict_proba(X_holdout_scaled)[:, 1]

predictions = holdout[['gsis_player_id', 'football_name', 'position']].copy()
predictions['pred_prob'] = holdout_proba
predictions = predictions.sort_values('pred_prob', ascending=False)
predictions['rank'] = range(1, len(predictions) + 1)

predictions.to_csv(MODELS_DIR / 'predictions_2025_FINAL.csv', index=False)
print(f'[OK] Saved: {MODELS_DIR / "predictions_2025_FINAL.csv"}')

print('\nTop 20 Predicted Contributors (2025):')
print(predictions[['rank', 'football_name', 'position', 'pred_prob']].head(20).to_string(index=False))

# =============================================================================
# SUMMARY
# =============================================================================

print('\n' + '='*70)
print('PHASE 4: COMPLETE')
print('='*70)
print(f'\nFinal Model Performance:')
print(f'  AUC: {val_auc:.4f}')
print(f'  Features: {len(feature_cols)}')
print(f'    Tracking: {len(tracking_cols)} ({tracking_importance*100:.0f}% importance)')
print(f'    Combine: {len(COMBINE_COLS)} ({combine_importance*100:.0f}% importance)')
print(f'\nDeliverables:')
print(f'  - Predictions: {MODELS_DIR / "predictions_2025_FINAL.csv"}')
print(f'  - Feature importance: {MODELS_DIR / "feature_importance.csv"}')
print(f'  - Visualization: {FIGURES_DIR / "feature_importance.png"}')
