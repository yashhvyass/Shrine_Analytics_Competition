"""
EAST-WEST SHRINE BOWL ANALYTICS COMPETITION
Phase 5: Position Foundation Features

Creates position-based features:
  5.1 Position groups (DB, SKILL, OL, DL, LB)
  5.2 Position z-scores for 8 combine metrics (FIT ON TRAIN ONLY)
  5.3 Threshold pass/fail features (6 features)
  5.4 Position group one-hot encoding (5 features)
  5.5 Evaluation

Expected AUC: 0.685-0.695
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt

from config import (
    PROCESSED_DIR, MODELS_DIR, FIGURES_DIR,
    EXCLUDE_COLS, COMBINE_COLS, ZSCORE_METRICS,
    POSITION_TO_GROUP, POSITION_GROUPS,
    THRESHOLD_DEFINITIONS, BEST_MODEL_PARAMS
)

print('='*70)
print('PHASE 5: POSITION FOUNDATION FEATURES')
print('='*70)

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================

print('\n' + '='*70)
print('STEP 1: LOADING DATA')
print('='*70)

master = pd.read_csv(PROCESSED_DIR / 'master_dataset_clean.csv')
master['gsis_player_id'] = master['gsis_player_id'].astype(str)

tracking = pd.read_csv(PROCESSED_DIR / 'tracking_features.csv')
tracking = tracking.dropna(subset=['gsis_id'])
tracking['gsis_id'] = tracking['gsis_id'].apply(lambda x: str(int(float(x))))

merged = master.merge(tracking, left_on='gsis_player_id', right_on='gsis_id', how='left')
print(f'[OK] Loaded {len(merged)} players')

# =============================================================================
# STEP 2: CREATE POSITION GROUPS (5.1)
# =============================================================================

print('\n' + '='*70)
print('STEP 2: CREATING POSITION GROUPS')
print('='*70)

def assign_position_group(position):
    if pd.isna(position):
        return 'UNKNOWN'
    return POSITION_TO_GROUP.get(position, 'UNKNOWN')

merged['position_group'] = merged['position'].apply(assign_position_group)

print('\nPosition group distribution:')
group_dist = merged['position_group'].value_counts()
for group, count in group_dist.items():
    print(f'  {group}: {count}')

# =============================================================================
# STEP 3: POSITION Z-SCORES (5.2) - FIT ON TRAIN ONLY
# =============================================================================

print('\n' + '='*70)
print('STEP 3: POSITION Z-SCORES (Fit on TRAIN only)')
print('='*70)

# Split data
train = merged[merged['cohort'] == 'TRAIN'].copy()
val = merged[merged['cohort'] == 'VALIDATE'].copy()
holdout = merged[merged['cohort'] == 'HOLDOUT'].copy()

# Filter to players with outcomes
train = train[train['target'].notna()]
val = val[val['target'].notna()]

print(f'TRAIN: {len(train)}, VALIDATE: {len(val)}, HOLDOUT: {len(holdout)}')

# Calculate position-group means and stds from TRAIN only
position_stats = {}

for group in POSITION_GROUPS.keys():
    group_train = train[train['position_group'] == group]
    if len(group_train) >= 3:  # Need at least 3 samples
        stats = {}
        for metric in ZSCORE_METRICS:
            if metric in group_train.columns:
                mean = group_train[metric].mean()
                std = group_train[metric].std()
                if std == 0 or pd.isna(std):
                    std = 1  # Prevent division by zero
                stats[metric] = {'mean': mean, 'std': std}
        position_stats[group] = stats
        print(f'  {group}: {len(group_train)} players, computed stats for {len(stats)} metrics')

# Calculate OVERALL stats for UNKNOWN positions (from TRAIN)
overall_stats = {}
for metric in ZSCORE_METRICS:
    if metric in train.columns:
        mean = train[metric].mean()
        std = train[metric].std()
        if std == 0 or pd.isna(std):
            std = 1
        overall_stats[metric] = {'mean': mean, 'std': std}
position_stats['UNKNOWN'] = overall_stats

# Apply z-scores to all data
def compute_position_zscores(df, position_stats):
    """Compute position-adjusted z-scores."""
    for metric in ZSCORE_METRICS:
        zscore_col = f'{metric}_zscore'
        df[zscore_col] = 0.0
        
        for group in df['position_group'].unique():
            mask = df['position_group'] == group
            stats = position_stats.get(group, position_stats['UNKNOWN'])
            
            if metric in stats:
                mean = stats[metric]['mean']
                std = stats[metric]['std']
                df.loc[mask, zscore_col] = (df.loc[mask, metric] - mean) / std
            else:
                # Use overall stats
                mean = position_stats['UNKNOWN'][metric]['mean']
                std = position_stats['UNKNOWN'][metric]['std']
                df.loc[mask, zscore_col] = (df.loc[mask, metric] - mean) / std
    
    return df

train = compute_position_zscores(train, position_stats)
val = compute_position_zscores(val, position_stats)
holdout = compute_position_zscores(holdout, position_stats)

zscore_cols = [f'{m}_zscore' for m in ZSCORE_METRICS]
print(f'\n[OK] Created {len(zscore_cols)} z-score features')

# =============================================================================
# STEP 4: THRESHOLD PASS/FAIL FEATURES (5.3)
# =============================================================================

print('\n' + '='*70)
print('STEP 4: THRESHOLD PASS/FAIL FEATURES')
print('='*70)

def compute_threshold_features(df):
    """Create binary pass/fail features based on position-specific thresholds."""
    for metric, config in THRESHOLD_DEFINITIONS.items():
        if metric not in df.columns:
            continue
            
        threshold_col = f'{metric}_elite'
        df[threshold_col] = 0
        
        for group in df['position_group'].unique():
            mask = df['position_group'] == group
            threshold = config['thresholds'].get(group, config['thresholds'].get('SKILL', None))
            
            if threshold is None:
                continue
                
            if config['direction'] == 'lower':
                df.loc[mask, threshold_col] = (df.loc[mask, metric] <= threshold).astype(int)
            else:
                df.loc[mask, threshold_col] = (df.loc[mask, metric] >= threshold).astype(int)
    
    return df

train = compute_threshold_features(train)
val = compute_threshold_features(val)
holdout = compute_threshold_features(holdout)

threshold_cols = [f'{m}_elite' for m in THRESHOLD_DEFINITIONS.keys() if f'{m}_elite' in train.columns]
print(f'[OK] Created {len(threshold_cols)} threshold features')

# Show threshold pass rates
print('\nThreshold pass rates (TRAIN):')
for col in threshold_cols:
    pass_rate = train[col].mean() * 100
    print(f'  {col}: {pass_rate:.1f}%')

# =============================================================================
# STEP 5: POSITION GROUP ONE-HOT ENCODING (5.4)
# =============================================================================

print('\n' + '='*70)
print('STEP 5: POSITION GROUP ONE-HOT ENCODING')
print('='*70)

# Create one-hot encoding
for group in POSITION_GROUPS.keys():
    col_name = f'pos_{group}'
    train[col_name] = (train['position_group'] == group).astype(int)
    val[col_name] = (val['position_group'] == group).astype(int)
    holdout[col_name] = (holdout['position_group'] == group).astype(int)

onehot_cols = [f'pos_{g}' for g in POSITION_GROUPS.keys()]
print(f'[OK] Created {len(onehot_cols)} one-hot features')

# =============================================================================
# STEP 6: PREPARE FEATURES FOR TRAINING
# =============================================================================

print('\n' + '='*70)
print('STEP 6: PREPARING FEATURES')
print('='*70)

# Get tracking columns
tracking_cols = [col for col in tracking.columns if col != 'gsis_id']

# Combine all feature sets
base_feature_cols = [col for col in merged.columns if col not in EXCLUDE_COLS and col in train.columns]
new_feature_cols = zscore_cols + threshold_cols + onehot_cols

# Filter to only include columns that exist
all_feature_cols = []
for col in base_feature_cols + new_feature_cols:
    if col in train.columns and col not in EXCLUDE_COLS:
        all_feature_cols.append(col)

# Remove duplicates
all_feature_cols = list(dict.fromkeys(all_feature_cols))

print(f'Feature breakdown:')
print(f'  Base features (combine + tracking): {len([c for c in all_feature_cols if c not in new_feature_cols])}')
print(f'  Z-score features: {len(zscore_cols)}')
print(f'  Threshold features: {len(threshold_cols)}')
print(f'  One-hot features: {len(onehot_cols)}')
print(f'  TOTAL: {len(all_feature_cols)}')

X_train = train[all_feature_cols].copy()
y_train = train['target'].copy()
X_val = val[all_feature_cols].copy()
y_val = val['target'].copy()
X_holdout = holdout[all_feature_cols].copy()

# Handle missing values
for df in [X_train, X_val, X_holdout]:
    df.fillna(0, inplace=True)

print(f'\nData shapes:')
print(f'  X_train: {X_train.shape}')
print(f'  X_val: {X_val.shape}')
print(f'  X_holdout: {X_holdout.shape}')

# =============================================================================
# STEP 7: TRAIN AND EVALUATE (5.5)
# =============================================================================

print('\n' + '='*70)
print('STEP 7: TRAINING AND EVALUATION')
print('='*70)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train model
model = RandomForestClassifier(**BEST_MODEL_PARAMS)
model.fit(X_train_scaled, y_train)

# Predictions
train_proba = model.predict_proba(X_train_scaled)[:, 1]
val_proba = model.predict_proba(X_val_scaled)[:, 1]

train_auc = roc_auc_score(y_train, train_proba)
val_auc = roc_auc_score(y_val, val_proba)

print(f'\nPerformance:')
print(f'  Train AUC: {train_auc:.4f}')
print(f'  Val AUC: {val_auc:.4f}')
print(f'  Overfit: {train_auc - val_auc:.4f}')

# Precision @ K
print('\nPrecision @ Top-K:')
for k in [10, 20]:
    top_k_idx = np.argsort(val_proba)[-k:]
    precision = y_val.iloc[top_k_idx].mean()
    print(f'  Top {k}: {precision*100:.1f}% ({int(y_val.iloc[top_k_idx].sum())}/{k})')

# =============================================================================
# STEP 8: FEATURE IMPORTANCE ANALYSIS
# =============================================================================

print('\n' + '='*70)
print('STEP 8: FEATURE IMPORTANCE')
print('='*70)

importance = pd.DataFrame({
    'feature': all_feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print('\nTop 20 Features:')
print(importance.head(20).to_string(index=False))

# Importance by feature category
zscore_imp = importance[importance['feature'].isin(zscore_cols)]['importance'].sum()
threshold_imp = importance[importance['feature'].isin(threshold_cols)]['importance'].sum()
onehot_imp = importance[importance['feature'].isin(onehot_cols)]['importance'].sum()
other_imp = 1 - zscore_imp - threshold_imp - onehot_imp

print(f'\nImportance by category:')
print(f'  Z-scores: {zscore_imp*100:.1f}%')
print(f'  Thresholds: {threshold_imp*100:.1f}%')
print(f'  Position one-hot: {onehot_imp*100:.1f}%')
print(f'  Other (combine + tracking): {other_imp*100:.1f}%')

# =============================================================================
# STEP 9: SAVE FEATURES
# =============================================================================

print('\n' + '='*70)
print('STEP 9: SAVING ENHANCED DATA')
print('='*70)

# Save enhanced datasets
train_enhanced = train[['gsis_player_id', 'target'] + all_feature_cols + ['position_group']]
val_enhanced = val[['gsis_player_id', 'target'] + all_feature_cols + ['position_group']]
holdout_enhanced = holdout[['gsis_player_id'] + all_feature_cols + ['position_group']]

train_enhanced.to_csv(PROCESSED_DIR / 'train_position_features.csv', index=False)
val_enhanced.to_csv(PROCESSED_DIR / 'val_position_features.csv', index=False)
holdout_enhanced.to_csv(PROCESSED_DIR / 'holdout_position_features.csv', index=False)

print(f'[OK] Saved: {PROCESSED_DIR / "train_position_features.csv"}')
print(f'[OK] Saved: {PROCESSED_DIR / "val_position_features.csv"}')
print(f'[OK] Saved: {PROCESSED_DIR / "holdout_position_features.csv"}')

# Save feature importance
importance.to_csv(MODELS_DIR / 'feature_importance_phase5.csv', index=False)

# =============================================================================
# SUMMARY
# =============================================================================

print('\n' + '='*70)
print('PHASE 5: COMPLETE')
print('='*70)

print(f'''
SUMMARY
=======
New Features Created:
  - Position groups: 5 (DB, SKILL, OL, DL, LB)
  - Z-score features: {len(zscore_cols)}
  - Threshold features: {len(threshold_cols)}
  - One-hot features: {len(onehot_cols)}
  - Total features: {len(all_feature_cols)}

Performance:
  - Val AUC: {val_auc:.4f}
  - Expected: 0.685-0.695
  - Status: {"MET" if 0.685 <= val_auc <= 0.695 else "ABOVE" if val_auc > 0.695 else "BELOW"} expectations

Key Implementation Notes:
  - Z-scores fitted on TRAIN only (no data leakage)
  - Used position_group, not raw position
  - Thresholds are position-specific

Next: Run phase06_composites.py
''')
