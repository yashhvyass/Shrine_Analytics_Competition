"""
EAST-WEST SHRINE BOWL ANALYTICS COMPETITION
Phase 3: Baseline Model (Combine Features Only)

Purpose: Establish baseline performance using only combine metrics
Target: >= 300 rookie snaps (HIGH-IMPACT contributor)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# =============================================================================
# CONFIGURATION
# =============================================================================

PROCESSED_DIR = Path(__file__).parent / 'processed'
MODELS_DIR = Path(__file__).parent / 'models'
MODELS_DIR.mkdir(exist_ok=True)

COMBINE_COLS = [
    'height', 'weight', 'forty_yd_dash', 'bench_reps_of_225',
    'standing_vertical', 'three_cone', 'twenty_yard_shuttle',
    'standing_broad_jump', 'hand_size', 'arm_length', 'wingspan'
]

print('='*70)
print('PHASE 3: BASELINE MODEL (COMBINE FEATURES ONLY)')
print('='*70)
print(f'\nTarget: >= 300 rookie snaps (HIGH-IMPACT contributor)')

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================

print('\n' + '='*70)
print('STEP 1: LOADING DATA')
print('='*70)

master = pd.read_csv(PROCESSED_DIR / 'master_dataset_clean.csv')
print(f'[OK] Loaded {len(master)} players')

# Split by cohort
train = master[master['cohort'] == 'TRAIN'].copy()
val = master[master['cohort'] == 'VALIDATE'].copy()
holdout = master[master['cohort'] == 'HOLDOUT'].copy()

print(f'\nCohort sizes:')
print(f'  TRAIN (2022): {len(train)} players')
print(f'  VALIDATE (2024): {len(val)} players')
print(f'  HOLDOUT (2025): {len(holdout)} players')

print('\n' + '='*70)
print('STEP 2: PREPARING FEATURES')
print('='*70)

# Filter to players with known outcomes (target not NaN)
train = train[train['target'].notna()]
val = val[val['target'].notna()]

X_train = train[COMBINE_COLS].copy()
y_train = train['target'].copy()

X_val = val[COMBINE_COLS].copy()
y_val = val['target'].copy()

X_holdout = holdout[COMBINE_COLS].copy()

print(f'\nFeatures: {len(COMBINE_COLS)} combine metrics')
print(f'Players with outcomes: TRAIN={len(train)}, VALIDATE={len(val)}')
print(f'Target distribution (TRAIN):')
print(f'  Contributors: {int(y_train.sum())} ({y_train.mean()*100:.1f}%)')
print(f'  Non-contributors: {int((y_train==0).sum())} ({(y_train==0).mean()*100:.1f}%)')

# =============================================================================
# STEP 3: HANDLE MISSING VALUES
# =============================================================================

print('\n' + '='*70)
print('STEP 3: HANDLING MISSING VALUES')
print('='*70)

# Fill with median from training set
medians = X_train.median()
X_train = X_train.fillna(medians)
X_val = X_val.fillna(medians)
X_holdout = X_holdout.fillna(medians)

print(f'[OK] Filled missing values with training medians')

# =============================================================================
# STEP 4: SCALE FEATURES
# =============================================================================

print('\n' + '='*70)
print('STEP 4: SCALING FEATURES')
print('='*70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_holdout_scaled = scaler.transform(X_holdout)

print(f'[OK] Features standardized (mean=0, std=1)')

# =============================================================================
# STEP 5: TRAIN MODELS
# =============================================================================

print('\n' + '='*70)
print('STEP 5: TRAINING MODELS')
print('='*70)

# Logistic Regression
print('\n--- Logistic Regression ---')
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr.fit(X_train_scaled, y_train)

lr_train_auc = roc_auc_score(y_train, lr.predict_proba(X_train_scaled)[:, 1])
lr_val_auc = roc_auc_score(y_val, lr.predict_proba(X_val_scaled)[:, 1])
print(f'  Train AUC: {lr_train_auc:.4f}')
print(f'  Val AUC: {lr_val_auc:.4f}')

# Random Forest
print('\n--- Random Forest ---')
rf = RandomForestClassifier(
    n_estimators=100, max_depth=5, min_samples_leaf=10,
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf.fit(X_train_scaled, y_train)

rf_train_auc = roc_auc_score(y_train, rf.predict_proba(X_train_scaled)[:, 1])
rf_val_auc = roc_auc_score(y_val, rf.predict_proba(X_val_scaled)[:, 1])
print(f'  Train AUC: {rf_train_auc:.4f}')
print(f'  Val AUC: {rf_val_auc:.4f}')

# =============================================================================
# STEP 6: SELECT BEST MODEL
# =============================================================================

print('\n' + '='*70)
print('STEP 6: MODEL COMPARISON')
print('='*70)

print(f'\n{"Model":<25} {"Train AUC":<12} {"Val AUC":<12} {"Overfit":<10}')
print('-' * 60)
print(f'{"Logistic Regression":<25} {lr_train_auc:<12.4f} {lr_val_auc:<12.4f} {lr_train_auc-lr_val_auc:<10.4f}')
print(f'{"Random Forest":<25} {rf_train_auc:<12.4f} {rf_val_auc:<12.4f} {rf_train_auc-rf_val_auc:<10.4f}')

best_model = rf if rf_val_auc > lr_val_auc else lr
best_name = 'Random Forest' if rf_val_auc > lr_val_auc else 'Logistic Regression'
best_val_auc = max(rf_val_auc, lr_val_auc)

print(f'\n[BEST] {best_name}: Val AUC = {best_val_auc:.4f}')

# =============================================================================
# STEP 7: SAVE PREDICTIONS
# =============================================================================

print('\n' + '='*70)
print('STEP 7: GENERATING PREDICTIONS')
print('='*70)

# Generate holdout predictions
holdout_proba = best_model.predict_proba(X_holdout_scaled)[:, 1]

predictions = holdout[['gsis_player_id', 'football_name', 'position']].copy()
predictions['pred_prob'] = holdout_proba
predictions = predictions.sort_values('pred_prob', ascending=False)
predictions['rank'] = range(1, len(predictions) + 1)

predictions.to_csv(MODELS_DIR / 'predictions_baseline.csv', index=False)
print(f'[OK] Saved: {MODELS_DIR / "predictions_baseline.csv"}')

print('\nTop 10 Predicted Contributors (2025):')
print(predictions[['rank', 'football_name', 'position', 'pred_prob']].head(10).to_string(index=False))

# =============================================================================
# SUMMARY
# =============================================================================

print('\n' + '='*70)
print('PHASE 3: BASELINE COMPLETE')
print('='*70)
print(f'\nBaseline Performance: AUC = {best_val_auc:.4f}')
print(f'Features used: {len(COMBINE_COLS)} combine metrics')
print(f'\nThis baseline will be compared against Phase 4 (with tracking features)')
