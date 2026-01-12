"""
EAST-WEST SHRINE BOWL ANALYTICS COMPETITION
Phase 7: Final Model & Feature Selection

Final steps:
  7.1 Feature importance analysis
  7.2 Feature selection (drop <1% importance)
  7.3 Calibration check by position
  7.4 Final evaluation

Expected AUC: 0.71-0.72
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

from config import (
    PROCESSED_DIR, MODELS_DIR, FIGURES_DIR,
    POSITION_GROUPS, BEST_MODEL_PARAMS
)

print('='*70)
print('PHASE 7: FINAL MODEL & FEATURE SELECTION')
print('='*70)

# =============================================================================
# STEP 1: LOAD COMPOSITE-ENHANCED DATA
# =============================================================================

print('\n' + '='*70)
print('STEP 1: LOADING DATA')
print('='*70)

train = pd.read_csv(PROCESSED_DIR / 'train_composite_features.csv')
val = pd.read_csv(PROCESSED_DIR / 'val_composite_features.csv')
holdout = pd.read_csv(PROCESSED_DIR / 'holdout_composite_features.csv')

print(f'[OK] TRAIN: {len(train)}, VAL: {len(val)}, HOLDOUT: {len(holdout)}')

# Get feature columns
feature_cols = [col for col in train.columns 
                if col not in ['gsis_player_id', 'target', 'position_group']]
print(f'Total features: {len(feature_cols)}')

# =============================================================================
# STEP 2: INITIAL MODEL FOR FEATURE IMPORTANCE (7.1)
# =============================================================================

print('\n' + '='*70)
print('STEP 2: FEATURE IMPORTANCE ANALYSIS')
print('='*70)

X_train = train[feature_cols].copy()
y_train = train['target'].copy()
X_val = val[feature_cols].copy()
y_val = val['target'].copy()

# Fill missing
X_train.fillna(0, inplace=True)
X_val.fillna(0, inplace=True)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train initial model
model = RandomForestClassifier(**BEST_MODEL_PARAMS)
model.fit(X_train_scaled, y_train)

# Get importance
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print('\nTop 25 Features:')
print(importance.head(25).to_string(index=False))

# Initial AUC
val_proba_initial = model.predict_proba(X_val_scaled)[:, 1]
initial_auc = roc_auc_score(y_val, val_proba_initial)
print(f'\nInitial Val AUC: {initial_auc:.4f}')

# =============================================================================
# STEP 3: FEATURE SELECTION (7.2)
# =============================================================================

print('\n' + '='*70)
print('STEP 3: FEATURE SELECTION (drop <1% importance)')
print('='*70)

# Identify features to keep
importance_threshold = 0.01
important_features = importance[importance['importance'] >= importance_threshold]['feature'].tolist()
dropped_features = importance[importance['importance'] < importance_threshold]['feature'].tolist()

print(f'Features with >= 1% importance: {len(important_features)}')
print(f'Features dropped (< 1%): {len(dropped_features)}')

# If too many dropped, adjust threshold
if len(important_features) < 10:
    importance_threshold = 0.005
    important_features = importance[importance['importance'] >= importance_threshold]['feature'].tolist()
    dropped_features = importance[importance['importance'] < importance_threshold]['feature'].tolist()
    print(f'\nAdjusted threshold to 0.5%:')
    print(f'  Keeping: {len(important_features)} features')
    print(f'  Dropping: {len(dropped_features)} features')

print('\nDropped features:')
for f in dropped_features[:15]:
    imp = importance[importance['feature'] == f]['importance'].values[0]
    print(f'  {f}: {imp*100:.2f}%')
if len(dropped_features) > 15:
    print(f'  ... and {len(dropped_features) - 15} more')

# Retrain with selected features
X_train_selected = train[important_features].copy()
X_val_selected = val[important_features].copy()
X_holdout_selected = holdout[important_features].copy()

# Fill and scale
X_train_selected.fillna(0, inplace=True)
X_val_selected.fillna(0, inplace=True)
X_holdout_selected.fillna(0, inplace=True)

scaler_selected = StandardScaler()
X_train_selected_scaled = scaler_selected.fit_transform(X_train_selected)
X_val_selected_scaled = scaler_selected.transform(X_val_selected)
X_holdout_selected_scaled = scaler_selected.transform(X_holdout_selected)

# Train new model
model_selected = RandomForestClassifier(**BEST_MODEL_PARAMS)
model_selected.fit(X_train_selected_scaled, y_train)

train_proba = model_selected.predict_proba(X_train_selected_scaled)[:, 1]
val_proba = model_selected.predict_proba(X_val_selected_scaled)[:, 1]

train_auc = roc_auc_score(y_train, train_proba)
val_auc = roc_auc_score(y_val, val_proba)

print(f'\nAfter Feature Selection:')
print(f'  Train AUC: {train_auc:.4f}')
print(f'  Val AUC: {val_auc:.4f}')
print(f'  Change: {val_auc - initial_auc:+.4f}')

# =============================================================================
# STEP 4: CALIBRATION CHECK BY POSITION (7.3)
# =============================================================================

print('\n' + '='*70)
print('STEP 4: CALIBRATION BY POSITION GROUP')
print('='*70)

# Add predictions to validation set
val_with_pred = val.copy()
val_with_pred['pred_prob'] = val_proba

# Calibration by position group
print('\nCalibration by position group:')
for group in POSITION_GROUPS.keys():
    group_data = val_with_pred[val_with_pred['position_group'] == group]
    if len(group_data) >= 5:
        actual_rate = group_data['target'].mean()
        predicted_rate = group_data['pred_prob'].mean()
        calibration_ratio = predicted_rate / actual_rate if actual_rate > 0 else np.nan
        print(f'  {group}: n={len(group_data)}, actual={actual_rate:.2f}, pred={predicted_rate:.2f}, ratio={calibration_ratio:.2f}')

# Overall calibration
print(f'\nOverall:')
print(f'  Actual contributor rate: {y_val.mean():.3f}')
print(f'  Predicted mean probability: {val_proba.mean():.3f}')
print(f'  Brier Score: {brier_score_loss(y_val, val_proba):.4f}')

# Calibration plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 1. Calibration curve
ax1 = axes[0]
fraction_of_positives, mean_predicted_value = calibration_curve(y_val, val_proba, n_bins=5)
ax1.plot(mean_predicted_value, fraction_of_positives, marker='o', label='Model')
ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
ax1.set_xlabel('Mean Predicted Probability')
ax1.set_ylabel('Fraction of Positives')
ax1.set_title('Calibration Curve')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Calibration by position
ax2 = axes[1]
groups = []
actual_rates = []
pred_rates = []
for group in POSITION_GROUPS.keys():
    group_data = val_with_pred[val_with_pred['position_group'] == group]
    if len(group_data) >= 3:
        groups.append(group)
        actual_rates.append(group_data['target'].mean())
        pred_rates.append(group_data['pred_prob'].mean())

x = np.arange(len(groups))
width = 0.35
ax2.bar(x - width/2, actual_rates, width, label='Actual', color='#2ecc71')
ax2.bar(x + width/2, pred_rates, width, label='Predicted', color='#3498db')
ax2.set_xlabel('Position Group')
ax2.set_ylabel('Rate')
ax2.set_title('Calibration by Position Group')
ax2.set_xticks(x)
ax2.set_xticklabels(groups)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'calibration_by_position.png', dpi=150)
print(f'\n[OK] Saved: {FIGURES_DIR / "calibration_by_position.png"}')
plt.close()

# =============================================================================
# STEP 5: FINAL EVALUATION (7.4)
# =============================================================================

print('\n' + '='*70)
print('STEP 5: FINAL EVALUATION')
print('='*70)

print(f'\nFinal Model Performance:')
print(f'  Features used: {len(important_features)}')
print(f'  Train AUC: {train_auc:.4f}')
print(f'  Val AUC: {val_auc:.4f}')
print(f'  Overfit: {train_auc - val_auc:.4f}')

# Detailed metrics
y_pred = (val_proba >= 0.5).astype(int)
print(f'\nClassification Report (threshold=0.5):')
print(classification_report(y_val, y_pred, target_names=['Non-contributor', 'Contributor']))

# Precision @ K
print('Precision @ Top-K:')
for k in [5, 10, 15, 20]:
    top_k_idx = np.argsort(val_proba)[-k:]
    precision = y_val.iloc[top_k_idx].mean()
    n_contrib = int(y_val.iloc[top_k_idx].sum())
    print(f'  Top {k}: {precision*100:.1f}% ({n_contrib}/{k} contributors)')

# =============================================================================
# STEP 6: GENERATE FINAL PREDICTIONS
# =============================================================================

print('\n' + '='*70)
print('STEP 6: FINAL PREDICTIONS')
print('='*70)

# Holdout predictions
holdout_proba = model_selected.predict_proba(X_holdout_selected_scaled)[:, 1]

# Create final predictions dataframe
final_predictions = holdout[['gsis_player_id', 'position_group']].copy()
final_predictions['pred_prob'] = holdout_proba
final_predictions = final_predictions.sort_values('pred_prob', ascending=False)
final_predictions['rank'] = range(1, len(final_predictions) + 1)

# Load names
master = pd.read_csv(PROCESSED_DIR / 'master_dataset_clean.csv')
master['gsis_player_id'] = master['gsis_player_id'].astype(str)
final_predictions['gsis_player_id'] = final_predictions['gsis_player_id'].astype(str)
final_predictions = final_predictions.merge(
    master[['gsis_player_id', 'football_name', 'position']],
    on='gsis_player_id',
    how='left'
)

# Reorder columns
final_predictions = final_predictions[['rank', 'gsis_player_id', 'football_name', 'position', 
                                        'position_group', 'pred_prob']]

print('\nTop 20 Predicted Contributors (2025):')
print(final_predictions.head(20).to_string(index=False))

# Save
final_predictions.to_csv(MODELS_DIR / 'predictions_2025_phase7.csv', index=False)
print(f'\n[OK] Saved: {MODELS_DIR / "predictions_2025_phase7.csv"}')

# =============================================================================
# STEP 7: SAVE FINAL MODEL ARTIFACTS
# =============================================================================

print('\n' + '='*70)
print('STEP 7: SAVING MODEL ARTIFACTS')
print('='*70)

# Save final feature importance
final_importance = pd.DataFrame({
    'feature': important_features,
    'importance': model_selected.feature_importances_
}).sort_values('importance', ascending=False)

final_importance.to_csv(MODELS_DIR / 'feature_importance_final.csv', index=False)
print(f'[OK] Saved feature importance')

# Visualize final feature importance
fig, ax = plt.subplots(figsize=(10, 12))
top_n = min(25, len(final_importance))
plot_data = final_importance.head(top_n).iloc[::-1]
ax.barh(range(top_n), plot_data['importance'], color='#3498db')
ax.set_yticks(range(top_n))
ax.set_yticklabels(plot_data['feature'])
ax.set_xlabel('Importance')
ax.set_title(f'Top {top_n} Feature Importance (Final Model)')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'feature_importance_final.png', dpi=150)
print(f'[OK] Saved: {FIGURES_DIR / "feature_importance_final.png"}')
plt.close()

# =============================================================================
# SUMMARY
# =============================================================================

print('\n' + '='*70)
print('PHASE 7: COMPLETE')
print('='*70)

target_met = 0.71 <= val_auc <= 0.72

print(f'''
FINAL MODEL SUMMARY
===================
Feature Selection:
  - Initial features: {len(feature_cols)}
  - After selection: {len(important_features)}
  - Dropped: {len(dropped_features)} (< 1% importance)

Performance:
  - Val AUC: {val_auc:.4f}
  - Expected: 0.71-0.72
  - Status: {"MET" if target_met else "ABOVE" if val_auc > 0.72 else "BELOW"} expectations

Calibration:
  - Brier Score: {brier_score_loss(y_val, val_proba):.4f}
  - Overall calibration: {val_proba.mean() / y_val.mean():.2f}x

Deliverables:
  - Predictions: {MODELS_DIR / "predictions_2025_phase7.csv"}
  - Feature importance: {MODELS_DIR / "feature_importance_final.csv"}
  - Calibration plot: {FIGURES_DIR / "calibration_by_position.png"}
  - Feature importance plot: {FIGURES_DIR / "feature_importance_final.png"}

Model is ready for deployment!
''')
