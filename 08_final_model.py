"""
EAST-WEST SHRINE BOWL ANALYTICS COMPETITION
Phase 8-9 Combined: Advanced Analysis & Model Evaluation

Consolidated file containing:
  1. Model Training (with Production Score)
  2. SHAP Analysis
  3. Bootstrap Confidence Intervals
  4. Position-Specific AUC
  5. ROC Curve
  6. Precision/Recall at K
  7. Calibration Analysis
  8. Threshold Analysis
  9. Final Predictions

Target: >= 300 NFL rookie snaps
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from config import (
    PROCESSED_DIR, MODELS_DIR, FIGURES_DIR,
    POSITION_GROUPS, POSITION_TO_GROUP, BEST_MODEL_PARAMS, COMBINE_COLS
)

print('='*70)
print('PHASE 8-9: COMPREHENSIVE MODEL EVALUATION')
print('='*70)


print('\n' + '='*70)
print('STEP 1: LOAD DATA AND TRAIN FINAL MODEL')
print('='*70)

train = pd.read_csv(PROCESSED_DIR / 'train_composite_features.csv')
val = pd.read_csv(PROCESSED_DIR / 'val_composite_features.csv')
holdout = pd.read_csv(PROCESSED_DIR / 'holdout_composite_features.csv')

# Get feature columns
importance = pd.read_csv(MODELS_DIR / 'feature_importance_final.csv')
important_features = importance[importance['importance'] >= 0.01]['feature'].tolist()

print(f'TRAIN: {len(train)}, VAL: {len(val)}, HOLDOUT: {len(holdout)}')
print(f'Features (after selection): {len(important_features)}')

# --- Add Production Score ---
college = pd.read_csv(PROCESSED_DIR / 'processed_college_stats.csv')

def create_production_score(df, college_stats):
    """Create position-normalized production scores."""
    college_agg = college_stats.groupby('gsis_player_id').agg({
        'receiving_yards': 'sum', 'rushing_yards': 'sum',
        'receiving_touchdowns': 'sum', 'rushing_touchdowns': 'sum',
        'passing_yards': 'sum', 'passing_touchdowns': 'sum',
        'defense_total_tackles': 'sum', 'defense_sacks': 'sum',
        'defense_interceptions': 'sum', 'season': 'count'
    }).reset_index()
    college_agg = college_agg.rename(columns={'season': 'seasons_played'})
    college_agg['gsis_player_id'] = college_agg['gsis_player_id'].astype(str)
    
    for col in ['receiving_yards', 'rushing_yards', 'defense_total_tackles']:
        if col in college_agg.columns:
            college_agg[f'{col}_per_season'] = college_agg[col] / college_agg['seasons_played'].clip(1)
    
    df = df.copy()
    df['gsis_player_id'] = df['gsis_player_id'].astype(str)
    df = df.merge(college_agg, on='gsis_player_id', how='left')
    
    df['production_score'] = 0.0
    
    # SKILL: receiving + rushing
    skill_mask = df['position_group'] == 'SKILL'
    if skill_mask.any():
        df.loc[skill_mask, 'production_score'] = (
            df.loc[skill_mask, 'receiving_yards_per_season'].fillna(0) / 1000 +
            df.loc[skill_mask, 'rushing_yards_per_season'].fillna(0) / 500 +
            df.loc[skill_mask, 'receiving_touchdowns'].fillna(0) / 10
        )
    
    # DB: interceptions + tackles
    db_mask = df['position_group'] == 'DB'
    if db_mask.any():
        df.loc[db_mask, 'production_score'] = (
            df.loc[db_mask, 'defense_interceptions'].fillna(0) / 5 +
            df.loc[db_mask, 'defense_total_tackles_per_season'].fillna(0) / 50
        )
    
    # DL/LB: sacks + tackles
    for group in ['DL', 'LB']:
        mask = df['position_group'] == group
        if mask.any():
            df.loc[mask, 'production_score'] = (
                df.loc[mask, 'defense_sacks'].fillna(0) / 10 +
                df.loc[mask, 'defense_total_tackles_per_season'].fillna(0) / 40
            )
    
    # OL: experience
    ol_mask = df['position_group'] == 'OL'
    if ol_mask.any():
        df.loc[ol_mask, 'production_score'] = df.loc[ol_mask, 'seasons_played'].fillna(0) / 4
    
    return df

train = create_production_score(train, college)
val = create_production_score(val, college)
holdout = create_production_score(holdout, college)

# Save datasets with production score for later phases
train.to_csv(PROCESSED_DIR / 'train_final_features.csv', index=False)
val.to_csv(PROCESSED_DIR / 'val_final_features.csv', index=False)
holdout.to_csv(PROCESSED_DIR / 'holdout_final_features.csv', index=False)

# Final feature list
final_features = important_features + ['production_score']
final_features = [f for f in final_features if f in train.columns]
print(f'Final features (with production): {len(final_features)}')

# Prepare data
X_train = train[final_features].fillna(0)
y_train = train['target']
X_val = val[final_features].fillna(0)
y_val = val['target']
X_holdout = holdout[final_features].fillna(0)

# Scale and train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_holdout_scaled = scaler.transform(X_holdout)

model = RandomForestClassifier(**BEST_MODEL_PARAMS)
model.fit(X_train_scaled, y_train)

y_proba = model.predict_proba(X_val_scaled)[:, 1]
baseline_auc = roc_auc_score(y_val, y_proba)
print(f'\nFinal Model Val AUC: {baseline_auc:.4f}')


print('\n' + '='*70)
print('STEP 2: SHAP ANALYSIS')
print('='*70)

try:
    import shap
    
    print('Creating SHAP explainer...')
    explainer = shap.TreeExplainer(model)
    
    # SHAP for validation set
    print('Computing SHAP values...')
    shap_values_all = explainer.shap_values(X_val_scaled)
    
    if isinstance(shap_values_all, list):
        shap_values_class1 = shap_values_all[1]
    elif len(shap_values_all.shape) == 3:
        shap_values_class1 = shap_values_all[:, :, 1]
    else:
        shap_values_class1 = shap_values_all
    
    # Top 5 holdout explanations
    print('\n--- Top 5 Predictions with SHAP Explanations ---')
    holdout_proba = model.predict_proba(X_holdout_scaled)[:, 1]
    top5_idx = np.argsort(-holdout_proba)[:5]
    
    master = pd.read_csv(PROCESSED_DIR / 'master_dataset_clean.csv')
    master['gsis_player_id'] = master['gsis_player_id'].astype(str)
    holdout_ids = holdout['gsis_player_id'].astype(str).values
    holdout_pos = holdout['position_group'].values
    
    X_top5_scaled = X_holdout_scaled[top5_idx]
    shap_top5 = explainer.shap_values(X_top5_scaled)
    if isinstance(shap_top5, list):
        shap_top5 = shap_top5[1]
    elif len(shap_top5.shape) == 3:
        shap_top5 = shap_top5[:, :, 1]
    
    for i in range(5):
        orig_idx = top5_idx[i]
        pid = holdout_ids[orig_idx]
        player_info = master[master['gsis_player_id'] == pid]
        name = player_info['football_name'].values[0] if len(player_info) > 0 else 'Unknown'
        pos = holdout_pos[orig_idx]
        prob = holdout_proba[orig_idx]
        
        player_shap = shap_top5[i]
        top3 = sorted(zip(final_features, player_shap), key=lambda x: abs(x[1]), reverse=True)[:3]
        
        print(f'\n{i+1}. {name} ({pos}) - {prob*100:.1f}%')
        for feat, sv in top3:
            sign = '+' if sv > 0 else '-'
            print(f'   {sign} {feat}: {sv:+.3f}')
    
    # SHAP Summary Plot - Use scaled data since SHAP values were computed on scaled data
    print('\nGenerating SHAP plots...')
    plt.figure(figsize=(12, 10))
    # Note: Using X_val_scaled with feature_names since SHAP was computed on scaled features
    shap.summary_plot(shap_values_class1, X_val_scaled, feature_names=final_features, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # SHAP Importance Bar Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap_importance = np.abs(shap_values_class1).mean(axis=0)
    shap_df = pd.DataFrame({'feature': final_features, 'importance': shap_importance})
    shap_df = shap_df.sort_values('importance', ascending=True).tail(15)
    ax.barh(range(len(shap_df)), shap_df['importance'], color='#e74c3c')
    ax.set_yticks(range(len(shap_df)))
    ax.set_yticklabels(shap_df['feature'])
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title('SHAP Feature Importance (Top 15)')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'shap_importance.png', dpi=150)
    plt.close()
    print(f'[OK] Saved SHAP plots')
    
except ImportError:
    print('[SKIP] SHAP not installed')
except Exception as e:
    print(f'[ERROR] SHAP: {e}')


print('\n' + '='*70)
print('STEP 3: BOOTSTRAP CONFIDENCE INTERVALS')
print('='*70)

def bootstrap_metric(y_true, y_pred, metric_fn, n_bootstrap=1000, random_state=42):
    """Calculate bootstrap CI for any metric."""
    np.random.seed(random_state)
    scores = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
        try:
            score = metric_fn(y_true.iloc[idx], y_pred[idx])
            scores.append(score)
        except:
            pass
    return {
        'mean': np.mean(scores),
        'ci_lower': np.percentile(scores, 2.5),
        'ci_upper': np.percentile(scores, 97.5)
    }

print('Running 1000 bootstrap iterations...')
auc_boot = bootstrap_metric(y_val, y_proba, roc_auc_score)

def precision_at_k(y_true, y_pred, k=10):
    top_k = np.argsort(y_pred)[-k:]
    return y_true.iloc[top_k].mean()

prec10_boot = bootstrap_metric(y_val, y_proba, lambda y, p: precision_at_k(y, p, 10))

print(f'\nVal AUC: {baseline_auc:.4f}')
print(f'  Bootstrap 95% CI: [{auc_boot["ci_lower"]:.3f}, {auc_boot["ci_upper"]:.3f}]')
print(f'\nPrecision@10: {precision_at_k(y_val, y_proba):.1%}')
print(f'  Bootstrap 95% CI: [{prec10_boot["ci_lower"]*100:.1f}%, {prec10_boot["ci_upper"]*100:.1f}%]')

baseline_rate = y_val.mean()
print(f'\nBaseline rate: {baseline_rate:.1%}')
print(f'Precision@10 lift: {precision_at_k(y_val, y_proba)/baseline_rate:.2f}x')

# Save CIs
ci_report = pd.DataFrame([
    {'metric': 'AUC', 'value': baseline_auc, 'ci_lower': auc_boot['ci_lower'], 'ci_upper': auc_boot['ci_upper']},
    {'metric': 'Precision@10', 'value': precision_at_k(y_val, y_proba), 'ci_lower': prec10_boot['ci_lower'], 'ci_upper': prec10_boot['ci_upper']}
])
ci_report.to_csv(MODELS_DIR / 'confidence_intervals.csv', index=False)


print('\n' + '='*70)
print('STEP 4: POSITION-SPECIFIC AUC')
print('='*70)

val_with_pred = val.copy()
val_with_pred['pred_prob'] = y_proba

print('\n--- AUC by Position Group ---')
print(f'{"Group":<10} {"N":>5} {"Contributors":>12} {"Rate":>8} {"AUC":>8}')
print('-' * 50)

position_results = []
for group in ['DB', 'SKILL', 'OL', 'DL', 'LB', 'UNKNOWN']:
    g = val_with_pred[val_with_pred['position_group'] == group]
    n = len(g)
    contrib = int(g['target'].sum())
    rate = g['target'].mean() * 100 if n > 0 else 0
    
    if n >= 5 and g['target'].nunique() == 2:
        auc = roc_auc_score(g['target'], g['pred_prob'])
        print(f'{group:<10} {n:>5} {contrib:>12} {rate:>7.1f}% {auc:>8.3f}')
    else:
        auc = None
        print(f'{group:<10} {n:>5} {contrib:>12} {rate:>7.1f}% {"N/A":>8}')
    
    position_results.append({'group': group, 'n': n, 'contributors': contrib, 'rate': rate, 'auc': auc})

pd.DataFrame(position_results).to_csv(MODELS_DIR / 'position_auc.csv', index=False)


print('\n' + '='*70)
print('STEP 5: ROC CURVE')
print('='*70)

fpr, tpr, thresholds = roc_curve(y_val, y_proba)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'Model (AUC = {baseline_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
ax.fill_between(fpr, tpr, alpha=0.3)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve - Shrine Bowl Contributor Prediction', fontsize=14)
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'roc_curve.png', dpi=150)
plt.close()
print(f'[OK] Saved: {FIGURES_DIR / "roc_curve.png"}')


print('\n' + '='*70)
print('STEP 6: PRECISION/RECALL AT K')
print('='*70)

k_values = [5, 10, 15, 20, 30, 40]
prec_at_k = []
recall_at_k = []

print(f'{"K":>5} {"Precision":>12} {"Recall":>10} {"Lift":>8}')
print('-' * 40)

for k in k_values:
    top_k = np.argsort(y_proba)[-k:]
    prec = y_val.iloc[top_k].mean()
    rec = y_val.iloc[top_k].sum() / y_val.sum() if y_val.sum() > 0 else 0
    lift = prec / baseline_rate
    prec_at_k.append(prec)
    recall_at_k.append(rec)
    print(f'{k:>5} {prec:>11.1%} {rec:>9.1%} {lift:>7.2f}x')

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(k_values, [p*100 for p in prec_at_k], 'b-o', linewidth=2, markersize=8, label='Precision@K')
ax.plot(k_values, [r*100 for r in recall_at_k], 'g-s', linewidth=2, markersize=8, label='Recall@K')
ax.axhline(y=baseline_rate*100, color='r', linestyle='--', label=f'Baseline ({baseline_rate:.1%})')
ax.set_xlabel('K (Top K Predictions)', fontsize=12)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('Precision and Recall at K', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'precision_recall_at_k.png', dpi=150)
plt.close()
print(f'[OK] Saved: {FIGURES_DIR / "precision_recall_at_k.png"}')


print('\n' + '='*70)
print('STEP 7: CALIBRATION ANALYSIS')
print('='*70)

brier = brier_score_loss(y_val, y_proba)
print(f'Brier Score: {brier:.4f} (0=perfect, 0.25=random)')

prob_true, prob_pred = calibration_curve(y_val, y_proba, n_bins=5, strategy='uniform')

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(prob_pred, prob_true, 'b-o', linewidth=2, markersize=10, label='Model')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfectly Calibrated')
ax.set_xlabel('Mean Predicted Probability', fontsize=12)
ax.set_ylabel('Fraction of Positives', fontsize=12)
ax.set_title(f'Calibration Curve (Brier Score: {brier:.3f})', fontsize=14)
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'calibration_curve.png', dpi=150)
plt.close()
print(f'[OK] Saved: {FIGURES_DIR / "calibration_curve.png"}')


print('\n' + '='*70)
print('STEP 8: THRESHOLD ANALYSIS')
print('='*70)

thresholds = np.arange(0.1, 0.9, 0.1)
threshold_results = []

print(f'{"Threshold":>10} {"Predicted":>10} {"True Pos":>10} {"Precision":>10} {"Recall":>10}')
print('-' * 55)

for t in thresholds:
    pred = (y_proba >= t).astype(int)
    tp = ((pred == 1) & (y_val == 1)).sum()
    fp = ((pred == 1) & (y_val == 0)).sum()
    fn = ((pred == 0) & (y_val == 1)).sum()
    
    predicted = pred.sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f'{t:>10.1f} {predicted:>10} {tp:>10} {precision:>9.1%} {recall:>9.1%}')
    threshold_results.append({
        'threshold': t, 'predicted': predicted, 'true_positives': tp,
        'precision': precision, 'recall': recall
    })

pd.DataFrame(threshold_results).to_csv(MODELS_DIR / 'threshold_analysis.csv', index=False)


print('\n' + '='*70)
print('STEP 9: FINAL PREDICTIONS (2025 HOLDOUT)')
print('='*70)

holdout_proba = model.predict_proba(X_holdout_scaled)[:, 1]
predictions = holdout[['gsis_player_id', 'position_group']].copy()
predictions['pred_prob'] = holdout_proba
predictions = predictions.sort_values('pred_prob', ascending=False).reset_index(drop=True)
predictions['rank'] = range(1, len(predictions) + 1)

# Add names
predictions['gsis_player_id'] = predictions['gsis_player_id'].astype(str)
predictions = predictions.merge(
    master[['gsis_player_id', 'football_name', 'position']],
    on='gsis_player_id', how='left'
)
predictions = predictions[['rank', 'gsis_player_id', 'football_name', 'position', 'position_group', 'pred_prob']]

# Top 20
print('\nTop 20 Predictions:')
print(predictions.head(20).to_string(index=False))

# Top 2 per position
print('\n--- Top 2 Per Position Group ---')
for group in ['DB', 'SKILL', 'OL', 'DL', 'LB']:
    g_pred = predictions[predictions['position_group'] == group].head(2)
    print(f'\n{group}:')
    for _, row in g_pred.iterrows():
        print(f'  {row["rank"]:3}. {row["football_name"]:12} ({row["position"]:3}) - {row["pred_prob"]*100:.1f}%')

predictions.to_csv(MODELS_DIR / 'predictions_final.csv', index=False)
print(f'\n[OK] Saved: {MODELS_DIR / "predictions_final.csv"}')

print('\n' + '='*70)
print('PHASE 8-9: COMPLETE')
print('='*70)

