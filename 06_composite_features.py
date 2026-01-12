"""
EAST-WEST SHRINE BOWL ANALYTICS COMPETITION
Phase 6: Composite Scores & Interactions

Creates advanced features:
  6.1 Composite scores (explosiveness, agility, size_weight_speed)
  6.2 Selective interactions (8 key position x metric)
  6.3 Evaluation

Expected AUC: 0.700-0.710
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from config import (
    PROCESSED_DIR, MODELS_DIR, FIGURES_DIR,
    EXCLUDE_COLS, COMBINE_COLS, ZSCORE_METRICS,
    POSITION_GROUPS, KEY_INTERACTIONS,
    COMPOSITE_DEFINITIONS, BEST_MODEL_PARAMS
)

print('='*70)
print('PHASE 6: COMPOSITE SCORES & INTERACTIONS')
print('='*70)


print('\n' + '='*70)
print('STEP 1: LOADING POSITION-ENHANCED DATA')
print('='*70)

train = pd.read_csv(PROCESSED_DIR / 'train_position_features.csv')
val = pd.read_csv(PROCESSED_DIR / 'val_position_features.csv')
holdout = pd.read_csv(PROCESSED_DIR / 'holdout_position_features.csv')

print(f'[OK] TRAIN: {len(train)}, VAL: {len(val)}, HOLDOUT: {len(holdout)}')

# Get existing feature columns
existing_features = [col for col in train.columns 
                     if col not in ['gsis_player_id', 'target', 'position_group']]
print(f'Existing features: {len(existing_features)}')


print('\n' + '='*70)
print('STEP 2: COMPOSITE SCORES')
print('='*70)

def create_composite_scores(df, fit_stats=None):
    """Create composite score features. Returns df and fit statistics."""
    stats = {} if fit_stats is None else fit_stats
    
    # 1. Explosiveness: normalized combination of vertical + broad jump
    metrics = ['standing_vertical', 'standing_broad_jump']
    if all(m in df.columns for m in metrics):
        if fit_stats is None:
            # Fit on this data (should be TRAIN)
            stats['explosiveness'] = {
                'vertical_mean': df['standing_vertical'].mean(),
                'vertical_std': df['standing_vertical'].std(),
                'broad_mean': df['standing_broad_jump'].mean(),
                'broad_std': df['standing_broad_jump'].std()
            }
        
        s = stats['explosiveness']
        vert_z = (df['standing_vertical'] - s['vertical_mean']) / s['vertical_std']
        broad_z = (df['standing_broad_jump'] - s['broad_mean']) / s['broad_std']
        df['composite_explosiveness'] = (vert_z + broad_z) / 2
    
    # 2. Agility: normalized combination of three_cone + shuttle (inverted)
    metrics = ['three_cone', 'twenty_yard_shuttle']
    if all(m in df.columns for m in metrics):
        if fit_stats is None:
            stats['agility'] = {
                'cone_mean': df['three_cone'].mean(),
                'cone_std': df['three_cone'].std(),
                'shuttle_mean': df['twenty_yard_shuttle'].mean(),
                'shuttle_std': df['twenty_yard_shuttle'].std()
            }
        
        s = stats['agility']
        cone_z = (df['three_cone'] - s['cone_mean']) / s['cone_std']
        shuttle_z = (df['twenty_yard_shuttle'] - s['shuttle_mean']) / s['shuttle_std']
        # Invert: lower is better, so negate
        df['composite_agility'] = -1 * (cone_z + shuttle_z) / 2
    
    # 3. Size-Weight-Speed: weight / forty_yd_dash (bigger AND faster = better)
    metrics = ['weight', 'forty_yd_dash']
    if all(m in df.columns for m in metrics):
        if fit_stats is None:
            sws = df['weight'] / df['forty_yd_dash'].replace(0, np.nan)
            stats['sws'] = {
                'mean': sws.mean(),
                'std': sws.std()
            }
        
        s = stats['sws']
        raw_sws = df['weight'] / df['forty_yd_dash'].replace(0, np.nan)
        df['composite_size_weight_speed'] = (raw_sws - s['mean']) / s['std']
    
    return df, stats

# Fit composite stats on TRAIN, apply to all
train, composite_stats = create_composite_scores(train, fit_stats=None)
val, _ = create_composite_scores(val, fit_stats=composite_stats)
holdout, _ = create_composite_scores(holdout, fit_stats=composite_stats)

composite_cols = ['composite_explosiveness', 'composite_agility', 'composite_size_weight_speed']
composite_cols = [c for c in composite_cols if c in train.columns]

print(f'[OK] Created {len(composite_cols)} composite features:')
for col in composite_cols:
    print(f'  - {col}')

# Show composite stats
print('\nComposite score statistics (TRAIN):')
for col in composite_cols:
    print(f'  {col}: mean={train[col].mean():.3f}, std={train[col].std():.3f}')


print('\n' + '='*70)
print('STEP 3: SELECTIVE INTERACTIONS (8 key position x metric)')
print('='*70)

def create_interactions(df, fit_stats=None):
    """Create position x metric interaction features."""
    stats = {} if fit_stats is None else fit_stats
    
    for pos_group, metric in KEY_INTERACTIONS:
        # Check if position one-hot and metric exist
        pos_col = f'pos_{pos_group}'
        if pos_col not in df.columns:
            print(f'  [SKIP] {pos_col} not found')
            continue
        if metric not in df.columns:
            print(f'  [SKIP] {metric} not found')
            continue
        
        # Create interaction column name
        interaction_col = f'int_{pos_group}_{metric}'
        
        # Normalize metric using TRAIN stats
        if fit_stats is None:
            stats[interaction_col] = {
                'mean': df[metric].mean(),
                'std': df[metric].std() if df[metric].std() > 0 else 1
            }
        
        s = stats[interaction_col]
        metric_normalized = (df[metric] - s['mean']) / s['std']
        
        # Interaction = position_indicator * normalized_metric
        df[interaction_col] = df[pos_col] * metric_normalized
    
    return df, stats

# Fit on TRAIN, apply to all
train, interaction_stats = create_interactions(train, fit_stats=None)
val, _ = create_interactions(val, fit_stats=interaction_stats)
holdout, _ = create_interactions(holdout, fit_stats=interaction_stats)

interaction_cols = [f'int_{pg}_{m}' for pg, m in KEY_INTERACTIONS if f'int_{pg}_{m}' in train.columns]
print(f'\n[OK] Created {len(interaction_cols)} interaction features:')
for col in interaction_cols:
    print(f'  - {col}')


print('\n' + '='*70)
print('STEP 4: PREPARING FEATURES')
print('='*70)

# Combine all features
new_features = composite_cols + interaction_cols
all_features = existing_features + new_features

# Filter to columns that exist
all_features = [col for col in all_features if col in train.columns]
all_features = list(dict.fromkeys(all_features))  # Remove duplicates

print(f'Feature breakdown:')
print(f'  Existing features: {len(existing_features)}')
print(f'  Composite features: {len(composite_cols)}')
print(f'  Interaction features: {len(interaction_cols)}')
print(f'  TOTAL: {len(all_features)}')

X_train = train[all_features].copy()
y_train = train['target'].copy()
X_val = val[all_features].copy()
y_val = val['target'].copy()
X_holdout = holdout[all_features].copy()

# Handle missing values
for df in [X_train, X_val, X_holdout]:
    df.fillna(0, inplace=True)


print('\n' + '='*70)
print('STEP 5: TRAINING AND EVALUATION')
print('='*70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

model = RandomForestClassifier(**BEST_MODEL_PARAMS)
model.fit(X_train_scaled, y_train)

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


print('\n' + '='*70)
print('STEP 6: FEATURE IMPORTANCE')
print('='*70)

importance = pd.DataFrame({
    'feature': all_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print('\nTop 20 Features:')
print(importance.head(20).to_string(index=False))

# Importance by category
composite_imp = importance[importance['feature'].isin(composite_cols)]['importance'].sum()
interaction_imp = importance[importance['feature'].isin(interaction_cols)]['importance'].sum()
other_imp = 1 - composite_imp - interaction_imp

print(f'\nImportance by category:')
print(f'  Composite scores: {composite_imp*100:.1f}%')
print(f'  Interactions: {interaction_imp*100:.1f}%')
print(f'  Other: {other_imp*100:.1f}%')


print('\n' + '='*70)
print('STEP 7: SAVING ENHANCED DATA')
print('='*70)

# Save for next phase
train_out = train[['gsis_player_id', 'target', 'position_group'] + all_features]
val_out = val[['gsis_player_id', 'target', 'position_group'] + all_features]
holdout_out = holdout[['gsis_player_id', 'position_group'] + all_features]

train_out.to_csv(PROCESSED_DIR / 'train_composite_features.csv', index=False)
val_out.to_csv(PROCESSED_DIR / 'val_composite_features.csv', index=False)
holdout_out.to_csv(PROCESSED_DIR / 'holdout_composite_features.csv', index=False)

print(f'[OK] Saved enhanced datasets')

importance.to_csv(MODELS_DIR / 'feature_importance_phase6.csv', index=False)

print('\n' + '='*70)
print('PHASE 6: COMPLETE')
print('='*70)

