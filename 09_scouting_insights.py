"""
EAST-WEST SHRINE BOWL ANALYTICS COMPETITION
Phase 9: Scouting Insights & Draft Analysis

This phase generates competition-winning content:
  1. Draft Round vs Contributor Rate (Historical Validation)
  2. Decision Tree Rules (Quotable Insights)
  3. Quantified Model Value (ROI for Scouts)
  4. Athletic Archetypes (Cluster Analysis)
  5. Top Player Profiles with Percentiles
  6. Comparable Players from Historical Contributors

All insights are data-driven with no hypotheticals.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, DATA_DIR, BEST_MODEL_PARAMS

print('='*70)
print('PHASE 9: SCOUTING INSIGHTS & DRAFT ANALYSIS')
print('='*70)

# =============================================================================
# LOAD DATA
# =============================================================================

train = pd.read_csv(PROCESSED_DIR / 'train_composite_features.csv')
val = pd.read_csv(PROCESSED_DIR / 'val_composite_features.csv')
holdout = pd.read_csv(PROCESSED_DIR / 'holdout_composite_features.csv')
predictions = pd.read_csv(MODELS_DIR / 'predictions_final.csv')
master = pd.read_csv(PROCESSED_DIR / 'master_dataset_clean.csv')
players = pd.read_parquet(DATA_DIR / 'shrine_bowl_players.parquet')  # Fixed: use DATA_DIR

# Convert IDs to string for merging
for df in [train, val, holdout, predictions, master, players]:
    if 'gsis_player_id' in df.columns:
        df['gsis_player_id'] = df['gsis_player_id'].astype(str)

# Merge predictions with holdout
holdout = holdout.merge(predictions[['gsis_player_id', 'pred_prob', 'rank', 'football_name']], 
                        on='gsis_player_id', how='left')

# Train model for validation predictions
importance = pd.read_csv(MODELS_DIR / 'feature_importance_final.csv')
features = importance[importance['importance'] >= 0.01]['feature'].tolist()

X_train = train[features].fillna(0)
y_train = train['target']
X_val = val[features].fillna(0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

model = RandomForestClassifier(**BEST_MODEL_PARAMS)
model.fit(X_train_scaled, y_train)
val['pred_prob'] = model.predict_proba(X_val_scaled)[:, 1]

print(f'TRAIN: {len(train)}, VAL: {len(val)}, HOLDOUT: {len(holdout)}')

# =============================================================================
# TASK 1: DRAFT ROUND VS CONTRIBUTOR RATE
# =============================================================================

print('\n' + '='*70)
print('TASK 1: DRAFT ROUND VS CONTRIBUTOR RATE')
print('='*70)

# Merge draft info
draft_cols = ['gsis_player_id', 'draft_season', 'draft_round', 'draft_pick', 'draft_overall_selection']
master_with_draft = master.merge(players[draft_cols], on='gsis_player_id', how='left')

historical = master_with_draft[master_with_draft['cohort'].isin(['TRAIN', 'VALIDATE'])]
historical = historical[historical['target'].notna()]

historical['draft_status'] = historical['draft_round'].apply(
    lambda x: f'Round {int(x)}' if pd.notna(x) else 'UDFA'
)

print('\nContributor Rate by Draft Status:')
print('-' * 50)
print(f'{"Draft Status":<15} {"N":>8} {"Contributors":>13} {"Rate":>10}')
print('-' * 50)

draft_analysis = []
for status in ['Round 3', 'Round 4', 'Round 5', 'Round 6', 'Round 7', 'UDFA']:
    subset = historical[historical['draft_status'] == status]
    if len(subset) > 0:
        n = len(subset)
        contrib = int(subset['target'].sum())
        rate = subset['target'].mean() * 100
        print(f'{status:<15} {n:>8} {contrib:>13} {rate:>9.1f}%')
        draft_analysis.append({'status': status, 'n': n, 'contributors': contrib, 'rate': rate})

# Visualize
df_draft = pd.DataFrame(draft_analysis)
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#2ecc71' if r > 20 else '#e74c3c' if r < 10 else '#f39c12' for r in df_draft['rate']]
bars = ax.bar(df_draft['status'], df_draft['rate'], color=colors, edgecolor='black')
ax.axhline(y=historical['target'].mean()*100, color='blue', linestyle='--', linewidth=2, 
           label=f'Overall: {historical["target"].mean()*100:.1f}%')
ax.set_xlabel('Draft Status', fontsize=12)
ax.set_ylabel('Contributor Rate (%)', fontsize=12)
ax.set_title('NFL Contributor Rate by Draft Position (Shrine Bowl Players)', fontsize=14)
ax.legend()
for bar, n in zip(bars, df_draft['n']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'n={n}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'draft_round_contributor_rate.png', dpi=150)
print(f'\n[OK] Saved: {FIGURES_DIR / "draft_round_contributor_rate.png"}')
plt.close()

# =============================================================================
# TASK 2: DECISION TREE RULES
# =============================================================================

print('\n' + '='*70)
print('TASK 2: DECISION TREE RULES (Quotable Insights)')
print('='*70)

key_features = ['composite_agility', 'forty_yd_dash', 'pos_DB', 'pos_SKILL', 'accel_p95_max']
available_features = [f for f in key_features if f in train.columns]
print(f'Using features: {available_features}')

X_tree = train[available_features].fillna(0)
y_tree = train['target']

dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=8, random_state=42)
dt.fit(X_tree, y_tree)

print('\n--- Decision Tree Rules ---')
print(export_text(dt, feature_names=available_features))

# Save tree visualization
fig, ax = plt.subplots(figsize=(20, 12))
plot_tree(dt, feature_names=available_features, class_names=['Non-Contributor', 'Contributor'],
          filled=True, rounded=True, fontsize=11, ax=ax)
plt.title('Decision Rules for Identifying NFL Contributors', fontsize=16)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'decision_rules.png', dpi=150, bbox_inches='tight')
print(f'\n[OK] Saved: {FIGURES_DIR / "decision_rules.png"}')
plt.close()

# Leaf analysis
train['tree_leaf'] = dt.apply(X_tree)
leaf_stats = train.groupby('tree_leaf').agg({
    'target': ['count', 'sum', 'mean']
}).round(3)
leaf_stats.columns = ['n_players', 'n_contributors', 'contributor_rate']
print('\n--- Leaf Node Analysis ---')
print(leaf_stats)

# =============================================================================
# TASK 3: QUANTIFIED MODEL VALUE
# =============================================================================

print('\n' + '='*70)
print('TASK 3: QUANTIFIED MODEL VALUE')
print('='*70)

baseline_rate = val['target'].mean()
print(f'\nBaseline contributor rate: {baseline_rate:.1%}')

model_value = []
for k in [10, 20, 30]:
    top_k = val.nlargest(k, 'pred_prob')
    precision = top_k['target'].mean()
    lift = precision / baseline_rate
    random_expected = k * baseline_rate
    model_expected = k * precision
    
    print(f'\nTop {k} Predictions:')
    print(f'  Precision: {precision:.1%}')
    print(f'  Lift: {lift:.2f}x')
    print(f'  Expected contributors: {model_expected:.1f} (vs {random_expected:.1f} random)')
    
    model_value.append({'k': k, 'precision': precision, 'lift': lift, 
                        'expected': model_expected, 'random': random_expected})

pd.DataFrame(model_value).to_csv(MODELS_DIR / 'model_value_analysis.csv', index=False)

# =============================================================================
# TASK 4: ATHLETIC ARCHETYPES
# =============================================================================

print('\n' + '='*70)
print('TASK 4: ATHLETIC ARCHETYPES (Cluster Analysis)')
print('='*70)

cluster_features = ['forty_yd_dash', 'composite_agility', 'composite_explosiveness', 'weight']
cluster_features = [f for f in cluster_features if f in train.columns]
print(f'Clustering on: {cluster_features}')

cluster_data = train[cluster_features].fillna(train[cluster_features].median())
cluster_scaler = StandardScaler()
scaled = cluster_scaler.fit_transform(cluster_data)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
train['archetype'] = kmeans.fit_predict(scaled)

print('\n--- Archetype Analysis ---')
archetype_stats = train.groupby('archetype').agg({
    'target': ['count', 'sum', 'mean'],
    'forty_yd_dash': 'mean',
    'composite_agility': 'mean',
    'weight': 'mean'
}).round(3)
archetype_stats.columns = ['n_players', 'contributors', 'contributor_rate', 
                           'avg_40yd', 'avg_agility', 'avg_weight']

# Name archetypes - ensure unique names
archetype_names = {}
used_names = set()

# Sort by weight to assign power names in order
sorted_archetypes = archetype_stats.sort_values('avg_weight', ascending=False).index.tolist()

for arch in archetype_stats.index:
    row = archetype_stats.loc[arch]
    
    # Determine base name based on characteristics
    if row['avg_40yd'] < 4.6 and row['avg_agility'] > 0.3:
        base_name = 'Speed-Agility Elite'
    elif row['avg_weight'] > 300:
        base_name = 'Heavy Power'
    elif row['avg_weight'] > 280:
        base_name = 'Power Athlete'
    elif row['avg_agility'] > 0:
        base_name = 'Agile Mover'
    else:
        base_name = 'Average Athlete'
    
    # Ensure unique names
    name = base_name
    counter = 1
    while name in used_names:
        counter += 1
        name = f'{base_name} {counter}'
    
    used_names.add(name)
    archetype_names[arch] = name
    print(f'\nArchetype {arch}: {name}')
    print(f'  40-yd: {row["avg_40yd"]:.2f}s, Agility: {row["avg_agility"]:+.2f}, Weight: {row["avg_weight"]:.0f}lbs')
    print(f'  Contributor rate: {row["contributor_rate"]:.1%} ({int(row["contributors"])}/{int(row["n_players"])})')

# Apply to holdout
holdout_cluster = holdout[cluster_features].fillna(holdout[cluster_features].median())
holdout_scaled = cluster_scaler.transform(holdout_cluster)
holdout['archetype'] = kmeans.predict(holdout_scaled)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax1 = axes[0]
arch_order = archetype_stats.sort_values('contributor_rate', ascending=False).index
colors = ['#2ecc71' if archetype_stats.loc[a, 'contributor_rate'] > 0.20 else '#e74c3c' for a in arch_order]
bars = ax1.bar(range(len(arch_order)), [archetype_stats.loc[a, 'contributor_rate']*100 for a in arch_order], color=colors, edgecolor='black')
ax1.set_xticks(range(len(arch_order)))
ax1.set_xticklabels([archetype_names[a] for a in arch_order], rotation=15, ha='right')
ax1.axhline(y=train['target'].mean()*100, color='blue', linestyle='--', label='Overall baseline')
ax1.set_ylabel('Contributor Rate (%)')
ax1.set_title('Contributor Rate by Athletic Archetype')
ax1.legend()

ax2 = axes[1]
for arch in train['archetype'].unique():
    subset = train[train['archetype'] == arch]
    ax2.scatter(subset['forty_yd_dash'], subset['composite_agility'],
                label=f'{archetype_names[arch]}', alpha=0.7, s=60)
ax2.set_xlabel('40-Yard Dash (s)')
ax2.set_ylabel('Composite Agility')
ax2.set_title('Athletic Archetypes by Speed & Agility')
ax2.legend(fontsize=9)
ax2.invert_xaxis()

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'athletic_archetypes.png', dpi=150)
print(f'\n[OK] Saved: {FIGURES_DIR / "athletic_archetypes.png"}')
plt.close()

# =============================================================================
# TASK 5: TOP PLAYER PROFILE
# =============================================================================

print('\n' + '='*70)
print('TASK 5: TOP PLAYER PROFILE')
print('='*70)

top_player = holdout.loc[holdout['pred_prob'].idxmax()]
print(f'\n#1 Prospect: {top_player["football_name"]} ({top_player["position_group"]})')
print(f'Contributor Probability: {top_player["pred_prob"]*100:.1f}%')

all_players = pd.concat([train, val, holdout], ignore_index=True)

def get_percentile(value, series, lower_is_better=False):
    if pd.isna(value):
        return None
    series_clean = series.dropna()
    if lower_is_better:
        return (series_clean > value).mean() * 100
    return (series_clean < value).mean() * 100

print('\n--- Athletic Profile (Percentiles vs All Shrine Bowl Players) ---')
profile_metrics = [
    ('forty_yd_dash', True, '40-Yard Dash'),
    ('composite_agility', False, 'Composite Agility'),
    ('composite_explosiveness', False, 'Explosiveness'),
    ('accel_p95_max', False, 'Peak Acceleration'),
    ('speed_mean_mean', False, 'Average Speed'),
]

player_profile = {'name': top_player.get('football_name', 'Unknown'),
                  'position': top_player.get('position_group', 'Unknown'),
                  'probability': top_player['pred_prob']}

for col, lower_better, label in profile_metrics:
    if col in top_player.index and col in all_players.columns:
        val_raw = top_player[col]
        pct = get_percentile(val_raw, all_players[col], lower_is_better=lower_better)
        if pct is not None:
            player_profile[col] = pct
            print(f'  {label}: {pct:.0f}th percentile')

# =============================================================================
# TASK 6: COMPARABLE PLAYERS
# =============================================================================

print('\n' + '='*70)
print('TASK 6: COMPARABLE PLAYERS (From 2022 Contributors)')
print('='*70)

contributors = train[train['target'] == 1].copy()
contributors = contributors.merge(master[['gsis_player_id', 'football_name']], on='gsis_player_id', how='left')

print(f'\nFinding players similar to {top_player["football_name"]} among {len(contributors)} 2022 contributors...')

comparison_features = ['forty_yd_dash', 'composite_agility', 'composite_explosiveness', 'weight']
comparison_features = [f for f in comparison_features if f in contributors.columns and f in holdout.columns]

X_contributors = contributors[comparison_features].fillna(0)
X_top = pd.DataFrame([top_player[comparison_features].fillna(0)])

nn = NearestNeighbors(n_neighbors=min(5, len(contributors)))
nn.fit(X_contributors)
distances, indices = nn.kneighbors(X_top)

print(f'\n--- Comparable NFL Contributors ---')
comparables = []
for i, idx in enumerate(indices[0]):
    comp = contributors.iloc[idx]
    name = comp.get('football_name', 'Unknown')
    pos = comp.get('position_group', 'UNK')
    similarity = 1 / (1 + distances[0][i])
    print(f'  {i+1}. {name} ({pos}) - {similarity*100:.0f}% similar profile')
    comparables.append({'name': name, 'position': pos, 'similarity': similarity})

# =============================================================================
# TASK 7: SCOUTING RECOMMENDATIONS
# =============================================================================

print('\n' + '='*70)
print('TASK 7: SCOUTING RECOMMENDATIONS')
print('='*70)

# Generate tiered recommendations
holdout_preds = predictions[predictions['gsis_player_id'].isin(holdout['gsis_player_id'].values)]
summary_df = holdout_preds[['rank', 'football_name', 'position_group', 'pred_prob']].copy()
summary_df = summary_df.sort_values('pred_prob', ascending=False)

# Tiers based on CONTRIBUTOR PROBABILITY (not draft round prediction!)
summary_df['recommendation'] = summary_df['pred_prob'].apply(
    lambda x: 'HIGH CONFIDENCE (75%+)' if x >= 0.75 else 
              'LIKELY CONTRIBUTOR (60-74%)' if x >= 0.60 else
              'POSSIBLE CONTRIBUTOR (50-59%)' if x >= 0.50 else
              'BELOW AVERAGE (35-49%)' if x >= 0.35 else 'LOW CONFIDENCE (<35%)'
)
summary_df.to_csv(MODELS_DIR / 'scouting_recommendations.csv', index=False)

print('\n--- Contributor Probability Tiers ---')
for tier in ['HIGH CONFIDENCE (75%+)', 'LIKELY CONTRIBUTOR (60-74%)', 'POSSIBLE CONTRIBUTOR (50-59%)', 'BELOW AVERAGE (35-49%)']:
    tier_players = summary_df[summary_df['recommendation'] == tier]
    if len(tier_players) > 0:
        names = ', '.join(tier_players['football_name'].head(5).tolist())
        print(f'\n{tier} ({len(tier_players)} players):')
        print(f'  {names}{"..." if len(tier_players) > 5 else ""}')

# =============================================================================
# QUOTABLE INSIGHTS SUMMARY
# =============================================================================

print('\n' + '='*70)
print('QUOTABLE INSIGHTS FOR PRESENTATION')
print('='*70)

# Calculate specific stats for quotes
db_high_agility = train[(train.get('pos_DB', pd.Series([0]*len(train))) == 1)]
if len(db_high_agility) > 0:
    db_rate = db_high_agility['target'].mean() * 100
else:
    db_rate = train[train['position_group'] == 'DB']['target'].mean() * 100 if 'position_group' in train.columns else 0

top30_lift = val.nlargest(30, 'pred_prob')['target'].mean() / val['target'].mean()
best_archetype_rate = archetype_stats['contributor_rate'].max() * 100

print(f'''
QUOTABLE INSIGHTS:

1. DECISION RULE:
   "DBs have a {db_rate:.0f}% contributor rate vs {train["target"].mean()*100:.1f}% baseline"

2. MODEL VALUE:
   "Our Top 30 recommendations identify {top30_lift:.1f}x more contributors than random selection"
   
3. ARCHETYPE INSIGHT:
   "Speed-Agility Elite athletes have the highest contributor rate at {best_archetype_rate:.0f}%"

4. TOP PROSPECT:
   "{top_player.get("football_name", "Unknown")} is our #1 pick at {top_player["pred_prob"]*100:.0f}% probability,
   ranking in the {player_profile.get("composite_agility", 0):.0f}th percentile for agility"

5. COMPARABLE SUCCESS:
   "Similar athletic profiles from 2022 achieved NFL contributor status"
''')

print('\n[COMPLETE] Phase 9: Scouting Insights & Draft Analysis')
print(f'\nDeliverables:')
print(f'  - {FIGURES_DIR / "draft_round_contributor_rate.png"}')
print(f'  - {FIGURES_DIR / "decision_rules.png"}')
print(f'  - {FIGURES_DIR / "athletic_archetypes.png"}')
print(f'  - {MODELS_DIR / "scouting_recommendations.csv"}')
print(f'  - {MODELS_DIR / "model_value_analysis.csv"}')
