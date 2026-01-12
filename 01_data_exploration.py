"""
EAST-WEST SHRINE BOWL ANALYTICS COMPETITION
Phase 1: Exploratory Data Analysis & Data Preparation

This script:
  PART A: Analyzes RAW organizer-provided files
  PART B: Processes and creates master dataset
  PART C: Analyzes PROCESSED data

Target: >= 300 rookie snaps (HIGH-IMPACT contributor)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path(__file__).parent
PROCESSED_DIR = PROJECT_DIR / 'processed'
FIGURES_DIR = PROJECT_DIR / 'figures'
MODELS_DIR = PROJECT_DIR / 'models'

PROCESSED_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

SNAP_THRESHOLD = 300

print('='*70)
print('PHASE 1: EXPLORATORY DATA ANALYSIS & DATA PREPARATION')
print('='*70)
print(f'\nTarget: >= {SNAP_THRESHOLD} rookie snaps (HIGH-IMPACT contributor)')

# #############################################################################
#                      PART A: RAW DATA ANALYSIS
# #############################################################################

print('\n')
print('#' * 70)
print('#  PART A: RAW DATA ANALYSIS')
print('#' * 70)

# =============================================================================
# A1: LOAD RAW FILES
# =============================================================================

print('\n' + '='*70)
print('A1: LOADING RAW FILES')
print('='*70)

# 1. Player combine data
raw_players = pd.read_parquet(PROJECT_DIR / 'shrine_bowl_players.parquet')
print(f'\n[1] shrine_bowl_players.parquet: {len(raw_players)} players, {len(raw_players.columns)} columns')

# 2. College stats
raw_college = pd.read_csv(PROJECT_DIR / 'shrine_bowl_players_college_stats.csv')
print(f'[2] shrine_bowl_players_college_stats.csv: {len(raw_college)} records, {raw_college["college_gsis_id"].nunique()} players')

# 3. NFL rookie stats
raw_nfl = pd.read_csv(PROJECT_DIR / 'shrine_bowl_players_nfl_rookie_stats.csv')
print(f'[3] shrine_bowl_players_nfl_rookie_stats.csv: {len(raw_nfl)} records, {raw_nfl["college_gsis_id"].nunique()} players')

# 4. Session timestamps
raw_sessions = pd.read_csv(PROJECT_DIR / 'session_timestamps.csv')
print(f'[4] session_timestamps.csv: {len(raw_sessions)} sessions')

# =============================================================================
# A2: PLAYER COMBINE DATA ANALYSIS
# =============================================================================

print('\n' + '='*70)
print('A2: PLAYER COMBINE DATA')
print('='*70)

print(f'\nColumns: {list(raw_players.columns)}')

# Shrine Bowl years (from draft_season)
print(f'\n--- Shrine Bowl Years (draft_season) ---')
year_counts = raw_players['draft_season'].value_counts().sort_index()
print(year_counts)

# Combine metrics coverage
combine_cols = ['height', 'weight', 'forty_yd_dash', 'bench_reps_of_225',
                'standing_vertical', 'three_cone', 'twenty_yard_shuttle',
                'standing_broad_jump', 'hand_size', 'arm_length', 'wingspan']

print(f'\n--- Combine Metrics Coverage ---')
for col in combine_cols:
    if col in raw_players.columns:
        # Convert to numeric in case of string values
        col_numeric = pd.to_numeric(raw_players[col], errors='coerce')
        coverage = col_numeric.notna().mean() * 100
        mean_val = col_numeric.mean()
        print(f'  {col:<25} {coverage:5.1f}% coverage, mean: {mean_val:.2f}')

# =============================================================================
# A3: COLLEGE STATS ANALYSIS
# =============================================================================

print('\n' + '='*70)
print('A3: COLLEGE STATS')
print('='*70)

print(f'\nColumns: {list(raw_college.columns)[:15]}...')
print(f'\n--- Seasons ---')
print(raw_college['season'].value_counts().sort_index())
print(f'\n--- Positions ---')
print(raw_college['position'].value_counts().head(10))

# =============================================================================
# A4: NFL ROOKIE STATS ANALYSIS  
# =============================================================================

print('\n' + '='*70)
print('A4: NFL ROOKIE STATS')
print('='*70)

print(f'\nColumns: {list(raw_nfl.columns)[:15]}...')
print(f'\n--- Rookie Seasons ---')
print(raw_nfl['rookie_season'].value_counts().sort_index())

# Check for snaps
if 'total_snaps' in raw_nfl.columns:
    snaps = raw_nfl['total_snaps'].dropna()
    print(f'\n--- Total Snaps ---')
    print(f'  Players with data: {len(snaps)}')
    print(f'  Range: {snaps.min():.0f} - {snaps.max():.0f}')
    print(f'  Mean: {snaps.mean():.0f}, Median: {snaps.median():.0f}')
    print(f'  >= {SNAP_THRESHOLD} snaps: {(snaps >= SNAP_THRESHOLD).sum()} players')

# Draft status
print(f'\n--- Draft Status ---')
drafted = raw_nfl['draft_round'].notna().sum()
print(f'  Drafted: {drafted} ({drafted/len(raw_nfl)*100:.1f}%)')
print(f'  Undrafted: {len(raw_nfl) - drafted}')

# =============================================================================
# A5: RAW DATA VISUALIZATIONS
# =============================================================================

print('\n' + '='*70)
print('A5: RAW DATA VISUALIZATIONS')
print('='*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Players by year
ax1 = axes[0, 0]
years = raw_players['draft_season'].value_counts().sort_index()
ax1.bar(years.index, years.values, color=['#1e3a5f', '#4a90d9', '#27ae60'])
ax1.set_title('Players by Shrine Bowl Year (Raw)', fontweight='bold')
ax1.set_ylabel('Number of Players')
for i, (yr, cnt) in enumerate(years.items()):
    ax1.annotate(f'{cnt}', (yr, cnt), ha='center', va='bottom', fontsize=12)

# 2. Combine coverage
ax2 = axes[0, 1]
coverage = [(col, raw_players[col].notna().mean()*100) for col in combine_cols if col in raw_players.columns]
cols, pcts = zip(*coverage)
bars = ax2.barh(range(len(cols)), pcts, color='#4a90d9')
ax2.set_yticks(range(len(cols)))
ax2.set_yticklabels(cols)
ax2.set_xlabel('Coverage (%)')
ax2.set_title('Combine Metric Coverage (Raw)', fontweight='bold')
ax2.axvline(100, color='green', linestyle='--', alpha=0.5)

# 3. 40-yard dash
ax3 = axes[1, 0]
forty = pd.to_numeric(raw_players['forty_yd_dash'], errors='coerce').dropna()
ax3.hist(forty, bins=25, color='#27ae60', edgecolor='white', alpha=0.8)
ax3.set_xlabel('Time (seconds)')
ax3.set_title('40-Yard Dash Distribution (Raw)', fontweight='bold')
ax3.axvline(forty.mean(), color='red', linestyle='--', label=f'Mean: {forty.mean():.2f}s')
ax3.legend()

# 4. Position distribution
ax4 = axes[1, 1]
pos_counts = raw_college.groupby('college_gsis_id')['position'].first().value_counts().head(10)
ax4.barh(range(len(pos_counts)), pos_counts.values[::-1], color='#e74c3c')
ax4.set_yticks(range(len(pos_counts)))
ax4.set_yticklabels(pos_counts.index[::-1])
ax4.set_xlabel('Number of Players')
ax4.set_title('Top 10 Positions (Raw College)', fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'eda_raw_data.png', dpi=150, bbox_inches='tight')
print(f'[OK] Saved: {FIGURES_DIR / "eda_raw_data.png"}')
plt.close()

# #############################################################################
#                      PART B: DATA PROCESSING
# #############################################################################

print('\n')
print('#' * 70)
print('#  PART B: DATA PROCESSING')
print('#' * 70)

# =============================================================================
# B1: CREATE MASTER DATASET
# =============================================================================

print('\n' + '='*70)
print('B1: CREATING MASTER DATASET')
print('='*70)

# Start with player data
master = raw_players.copy()
master['gsis_player_id'] = master['gsis_player_id'].astype(str)

# Convert combine metrics to numeric (some may be strings)
for col in combine_cols:
    if col in master.columns:
        master[col] = pd.to_numeric(master[col], errors='coerce')

# Extract shrine bowl year from draft_season
master['shrine_bowl_year'] = pd.to_numeric(master['draft_season'], errors='coerce').astype('Int64')

print(f'[OK] Base players: {len(master)}')
print(f'     Years: {master["shrine_bowl_year"].value_counts().sort_index().to_dict()}')

# =============================================================================
# B2: ADD NFL OUTCOMES
# =============================================================================

print('\n' + '='*70)
print('B2: ADDING NFL OUTCOMES')
print('='*70)

# Prepare NFL data
nfl = raw_nfl.copy()
nfl['college_gsis_id'] = nfl['college_gsis_id'].astype(str)

# Rename to match master
nfl = nfl.rename(columns={'college_gsis_id': 'gsis_player_id'})

# Select columns to merge
nfl_cols = ['gsis_player_id', 'total_snaps', 'rookie_season', 'position']
nfl_merge = nfl[nfl_cols].drop_duplicates('gsis_player_id')

print(f'NFL players with data: {len(nfl_merge)}')

# Merge
master = master.merge(nfl_merge, on='gsis_player_id', how='left', suffixes=('', '_nfl'))

# Fill position from NFL if missing
if 'position_nfl' in master.columns:
    master['position'] = master.get('position', master['position_nfl'])
    if 'position' not in master.columns or master['position'].isna().all():
        master['position'] = master['position_nfl']
    else:
        master['position'] = master['position'].fillna(master['position_nfl'])
    master = master.drop(columns=['position_nfl'], errors='ignore')

# Also fill position from college stats where missing
college_pos = raw_college.groupby('college_gsis_id')['position'].first().reset_index()
college_pos['college_gsis_id'] = college_pos['college_gsis_id'].astype(str)
college_pos = college_pos.rename(columns={'college_gsis_id': 'gsis_player_id', 'position': 'position_college'})

master = master.merge(college_pos, on='gsis_player_id', how='left')
if 'position_college' in master.columns:
    master['position'] = master['position'].fillna(master['position_college'])
    master = master.drop(columns=['position_college'], errors='ignore')

print(f'After merge: {len(master)} players')
print(f'Players with NFL data: {master["total_snaps"].notna().sum()}')
print(f'Players with position: {master["position"].notna().sum()}')

# =============================================================================
# B3: CREATE TARGET VARIABLE
# =============================================================================

print('\n' + '='*70)
print('B3: CREATING TARGET VARIABLE')
print('='*70)

# First, assign cohorts so we can handle target differently by cohort
def assign_cohort(year):
    if year == 2022:
        return 'TRAIN'
    elif year == 2024:
        return 'VALIDATE'
    elif year == 2025:
        return 'HOLDOUT'
    else:
        return 'OTHER'

master['cohort'] = master['shrine_bowl_year'].apply(assign_cohort)

# Target: >= SNAP_THRESHOLD snaps
# For TRAIN and VALIDATE: missing NFL data = 0 snaps = non-contributor (target=0)
# Rationale: If not in NFL stats, they definitionally did NOT get 300+ NFL snaps
# For HOLDOUT: we don't know yet (target=NaN)

master['target'] = np.where(
    master['cohort'] == 'HOLDOUT',
    np.nan,  # HOLDOUT: unknown
    np.where(
        master['total_snaps'].notna(),
        (master['total_snaps'] >= SNAP_THRESHOLD).astype(float),
        0.0  # Missing = non-contributor (0 snaps)
    )
)

n_train_val = master[master['cohort'].isin(['TRAIN', 'VALIDATE'])]
n_contrib = int(n_train_val['target'].sum())
n_total = len(n_train_val)
print(f'TRAIN+VALIDATE players: {n_total}')
print(f'Contributors (>={SNAP_THRESHOLD} snaps): {n_contrib} ({n_contrib/n_total*100:.1f}%)')
print(f'Non-contributors: {n_total - n_contrib} ({(n_total-n_contrib)/n_total*100:.1f}%)')
print(f'  - Had <300 snaps: {n_train_val["total_snaps"].notna().sum() - n_contrib}')
print(f'  - Not in NFL stats: {n_train_val["total_snaps"].isna().sum()}')

# =============================================================================
# B4: COHORT SUMMARY
# =============================================================================

print('\n' + '='*70)
print('B4: COHORT SUMMARY')
print('='*70)

print('Cohort distribution:')
for cohort in ['TRAIN', 'VALIDATE', 'HOLDOUT']:
    sub = master[master['cohort'] == cohort]
    n_contrib = int(sub['target'].sum()) if sub['target'].notna().any() else 0
    n_with_target = sub['target'].notna().sum()
    if n_with_target > 0:
        rate = n_contrib / n_with_target * 100
        print(f'  {cohort}: {len(sub)} players, {n_contrib} contributors ({rate:.1f}%)')
    else:
        print(f'  {cohort}: {len(sub)} players, outcomes unknown')

# =============================================================================
# B5: SAVE PROCESSED DATA
# =============================================================================

print('\n' + '='*70)
print('B5: SAVING PROCESSED DATA')
print('='*70)

# Select and reorder columns
output_cols = [
    'gsis_player_id', 'football_name', 'first_name', 'last_name', 'position',
    'target', 'total_snaps', 'rookie_season', 'shrine_bowl_year', 'cohort',
    'height', 'weight', 'forty_yd_dash', 'bench_reps_of_225', 'standing_vertical',
    'three_cone', 'twenty_yard_shuttle', 'standing_broad_jump',
    'hand_size', 'arm_length', 'wingspan', 'team_name', 'team_code', 'conference'
]

# Only include columns that exist
available_cols = [col for col in output_cols if col in master.columns]
master_clean = master[available_cols].copy()

# Save master dataset
master_clean.to_csv(PROCESSED_DIR / 'master_dataset_clean.csv', index=False)
print(f'[OK] Saved: {PROCESSED_DIR / "master_dataset_clean.csv"}')
print(f'     Shape: {master_clean.shape}')

# Save processed versions of college and NFL stats (with renamed IDs)
raw_college_processed = raw_college.copy()
raw_college_processed = raw_college_processed.rename(columns={'college_gsis_id': 'gsis_player_id'})
raw_college_processed.to_csv(PROCESSED_DIR / 'processed_college_stats.csv', index=False)

raw_nfl_processed = raw_nfl.copy()
raw_nfl_processed = raw_nfl_processed.rename(columns={'college_gsis_id': 'gsis_player_id'})
raw_nfl_processed.to_csv(PROCESSED_DIR / 'processed_nfl_rookie_stats.csv', index=False)

print(f'[OK] Saved: {PROCESSED_DIR / "processed_college_stats.csv"}')
print(f'[OK] Saved: {PROCESSED_DIR / "processed_nfl_rookie_stats.csv"}')

# #############################################################################
#                      PART C: PROCESSED DATA ANALYSIS
# #############################################################################

print('\n')
print('#' * 70)
print('#  PART C: PROCESSED DATA ANALYSIS')
print('#' * 70)

# =============================================================================
# C1: PROCESSED DATA OVERVIEW
# =============================================================================

print('\n' + '='*70)
print('C1: PROCESSED DATA OVERVIEW')
print('='*70)

print(f'\nMaster dataset: {len(master_clean)} players, {len(master_clean.columns)} columns')
print(f'Columns: {list(master_clean.columns)}')

# =============================================================================
# C2: TARGET ANALYSIS
# =============================================================================

print('\n' + '='*70)
print('C2: TARGET VARIABLE ANALYSIS')
print('='*70)

trainable = master_clean[master_clean['target'].notna()]
contributors = trainable[trainable['target'] == 1]
non_contributors = trainable[trainable['target'] == 0]

print(f'\nTrainable players: {len(trainable)}')
print(f'  Contributors: {len(contributors)} ({len(contributors)/len(trainable)*100:.1f}%)')
print(f'  Non-contributors: {len(non_contributors)} ({len(non_contributors)/len(trainable)*100:.1f}%)')

if len(contributors) > 0:
    print(f'\nContributor snap statistics:')
    print(f'  Min: {contributors["total_snaps"].min():.0f}')
    print(f'  Max: {contributors["total_snaps"].max():.0f}')  
    print(f'  Mean: {contributors["total_snaps"].mean():.0f}')

# =============================================================================
# C3: COHORT ANALYSIS
# =============================================================================

print('\n' + '='*70)
print('C3: COHORT ANALYSIS')
print('='*70)

for cohort in ['TRAIN', 'VALIDATE', 'HOLDOUT']:
    sub = master_clean[master_clean['cohort'] == cohort]
    year = sub['shrine_bowl_year'].iloc[0] if len(sub) > 0 else 'N/A'
    n_contrib = int(sub['target'].sum()) if sub['target'].notna().any() else 0
    rate = sub['target'].mean() * 100 if sub['target'].notna().any() else 0
    print(f'{cohort} ({year}): {len(sub)} players, {n_contrib} contributors ({rate:.1f}%)')

# =============================================================================
# C4: FEATURE COVERAGE
# =============================================================================

print('\n' + '='*70)
print('C4: FEATURE COVERAGE')
print('='*70)

print('\n--- Combine Metrics ---')
for col in combine_cols:
    if col in master_clean.columns:
        cov = master_clean[col].notna().mean() * 100
        print(f'  {col:<25} {cov:.1f}%')

print(f'\n--- Position Coverage ---')
print(f'  Overall: {master_clean["position"].notna().mean()*100:.1f}%')
for cohort in ['TRAIN', 'VALIDATE', 'HOLDOUT']:
    sub = master_clean[master_clean['cohort'] == cohort]
    cov = sub['position'].notna().mean() * 100
    print(f'  {cohort}: {cov:.1f}%')

# =============================================================================
# C5: CONTRIBUTOR COMPARISON
# =============================================================================

print('\n' + '='*70)
print('C5: CONTRIBUTORS VS NON-CONTRIBUTORS')
print('='*70)

if len(contributors) > 0 and len(non_contributors) > 0:
    print('\nCombine metrics comparison:')
    for col in combine_cols:
        if col in trainable.columns:
            contrib_mean = contributors[col].mean()
            non_mean = non_contributors[col].mean()
            diff = contrib_mean - non_mean
            print(f'  {col:<25} Contrib: {contrib_mean:7.2f}, Non: {non_mean:7.2f}, Diff: {diff:+.2f}')

# =============================================================================
# C6: PROCESSED DATA VISUALIZATIONS
# =============================================================================

print('\n' + '='*70)
print('C6: PROCESSED DATA VISUALIZATIONS')
print('='*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Cohort distribution
ax1 = axes[0, 0]
cohort_order = ['TRAIN', 'VALIDATE', 'HOLDOUT']
cohort_counts = master_clean['cohort'].value_counts()[cohort_order]
colors = ['#1e3a5f', '#4a90d9', '#27ae60']
bars = ax1.bar(cohort_counts.index, cohort_counts.values, color=colors)
ax1.set_title('Players by Cohort (Processed)', fontweight='bold')
ax1.set_ylabel('Number of Players')
for bar, cnt in zip(bars, cohort_counts.values):
    ax1.annotate(f'{cnt}', (bar.get_x() + bar.get_width()/2, bar.get_height()), 
                 ha='center', va='bottom', fontsize=12)

# 2. Target distribution
ax2 = axes[0, 1]
target_counts = trainable['target'].value_counts().sort_index()
labels = [f'Non-contrib\n(<{SNAP_THRESHOLD})', f'Contrib\n(>={SNAP_THRESHOLD})']
colors_pie = ['#e74c3c', '#27ae60']
ax2.pie(target_counts.values, labels=labels, autopct='%1.1f%%', 
        colors=colors_pie, explode=(0, 0.05))
ax2.set_title('Target Distribution (Processed)', fontweight='bold')

# 3. Snap distribution
ax3 = axes[1, 0]
if 'total_snaps' in trainable.columns:
    snaps = trainable['total_snaps'].dropna()
    ax3.hist(snaps, bins=30, color='#4a90d9', edgecolor='white', alpha=0.8)
    ax3.axvline(SNAP_THRESHOLD, color='red', linestyle='--', linewidth=2,
                label=f'Threshold: {SNAP_THRESHOLD}')
    ax3.set_xlabel('Total Rookie Snaps')
    ax3.set_ylabel('Number of Players')
    ax3.set_title('Rookie Snap Distribution (Processed)', fontweight='bold')
    ax3.legend()

# 4. Contributor rate by cohort
ax4 = axes[1, 1]
rates = trainable.groupby('cohort')['target'].mean() * 100
cohorts_with_rates = [c for c in ['TRAIN', 'VALIDATE'] if c in rates.index]
bars = ax4.bar(cohorts_with_rates, [rates[c] for c in cohorts_with_rates], 
               color=['#1e3a5f', '#4a90d9'][:len(cohorts_with_rates)])
ax4.axhline(trainable['target'].mean()*100, color='red', linestyle='--',
            label=f'Overall: {trainable["target"].mean()*100:.1f}%')
ax4.set_ylabel('Contributor Rate (%)')
ax4.set_title('Contributor Rate by Cohort (Processed)', fontweight='bold')
ax4.legend()
for bar, cohort in zip(bars, cohorts_with_rates):
    ax4.annotate(f'{rates[cohort]:.1f}%', 
                 (bar.get_x() + bar.get_width()/2, bar.get_height()),
                 ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'eda_processed_data.png', dpi=150, bbox_inches='tight')
print(f'[OK] Saved: {FIGURES_DIR / "eda_processed_data.png"}')
plt.close()

# =============================================================================
# C7: DATA QUALITY CHECKS
# =============================================================================

print('\n' + '='*70)
print('C7: DATA QUALITY CHECKS')
print('='*70)

checks = [
    ('All players have ID', master_clean['gsis_player_id'].notna().all()),
    ('All players have cohort', master_clean['cohort'].notna().all()),
    ('TRAIN/VAL have target', trainable['target'].notna().all()),
    ('HOLDOUT has no target', master_clean[master_clean['cohort']=='HOLDOUT']['target'].isna().all()),
    ('Target is binary (0/1)', trainable['target'].isin([0, 1]).all()),
    ('Valid cohorts only', master_clean['cohort'].isin(['TRAIN', 'VALIDATE', 'HOLDOUT']).all()),
]

all_passed = True
for name, result in checks:
    status = 'PASS' if result else 'FAIL'
    print(f'  [{status}] {name}')
    if not result:
        all_passed = False

# =============================================================================
# SUMMARY
# =============================================================================

print('\n' + '='*70)
print('PHASE 1: COMPLETE')
print('='*70)

print(f'''
SUMMARY
=======
RAW DATA ANALYZED:
  - shrine_bowl_players.parquet: {len(raw_players)} players
  - shrine_bowl_players_college_stats.csv: {len(raw_college)} records
  - shrine_bowl_players_nfl_rookie_stats.csv: {len(raw_nfl)} records

PROCESSED DATA CREATED:
  - master_dataset_clean.csv: {len(master_clean)} players
  - processed_college_stats.csv
  - processed_nfl_rookie_stats.csv

TARGET VARIABLE (>={SNAP_THRESHOLD} snaps):
  - Contributors: {len(contributors)} ({len(contributors)/len(trainable)*100:.1f}%)
  - Non-contributors: {len(non_contributors)} ({len(non_contributors)/len(trainable)*100:.1f}%)
  - Min contributor snaps: {int(contributors["total_snaps"].min()) if len(contributors) > 0 else "N/A"}

COHORTS:
  - TRAIN (2022): {len(master_clean[master_clean["cohort"]=="TRAIN"])} players
  - VALIDATE (2024): {len(master_clean[master_clean["cohort"]=="VALIDATE"])} players
  - HOLDOUT (2025): {len(master_clean[master_clean["cohort"]=="HOLDOUT"])} players

VISUALIZATIONS:
  - {FIGURES_DIR / "eda_raw_data.png"}
  - {FIGURES_DIR / "eda_processed_data.png"}

DATA QUALITY: {"ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED"}

Next: Run phase03_baseline_model.py
''')
