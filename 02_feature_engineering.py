"""
EAST-WEST SHRINE BOWL ANALYTICS COMPETITION
Phase 2: Feature Engineering (Tracking Data)

Uses Polars for efficient processing of large parquet files.
Extracts player-level features from practice/game tracking data.
"""

import polars as pl
import pandas as pd
from pathlib import Path
import time


PROJECT_DIR = Path(__file__).parent
PROCESSED_DIR = PROJECT_DIR / 'processed'
PRACTICE_DIR = PROJECT_DIR / 'practice_data'
GAME_DIR = PROJECT_DIR / 'game_data'

PROCESSED_DIR.mkdir(exist_ok=True)

print('='*70)
print('PHASE 2: FEATURE ENGINEERING (TRACKING DATA)')
print('='*70)
print(f'\nUsing Polars for efficient processing')


print('\n' + '='*70)
print('STEP 1: FINDING TRACKING FILES')
print('='*70)

practice_files = sorted(PRACTICE_DIR.glob('*.parquet'))
game_files = sorted(GAME_DIR.glob('*.parquet'))

print(f'\nPractice files: {len(practice_files)}')
for f in practice_files:
    size_gb = f.stat().st_size / (1024**3)
    print(f'  {f.name}: {size_gb:.2f} GB')

print(f'\nGame files: {len(game_files)}')
for f in game_files:
    size_gb = f.stat().st_size / (1024**3)
    print(f'  {f.name}: {size_gb:.2f} GB')

all_files = practice_files + game_files
total_size_gb = sum(f.stat().st_size for f in all_files) / (1024**3)
print(f'\nTotal: {len(all_files)} files, {total_size_gb:.2f} GB')


def extract_features_from_file(file_path):
    """Extract player-level features from a single tracking file using Polars."""
    
    try:
        # Read parquet with lazy evaluation
        lf = pl.scan_parquet(file_path)
        
        # Check if gsis_id column exists
        schema = lf.collect_schema()
        if 'gsis_id' not in schema.names():
            print(f'    [SKIP] No gsis_id column')
            return None
        
        # Filter to players only (not ball)
        if 'entity_type' in schema.names():
            lf = lf.filter(pl.col('entity_type') == 'player')
        
        # Extract features per player
        features = lf.group_by('gsis_id').agg([
            # Speed features
            pl.col('s').max().alias('speed_max'),
            pl.col('s').mean().alias('speed_mean'),
            pl.col('s').std().alias('speed_std'),
            pl.col('s').quantile(0.5).alias('speed_p50'),
            pl.col('s').quantile(0.75).alias('speed_p75'),
            pl.col('s').quantile(0.95).alias('speed_p95'),
            
            # Acceleration features
            pl.col('a').max().alias('accel_max'),
            pl.col('a').mean().alias('accel_mean'),
            pl.col('a').std().alias('accel_std'),
            pl.col('a').quantile(0.95).alias('accel_p95'),
            
            # Movement features
            pl.col('dis').sum().alias('total_distance'),
            pl.col('dis').mean().alias('avg_distance_per_frame'),
            
            # Direction changes
            pl.col('dir').std().alias('dir_change_std'),
            
            # Count of frames
            pl.len().alias('total_frames'),
        ]).collect()
        
        return features
        
    except Exception as e:
        print(f'    [ERROR] {e}')
        return None


print('\n' + '='*70)
print('STEP 2: EXTRACTING FEATURES FROM TRACKING DATA')
print('='*70)

all_features = []
start_time = time.time()

for i, file_path in enumerate(all_files, 1):
    print(f'\n[{i}/{len(all_files)}] Processing {file_path.name}...')
    file_start = time.time()
    
    features = extract_features_from_file(file_path)
    
    if features is not None and len(features) > 0:
        all_features.append(features)
        print(f'    [OK] {len(features)} players, {time.time() - file_start:.1f}s')
    else:
        print(f'    [SKIP] No features extracted')

print(f'\n[OK] Processed {len(all_features)}/{len(all_files)} files in {time.time() - start_time:.1f}s')


print('\n' + '='*70)
print('STEP 3: AGGREGATING FEATURES')
print('='*70)

if len(all_features) == 0:
    print('[ERROR] No features extracted from any file!')
    exit(1)

# Concatenate all dataframes
combined = pl.concat(all_features)
print(f'Combined records: {len(combined)}')
print(f'Unique players: {combined["gsis_id"].n_unique()}')

# Aggregate across all sessions per player
player_features = combined.group_by('gsis_id').agg([
    # Speed
    pl.col('speed_max').max().alias('speed_max_max'),
    pl.col('speed_mean').mean().alias('speed_mean_mean'),
    pl.col('speed_std').mean().alias('speed_std_mean'),
    pl.col('speed_p50').mean().alias('speed_p50_mean'),
    pl.col('speed_p75').max().alias('speed_p75_max'),
    pl.col('speed_p95').max().alias('speed_p95_max'),
    
    # Acceleration
    pl.col('accel_max').max().alias('accel_max_max'),
    pl.col('accel_mean').mean().alias('accel_mean_mean'),
    pl.col('accel_std').mean().alias('accel_std_mean'),
    pl.col('accel_p95').max().alias('accel_p95_max'),
    
    # Movement
    pl.col('total_distance').sum().alias('total_distance_sum'),
    pl.col('avg_distance_per_frame').mean().alias('avg_distance_per_frame_mean'),
    
    # Direction
    pl.col('dir_change_std').mean().alias('dir_change_avg_mean'),
    pl.col('dir_change_std').max().alias('dir_change_max_max'),
    
    # Activity
    pl.col('total_frames').sum().alias('total_frames_sum'),
])

# Calculate derived features
player_features = player_features.with_columns([
    # Work efficiency
    (pl.col('total_distance_sum') / pl.col('total_frames_sum')).fill_null(0).alias('work_efficiency'),
    
    # Speed consistency (inverse of std/mean ratio)
    (pl.col('speed_mean_mean') / pl.col('speed_std_mean').fill_null(1).clip(0.01, None)).alias('speed_consistency'),
    
    # High-intensity ratio
    (pl.col('speed_p95_max') / pl.col('speed_mean_mean').fill_null(1).clip(0.01, None)).alias('intensity_ratio'),
])

print(f'\n[OK] Final player features: {len(player_features)} players, {len(player_features.columns)} features')


print('\n' + '='*70)
print('STEP 4: SAVING TRACKING FEATURES')
print('='*70)

# Convert to pandas and save
tracking_df = player_features.to_pandas()

# Clean gsis_id (remove .0 suffix from float conversion)
tracking_df['gsis_id'] = tracking_df['gsis_id'].astype(str).str.replace('.0', '', regex=False)

tracking_df.to_csv(PROCESSED_DIR / 'tracking_features.csv', index=False)

print(f'[OK] Saved: {PROCESSED_DIR / "tracking_features.csv"}')
print(f'     Shape: {tracking_df.shape}')


print('\n' + '='*70)
print('STEP 5: FEATURE SUMMARY')
print('='*70)

print('\nTracking features created:')
for col in tracking_df.columns:
    if col != 'gsis_id':
        mean_val = tracking_df[col].mean()
        std_val = tracking_df[col].std()
        print(f'  {col:<30} mean: {mean_val:8.2f}, std: {std_val:8.2f}')

print('\n' + '='*70)
print('PHASE 2: COMPLETE')
print('='*70)

