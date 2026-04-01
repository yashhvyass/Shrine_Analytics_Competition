# East-West Shrine Bowl Analytics Competition
Predicting which East-West Shrine Bowl participants will become high-impact NFL contributors (≥300 rookie snaps) using combine measurements, practice/game tracking data, college production stats, and position-aware feature engineering.
Final Model: Validation AUC = 0.743 (95% CI: 0.603–0.865)

## Problem Statement
The East-West Shrine Bowl showcases college football prospects for NFL scouts each year. Given historical player data from the 2022 and 2024 Shrine Bowls, the goal is to predict which 2025 participants are most likely to earn ≥300 rookie snaps — a threshold indicating meaningful NFL contribution.
This is a binary classification problem with significant class imbalance (contributors represent a minority of participants) and limited sample sizes across just three cohort years.
## Approach
The project follows a 9-phase pipeline, each building on the previous:
PhaseScriptPurpose101_data_exploration.pyEDA on raw data, master dataset creation, data quality checks202_feature_engineering.pyTracking data feature extraction (speed, acceleration, movement) using Polars303_baseline_model.pyBaseline model with combine metrics only (Logistic Regression vs Random Forest)404_tracking_features.pyEnhanced model combining combine + tracking features505_position_features.pyPosition groups, z-scores, threshold pass/fail features606_composite_features.pyComposite scores (explosiveness, agility, size-weight-speed) and position×metric interactions707_feature_selection.pyFeature importance filtering (<1% importance dropped), calibration checks808_final_model.pyFull evaluation suite: SHAP analysis, bootstrap CIs, ROC, precision/recall, threshold tuning, final predictions909_scouting_insights.pyDraft analysis, decision tree rules, athletic archetypes (K-Means), player comparables, scouting recommendations
## Data Sources
The competition provides four raw data files (not included in this repo):

shrine_bowl_players.parquet — Player demographics and combine metrics (height, weight, 40-yard dash, bench press, vertical, 3-cone, shuttle, broad jump, hand size, arm length, wingspan)
shrine_bowl_players_college_stats.csv — Multi-season college statistics (rushing, receiving, passing, defense)
shrine_bowl_players_nfl_rookie_stats.csv — NFL rookie outcomes including total snaps (the target variable)
session_timestamps.csv — Practice/game session metadata
practice_data/ and game_data/ — Player tracking parquet files with frame-level speed, acceleration, distance, and direction

## Cohort Structure
CohortYearRoleTRAIN2022Model training (known outcomes)VALIDATE2024Model evaluation (known outcomes)HOLDOUT2025Final predictions (outcomes unknown)
## Feature Engineering
The final model uses 39 features across five categories:
Combine Metrics (11) — Raw physical measurements: 40-yard dash, 3-cone, shuttle, vertical, broad jump, bench press, height, weight, hand size, arm length, wingspan.
Tracking Features (15) — Extracted from practice/game tracking data using Polars for efficient large-file processing. Includes max/mean/percentile speed, peak/mean acceleration, total distance, direction change variability, work efficiency, speed consistency, and intensity ratio.
Position Features (13) — Position group one-hot encoding (DB, SKILL, OL, DL, LB), position-normalized z-scores for key combine metrics, and threshold pass/fail indicators.
Composite Scores (3) — Derived metrics combining multiple measurements: explosiveness (vertical + broad jump + acceleration), agility (3-cone + shuttle + direction change), and size-weight-speed index.
Position × Metric Interactions (8) — Selective interaction terms (e.g., DB × 40-yard dash, OL × bench press) capturing position-specific athletic value.
College Production Score (1) — Position-normalized college stats (receiving/rushing yards for SKILL, interceptions/tackles for DB, sacks/tackles for DL/LB, experience for OL).
Model
## Algorithm: Random Forest Classifier with balanced class weights
Hyperparameters:

75 estimators, max depth 8, min samples split 15, min samples leaf 7
max_features='sqrt', class_weight='balanced'

## Feature selection: Features with <1% importance dropped after initial training.
Key performance metrics:

Validation AUC: 0.743 (95% CI: 0.603–0.865 via bootstrap)
Calibration verified across position groups
SHAP analysis confirms feature importance alignment

## Top Features by Importance

40-yard dash (7.2%)
Composite agility (5.6%)
Peak acceleration (4.8%)
Composite explosiveness (4.6%)
Average speed (4.3%)

## Scouting Outputs
The final phase generates actionable scouting deliverables:

Tiered Recommendations — Players bucketed into HIGH CONFIDENCE (75%+), LIKELY CONTRIBUTOR (60–74%), POSSIBLE CONTRIBUTOR (50–59%), and BELOW AVERAGE (<50%)
Decision Tree Rules — Interpretable 3-level decision tree for quick screening
Athletic Archetypes — K-Means clustering (4 clusters) identifying Speed-Agility Elite, Power Athletes, Agile Movers, etc.
Player Comparables — Nearest-neighbor matching against historical contributors
Draft Round Analysis — Contributor rate by draft position for ROI context

## Repository Structure

├── 01_data_exploration.py          # EDA and master dataset creation
├── 02_feature_engineering.py       # Tracking data feature extraction (Polars)
├── 03_baseline_model.py            # Baseline model (combine-only)
├── 04_tracking_features.py         # Combined model (combine + tracking)
├── 05_position_features.py         # Position-aware features
├── 06_composite_features.py        # Composite scores and interactions
├── 07_feature_selection.py         # Feature selection and calibration
├── 08_final_model.py               # Full evaluation (SHAP, bootstrap, ROC, predictions)
├── 09_scouting_insights.py         # Scouting deliverables and archetypes
├── processed/                      # Intermediate datasets
│   ├── master_dataset_clean.csv
│   ├── tracking_features.csv
│   ├── train_*.csv / val_*.csv / holdout_*.csv
│   └── processed_college_stats.csv
├── models/                         # Model outputs
│   ├── predictions_2025_FINAL.csv
│   ├── scouting_recommendations.csv
│   ├── feature_importance_final.csv
│   ├── confidence_intervals.csv
│   └── threshold_analysis.csv
├── figures/                        # Visualizations
│   ├── shap_summary.png
│   ├── roc_curve.png
│   ├── calibration_curve.png
│   ├── athletic_archetypes.png
│   ├── decision_rules.png
│   ├── draft_round_contributor_rate.png
│   └── ...
├── LICENSE
└── README.md

## Requirements
Python 3.9+
pandas, numpy, scikit-learn, matplotlib
polars (for tracking data processing)
shap (for model interpretability)

## Install dependencies:
bashpip install pandas numpy scikit-learn matplotlib polars shap
Usage
Run scripts sequentially — each phase depends on outputs from the previous:
bashpython 01_data_exploration.py
python 02_feature_engineering.py
python 03_baseline_model.py
python 04_tracking_features.py
python 05_position_features.py
python 06_composite_features.py
python 07_feature_selection.py
python 08_final_model.py
python 09_scouting_insights.py

## Note: 
Raw data files (shrine_bowl_players.parquet, tracking data directories, etc.) are not included in this repo. Place them in the project root before running. Phases 5–9 also require a config.py file with shared constants (position mappings, model params, directory paths).
