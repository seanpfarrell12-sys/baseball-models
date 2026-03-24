"""
=============================================================================
MONEYLINE MODEL — FILE 3 OF 4: MODEL TRAINING AND SCORING  (REFACTORED)
=============================================================================
Trains an XGBoost classifier on the refactored feature set using
walk-forward cross-validation to simulate real-world deployment.

Walk-forward CV folds:
  Fold 1 : Train = 2023        →  Test = 2024
  Fold 2 : Train = 2023+2024   →  Test = 2025
  Final  : Train = all seasons →  Full model for 2026 scoring

Model: XGBoost with isotonic probability calibration (CalibratedClassifierCV)
  - Regularization: reg_alpha (L1), reg_lambda (L2), min_child_weight
  - Objective: binary:logistic
  - Calibration: isotonic regression on a holdout set

Evaluation:
  AUC-ROC   : discrimination ability (target > 0.54)
  Log Loss  : probability calibration (target < 0.685)
  Brier     : mean squared error of probabilities (target < 0.248)
  Accuracy  : % games predicted correctly (target > 0.55)

Input  : data/processed/moneyline_dataset.csv
Output : models/moneyline_model.json
         models/moneyline_features.json
         models/moneyline_metrics.json
         models/moneyline_feature_importance.csv
=============================================================================
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.calibration   import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, log_loss, brier_score_loss, accuracy_score
)

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
EXPORT_DIR = os.path.join(BASE_DIR, "exports")

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)


# =============================================================================
# LOAD DATA
# =============================================================================
def load_data(path: str) -> tuple:
    df = pd.read_csv(path)
    print(f"  {len(df):,} games loaded across seasons: {sorted(df['season'].unique())}")
    print(f"  Home win rate: {df['home_win'].mean():.3f}")

    exclude = {"game_date", "season", "home_team", "away_team",
               "home_win", "total_runs", "feature_season"}
    feature_cols = [c for c in df.columns
                    if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    X = df[feature_cols].copy()
    y = df["home_win"].copy()
    return X, y, df, feature_cols


# =============================================================================
# WALK-FORWARD CV
# =============================================================================
def walk_forward_cv(df: pd.DataFrame, X: pd.DataFrame,
                    y: pd.Series, feature_cols: list) -> list:
    """
    Walk-forward validation across all seasons in the dataset.

    For each test season (from second to last), trains on all prior seasons
    and evaluates on the test season.  Returns a list of per-fold metrics.
    """
    seasons  = sorted(df["season"].unique())
    if len(seasons) < 2:
        print("  Need at least 2 seasons for walk-forward CV.")
        return []

    fold_metrics = []

    for i in range(1, len(seasons)):
        test_season  = seasons[i]
        train_mask   = df["season"] < test_season
        test_mask    = df["season"] == test_season

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        if len(X_train) < 100 or len(X_test) < 50:
            continue

        model = _train_xgb(X_train, y_train, X_test, y_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        m = {
            "fold":       f"train<{test_season} / test={test_season}",
            "n_train":    len(y_train),
            "n_test":     len(y_test),
            "auc":        round(roc_auc_score(y_test, y_prob), 4),
            "log_loss":   round(log_loss(y_test, y_prob), 4),
            "brier":      round(brier_score_loss(y_test, y_prob), 4),
            "accuracy":   round(accuracy_score(y_test, y_pred), 4),
        }
        fold_metrics.append(m)

        print(f"\n  Fold: {m['fold']}")
        print(f"    Train {m['n_train']:,}  |  Test {m['n_test']:,}")
        print(f"    AUC       {m['auc']:.4f}   (random=0.500)")
        print(f"    Log Loss  {m['log_loss']:.4f}   (random=0.693)")
        print(f"    Brier     {m['brier']:.4f}   (random=0.250)")
        print(f"    Accuracy  {m['accuracy']:.4f}   (baseline≈0.540)")

    return fold_metrics


# =============================================================================
# TRAIN XGBoost (internal helper)
# =============================================================================
def _train_xgb(X_train, y_train, X_val=None, y_val=None) -> xgb.XGBClassifier:
    params = {
        "n_estimators":     400,
        "max_depth":        4,
        "learning_rate":    0.03,
        "subsample":        0.75,
        "colsample_bytree": 0.75,
        "min_child_weight": 10,      # forces larger leaf nodes → reduces overfitting
        "reg_alpha":        0.5,     # L1: sparsity
        "reg_lambda":       2.0,     # L2: shrinkage
        "objective":        "binary:logistic",
        "eval_metric":      "logloss",
        "random_state":     42,
        "tree_method":      "hist",
        "verbosity":        0,
    }
    model = xgb.XGBClassifier(**params)

    if X_val is not None and y_val is not None:
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)
    else:
        model.fit(X_train, y_train, verbose=False)

    return model


# =============================================================================
# TRAIN FINAL MODEL WITH CALIBRATION
# =============================================================================
def train_final_model(X: pd.DataFrame, y: pd.Series,
                      df: pd.DataFrame) -> tuple:
    """
    Train the production model on all available seasons.

    Calibration strategy:
      Reserve 15% of the most-recent-season data as a calibration holdout.
      Train XGBoost on the remaining 85%.
      Wrap with CalibratedClassifierCV (isotonic) on the holdout.
      This produces well-calibrated probabilities rather than raw logit scores.

    Returns: (calibrated_model, uncalibrated_xgb_model)
    """
    seasons = sorted(df["season"].unique())
    last_season = seasons[-1]

    # Calibration holdout: 15% of the most recent season, randomly
    last_season_mask = df["season"] == last_season
    X_last = X[last_season_mask]
    y_last = y[last_season_mask]

    X_cal_holdout, X_full_last, y_cal_holdout, y_full_last = train_test_split(
        X_last, y_last, test_size=0.85, random_state=42, stratify=y_last
    )

    # Combine non-holdout last-season rows with all earlier seasons
    X_train_base = pd.concat([X[~last_season_mask], X_full_last])
    y_train_base = pd.concat([y[~last_season_mask], y_full_last])

    print(f"\n  Training base XGBoost on {len(X_train_base):,} games...")
    base_model = _train_xgb(X_train_base, y_train_base)

    # Isotonic calibration on the holdout set
    print(f"  Calibrating probabilities on {len(X_cal_holdout):,}-game holdout...")
    cal_model = CalibratedClassifierCV(base_model, cv="prefit", method="isotonic")
    cal_model.fit(X_cal_holdout, y_cal_holdout)
    print("  ✓ Calibration complete.")

    return cal_model, base_model


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================
def feature_importance_report(base_model: xgb.XGBClassifier,
                               feature_cols: list) -> pd.DataFrame:
    imp = pd.DataFrame({
        "feature":    feature_cols,
        "importance": base_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\n  ── Top 15 Feature Importances ─────────────────────────────────")
    for _, row in imp.head(15).iterrows():
        bar = "█" * max(1, int(row["importance"] * 300))
        print(f"  {row['feature']:40s} | {row['importance']:.4f} | {bar}")
    print("  ────────────────────────────────────────────────────────────────")

    return imp


# =============================================================================
# SCORE TEMPLATE BUILDER
# =============================================================================
def build_scoring_template(feature_cols: list, df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a placeholder scoring template for today's games.

    In production (run_daily.py / 04_export_moneyline.py), this template
    is replaced with live values:
      - home/away SP MLBAM ID → looked up in Statcast expected/arsenal
      - home/away team → looked up in platoon splits + bullpen lookup
    """
    col_means = df[feature_cols].mean().to_dict()
    example_games = [
        {"game_date": "2026-04-01", "home_team": "NYY", "away_team": "BOS", **col_means},
        {"game_date": "2026-04-01", "home_team": "LAD", "away_team": "SFG", **col_means},
        {"game_date": "2026-04-01", "home_team": "HOU", "away_team": "TEX", **col_means},
    ]
    return pd.DataFrame(example_games)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("MONEYLINE MODEL — STEP 3: MODEL TRAINING (REFACTORED)")
    print("=" * 70)

    data_path = os.path.join(PROC_DIR, "moneyline_dataset.csv")
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found.  Run 02_build_moneyline.py first.")
        exit(1)

    # Load
    print("\n[ 1/5 ] Loading dataset...")
    X, y, df, feature_cols = load_data(data_path)
    print(f"  Feature count: {len(feature_cols)}")
    print(f"  Features: {feature_cols}")

    # Walk-forward CV
    print("\n[ 2/5 ] Walk-forward cross-validation...")
    cv_results = walk_forward_cv(df, X, y, feature_cols)

    if cv_results:
        avg_auc = np.mean([m["auc"]      for m in cv_results])
        avg_ll  = np.mean([m["log_loss"] for m in cv_results])
        avg_br  = np.mean([m["brier"]    for m in cv_results])
        avg_acc = np.mean([m["accuracy"] for m in cv_results])
        print(f"\n  ── Walk-Forward CV Summary ({'–'.join(str(m['fold']) for m in cv_results)}) ──")
        print(f"  Avg AUC-ROC  : {avg_auc:.4f}")
        print(f"  Avg Log Loss : {avg_ll:.4f}")
        print(f"  Avg Brier    : {avg_br:.4f}")
        print(f"  Avg Accuracy : {avg_acc:.4f}")

    # Train final model with calibration
    print("\n[ 3/5 ] Training final calibrated model (all seasons)...")
    cal_model, base_model = train_final_model(X, y, df)

    # Evaluate calibrated model on last season
    print("\n[ 4/5 ] Evaluating calibrated model on most recent season...")
    last_season = sorted(df["season"].unique())[-1]
    test_mask   = df["season"] == last_season
    X_test = X[test_mask]
    y_test = y[test_mask]

    y_prob_cal = cal_model.predict_proba(X_test)[:, 1]
    y_pred_cal = (y_prob_cal >= 0.5).astype(int)
    final_metrics = {
        "calibrated_auc":      round(roc_auc_score(y_test, y_prob_cal), 4),
        "calibrated_log_loss": round(log_loss(y_test, y_prob_cal), 4),
        "calibrated_brier":    round(brier_score_loss(y_test, y_prob_cal), 4),
        "calibrated_accuracy": round(accuracy_score(y_test, y_pred_cal), 4),
        "n_test":              int(len(y_test)),
        "test_season":         int(last_season),
        "n_features":          len(feature_cols),
        "cv_folds":            cv_results,
    }
    print(f"\n  Calibrated model — test season {last_season}:")
    print(f"    AUC-ROC  : {final_metrics['calibrated_auc']:.4f}")
    print(f"    Log Loss : {final_metrics['calibrated_log_loss']:.4f}")
    print(f"    Brier    : {final_metrics['calibrated_brier']:.4f}")
    print(f"    Accuracy : {final_metrics['calibrated_accuracy']:.4f}")

    # Feature importance (from uncalibrated base model)
    print("\n[ 5/5 ] Feature importances + saving artifacts...")
    imp_df = feature_importance_report(base_model, feature_cols)

    # ── Save artifacts ──────────────────────────────────────────────────────
    # XGBoost base model (native format for fast loading)
    model_path = os.path.join(MODEL_DIR, "moneyline_model.json")
    base_model.save_model(model_path)
    print(f"  ✓ Base XGBoost model   : {model_path}")

    # Calibrated model (sklearn format via joblib)
    import joblib
    cal_path = os.path.join(MODEL_DIR, "moneyline_calibrated.pkl")
    joblib.dump(cal_model, cal_path)
    print(f"  ✓ Calibrated model     : {cal_path}")

    # Feature list (must match between train and score)
    feat_path = os.path.join(MODEL_DIR, "moneyline_features.json")
    with open(feat_path, "w") as f:
        json.dump(feature_cols, f)
    print(f"  ✓ Feature list         : {feat_path}")

    # Metrics
    met_path = os.path.join(MODEL_DIR, "moneyline_metrics.json")
    with open(met_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"  ✓ Metrics              : {met_path}")

    # Feature importance CSV
    imp_path = os.path.join(MODEL_DIR, "moneyline_feature_importance.csv")
    imp_df.to_csv(imp_path, index=False)
    print(f"  ✓ Feature importances  : {imp_path}")

    # Scoring template (placeholder for production export)
    template = build_scoring_template(feature_cols, df)
    tmpl_path = os.path.join(PROC_DIR, "moneyline_today_template.csv")
    template.to_csv(tmpl_path, index=False)
    print(f"  ✓ Scoring template     : {tmpl_path}")

    print("\n" + "=" * 70)
    print("STEP 3 COMPLETE — Run 04_export_moneyline.py for today's picks.")
    print("=" * 70)
