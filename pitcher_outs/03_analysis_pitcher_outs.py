"""
=============================================================================
PITCHER TOTAL OUTS MODEL — FILE 3 OF 4: MODEL TRAINING AND SCORING
=============================================================================
Purpose : Train XGBoost regressor to predict pitcher outs per start;
          score individual pitcher props for upcoming games.
Input   : ../data/processed/pitcher_outs_dataset.csv
Output  : ../models/pitcher_outs_model.json
          ../data/processed/pitcher_outs_predictions.csv

Modeling philosophy:
  - We predict EXPECTED OUTS per start (e.g., 16.2 outs = 5.4 IP)
  - The model captures BOTH pitcher skill (K%, BB%) AND managerial tendencies
  - Manager hook rate is the most differentiated variable vs market pricing
    (books price outs mainly on ERA — we add managerial context)

Key market inefficiency this model targets:
  - Books underprice (set Over too low) for pitchers who:
    1. Have elite K%-BB% but are on teams with aggressive managers
    2. Face weak lineups (high K% lineups) → even modest pitchers can go deep
    3. Come off a rest week with bullpen tired → manager forced to leave SP in

  - Books overprice (set Over too high) for pitchers who:
    1. Have flashy ERA but allow hard contact (low xwOBA suppressor)
    2. Pitch in pitcher-friendly parks so ERA looks better than true talent
    3. Have managers who habitually pull starters early (Kevin Cash, TBR)

For R users:
  - XGBRegressor is nearly identical to XGBClassifier but for continuous targets
  - SHAP values (SHapley Additive exPlanations) explain individual predictions
    Similar to breakdown() in R's iml or DALEX packages
=============================================================================
"""

import os
import json
import warnings
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

warnings.filterwarnings("ignore")

# --- Configuration ----------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
EXPORT_DIR = os.path.join(BASE_DIR, "exports")

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# Standard outs prop lines (sportsbooks typically offer these)
OUTS_PROP_LINES = [14.5, 15.5, 16.5, 17.5, 18.5]


# =============================================================================
# FUNCTION 1: Load and Prepare Data
# =============================================================================
def load_data(path: str) -> tuple:
    """
    Load the pitcher outs dataset and prepare for XGBoost regression.

    Primary target: outs_per_start (continuous float, typically 12–21)
    Features: efficiency metrics + arsenal quality + manager + opponent

    Returns
    -------
    tuple : (X, y, df, feature_cols)
    """
    print(f"  Loading data from: {path}")
    df = pd.read_csv(path)
    print(f"  ✓ {len(df):,} pitcher-seasons loaded.")

    # Exclude target columns and identifier columns
    exclude_cols = {
        "Name", "team_std", "Season", "GS", "IP",
        "outs_per_start", "ip_per_start",
        "over_15_5_outs_rate", "over_17_5_outs_rate", "over_18_5_outs_rate",
        "manager",  # String column
    }

    feature_cols = [c for c in df.columns
                    if c not in exclude_cols
                    and pd.api.types.is_numeric_dtype(df[c])
                    and df[c].nunique() > 1]

    # Primary regression target
    if "outs_per_start" not in df.columns:
        print("  ERROR: 'outs_per_start' not found. Check build step output.")
        return None, None, df, feature_cols

    y = df["outs_per_start"].copy()
    X = df[feature_cols].copy()

    print(f"  ✓ Features: {len(feature_cols)} columns")
    print(f"  ✓ Mean outs/start: {y.mean():.2f} | Std: {y.std():.2f}")
    print(f"  ✓ Range: {y.min():.1f}–{y.max():.1f} outs")
    print(f"  ✓ Feature names: {feature_cols[:6]}...")

    return X, y, df, feature_cols


# =============================================================================
# FUNCTION 2: Train XGBoost Regressor
# =============================================================================
def train_xgboost_regressor(X_train: pd.DataFrame, y_train: pd.Series,
                             X_test:  pd.DataFrame, y_test:  pd.Series
                             ) -> xgb.XGBRegressor:
    """
    Train XGBoost to predict expected outs per start.

    Hyperparameter tuning rationale:
      - max_depth=4: Pitcher outs has fewer complex interactions than moneyline.
        Depth-4 trees can capture K%-BB% × manager_hook interactions without
        overfitting on our limited sample of 3 × 30 × ~12 SPs ≈ ~1,000 rows.
      - min_child_weight=8: Requires 8 samples per leaf. Prevents the model
        from learning noise from pitchers with few starts.
      - n_estimators=300: Enough trees to find important interactions.
      - subsample=0.75: Each tree sees 75% of data → reduces variance.

    R equivalent:
      library(xgboost)
      dtrain <- xgb.DMatrix(as.matrix(X_train), label=y_train)
      params <- list(objective="reg:squarederror", eta=0.05, max_depth=4)
      model  <- xgb.train(params, dtrain, nrounds=300)
      outs   <- predict(model, xgb.DMatrix(as.matrix(X_test)))

    Returns
    -------
    xgb.XGBRegressor
        Fitted regressor model.
    """
    print("  Training XGBoost regressor for pitcher total outs...")

    model = xgb.XGBRegressor(
        n_estimators     = 300,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.75,
        colsample_bytree = 0.80,
        min_child_weight = 8,
        reg_alpha        = 0.2,
        reg_lambda       = 1.5,
        objective        = "reg:squarederror",
        random_state     = 42,
        tree_method      = "hist",
        verbosity        = 0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    print(f"  ✓ XGBoost regressor trained.")
    return model


# =============================================================================
# FUNCTION 3: Cross-Validation with Season-Based Folds
# =============================================================================
def cross_validate_model(X: pd.DataFrame, y: pd.Series, df: pd.DataFrame,
                          n_folds: int = 5) -> dict:
    """
    Cross-validate using KFold on the pitcher outs dataset.

    Metrics:
      - MAE: how many outs off per prediction
        MAE=1.5 outs = off by about half an inning on average
      - RMSE: penalizes big misses (a 5-out miss = blown start prediction)
      - R²: how much variance the model explains vs predicting the mean
        R²=0.50 means the model explains 50% of variance in outs/start

    The most important diagnostic:
      - Plot actual vs predicted outs to check for systematic bias
      - If model underpredicts for "workhorse" pitchers, add an IP/GS feature
    """
    print(f"  Running {n_folds}-fold cross-validation...")

    kf    = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.75, colsample_bytree=0.80, min_child_weight=8,
        random_state=42, tree_method="hist", verbosity=0,
    )

    mae_scores  = -cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")
    rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error"))
    r2_scores   = cross_val_score(model, X, y, cv=kf, scoring="r2")

    cv_metrics = {
        "cv_mae":  float(mae_scores.mean()),
        "cv_rmse": float(rmse_scores.mean()),
        "cv_r2":   float(r2_scores.mean()),
    }

    print(f"\n  ── Cross-Validation Results ─────────────────────────────────")
    print(f"  MAE  : {cv_metrics['cv_mae']:.3f} outs/start")
    print(f"  RMSE : {cv_metrics['cv_rmse']:.3f} outs/start")
    print(f"  R²   : {cv_metrics['cv_r2']:.4f}")
    print(f"\n  Interpretation:")
    print(f"  MAE of {cv_metrics['cv_mae']:.1f} outs ≈ "
          f"{cv_metrics['cv_mae']/3:.2f} IP error per prediction.")
    print(f"  ─────────────────────────────────────────────────────────────")
    return cv_metrics


# =============================================================================
# FUNCTION 4: Convert E[Outs] to P(Over/Under Line)
# =============================================================================
def eouts_to_probability(expected_outs: float, line: float,
                          outs_std: float = 3.5) -> dict:
    """
    Convert model's expected outs to P(over/under) for a specific prop line.

    Outs per start follows a roughly normal distribution within the range
    of a single pitcher's starts (given a fixed skill level). We use a
    normal approximation with:
      μ = expected_outs (from model)
      σ = outs_std ≈ 3.5 (typical within-pitcher start-to-start std dev)

    Example: expected_outs=16.5, line=15.5, sigma=3.5
      z = (15.5 - 16.5) / 3.5 = -0.286
      P(over) = 1 - Φ(-0.286) = Φ(0.286) ≈ 0.613
      P(under) ≈ 0.387

    For half-point lines, no push is possible. For whole-number lines,
    we account for a small probability of landing exactly on the line.

    Parameters
    ----------
    expected_outs : float    Model's predicted outs per start
    line : float             Market prop line (e.g., 15.5)
    outs_std : float         Within-pitcher standard deviation (default 3.5)

    Returns
    -------
    dict with p_over, p_under, expected_outs
    """
    # Ensure we don't divide by zero
    sigma = max(outs_std, 0.5)

    z_score = (line - expected_outs) / sigma

    # Normal CDF (stats.norm.cdf = pnorm() in R)
    p_under = float(stats.norm.cdf(z_score))
    p_over  = float(1 - p_under)

    # For whole-number lines (push is possible)
    if abs(line - round(line)) < 0.01:
        # Small push probability for exact landing
        p_push  = float(stats.norm.pdf(z_score) / sigma) * 0.5  # Approximate
        p_over  = max(0, p_over - p_push / 2)
        p_under = max(0, p_under - p_push / 2)
    else:
        p_push = 0.0

    return {
        "p_over":        round(p_over, 4),
        "p_under":       round(p_under, 4),
        "p_push":        round(p_push, 4),
        "expected_outs": round(expected_outs, 2),
    }


# =============================================================================
# FUNCTION 5: Score Today's Starting Pitchers
# =============================================================================
def score_pitchers(model: xgb.XGBRegressor,
                   feature_cols: list,
                   pitchers_df: pd.DataFrame,
                   prop_line: float = 15.5) -> pd.DataFrame:
    """
    Score today's starting pitchers against their prop outs line.

    Required columns in pitchers_df:
      - pitcher_name     : SP's name
      - team             : SP's team abbreviation
      - opp_team         : Opposing team
      - k_pct            : SP's current K%
      - bb_pct           : SP's current BB%
      - k_minus_bb_pct   : SP's current K-BB%
      - siera / xfip     : SP's current SIERA or xFIP
      - depth_score      : Manager's historical depth tendency
      - avg_sp_outs      : Manager's historical avg SP outs allowed

    Optional (improve prediction if available):
      - opp_lineup_wrc_plus : Today's opponent lineup's wRC+
      - bullpen_days_rest   : Relief corps fatigue level (higher = push SP deeper)
      - swstr_pct, fstrike_pct : Pitch efficiency metrics

    Returns
    -------
    pd.DataFrame
        pitchers_df with expected_outs, p_over, p_under for each prop line.
    """
    # Fill missing features with 0 (will use training means if available)
    missing = [c for c in feature_cols if c not in pitchers_df.columns]
    if missing:
        print(f"  WARNING: Missing {len(missing)} features — filling with 0: {missing[:5]}")
        for col in missing:
            pitchers_df[col] = 0.0

    X_score = pitchers_df[feature_cols].fillna(0)

    # Predict expected outs
    expected_outs = model.predict(X_score)
    pitchers_df["expected_outs"] = expected_outs.round(2)

    # Convert to probabilities for all standard prop lines
    for line in OUTS_PROP_LINES:
        probs = [eouts_to_probability(eo, line) for eo in expected_outs]
        col   = f"line_{str(line).replace('.', '_')}"
        pitchers_df[f"p_over_{col}"]  = [p["p_over"]  for p in probs]
        pitchers_df[f"p_under_{col}"] = [p["p_under"] for p in probs]

    # Primary line probabilities
    main_probs = [eouts_to_probability(eo, prop_line) for eo in expected_outs]
    pitchers_df["p_over_main"]  = [p["p_over"]  for p in main_probs]
    pitchers_df["p_under_main"] = [p["p_under"] for p in main_probs]

    # Convert expected outs to IP for display (IP = outs / 3)
    pitchers_df["expected_ip"] = (pitchers_df["expected_outs"] / 3).round(2)

    return pitchers_df


# =============================================================================
# FUNCTION 6: Feature Importance Analysis
# =============================================================================
def analyze_feature_importance(model: xgb.XGBRegressor,
                                feature_cols: list) -> pd.DataFrame:
    """
    Analyze which features drive pitcher outs predictions.

    Expected top features:
      1. k_minus_bb_pct   — K-BB% is the best composite efficiency metric
      2. siera / xfip     — Overall run prevention quality
      3. depth_score      — Manager tendency (most differentiating)
      4. avg_sp_outs      — Historical baseline for this team's SP usage
      5. k_pct            — More K's = faster, more consistent outs
      6. swstr_pct        — Swing-and-miss = shorter PA = deeper starts
      7. avg_fb_velo      — Velocity correlates with K and longevity in starts

    If depth_score appears in top 3, the managerial context is being learned.
    If it's not in top 10, it may need more historical data or better encoding.
    """
    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print("\n  ── Feature Importances (Top 12) ─────────────────────────────")
    for i, row in imp_df.head(12).iterrows():
        bar = "█" * int(row["importance"] * 300)
        print(f"  {i+1:2d}. {row['feature']:35s} | {row['importance']:.4f} | {bar}")
    print("  ─────────────────────────────────────────────────────────────")

    return imp_df


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("PITCHER TOTAL OUTS MODEL — STEP 3: MODEL TRAINING AND SCORING")
    print("=" * 70)

    data_path = os.path.join(PROC_DIR, "pitcher_outs_dataset.csv")
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found. Run 02_build_pitcher_outs.py first.")
        exit(1)

    print("\n[ 1/6 ] Loading data...")
    X, y, df, feature_cols = load_data(data_path)

    if X is None:
        print("ERROR: Failed to load data. Exiting.")
        exit(1)

    # Time-based split
    print("\n[ 2/6 ] Splitting by season...")
    if "Season" in df.columns:
        train_mask = df["Season"] < 2025
        test_mask  = df["Season"] == 2025
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  Training: {len(X_train):,} | Test: {len(X_test):,} pitcher-seasons")

    # Cross-validation
    print("\n[ 3/6 ] Cross-validating...")
    cv_metrics = cross_validate_model(X_train, y_train, df[df["Season"] < 2025] if "Season" in df.columns else df)

    # Train final model
    print("\n[ 4/6 ] Training final model...")
    model = train_xgboost_regressor(X_train, y_train, X_test, y_test)

    # Test evaluation
    test_preds = model.predict(X_test)
    test_mae   = mean_absolute_error(y_test, test_preds)
    test_rmse  = np.sqrt(mean_squared_error(y_test, test_preds))
    test_r2    = r2_score(y_test, test_preds)
    print(f"\n  ── Test Set Performance ────────────────────────────────────")
    print(f"  MAE  : {test_mae:.3f} outs ({test_mae/3:.2f} IP per start)")
    print(f"  RMSE : {test_rmse:.3f} outs")
    print(f"  R²   : {test_r2:.4f}")
    print(f"  ─────────────────────────────────────────────────────────────")

    # Feature importance
    print("\n[ 5/6 ] Analyzing feature importances...")
    imp_df = analyze_feature_importance(model, feature_cols)

    # Save all artifacts
    print("\n[ 6/6 ] Saving model...")
    model.save_model(os.path.join(MODEL_DIR, "pitcher_outs_model.json"))
    with open(os.path.join(MODEL_DIR, "pitcher_outs_features.json"), "w") as f:
        json.dump(feature_cols, f)

    metrics_all = {**cv_metrics, "test_mae": test_mae, "test_rmse": test_rmse, "test_r2": test_r2}
    with open(os.path.join(MODEL_DIR, "pitcher_outs_metrics.json"), "w") as f:
        json.dump(metrics_all, f, indent=2)
    imp_df.to_csv(os.path.join(MODEL_DIR, "pitcher_outs_feature_importance.csv"), index=False)

    # Build scoring template with example pitchers
    col_means = X.mean().to_dict()
    example_pitchers = [
        {"pitcher_name": "Gerrit Cole",    "team": "NYY", "opp_team": "BOS",
         "depth_score": 0.50, "avg_sp_outs": 14.5, **col_means},
        {"pitcher_name": "Shane McClanahan","team": "TBR","opp_team": "BAL",
         "depth_score": 0.30, "avg_sp_outs": 12.5, **col_means},  # Kevin Cash = low
        {"pitcher_name": "Dylan Cease",    "team": "SDP", "opp_team": "LAD",
         "depth_score": 0.48, "avg_sp_outs": 14.3, **col_means},
    ]
    template = pd.DataFrame(example_pitchers)

    scored = score_pitchers(model, feature_cols, template, prop_line=15.5)
    pred_path = os.path.join(PROC_DIR, "pitcher_outs_predictions.csv")
    scored.to_csv(pred_path, index=False)
    print(f"  ✓ Predictions saved: {pred_path}")

    display_cols = ["pitcher_name", "team", "opp_team", "expected_outs",
                    "expected_ip", "p_over_line_15_5", "p_under_line_15_5"]
    display_cols = [c for c in display_cols if c in scored.columns]
    print(f"\n  Sample predictions (15.5 outs prop line):")
    print(scored[display_cols].to_string(index=False))

    print("\n" + "=" * 70)
    print("STEP 3 COMPLETE — Run 04_export_pitcher_outs.py next.")
    print("=" * 70)
