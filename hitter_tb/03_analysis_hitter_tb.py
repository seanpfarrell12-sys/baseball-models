"""
=============================================================================
HITTER TOTAL BASES MODEL — FILE 3 OF 4: MODEL TRAINING AND SCORING
=============================================================================
Purpose : Train XGBoost regressor to predict hitter total bases per game;
          score specific player prop bets for upcoming games.
Input   : ../data/processed/hitter_tb_dataset.csv
          ../data/processed/pitcher_matchup_lookup.csv
Output  : ../models/hitter_tb_model.json
          ../data/processed/hitter_tb_predictions.csv

Model structure:
  - PRIMARY MODEL: XGBoost REGRESSOR → predicts E[TB] (continuous)
    This gives us expected total bases (e.g., 1.23 expected TB).
    We compare this to the market line (e.g., 1.5 over/under).

  - SECONDARY PROBABILITY: We convert E[TB] to P(over/under line) using
    a negative binomial distribution (more appropriate than Poisson for
    count data with overdispersion — TB can be 0,1,2,3,4 per game).

Why XGBoost over Poisson regression here?
  - TB has more features and non-linear interactions (Barrel% × park factor)
  - XGBoost handles these non-linearities better than a GLM
  - We have enough data (hundreds of player-seasons) to benefit from boosting

Daily workflow:
  1. Pull today's confirmed lineups
  2. For each player, look up their current Barrel%, EV, etc.
  3. Look up today's opposing SP stats (K%, SIERA, handedness)
  4. Run through model → get E[TB]
  5. Compare to market prop line → calculate edge

For R users:
  - XGBRegressor uses the same API as XGBClassifier but predicts continuous values
  - objective="reg:squarederror" = minimizing MSE (like OLS regression)
  - The predict() output is E[TB] directly (no need for predict_proba())
=============================================================================
"""

import os
import json
import warnings
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

from sklearn.model_selection import cross_val_score, KFold
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

# Typical prop betting lines for hitter TB
TB_PROP_LINES = [0.5, 1.5, 2.5]


# =============================================================================
# FUNCTION 1: Load and Prepare Data
# =============================================================================
def load_data(path: str) -> tuple:
    """
    Load the hitter TB dataset and prepare for XGBoost regression.

    Primary target: tb_per_game (continuous, expected TB per game)
    Feature set: Statcast metrics + traditional stats + matchup context

    Returns
    -------
    tuple : (X, y, df, feature_cols)
    """
    print(f"  Loading data from: {path}")
    df = pd.read_csv(path)
    print(f"  ✓ {len(df):,} batter-seasons loaded.")

    exclude_cols = {
        "Name", "Team", "team_std", "Season", "IDfg",
        "tb_per_game", "total_bases", "over_0_5_tb_rate",
        "over_1_5_tb_rate", "over_2_5_tb_rate",
        "name_clean",
        # Also exclude raw counting stats (already in rate form)
        "G", "PA", "AB", "H", "1B", "2B", "3B", "HR", "R", "RBI", "BB",
        "SO", "SB", "CS", "singles", "total_bases"
    }

    feature_cols = [c for c in df.columns
                    if c not in exclude_cols
                    and pd.api.types.is_numeric_dtype(df[c])
                    and df[c].nunique() > 1]   # Drop constant columns

    y = df["tb_per_game"].copy()
    X = df[feature_cols].copy()

    print(f"  ✓ Features: {len(feature_cols)} columns")
    print(f"  ✓ Target mean: {y.mean():.3f} TB/game | std: {y.std():.3f}")
    print(f"  ✓ Sample features: {feature_cols[:5]}")

    return X, y, df, feature_cols


# =============================================================================
# FUNCTION 2: Train XGBoost Regressor
# =============================================================================
def train_xgboost_regressor(X_train: pd.DataFrame, y_train: pd.Series,
                             X_test:  pd.DataFrame, y_test:  pd.Series
                             ) -> xgb.XGBRegressor:
    """
    Train XGBoost regression model to predict average TB per game.

    For regression, XGBoost minimizes MSE (mean squared error):
      L(θ) = Σ (y_i - ŷ_i)² + Ω(θ)
    where Ω is the regularization term that prevents overfitting.

    Key difference from classifier:
      - objective = "reg:squarederror" (MSE, not binary cross-entropy)
      - Output is a continuous value E[TB], not a probability
      - We don't need predict_proba() — just predict()

    Hyperparameter notes for regression:
      - max_depth=4 works well for player-level data (simpler interactions)
      - learning_rate=0.03 with 400 trees is a good starting point
      - min_child_weight=10 prevents overfitting on small player samples

    R equivalent:
      library(xgboost)
      dtrain <- xgb.DMatrix(data=as.matrix(X_train), label=y_train)
      model  <- xgboost(dtrain, nrounds=400, objective="reg:squarederror",
                         eta=0.03, max_depth=4)

    Returns
    -------
    xgb.XGBRegressor
        Trained regressor model.
    """
    print("  Training XGBoost regressor for hitter total bases...")

    model = xgb.XGBRegressor(
        n_estimators     = 400,
        max_depth        = 4,
        learning_rate    = 0.03,
        subsample        = 0.8,
        colsample_bytree = 0.7,
        min_child_weight = 10,   # Min samples per leaf (prevents overfit)
        reg_alpha        = 0.1,
        reg_lambda       = 1.0,
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
# FUNCTION 3: Cross-Validation
# =============================================================================
def cross_validate_model(model_params: dict, X: pd.DataFrame, y: pd.Series,
                          n_folds: int = 5) -> dict:
    """
    Perform k-fold cross-validation for the TB regression model.

    We use a simple KFold here (not time-based) because player-season
    data doesn't have a strict temporal ordering at the row level.
    Each row is a player-season summary, not a specific game event.

    Metrics for regression:
      - MAE  : Mean Absolute Error — how many TB off on average
                (MAE of 0.15 TB is realistic for a good model)
      - RMSE : Root Mean Squared Error — penalizes big misses more
      - R²   : Variance explained — how much better than predicting the mean
                (R² of 0.4–0.6 is strong for player-level prediction)

    Returns
    -------
    dict with cv_mae, cv_rmse, cv_r2
    """
    print(f"  Running {n_folds}-fold cross-validation...")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_model = xgb.XGBRegressor(**model_params, verbosity=0)

    # cross_val_score uses negative metrics so we negate for readable values
    # In R: cv_results <- trainControl(method="cv", number=5); train(...)
    mae_scores  = -cross_val_score(cv_model, X, y, cv=kf, scoring="neg_mean_absolute_error")
    rmse_scores = np.sqrt(-cross_val_score(cv_model, X, y, cv=kf, scoring="neg_mean_squared_error"))
    r2_scores   = cross_val_score(cv_model, X, y, cv=kf, scoring="r2")

    cv_metrics = {
        "cv_mae":  float(mae_scores.mean()),
        "cv_rmse": float(rmse_scores.mean()),
        "cv_r2":   float(r2_scores.mean()),
    }
    print(f"  ── Cross-Validation Results ─────────────────────────────────")
    print(f"  MAE  : {cv_metrics['cv_mae']:.4f} TB (avg error per player)")
    print(f"  RMSE : {cv_metrics['cv_rmse']:.4f} TB")
    print(f"  R²   : {cv_metrics['cv_r2']:.4f} (variance explained; 1.0 = perfect)")
    print(f"  ─────────────────────────────────────────────────────────────")

    return cv_metrics


# =============================================================================
# FUNCTION 4: Convert E[TB] to P(Over/Under Line)
# =============================================================================
def etb_to_probability(expected_tb: float, line: float,
                        dispersion: float = 0.8) -> dict:
    """
    Convert expected TB (continuous) to P(over/under line).

    We model TB per game as a Negative Binomial distribution:
      - More appropriate than Poisson because TB is overdispersed
        (variance > mean, especially for power hitters who have feast/famine games)
      - Parameters: μ = E[TB], θ = dispersion parameter

    The dispersion parameter (θ) is estimated from training data.
    θ = 0.8 is a reasonable default for hitter TB data.

    P(over line) = P(TB >= ceil(line)) if line is not a whole number
                 = 1 - P(TB ≤ floor(line) - 1) + P(push) for whole numbers

    Alternative: Use a simpler normal approximation when E[TB] is moderate:
      P(over line) ≈ 1 - Φ((line - E[TB]) / σ)  where σ = std of TB per game

    We use the normal approximation here for simplicity:

    Parameters
    ----------
    expected_tb : float
        Model prediction E[TB] for this player in this game.
    line : float
        The market prop line (e.g., 1.5).
    dispersion : float
        Standard deviation divisor for the normal approximation.
        Typical TB std ≈ 1.0–1.2 per game for regular starters.

    Returns
    -------
    dict with p_over, p_under, expected_tb
    """
    # Empirical standard deviation of TB per game ≈ 1.1
    # (this could be refined by computing from training data)
    sigma = max(np.sqrt(expected_tb * dispersion), 0.5)

    # Normal approximation: P(TB > line) = 1 - Φ((line - μ) / σ)
    # In R: 1 - pnorm(line, mean=expected_tb, sd=sigma)
    z_score = (line - expected_tb) / sigma
    p_over  = 1 - stats.norm.cdf(z_score)   # P(TB > line)
    p_under = stats.norm.cdf(z_score)        # P(TB < line)

    # Continuity correction for discrete distributions
    # (approximates that TB is actually discrete 0,1,2,3,4)
    if abs(line - round(line)) < 0.1:  # If line is near a whole number (push possible)
        p_push  = stats.norm.pdf(z_score) / sigma
        p_over  = max(0, p_over - p_push / 2)
        p_under = max(0, p_under - p_push / 2)

    return {
        "p_over":       round(float(p_over),  4),
        "p_under":      round(float(p_under), 4),
        "expected_tb":  round(float(expected_tb), 3),
    }


# =============================================================================
# FUNCTION 5: Score Individual Players for Upcoming Games
# =============================================================================
def score_player_props(model: xgb.XGBRegressor,
                        feature_cols: list,
                        players_df: pd.DataFrame,
                        prop_line: float = 1.5) -> pd.DataFrame:
    """
    Score individual player prop bets for today's games.

    inputs_df should have columns:
      - player_name  : Player name for identification
      - team         : Player's team abbreviation
      - opp_team     : Opposing team
      - opp_sp_name  : Opposing SP name (for matchup lookup)
      - All feature columns that the model was trained on

    For platoon adjustments, set 'platoon_advantage' column:
      - 1 if batter has platoon advantage (LHH vs RHP or RHH vs LHP)
      - 0 if no advantage (same-hand matchup)
      - -1 if reverse platoon (rare but relevant for extreme platoon hitters)

    Returns
    -------
    pd.DataFrame
        Player props with expected TB and P(over/under) for each line.
    """
    # Ensure all feature columns are present
    missing = [c for c in feature_cols if c not in players_df.columns]
    if missing:
        print(f"  WARNING: Missing features {missing[:5]}... filling with 0")
        for col in missing:
            players_df[col] = 0.0

    X_score = players_df[feature_cols].fillna(0)

    # Predict expected total bases per game
    expected_tb = model.predict(X_score)
    players_df["expected_tb"] = expected_tb.round(3)

    # Convert to P(over/under) for each standard prop line
    # In R: mapply(function(etb) etb_to_probability(etb, line), expected_tb)
    for line in TB_PROP_LINES:
        probs_list = [etb_to_probability(etb, line) for etb in expected_tb]

        col_name = f"line_{str(line).replace('.', '_')}"
        players_df[f"p_over_{col_name}"]  = [p["p_over"]  for p in probs_list]
        players_df[f"p_under_{col_name}"] = [p["p_under"] for p in probs_list]

    # Primary line for edge calculations (most common prop line is 1.5)
    main_probs = [etb_to_probability(etb, prop_line) for etb in expected_tb]
    players_df["p_over_main"]  = [p["p_over"]  for p in main_probs]
    players_df["p_under_main"] = [p["p_under"] for p in main_probs]

    return players_df


# =============================================================================
# FUNCTION 6: Feature Importance Analysis
# =============================================================================
def analyze_feature_importance(model: xgb.XGBRegressor,
                                feature_cols: list) -> pd.DataFrame:
    """
    Show which Statcast/traditional features drive the XGBoost TB predictions.

    Expected top features for hitter TB:
      1. barrel_pct / brl_percent    — best single predictor of XBH + HR
      2. ISO / SLG                   — direct power metrics
      3. avg_exit_velo               — hard contact = more extra base hits
      4. hr_per_game                 — direct component of TB
      5. wRC+ / wOBA                 — overall hitting quality

    If opp_avg_sp_siera is in the top 5, the matchup features are working.
    """
    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print("\n  ── Feature Importances (Top 15) ─────────────────────────────")
    for i, row in imp_df.head(15).iterrows():
        bar = "█" * int(row["importance"] * 300)
        print(f"  {i+1:2d}. {row['feature']:35s} | {row['importance']:.4f} | {bar}")
    print("  ─────────────────────────────────────────────────────────────")

    return imp_df


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("HITTER TOTAL BASES MODEL — STEP 3: MODEL TRAINING AND SCORING")
    print("=" * 70)

    data_path = os.path.join(PROC_DIR, "hitter_tb_dataset.csv")
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found. Run 02_build_hitter_tb.py first.")
        exit(1)

    print("\n[ 1/6 ] Loading data...")
    X, y, df, feature_cols = load_data(data_path)

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
    print(f"  Training: {len(X_train):,} | Test: {len(X_test):,} batter-seasons")

    # Cross-validation
    print("\n[ 3/6 ] Cross-validating...")
    xgb_params = {
        "n_estimators": 400, "max_depth": 4, "learning_rate": 0.03,
        "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 10,
        "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42,
        "tree_method": "hist"
    }
    cv_metrics = cross_validate_model(xgb_params, X_train, y_train, n_folds=5)

    # Train final model
    print("\n[ 4/6 ] Training final XGBoost model...")
    model = train_xgboost_regressor(X_train, y_train, X_test, y_test)

    # Evaluate on test set
    test_preds = model.predict(X_test)
    test_mae   = mean_absolute_error(y_test, test_preds)
    test_rmse  = np.sqrt(mean_squared_error(y_test, test_preds))
    test_r2    = r2_score(y_test, test_preds)
    print(f"\n  ── Test Set (2024) ────────────────────────────────────────")
    print(f"  MAE  : {test_mae:.4f} TB | RMSE: {test_rmse:.4f} TB | R²: {test_r2:.4f}")
    print(f"  ─────────────────────────────────────────────────────────────")

    # Feature importance
    print("\n[ 5/6 ] Analyzing feature importances...")
    imp_df = analyze_feature_importance(model, feature_cols)

    # Save model
    print("\n[ 6/6 ] Saving model and building scoring template...")
    model.save_model(os.path.join(MODEL_DIR, "hitter_tb_model.json"))
    with open(os.path.join(MODEL_DIR, "hitter_tb_features.json"), "w") as f:
        json.dump(feature_cols, f)

    metrics_all = {**cv_metrics, "test_mae": test_mae, "test_rmse": test_rmse, "test_r2": test_r2}
    with open(os.path.join(MODEL_DIR, "hitter_tb_metrics.json"), "w") as f:
        json.dump(metrics_all, f, indent=2)
    imp_df.to_csv(os.path.join(MODEL_DIR, "hitter_tb_feature_importance.csv"), index=False)

    # Scoring template — build example player slate
    col_means = X.mean().to_dict()
    example_players = [
        {"player_name": "Juan Soto",    "team": "NYM", "opp_team": "ATL", "platoon_advantage": 1, **col_means},
        {"player_name": "Aaron Judge",  "team": "NYY", "opp_team": "BOS", "platoon_advantage": 0, **col_means},
        {"player_name": "Freddie Freeman","team": "LAD","opp_team": "SFG","platoon_advantage": 1, **col_means},
    ]
    template = pd.DataFrame(example_players)

    scored = score_player_props(model, feature_cols, template, prop_line=1.5)
    predictions_path = os.path.join(PROC_DIR, "hitter_tb_predictions.csv")
    scored.to_csv(predictions_path, index=False)
    print(f"  ✓ Predictions saved: {predictions_path}")

    display_cols = ["player_name", "team", "opp_team", "expected_tb",
                    "p_over_line_1_5", "p_under_line_1_5"]
    display_cols = [c for c in display_cols if c in scored.columns]
    print(f"\n  Sample predictions (1.5 TB prop line):")
    print(scored[display_cols].to_string(index=False))

    print("\n" + "=" * 70)
    print("STEP 3 COMPLETE — Run 04_export_hitter_tb.py next.")
    print("=" * 70)
