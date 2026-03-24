"""
=============================================================================
OVER/UNDER TOTALS MODEL — FILE 3 OF 4: MODEL TRAINING AND SCORING
=============================================================================
Purpose : Train Poisson regression on game-level data; score upcoming games.
Input   : ../data/processed/totals_dataset.csv
Output  : ../models/totals_model.pkl
          ../data/processed/totals_predictions.csv

Why Poisson Regression?
  Baseball runs are count data (0, 1, 2, 3, ...) — non-negative integers.
  Poisson regression models the RATE (λ) of run scoring as:
    log(λ) = β₀ + β₁×wRC+ + β₂×SIERA + β₃×park_factor + ...
    λ       = expected run total (e.g., λ=8.4 means expect 8.4 runs)

  This is mathematically appropriate because:
    1. Counts can't be negative (runs ≥ 0)
    2. The conditional mean ≈ conditional variance (property of run scoring)
    3. The log link naturally prevents negative predictions

  The key output is λ̂ (lambda hat) = predicted total runs for each game.
  We then compare λ̂ to the market's over/under line to find edges.

  R equivalent:
    library(glm)
    model <- glm(total_runs ~ ., data=train_df, family=poisson(link="log"))
    preds <- predict(model, newdata=test_df, type="response")

Two-model approach:
  Rather than modeling total runs directly, we model each team's runs
  separately (home runs + away runs) and sum:
    λ_total = λ_home + λ_away
  This captures team-specific offensive/defensive asymmetries better.

For R users:
  - statsmodels is Python's equivalent of R's base stats/glm functions
  - sm.GLM() = glm() in R; Poisson() family = family=poisson in R
  - model.params = coef(model) in R
  - model.summary() = summary(model) in R
=============================================================================
"""

import os
import json
import warnings
import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm                    # R's equivalent: base stats package
from statsmodels.genmod.families import Poisson # R: family=poisson

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats                         # For probability distributions

warnings.filterwarnings("ignore")

# --- Configuration ----------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
EXPORT_DIR = os.path.join(BASE_DIR, "exports")

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)


# =============================================================================
# FUNCTION 1: Load and Prepare Data
# =============================================================================
def load_data(path: str) -> tuple:
    """
    Load the totals dataset and split into features/targets.

    For Poisson regression:
      y = total_runs (count, non-negative integer)
      X = feature matrix (all predictive columns)

    Returns
    -------
    tuple : (X, y, df, feature_cols)
    """
    print(f"  Loading data from: {path}")
    df = pd.read_csv(path)
    print(f"  ✓ {len(df):,} games loaded.")
    print(f"  ✓ Mean total runs: {df['total_runs'].mean():.2f}")

    exclude_cols = {
        "game_date", "Season", "home_team", "away_team",
        "home_runs", "away_runs", "total_runs"
    }
    feature_cols = [c for c in df.columns if c not in exclude_cols
                    and pd.api.types.is_numeric_dtype(df[c])]

    X = df[feature_cols].copy()
    y = df["total_runs"].copy()

    return X, y, df, feature_cols


# =============================================================================
# FUNCTION 2: Fit Poisson GLM (statsmodels)
# =============================================================================
def fit_poisson_model(X_train: pd.DataFrame, y_train: pd.Series) -> sm.GLM:
    """
    Fit a Poisson generalized linear model using statsmodels.

    The model formula (implicit):
      log(E[runs]) = β₀ + β₁×feature₁ + β₂×feature₂ + ...

    Statsmodels adds a constant term (intercept) via sm.add_constant().
    In R, glm() includes an intercept by default.

    Interpretation of Poisson coefficients:
      - A coefficient of 0.02 for wRC+ means: for every 1-unit increase
        in wRC+, expected runs increase by e^0.02 - 1 ≈ 2%.
      - Positive coefficients → more runs (park factor, wRC+)
      - Negative coefficients → fewer runs (low SIERA = better pitcher)

    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix for training.
    y_train : pd.Series
        Run totals (target variable — integer counts).

    Returns
    -------
    statsmodels.GLMResultsWrapper
        Fitted Poisson GLM with coefficients, p-values, and diagnostics.
    """
    print("  Fitting Poisson GLM...")

    # Add constant (intercept) to feature matrix
    # In R, glm() does this automatically with the formula interface
    # In statsmodels, we must be explicit: sm.add_constant(X) = cbind(1, X) in R
    X_const = sm.add_constant(X_train.astype(float))

    # Fit Poisson GLM
    # In R: glm(y ~ ., data=train_df, family=poisson(link="log"))
    glm_model = sm.GLM(
        y_train.astype(float),
        X_const,
        family=Poisson()          # Poisson family with log link (default)
    )

    result = glm_model.fit(
        maxiter=100,              # Maximum iterations for IRLS solver
        disp=False                # Don't print optimization details
    )

    print(f"  ✓ Poisson GLM fitted successfully.")
    print(f"  ✓ Pseudo R² (McFadden): {1 - result.llf/result.llnull:.4f}")
    print(f"  ✓ AIC: {result.aic:.2f}")
    print(f"  ✓ Deviance: {result.deviance:.2f} (df: {result.df_resid:.0f})")

    # Display coefficient summary (like summary(model) in R)
    print("\n  ── Significant Coefficients (p < 0.10) ──────────────────────")
    coef_df = pd.DataFrame({
        "feature":   result.params.index,
        "coef":      result.params.values,
        "p_value":   result.pvalues.values,
        "exp_coef":  np.exp(result.params.values),  # Multiplicative effect
    })
    sig_coef = coef_df[coef_df["p_value"] < 0.10].sort_values("p_value")
    for _, row in sig_coef.iterrows():
        direction = "↑" if row["coef"] > 0 else "↓"
        print(f"  {row['feature']:35s} | coef={row['coef']:+.4f} | "
              f"effect={row['exp_coef']:.4f}× | p={row['p_value']:.4f} {direction}")
    print("  ─────────────────────────────────────────────────────────────")

    return result


# =============================================================================
# FUNCTION 3: Cross-Validation for Poisson Model
# =============================================================================
def cross_validate_poisson(X: pd.DataFrame, y: pd.Series, n_folds: int = 5) -> dict:
    """
    Perform k-fold cross-validation for the Poisson GLM.

    Unlike XGBoost which has built-in CV, statsmodels requires manual CV.
    We use KFold from scikit-learn to create the folds and statsmodels
    to fit the model on each fold.

    Metrics for run total prediction:
      - MAE  : Mean Absolute Error (how far off in runs on average)
                e.g., MAE=1.5 means predictions are off by 1.5 runs on average
      - RMSE : Root Mean Squared Error (penalizes large errors more heavily)
      - Pearson r: Correlation between predicted and actual totals
                   (should be 0.3–0.5 for a good model)

    R equivalent:
      library(caret)
      tc <- trainControl(method="cv", number=5)
      cv_result <- train(total_runs ~ ., data=df, method="glm",
                          family=poisson, trControl=tc)

    Returns
    -------
    dict
        Cross-validation metrics (mae, rmse, correlation).
    """
    print(f"  Running {n_folds}-fold cross-validation...")

    kf   = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    maes, rmses, corrs = [], [], []

    # In Python, enumerate() gives (index, value) pairs when iterating
    # In R: for (i in seq_along(folds)) { ... }
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Fit model on training fold
        X_tr_c  = sm.add_constant(X_tr.astype(float))
        X_val_c = sm.add_constant(X_val.astype(float), has_constant='add')

        try:
            fold_model  = sm.GLM(y_tr.astype(float), X_tr_c, family=Poisson()).fit()
            fold_preds  = fold_model.predict(X_val_c)

            fold_mae    = mean_absolute_error(y_val, fold_preds)
            fold_rmse   = np.sqrt(mean_squared_error(y_val, fold_preds))
            fold_corr   = np.corrcoef(y_val, fold_preds)[0, 1]

            maes.append(fold_mae)
            rmses.append(fold_rmse)
            corrs.append(fold_corr)
            print(f"    Fold {fold_idx}: MAE={fold_mae:.3f}, RMSE={fold_rmse:.3f}, r={fold_corr:.3f}")
        except Exception as e:
            print(f"    Fold {fold_idx}: FAILED ({e})")

    cv_metrics = {
        "cv_mae":  np.mean(maes),
        "cv_rmse": np.mean(rmses),
        "cv_corr": np.mean(corrs),
    }
    print(f"\n  ── Cross-Validation Summary ─────────────────────────────────")
    print(f"  Mean MAE  : {cv_metrics['cv_mae']:.3f} runs (how many runs off on avg)")
    print(f"  Mean RMSE : {cv_metrics['cv_rmse']:.3f} runs")
    print(f"  Mean r    : {cv_metrics['cv_corr']:.4f} (correlation)")
    print(f"  ─────────────────────────────────────────────────────────────")

    return cv_metrics


# =============================================================================
# FUNCTION 4: Convert λ to Over/Under Probabilities (via Poisson CDF)
# =============================================================================
def poisson_ou_probability(lambda_hat: float, ou_line: float) -> dict:
    """
    Convert Poisson lambda (expected runs) to over/under probabilities.

    Given a predicted λ (e.g., 8.4 expected runs), what's the probability
    that the game goes OVER a specific total (e.g., 8.5)?

    Since runs are integers, P(total > 8.5) = P(total >= 9) = 1 - CDF(8).

    The Poisson CDF gives P(X ≤ k) for a Poisson(λ) random variable.
    In R: ppois(k, lambda) = P(X ≤ k); 1 - ppois(k, lambda) = P(X > k)

    Parameters
    ----------
    lambda_hat : float
        Model's predicted expected total runs (e.g., 8.43).
    ou_line : float
        The market's over/under line (e.g., 8.5).

    Returns
    -------
    dict with keys: p_over, p_under, p_push (exact), lambda_hat

    Example:
        lambda_hat = 8.4, line = 8.5
        P(over) = P(X ≥ 9) = 1 - P(X ≤ 8) = 1 - ppois(8, 8.4) ≈ 0.465
        P(under) = P(X ≤ 8) = ppois(8, 8.4) ≈ 0.535
    """
    # stats.poisson.cdf(k, mu) = P(X ≤ k) when X ~ Poisson(mu)
    # This is equivalent to ppois(k, lambda) in R
    k_floor = int(np.floor(ou_line))  # e.g., 8.5 → 8; 9.0 → 9

    if ou_line == k_floor:
        # Whole number line (e.g., 9.0): push is possible
        p_under = stats.poisson.cdf(k_floor - 1, lambda_hat)  # P(X ≤ 8) for line=9
        p_push  = stats.poisson.pmf(k_floor, lambda_hat)       # P(X = 9) — push
        p_over  = 1 - p_under - p_push                         # P(X ≥ 10)
    else:
        # Half-point line (e.g., 8.5): no push possible
        p_under = stats.poisson.cdf(k_floor, lambda_hat)       # P(X ≤ 8) for line=8.5
        p_push  = 0.0
        p_over  = 1 - p_under                                  # P(X ≥ 9)

    return {
        "p_over":   round(p_over,  4),
        "p_under":  round(p_under, 4),
        "p_push":   round(p_push,  4),
        "lambda_hat": round(lambda_hat, 3),
    }


# =============================================================================
# FUNCTION 5: Score Upcoming Games
# =============================================================================
def score_todays_games(model_result, feature_cols: list,
                       matchup_df: pd.DataFrame,
                       ou_lines: dict = None) -> pd.DataFrame:
    """
    Apply the trained Poisson model to today's upcoming games.

    Parameters
    ----------
    model_result : statsmodels GLMResultsWrapper
        Fitted Poisson model.
    feature_cols : list of str
        Features the model was trained on.
    matchup_df : pd.DataFrame
        Today's games with feature values populated.
    ou_lines : dict, optional
        Market over/under lines, keyed by "home_team-away_team" or game index.
        Example: {"NYY-BOS": 8.5, "LAD-SFG": 7.5}

    Returns
    -------
    pd.DataFrame
        matchup_df with predicted lambda, p_over, p_under added.
    """
    if ou_lines is None:
        ou_lines = {}

    X_score = matchup_df[feature_cols].copy()
    X_const = sm.add_constant(X_score.astype(float), has_constant='add')

    # Get predicted run totals (lambda)
    lambdas = model_result.predict(X_const)
    matchup_df["lambda_hat"] = lambdas.values

    # For each game, compute over/under probabilities
    p_overs, p_unders, default_line = [], [], 8.5

    for idx, row in matchup_df.iterrows():
        # Look up the market O/U line for this game
        game_key = f"{row.get('home_team', 'HOME')}-{row.get('away_team', 'AWAY')}"
        line     = ou_lines.get(game_key, ou_lines.get(str(idx), default_line))

        probs = poisson_ou_probability(row["lambda_hat"], line)
        p_overs.append(probs["p_over"])
        p_unders.append(probs["p_under"])

    matchup_df["p_over"]  = p_overs
    matchup_df["p_under"] = p_unders

    return matchup_df


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("TOTALS MODEL — STEP 3: MODEL TRAINING AND SCORING")
    print("=" * 70)

    data_path = os.path.join(PROC_DIR, "totals_dataset.csv")
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found. Run 02_build_totals.py first.")
        exit(1)

    # Load data
    print("\n[ 1/5 ] Loading training data...")
    X, y, df, feature_cols = load_data(data_path)

    # Time-based split
    print("\n[ 2/5 ] Splitting by season...")
    if "Season" in df.columns:
        train_mask = df["Season"] < 2025
        test_mask  = df["Season"] == 2025
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        print(f"  Training: {len(X_train):,} games | Test: {len(X_test):,} games")
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Cross-validation
    print("\n[ 3/5 ] Cross-validating Poisson model...")
    cv_metrics = cross_validate_poisson(X_train, y_train, n_folds=5)

    # Train final model on full training data
    print("\n[ 4/5 ] Training final Poisson GLM...")
    model_result = fit_poisson_model(X_train, y_train)

    # Evaluate on test set
    X_test_c  = sm.add_constant(X_test.astype(float), has_constant='add')
    test_preds = model_result.predict(X_test_c)
    test_mae   = mean_absolute_error(y_test, test_preds)
    test_rmse  = np.sqrt(mean_squared_error(y_test, test_preds))
    test_corr  = np.corrcoef(y_test, test_preds)[0, 1]
    print(f"\n  ── Test Set Performance (2024 season) ───────────────────────")
    print(f"  MAE     : {test_mae:.3f} runs")
    print(f"  RMSE    : {test_rmse:.3f} runs")
    print(f"  r       : {test_corr:.4f}")
    print(f"  ─────────────────────────────────────────────────────────────")

    # Save model
    print("\n[ 5/5 ] Saving model and generating scoring template...")
    model_path = os.path.join(MODEL_DIR, "totals_model.pkl")
    joblib.dump(model_result, model_path)
    print(f"  ✓ Model saved: {model_path}")

    feature_path = os.path.join(MODEL_DIR, "totals_features.json")
    with open(feature_path, "w") as f:
        json.dump(feature_cols, f)
    print(f"  ✓ Features saved: {feature_path}")

    metrics_all = {**cv_metrics, "test_mae": test_mae, "test_rmse": test_rmse, "test_corr": test_corr}
    with open(os.path.join(MODEL_DIR, "totals_metrics.json"), "w") as f:
        json.dump(metrics_all, f, indent=2)

    # Generate scoring template
    col_means = X.mean().to_dict()
    example_games = [
        {"game_date": "2025-04-01", "home_team": "NYY", "away_team": "BOS",
         "ou_line": 9.0, **col_means},
        {"game_date": "2025-04-01", "home_team": "LAD", "away_team": "SFG",
         "ou_line": 7.5, **col_means},
        {"game_date": "2025-04-01", "home_team": "COL", "away_team": "ARI",
         "ou_line": 11.0, **col_means},
    ]
    template = pd.DataFrame(example_games)

    # Score template
    ou_lines = {f"{g['home_team']}-{g['away_team']}": g["ou_line"]
                for g in example_games}
    scored = score_todays_games(model_result, feature_cols, template, ou_lines)

    predictions_path = os.path.join(PROC_DIR, "totals_predictions.csv")
    scored.to_csv(predictions_path, index=False)
    print(f"  ✓ Predictions saved: {predictions_path}")

    # Show sample output
    display_cols = ["home_team", "away_team", "lambda_hat", "p_over", "p_under"]
    display_cols = [c for c in display_cols if c in scored.columns]
    print(f"\n  Sample predictions (λ = expected total runs):")
    print(scored[display_cols].to_string(index=False))

    print("\n" + "=" * 70)
    print("STEP 3 COMPLETE — Run 04_export_totals.py next.")
    print("=" * 70)
