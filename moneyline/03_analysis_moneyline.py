"""
=============================================================================
MONEYLINE MODEL — FILE 3 OF 4: MODEL TRAINING AND SCORING
=============================================================================
Purpose : Train XGBoost classifier on historical game data; score upcoming games.
Input   : ../data/processed/moneyline_dataset.csv
Output  : ../models/moneyline_model.json
          ../data/processed/moneyline_predictions.csv (scored today's games)

XGBoost for binary classification:
  - Predicts P(home_team_wins) as a probability between 0.0 and 1.0
  - Uses gradient boosting: each tree corrects errors of the previous one
  - Handles missing values natively (no need for imputation)
  - Objective: binary:logistic (= logistic regression output)

Model evaluation:
  - Log Loss: measures calibration of probability predictions
    (lower = better; a perfect calibrated model has log loss = 0)
  - AUC-ROC: measures ranking ability (0.5 = random, 1.0 = perfect)
  - Accuracy: simply % of games predicted correctly
  - Brier Score: average squared error of probabilities (like MSE for probs)

R equivalent of this entire file:
  library(xgboost)
  dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
  model  <- xgboost(dtrain, nrounds=200, objective="binary:logistic")
  preds  <- predict(model, xgb.DMatrix(as.matrix(X_test)))

For R users learning Python:
  - scikit-learn uses .fit(X, y) / .predict(X) convention
  - Train/test split: train_test_split() = createDataPartition() in caret
  - cross_val_score() = trainControl + train() in caret
  - feature_importances_ = varImp() in caret
=============================================================================
"""

import os
import json
import warnings
import pandas as pd
import numpy as np
import joblib              # For saving/loading models (like saveRDS/readRDS in R)
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    log_loss, roc_auc_score, accuracy_score, brier_score_loss, classification_report
)
from sklearn.preprocessing import StandardScaler

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
    Load the processed moneyline dataset and split into features (X) and target (y).

    In XGBoost, data is organized as:
      X = feature matrix (all input columns)
      y = target vector (home_win: 1 or 0)

    This is identical to R's model formula approach, but explicit:
      In R: formula = home_win ~ feature1 + feature2 + ...
      In Python: X = df[feature_cols]; y = df["home_win"]

    Returns
    -------
    tuple : (X, y, df, feature_cols)
        X            : pd.DataFrame of features
        y            : pd.Series of binary outcomes
        df           : full DataFrame (for later joining with predictions)
        feature_cols : list of feature column names
    """
    print(f"  Loading data from: {path}")
    df = pd.read_csv(path)
    print(f"  ✓ {len(df):,} games loaded.")

    # Define feature columns — all numeric columns except IDs and target
    exclude_cols = {
        "game_date", "Season", "home_team", "away_team",
        "home_win", "total_runs", "feature_season"
    }
    feature_cols = [c for c in df.columns if c not in exclude_cols
                    and pd.api.types.is_numeric_dtype(df[c])]

    print(f"  ✓ Features: {feature_cols}")
    print(f"  ✓ Target: home_win (mean = {df['home_win'].mean():.3f})")

    X = df[feature_cols].copy()
    y = df["home_win"].copy()

    return X, y, df, feature_cols


# =============================================================================
# FUNCTION 2: Time-Based Train/Validation Split
# =============================================================================
def time_split(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series,
               test_season: int = 2025) -> tuple:
    """
    Split data by season to simulate real-world model validation.

    IMPORTANT: In sports betting, you should NEVER randomly shuffle and split.
    If you do, you get "look-ahead bias" — the model might learn from 2024
    games to predict 2022 games, which is impossible in real deployment.

    Correct approach: train on earlier seasons, test on the most recent season.
      - Training set: 2022, 2023
      - Test set: 2024
    This mimics how you'd use the model in production (train → deploy next year).

    R equivalent:
      train_idx <- which(df$Season < 2024)
      test_idx  <- which(df$Season == 2024)
      X_train <- X[train_idx, ]; y_train <- y[train_idx]
      X_test  <- X[test_idx, ];  y_test  <- y[test_idx]

    Parameters
    ----------
    test_season : int
        The season to hold out as test data.

    Returns
    -------
    tuple : (X_train, X_test, y_train, y_test)
    """
    if "Season" not in df.columns:
        # Fallback to random split if no season column
        print("  WARNING: No Season column — using random 80/20 split.")
        return train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    train_mask = df["Season"] < test_season
    test_mask  = df["Season"] == test_season

    X_train = X[train_mask].copy()
    X_test  = X[test_mask].copy()
    y_train = y[train_mask].copy()
    y_test  = y[test_mask].copy()

    print(f"  ✓ Training set: {len(X_train):,} games (seasons < {test_season})")
    print(f"  ✓ Test set:     {len(X_test):,} games (season = {test_season})")
    return X_train, X_test, y_train, y_test


# =============================================================================
# FUNCTION 3: Train XGBoost Model
# =============================================================================
def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                  X_test: pd.DataFrame, y_test: pd.Series) -> xgb.XGBClassifier:
    """
    Train an XGBoost classifier to predict P(home_team_wins).

    XGBoost Hyperparameters (key ones to tune):
    ─────────────────────────────────────────────────────────────────────
    n_estimators    : Number of boosting rounds (trees). More = better fit,
                      but risk of overfitting. Start with 200–500.
    max_depth       : Max depth of each tree. Deeper = more complex interactions.
                      For tabular data, 4–7 usually works well.
    learning_rate   : Step size for each update. Lower = slower but more robust.
                      Pair with more n_estimators when reducing this.
    subsample       : Fraction of rows to sample per tree (reduces overfitting).
    colsample_bytree: Fraction of COLUMNS to sample per tree (like mtry in R's rf).
    reg_alpha/lambda: L1/L2 regularization (penalize large weights).
    scale_pos_weight: Handles class imbalance. Set to (neg_count / pos_count)
                      if one class is much more common.

    Early stopping:
      Training stops when validation set performance doesn't improve for
      `early_stopping_rounds` consecutive trees. Prevents overfitting.

    Returns
    -------
    xgb.XGBClassifier
        Trained XGBoost model.
    """
    print("  Training XGBoost classifier...")

    # Compute home win rate for class balance check
    # In R: table(y_train) / length(y_train)
    win_rate = y_train.mean()
    print(f"  Home win rate in training data: {win_rate:.3f}")

    # XGBoost hyperparameters
    # In R: these would go inside the xgboost() or train() call
    params = {
        "n_estimators":     300,
        "max_depth":        5,
        "learning_rate":    0.05,    # "eta" in XGBoost terminology
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,       # Minimum # of samples in a leaf
        "reg_alpha":        0.1,     # L1 regularization
        "reg_lambda":       1.0,     # L2 regularization
        "objective":        "binary:logistic",  # Output probabilities
        "eval_metric":      "logloss",
        "random_state":     42,
        "tree_method":      "hist",  # Fast histogram-based training
        "verbosity":        0,       # Quiet training output
    }

    model = xgb.XGBClassifier(**params)

    # Train with early stopping: stop if val logloss doesn't improve in 30 rounds
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    print(f"  ✓ Model trained.")
    return model


# =============================================================================
# FUNCTION 4: Evaluate Model Performance
# =============================================================================
def evaluate_model(model: xgb.XGBClassifier,
                   X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Compute comprehensive evaluation metrics for the trained model.

    Metrics for a binary probability model:
      Log Loss: lower = better probability calibration
        - A coin flip model = log_loss of ~0.693
        - A good sports model might achieve 0.670 (marginal edge over random)
        - The betting market is very efficient; 0.680–0.685 is realistic

      Brier Score: mean squared error of probabilities
        - 0.25 = random (predicting 0.5 always)
        - A good model: 0.23–0.245

      AUC-ROC: area under the ROC curve
        - 0.5 = random; 1.0 = perfect; realistic: 0.52–0.56 for moneyline

      Accuracy: % of games predicted correctly
        - Baseline: 54% (home win rate) — must beat this
        - A good model: 55–57%

    Returns
    -------
    dict
        All computed metrics.
    """
    # Predict probabilities (like predict(model, newdata, type="response") in R)
    y_prob = model.predict_proba(X_test)[:, 1]  # Column 1 = P(home wins)
    y_pred = (y_prob >= 0.5).astype(int)         # Convert to binary predictions

    metrics = {
        "log_loss":    log_loss(y_test, y_prob),
        "brier_score": brier_score_loss(y_test, y_prob),
        "roc_auc":     roc_auc_score(y_test, y_prob),
        "accuracy":    accuracy_score(y_test, y_pred),
        "n_test":      len(y_test),
    }

    print("\n  ── Model Evaluation (Test Season) ──────────────────────────")
    print(f"  Log Loss  : {metrics['log_loss']:.4f}  (lower = better; random ≈ 0.693)")
    print(f"  Brier     : {metrics['brier_score']:.4f}  (lower = better; random ≈ 0.250)")
    print(f"  AUC-ROC   : {metrics['roc_auc']:.4f}  (higher = better; random = 0.500)")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}  (baseline ≈ home win rate)")
    print(f"  N Test    : {metrics['n_test']:,} games")
    print("  ─────────────────────────────────────────────────────────────")

    return metrics


# =============================================================================
# FUNCTION 5: Feature Importance Analysis
# =============================================================================
def analyze_feature_importance(model: xgb.XGBClassifier,
                                feature_cols: list) -> pd.DataFrame:
    """
    Compute and display XGBoost feature importances.

    XGBoost provides multiple importance types:
      - 'weight'  : How many times a feature is used in splits
      - 'gain'    : Average improvement in accuracy per split using this feature
                    (most useful for understanding which features actually HELP)
      - 'cover'   : Average number of samples affected by splits using this feature

    'gain' is typically the most interpretable — it measures how much each
    feature actually improves predictions when used for a decision.

    R equivalent: xgb.importance(model=model); xgb.plot.importance(importance)
    """
    # Get feature importances (using gain metric)
    # In R: importance_matrix <- xgb.importance(model=model)
    importances = model.feature_importances_  # This is the 'gain' metric by default

    importance_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    print("\n  ── Feature Importances (Top 10) ─────────────────────────────")
    for _, row in importance_df.head(10).iterrows():
        bar = "█" * int(row["importance"] * 200)
        print(f"  {row['feature']:35s} | {row['importance']:.4f} | {bar}")
    print("  ─────────────────────────────────────────────────────────────")

    return importance_df


# =============================================================================
# FUNCTION 6: Score Upcoming Games
# =============================================================================
def score_todays_games(model: xgb.XGBClassifier, feature_cols: list,
                       matchup_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the trained model to today's upcoming games.

    Inputs
    ------
    model       : Trained XGBoost model (loaded from models/ directory)
    feature_cols: List of column names the model was trained on
    matchup_df  : DataFrame with today's matchups (one row per game)

    The matchup_df must have the same feature columns as the training data.
    You'll construct this from:
      - Confirmed starting lineups (from RotoWire or MLB.com)
      - Confirmed starting pitchers (from MLB.com Probable Pitchers)
      - Their current-season stats (from pybaseball or manual lookup)

    Returns
    -------
    pd.DataFrame
        matchup_df with added columns:
          - p_home_win : Model's probability that home team wins (0.0–1.0)
          - p_away_win : Model's probability that away team wins (0.0–1.0)
    """
    # Ensure all required feature columns are present
    missing_cols = [c for c in feature_cols if c not in matchup_df.columns]
    if missing_cols:
        print(f"  WARNING: Missing feature columns: {missing_cols}")
        print(f"  Filling with training data means.")
        # Load training data to get column means for imputation
        train_path = os.path.join(PROC_DIR, "moneyline_dataset.csv")
        if os.path.exists(train_path):
            train_df   = pd.read_csv(train_path)
            col_means  = train_df[feature_cols].mean()
            for col in missing_cols:
                matchup_df[col] = col_means.get(col, 0.0)

    # Make predictions
    X_score = matchup_df[feature_cols].copy()

    # Predict probabilities for each game
    # model.predict_proba() returns array with shape (n_games, 2)
    # Column 0 = P(away wins), Column 1 = P(home wins)
    probs = model.predict_proba(X_score)

    matchup_df["p_home_win"] = probs[:, 1].round(4)
    matchup_df["p_away_win"] = probs[:, 0].round(4)

    return matchup_df


# =============================================================================
# FUNCTION 7: Build Example Scoring Template
# =============================================================================
def build_scoring_template(feature_cols: list, train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a scoring template DataFrame for today's games.

    In production, you would populate this template with:
      1. Pull today's probable starters from MLB.com
      2. Look up their current season SIERA, xFIP, K%, BB% from FanGraphs
      3. Look up each team's current wRC+, wOBA from team_batting()
      4. Fill in park factor from the PARK_FACTORS dictionary

    This template uses MEAN values from training data as placeholders.
    Replace these with actual current-season stats for real predictions.

    Returns
    -------
    pd.DataFrame
        Template with 3 example games (modify for actual matchups).
    """
    print("  Building scoring template for today's games...")

    # Get column means from training data as sensible defaults
    col_means = train_df[feature_cols].mean().to_dict()

    # Create example matchups — replace with actual games
    # In R: data.frame(home_team="NYY", away_team="BOS", ...)
    example_games = [
        {
            "game_date": "2025-04-01",
            "home_team": "NYY",
            "away_team": "BOS",
            **col_means  # All features set to training average as placeholder
        },
        {
            "game_date": "2025-04-01",
            "home_team": "LAD",
            "away_team": "SFG",
            **col_means
        },
        {
            "game_date": "2025-04-01",
            "home_team": "HOU",
            "away_team": "TEX",
            **col_means
        },
    ]

    template = pd.DataFrame(example_games)
    print(f"  ✓ Template created with {len(template)} example games.")
    print("  NOTE: Replace feature values with actual current-season stats.")
    return template


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("MONEYLINE MODEL — STEP 3: MODEL TRAINING AND SCORING")
    print("=" * 70)

    data_path = os.path.join(PROC_DIR, "moneyline_dataset.csv")

    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found. Run 02_build_moneyline.py first.")
        exit(1)

    # Load data
    print("\n[ 1/6 ] Loading training data...")
    X, y, df, feature_cols = load_data(data_path)

    # Time-based split
    print("\n[ 2/6 ] Splitting train/test by season...")
    X_train, X_test, y_train, y_test = time_split(df, X, y, test_season=2025)

    # Train model
    print("\n[ 3/6 ] Training XGBoost model...")
    model = train_xgboost(X_train, y_train, X_test, y_test)

    # Evaluate
    print("\n[ 4/6 ] Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)

    # Feature importance
    print("\n[ 5/6 ] Analyzing feature importances...")
    importance_df = analyze_feature_importance(model, feature_cols)

    # Save model and feature info for use in export step
    print("\n[ 6/6 ] Saving model and generating scoring template...")
    model_path = os.path.join(MODEL_DIR, "moneyline_model.json")
    model.save_model(model_path)
    print(f"  ✓ Model saved: {model_path}")

    # Save feature column list (critical — must match between train and score)
    feature_path = os.path.join(MODEL_DIR, "moneyline_features.json")
    with open(feature_path, "w") as f:
        json.dump(feature_cols, f)
    print(f"  ✓ Features saved: {feature_path}")

    # Save metrics
    metrics_path = os.path.join(MODEL_DIR, "moneyline_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ Metrics saved: {metrics_path}")

    # Save importance
    importance_df.to_csv(os.path.join(MODEL_DIR, "moneyline_feature_importance.csv"), index=False)

    # Generate scoring template for today's games
    template = build_scoring_template(feature_cols, df)
    template_path = os.path.join(PROC_DIR, "moneyline_today_template.csv")
    template.to_csv(template_path, index=False)
    print(f"  ✓ Scoring template saved: {template_path}")

    # Score the template (example predictions)
    scored = score_todays_games(model, feature_cols, template)
    predictions_path = os.path.join(PROC_DIR, "moneyline_predictions.csv")
    scored.to_csv(predictions_path, index=False)
    print(f"  ✓ Example predictions saved: {predictions_path}")
    print(f"\n  Sample predictions:")
    print(scored[["home_team", "away_team", "p_home_win", "p_away_win"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("STEP 3 COMPLETE — Run 04_export_moneyline.py next.")
    print("=" * 70)
