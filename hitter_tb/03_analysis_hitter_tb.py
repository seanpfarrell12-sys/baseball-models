"""
=============================================================================
HITTER TOTAL BASES MODEL — FILE 3 OF 4: MODEL TRAINING  (REFACTORED)
=============================================================================
Purpose : Train an XGBoost multinomial classifier that outputs a full
          discrete probability distribution over total-bases outcomes.

Output per batter-game:
  p_tb_0    P(total bases = 0)
  p_tb_1    P(total bases = 1)
  p_tb_2    P(total bases = 2)
  p_tb_3    P(total bases = 3)
  p_tb_4p   P(total bases ≥ 4)

  Derived from the probability vector:
  p_over_0_5  = 1 - p_tb_0
  p_over_1_5  = 1 - p_tb_0 - p_tb_1
  p_over_2_5  = p_tb_3 + p_tb_4p
  p_over_3_5  = p_tb_4p

PA-volume weighting:
  Each batter's game-level probability vector P(TB=k | 1 PA) is convolved
  over their projected plate-appearance volume (pa_proj from batting slot)
  using a Monte Carlo simulation.  This correctly handles the fact that a
  leadoff hitter seeing 4.3 PA has more total-bases opportunities than a
  #9 hitter seeing 3.9 PA.

Model: XGBoost multi:softprob
  - num_class = 5  (classes 0, 1, 2, 3, 4+)
  - Objective: multi:softprob  → raw output = n_samples × 5 probability matrix
  - Walk-forward CV (train < year N, test = year N)
  - Class weights computed per fold to correct class imbalance
    (TB=0 ~40%, TB=1 ~30%, TB=2 ~18%, TB=3 ~8%, TB=4+ ~4%)
  - Evaluation: multi-class log loss, accuracy, and calibration per class

MC PA convolution:
  For a batter with pa_proj = 4.1 and per-PA probabilities p = [p0,p1,p2,p3,p4+]:
    1. Draw n_sim Poisson(4.1) samples → integer PA counts per simulation
    2. For each draw of k PA, sample k outcomes i.i.d. from p
    3. Sum bases (0×n0 + 1×n1 + 2×n2 + 3×n3 + 4×n4+) → game TB distribution
    4. Compute P(game_TB = 0), P(=1), ... from simulation histogram

Input  : data/processed/hitter_tb_dataset.csv
Output : models/hitter_tb_model.json
         models/hitter_tb_features.json
         models/hitter_tb_metrics.json
         models/hitter_tb_feature_importance.csv
=============================================================================
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import log_loss, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR  = os.path.join(BASE_DIR, "data", "models")
EXPORT_DIR = os.path.join(BASE_DIR, "exports")

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

N_CLASSES  = 5          # TB = 0, 1, 2, 3, 4+
N_SIM      = 100_000    # MC simulations for PA convolution
RNG        = np.random.default_rng(42)

# Batting slot → projected PA (from 02_build)
SLOT_PA_PROJ = {
    1: 4.33, 2: 4.27, 3: 4.22, 4: 4.17,
    5: 4.10, 6: 4.05, 7: 4.00, 8: 3.95, 9: 3.90,
}
DEFAULT_PA_PROJ = 4.10


# =============================================================================
# LOAD DATA
# =============================================================================
def load_data(path: str) -> tuple:
    df = pd.read_csv(path, low_memory=False)
    print(f"  {len(df):,} batter-games loaded across seasons: "
          f"{sorted(df['season'].unique())}")
    print(f"  TB class distribution:\n{df['tb_actual'].value_counts().sort_index()}")

    exclude = {"game_date", "season", "team", "batter_mlbam", "batter_retro",
               "team_retro", "sp_mlbam", "sp_retro", "sp_hand", "home_flag",
               "tb_actual", "feature_season"}
    feature_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

    X = df[feature_cols].copy()
    y = df["tb_actual"].astype(int).copy()
    return X, y, df, feature_cols


# =============================================================================
# TRAIN XGBoost MULTICLASS (internal helper)
# =============================================================================
def _train_xgb(X_train: pd.DataFrame, y_train: pd.Series,
               X_val: pd.DataFrame = None,
               y_val: pd.Series = None) -> xgb.XGBClassifier:
    # Class imbalance correction via sample weights
    sample_weights = compute_sample_weight("balanced", y_train)

    params = {
        "n_estimators":     500,
        "max_depth":        4,
        "learning_rate":    0.03,
        "subsample":        0.75,
        "colsample_bytree": 0.75,
        "min_child_weight": 8,
        "reg_alpha":        0.5,
        "reg_lambda":       2.0,
        "objective":        "multi:softprob",
        "num_class":        N_CLASSES,
        "eval_metric":      "mlogloss",
        "random_state":     42,
        "tree_method":      "hist",
        "verbosity":        0,
    }
    model = xgb.XGBClassifier(**params)

    if X_val is not None and y_val is not None:
        model.fit(X_train, y_train,
                  sample_weight=sample_weights,
                  eval_set=[(X_val, y_val)],
                  verbose=False)
    else:
        model.fit(X_train, y_train,
                  sample_weight=sample_weights,
                  verbose=False)
    return model


# =============================================================================
# WALK-FORWARD CROSS-VALIDATION
# =============================================================================
def walk_forward_cv(df: pd.DataFrame, X: pd.DataFrame,
                    y: pd.Series, feature_cols: list) -> list:
    seasons = sorted(df["season"].unique())
    if len(seasons) < 2:
        print("  Need at least 2 seasons for walk-forward CV.")
        return []

    fold_metrics = []
    for i in range(1, len(seasons)):
        test_season = seasons[i]
        train_mask  = df["season"] < test_season
        test_mask   = df["season"] == test_season

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        if len(X_train) < 200 or len(X_test) < 50:
            continue

        model  = _train_xgb(X_train, y_train, X_test, y_test)
        y_prob = model.predict_proba(X_test)   # shape (n, 5)
        y_pred = y_prob.argmax(axis=1)

        # Per-class calibration: mean predicted vs actual rate for each class
        class_cal = {}
        for k in range(N_CLASSES):
            actual_rate    = (y_test == k).mean()
            predicted_rate = y_prob[:, k].mean()
            class_cal[f"class_{k}_actual"]    = round(actual_rate,    4)
            class_cal[f"class_{k}_predicted"] = round(predicted_rate, 4)

        m = {
            "fold":      f"train<{test_season} / test={test_season}",
            "n_train":   len(y_train),
            "n_test":    len(y_test),
            "log_loss":  round(log_loss(y_test, y_prob, labels=list(range(N_CLASSES))), 4),
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            **class_cal,
        }
        fold_metrics.append(m)

        print(f"\n  Fold: {m['fold']}")
        print(f"    Train {m['n_train']:,}  |  Test {m['n_test']:,}")
        print(f"    Log Loss : {m['log_loss']:.4f}  (5-class random ≈ {np.log(N_CLASSES):.3f})")
        print(f"    Accuracy : {m['accuracy']:.4f}")
        print(f"    Class calibration (actual → predicted):")
        for k in range(N_CLASSES):
            label = f"TB={k}" if k < 4 else "TB=4+"
            act   = m[f"class_{k}_actual"]
            pred  = m[f"class_{k}_predicted"]
            print(f"      {label}:  {act:.3f} → {pred:.3f}")

    return fold_metrics


# =============================================================================
# TRAIN FINAL MODEL
# =============================================================================
def train_final_model(X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
    print(f"\n  Training final XGBoost on all {len(X):,} batter-games...")
    model = _train_xgb(X, y)
    print("  ✓ Final model trained.")
    return model


# =============================================================================
# MC PA CONVOLUTION — game-level TB distribution
# =============================================================================
def pa_convolution(per_pa_probs: np.ndarray, pa_proj: float,
                   n_sim: int = N_SIM) -> dict:
    """
    Given:
      per_pa_probs : array shape (5,) — P(TB=k | single PA) for k in 0..4+
      pa_proj      : expected number of PA (float, e.g. 4.27 for slot-2)

    Returns a dict with:
      p_tb_0 .. p_tb_4p      — P(game TB = k)
      p_over_0_5 .. p_over_3_5 — derived O/U probabilities
      exp_tb                 — E[game TB]

    Method:
      1. Draw PA counts from Poisson(pa_proj) for each simulation
      2. For each simulation, draw n_pa outcomes i.i.d. from per_pa_probs
      3. Compute game TB = sum of bases (1B→1, 2B→2, 3B→3, HR→4)
      4. Histogram over N_SIM draws
    """
    per_pa_probs = np.asarray(per_pa_probs, dtype=float)
    per_pa_probs = np.clip(per_pa_probs, 0, None)
    per_pa_probs /= per_pa_probs.sum()  # re-normalise for safety

    # PA counts per simulation ~ Poisson(pa_proj)
    pa_counts = RNG.poisson(pa_proj, size=n_sim)

    # Bases produced per PA: 0, 1, 2, 3, 4 (4+ capped at 4 for expected value)
    bases_per_outcome = np.array([0, 1, 2, 3, 4], dtype=float)

    # Vectorised simulation:
    # Build a lookup of game-TB for each (sim, pa_count) combination
    # For each unique pa_count, sample that many outcomes from per_pa_probs
    game_tb = np.zeros(n_sim, dtype=float)
    for k in np.unique(pa_counts):
        if k == 0:
            continue
        mask = pa_counts == k
        n_at_k = mask.sum()
        # shape (n_at_k, k) — each row is one sim with k PA
        outcomes = RNG.choice(
            a=len(per_pa_probs),
            size=(n_at_k, k),
            p=per_pa_probs
        )
        game_tb[mask] = bases_per_outcome[outcomes].sum(axis=1)

    # Histogram
    bins = np.arange(0, game_tb.max() + 2)
    hist, _ = np.histogram(game_tb, bins=bins - 0.5)
    p_vec = hist / hist.sum()

    # Collapse to at most 5 classes (0, 1, 2, 3, 4+)
    p = np.zeros(5)
    for i, prob in enumerate(p_vec):
        cls = min(i, 4)
        p[cls] += prob

    return {
        "p_tb_0":     float(p[0]),
        "p_tb_1":     float(p[1]),
        "p_tb_2":     float(p[2]),
        "p_tb_3":     float(p[3]),
        "p_tb_4p":    float(p[4]),
        "p_over_0_5": float(1.0 - p[0]),
        "p_over_1_5": float(1.0 - p[0] - p[1]),
        "p_over_2_5": float(p[3] + p[4]),
        "p_over_3_5": float(p[4]),
        "exp_tb":     float(np.dot(np.arange(5), p)),
    }


# =============================================================================
# SCORE DATASET  (attach game-level probabilities to each batter row)
# =============================================================================
def score_dataset(model: xgb.XGBClassifier,
                  X: pd.DataFrame,
                  df: pd.DataFrame,
                  feature_cols: list) -> pd.DataFrame:
    """
    Compute per-PA probabilities from the model, then run PA convolution
    for each row to produce game-level probability vectors.
    """
    print("\n  Scoring dataset with PA convolution...")
    per_pa_probs = model.predict_proba(X[feature_cols])  # (n, 5)

    result_rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        pa_proj = row.get("pa_proj", DEFAULT_PA_PROJ)
        if pd.isna(pa_proj):
            pa_proj = DEFAULT_PA_PROJ

        game_probs = pa_convolution(per_pa_probs[i], float(pa_proj))
        result_rows.append({
            "game_date":    row.get("game_date"),
            "season":       row.get("season"),
            "team":         row.get("team"),
            "batter_mlbam": row.get("batter_mlbam"),
            "batting_slot": row.get("batting_slot"),
            "pa_proj":      pa_proj,
            "tb_actual":    row.get("tb_actual"),
            **game_probs,
        })

    scored = pd.DataFrame(result_rows)
    print(f"  ✓ {len(scored):,} rows scored")
    return scored


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================
def feature_importance_report(model: xgb.XGBClassifier,
                               feature_cols: list) -> pd.DataFrame:
    imp = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\n  ── Top 15 Feature Importances ─────────────────────────────────")
    for _, row in imp.head(15).iterrows():
        bar = "█" * max(1, int(row["importance"] * 300))
        print(f"  {row['feature']:40s} | {row['importance']:.4f} | {bar}")
    print("  ────────────────────────────────────────────────────────────────")
    return imp


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("HITTER TB MODEL — STEP 3: MODEL TRAINING (REFACTORED)")
    print("=" * 70)

    data_path = os.path.join(PROC_DIR, "hitter_tb_dataset.csv")
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found.  Run 02_build_hitter_tb.py first.")
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
        avg_ll  = np.mean([m["log_loss"]  for m in cv_results])
        avg_acc = np.mean([m["accuracy"]  for m in cv_results])
        print(f"\n  ── Walk-Forward CV Summary ──")
        print(f"  Avg Log Loss : {avg_ll:.4f}  (5-class random = {np.log(N_CLASSES):.3f})")
        print(f"  Avg Accuracy : {avg_acc:.4f}")

    # Train final model
    print("\n[ 3/5 ] Training final model (all seasons)...")
    final_model = train_final_model(X, y)

    # Score with PA convolution + evaluate on most-recent season
    print("\n[ 4/5 ] PA convolution scoring + evaluation on most-recent season...")
    scored = score_dataset(final_model, X, df, feature_cols)

    last_season  = sorted(df["season"].unique())[-1]
    scored_last  = scored[scored["season"] == last_season].copy()
    labeled_last = scored_last.dropna(subset=["tb_actual"])

    if len(labeled_last) > 0:
        y_true_last = labeled_last["tb_actual"].astype(int).values
        per_pa_last = final_model.predict_proba(
            X[df["season"] == last_season][feature_cols]
        )
        ll_last  = log_loss(y_true_last, per_pa_last,
                            labels=list(range(N_CLASSES)))
        acc_last = accuracy_score(y_true_last, per_pa_last.argmax(axis=1))

        print(f"\n  Final model — season {last_season} ({len(labeled_last):,} games):")
        print(f"    Log Loss : {ll_last:.4f}")
        print(f"    Accuracy : {acc_last:.4f}")

        # O/U calibration on scored data
        print(f"\n    O/U line calibration (mean model P vs actual rate):")
        for line_col, desc in [("p_over_0_5", "Over 0.5 TB"),
                                ("p_over_1_5", "Over 1.5 TB"),
                                ("p_over_2_5", "Over 2.5 TB"),
                                ("p_over_3_5", "Over 3.5 TB")]:
            threshold = float(line_col.split("_")[2]) / 10.0
            actual_rate = (labeled_last["tb_actual"] > threshold + 0.5 - 1).mean()
            pred_rate   = labeled_last[line_col].mean()
            # Recalculate actual over rate from raw tb_actual
            if "0.5" in line_col:
                actual_rate = (labeled_last["tb_actual"] > 0).mean()
            elif "1.5" in line_col:
                actual_rate = (labeled_last["tb_actual"] > 1).mean()
            elif "2.5" in line_col:
                actual_rate = (labeled_last["tb_actual"] > 2).mean()
            elif "3.5" in line_col:
                actual_rate = (labeled_last["tb_actual"] > 3).mean()
            print(f"    {desc}: model={pred_rate:.3f}  actual={actual_rate:.3f}  "
                  f"diff={pred_rate - actual_rate:+.3f}")

        final_metrics = {
            "log_loss_last_season":  round(ll_last, 4),
            "accuracy_last_season":  round(acc_last, 4),
            "n_test":                int(len(labeled_last)),
            "test_season":           int(last_season),
            "n_features":            len(feature_cols),
            "n_classes":             N_CLASSES,
            "cv_folds":              cv_results,
        }
    else:
        final_metrics = {
            "n_features":  len(feature_cols),
            "n_classes":   N_CLASSES,
            "cv_folds":    cv_results,
        }

    # Feature importance
    print("\n[ 5/5 ] Feature importances + saving artifacts...")
    imp_df = feature_importance_report(final_model, feature_cols)

    # ── Save artifacts ─────────────────────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, "hitter_tb_model.json")
    final_model.save_model(model_path)
    print(f"  ✓ XGBoost model         : {model_path}")

    feat_path = os.path.join(MODEL_DIR, "hitter_tb_features.json")
    with open(feat_path, "w") as f:
        json.dump(feature_cols, f)
    print(f"  ✓ Feature list          : {feat_path}")

    met_path = os.path.join(MODEL_DIR, "hitter_tb_metrics.json")
    with open(met_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"  ✓ Metrics               : {met_path}")

    imp_path = os.path.join(MODEL_DIR, "hitter_tb_feature_importance.csv")
    imp_df.to_csv(imp_path, index=False)
    print(f"  ✓ Feature importances   : {imp_path}")

    # Save scored training data for inspection
    scored_path = os.path.join(PROC_DIR, "hitter_tb_scored_train.csv")
    scored.to_csv(scored_path, index=False)
    print(f"  ✓ Scored training data  : {scored_path}")

    print("\n" + "=" * 70)
    print("STEP 3 COMPLETE — Run 04_export_hitter_tb.py for today's picks.")
    print("=" * 70)
