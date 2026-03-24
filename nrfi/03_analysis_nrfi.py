"""
=============================================================================
NRFI / YRFI MODEL — FILE 3 OF 4: MODEL TRAINING AND CALIBRATION
=============================================================================
Model: XGBoost binary classifier with Isotonic Regression calibration.

Target: yrfi ∈ {0, 1}
  1 = at least one run scored in the 1st inning by either team (YRFI)
  0 = no runs in the 1st inning (NRFI)

Architecture:
  - XGBoost binary:logistic base model captures non-linear feature interactions
    (e.g., the combined effect of a high-ISO lineup + HR-friendly park).
  - Isotonic regression post-hoc calibration maps raw XGBoost probabilities
    to well-calibrated P(YRFI) values.
    Rationale: raw XGBoost probabilities are often overconfident; isotonic
    regression is monotonic and does not reduce predictive rank order.
  - Walk-forward temporal CV: train on prior seasons, evaluate on holdout season.
    This mimics the actual deployment scenario and avoids future leakage.

Walk-forward CV design:
  e.g.,  Fold 1: train 2023,       eval 2024
         Fold 2: train 2023-2024,  eval 2025
  Evaluation metrics per fold:
    - AUC-ROC: overall discrimination between YRFI and NRFI games
    - Brier score: probability calibration quality
    - Log-loss: sharpness + calibration combined
    - Calibration curve: reliability diagram binned into deciles

Outputs (data/models/):
  nrfi_model.json           — trained XGBoost model
  nrfi_calibrator.pkl       — fitted isotonic calibration transformer
  nrfi_features.json        — ordered feature list for scoring
  nrfi_cv_metrics.json      — walk-forward CV metrics per fold
  nrfi_calibration_plot.png — reliability diagram (calibration curve)

Input  : data/processed/nrfi_dataset.csv
=============================================================================
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless backend — no display required
import matplotlib.pyplot as plt

from sklearn.isotonic           import IsotonicRegression
from sklearn.calibration        import calibration_curve
from sklearn.metrics            import (roc_auc_score, brier_score_loss,
                                        log_loss)
from sklearn.model_selection    import train_test_split
import xgboost as xgb

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR  = os.path.join(BASE_DIR, "data", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_SEED = 42
RNG         = np.random.default_rng(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUPS
# ─────────────────────────────────────────────────────────────────────────────
# All feature names must match columns produced by 02_build_nrfi.py.
# Groups are defined for interpretability and ablation studies.

SP_FEATURES = [
    # Home SP (pitching in the top half of 1st; limiting away team)
    "home_sp_fi_era",           # first-inning ERA (prior year)
    "home_sp_fi_k_pct",         # first-inning strikeout rate
    "home_sp_fi_bb_pct",        # first-inning walk rate
    "home_sp_fi_hr_per_9",      # first-inning HR/9
    "home_sp_fi_whiff_pct",     # first-inning swinging strike rate
    "home_sp_stuff_plus",       # FanGraphs Stuff+ (overall pitch quality)
    "home_sp_location_plus",    # FanGraphs Location+ (command/control)
    "home_sp_swstr_pct",        # SwStr% (proxy if Stuff+ unavailable)
    "home_sp_f_strike_pct",     # First-pitch strike rate
    "home_sp_is_lhp",           # 1 = LHP, 0 = RHP (handedness)
    # Away SP (pitching in the bottom half of 1st; limiting home team)
    "away_sp_fi_era",
    "away_sp_fi_k_pct",
    "away_sp_fi_bb_pct",
    "away_sp_fi_hr_per_9",
    "away_sp_fi_whiff_pct",
    "away_sp_stuff_plus",
    "away_sp_location_plus",
    "away_sp_swstr_pct",
    "away_sp_f_strike_pct",
    "away_sp_is_lhp",
]

LINEUP_FEATURES = [
    # Home top-3 batting stats vs opposing SP hand
    "home_top3_wrc_plus",       # weighted runs created+ (vs SP hand)
    "home_top3_obp",            # on-base percentage
    "home_top3_iso",            # isolated power (extra bases per AB)
    "home_top3_k_pct",          # strikeout rate
    "home_top3_bb_pct",         # walk rate
    # Away top-3 batting stats vs opposing SP hand
    "away_top3_wrc_plus",
    "away_top3_obp",
    "away_top3_iso",
    "away_top3_k_pct",
    "away_top3_bb_pct",
]

ENVIRONMENTAL_FEATURES = [
    "temperature_f",            # game-time temperature
    "wind_toward_cf",           # wind component toward CF (+ve = tailwind)
    "humidity_pct",             # relative humidity
    "hr_park_factor",           # park HR factor (100 = average)
    "hr_environment",           # composite HR index
    "temp_carry_factor",        # temperature HR carry boost
    "alt_carry_factor",         # altitude HR carry boost
    "altitude_ft",              # park altitude
    "is_dome",                  # 1 = dome/retractable (weather irrelevant)
]

INTERACTION_FEATURES = [
    "combined_fi_bb_pct",       # avg BB% from both SPs (walk pressure)
    "home_lineup_vs_away_sp",   # home wRC+ × (1 - away SP K%)
    "away_lineup_vs_home_sp",   # away wRC+ × (1 - home SP K%)
    "home_hr_threat",           # hr_environment × home top3 ISO
    "away_hr_threat",           # hr_environment × away top3 ISO
]

ALL_FEATURES = (SP_FEATURES + LINEUP_FEATURES +
                ENVIRONMENTAL_FEATURES + INTERACTION_FEATURES)

TARGET       = "yrfi"
GROUP_COL    = "season"
ID_COLS      = ["game_pk", "game_date", "home_team", "away_team"]

# ─────────────────────────────────────────────────────────────────────────────
# XGBOOST PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
XGB_PARAMS = {
    "objective":        "binary:logistic",
    "eval_metric":      "logloss",
    "n_estimators":     400,
    "learning_rate":    0.04,
    "max_depth":        4,          # shallow trees → less overfitting on ~2K games/yr
    "subsample":        0.80,
    "colsample_bytree": 0.75,
    "min_child_weight": 20,         # conservative: each leaf requires 20+ games
    "gamma":            1.0,        # minimum loss reduction to make a split
    "reg_alpha":        0.1,
    "reg_lambda":       1.5,
    "random_state":     RANDOM_SEED,
    "verbosity":        0,
    "use_label_encoder": False,
}


# =============================================================================
# LOAD DATASET
# =============================================================================

def load_dataset() -> pd.DataFrame:
    path = os.path.join(PROC_DIR, "nrfi_dataset.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found — run 02_build_nrfi.py first")
    df = pd.read_csv(path, low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["season"]    = pd.to_numeric(df["season"], errors="coerce")

    # Keep only columns we need (features + target + IDs)
    keep = [c for c in ALL_FEATURES + [TARGET, GROUP_COL] + ID_COLS
            if c in df.columns]
    df = df[keep].copy()

    # Final guard: ensure all feature columns exist (fill with NaN if absent)
    for feat in ALL_FEATURES:
        if feat not in df.columns:
            df[feat] = np.nan

    print(f"  Dataset loaded: {len(df):,} games | "
          f"YRFI rate: {df[TARGET].mean():.3f}")
    return df


# =============================================================================
# WALK-FORWARD CROSS-VALIDATION
# =============================================================================

def walk_forward_cv(df: pd.DataFrame) -> dict:
    """
    Walk-forward temporal CV: train on all prior seasons, evaluate on the
    next season. Returns a dict of per-fold metrics.

    Calibration is applied within each fold (fit isotonic on val set ← leaky
    for real deployment; used here only for diagnostic purposes).
    For the final production model, calibration is fit on a held-out 20%
    sample of the full training set.
    """
    seasons   = sorted(df[GROUP_COL].dropna().unique())
    n_seasons = len(seasons)

    if n_seasons < 2:
        print("  WARNING: need ≥2 seasons for walk-forward CV — skipping")
        return {}

    all_metrics = {}

    # Use only columns that exist in the dataset
    feat_cols = [c for c in ALL_FEATURES if c in df.columns]

    for i in range(1, n_seasons):
        train_seasons = seasons[:i]
        eval_season   = seasons[i]

        train_df = df[df[GROUP_COL].isin(train_seasons)].copy()
        eval_df  = df[df[GROUP_COL] == eval_season].copy()

        if len(train_df) < 100 or len(eval_df) < 30:
            print(f"  Fold train={train_seasons}→eval={eval_season}: "
                  f"insufficient data, skipping")
            continue

        X_train = train_df[feat_cols].values
        y_train = train_df[TARGET].values
        X_eval  = eval_df[feat_cols].values
        y_eval  = eval_df[TARGET].values

        # ── Train model ────────────────────────────────────────────────────
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_eval, y_eval)],
            verbose=False,
        )

        raw_probs  = model.predict_proba(X_eval)[:, 1]

        # ── Isotonic calibration (diagnostic only — fit on eval for display) -
        # In production calibration is fit on a hold-out split of the full set.
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        # Use a 50/50 split of the eval set for diagnostic calibration
        n_cal   = len(y_eval) // 2
        iso.fit(raw_probs[:n_cal], y_eval[:n_cal])
        cal_probs = iso.predict(raw_probs[n_cal:])
        y_eval_h  = y_eval[n_cal:]

        # ── Metrics ────────────────────────────────────────────────────────
        auc    = roc_auc_score(y_eval, raw_probs)
        brier  = brier_score_loss(y_eval, raw_probs)
        ll     = log_loss(y_eval, raw_probs)
        cal_bs = brier_score_loss(y_eval_h, cal_probs) if len(y_eval_h) > 10 else np.nan

        fold_key = f"train_{min(train_seasons)}-{max(train_seasons)}_eval_{eval_season}"
        all_metrics[fold_key] = {
            "train_seasons":  [int(s) for s in train_seasons],
            "eval_season":    int(eval_season),
            "n_train":        int(len(train_df)),
            "n_eval":         int(len(eval_df)),
            "auc_roc":        round(float(auc), 4),
            "brier_score":    round(float(brier), 4),
            "log_loss":       round(float(ll), 4),
            "cal_brier":      round(float(cal_bs), 4) if not np.isnan(cal_bs) else None,
            "yrfi_rate_eval": round(float(y_eval.mean()), 4),
        }
        print(f"  Fold {fold_key}: "
              f"AUC={auc:.4f}  Brier={brier:.4f}  LL={ll:.4f}")

    return all_metrics


# =============================================================================
# FINAL MODEL TRAINING
# =============================================================================

def train_final_model(df: pd.DataFrame) -> tuple:
    """
    Train XGBoost on the full dataset (all available seasons) with isotonic
    calibration on a held-out 20% validation split.

    Returns:
      model       — trained XGBClassifier
      calibrator  — fitted IsotonicRegression
      feat_cols   — ordered list of feature names used
      importances — dict of feature → importance score
    """
    feat_cols = [c for c in ALL_FEATURES if c in df.columns]

    X = df[feat_cols].values
    y = df[TARGET].values

    # ── Train / calibration split (temporal: use last ~20% of data) ──────────
    seasons     = sorted(df[GROUP_COL].dropna().unique())
    cal_season  = seasons[-1]  # hold out most recent season for calibration

    train_mask  = df[GROUP_COL] < cal_season
    cal_mask    = df[GROUP_COL] == cal_season

    X_train, y_train = X[train_mask], y[train_mask]
    X_cal,   y_cal   = X[cal_mask],   y[cal_mask]

    if len(X_cal) < 50:
        # Not enough data in holdout → fall back to random 20% split
        print("  NOTE: last-season holdout too small — using random 20% for calibration")
        X_train, X_cal, y_train, y_cal = train_test_split(
            X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
        )

    print(f"  Training set:     {len(X_train):,} games")
    print(f"  Calibration set:  {len(X_cal):,} games")

    # ── XGBoost ──────────────────────────────────────────────────────────────
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_cal, y_cal)],
        verbose=False,
    )

    # ── Isotonic calibration ─────────────────────────────────────────────────
    raw_cal = model.predict_proba(X_cal)[:, 1]
    iso     = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_cal, y_cal)

    # ── Feature importances (gain-based) ─────────────────────────────────────
    imp_raw   = model.get_booster().get_score(importance_type="gain")
    # Map f0, f1, ... back to feature names
    imp_named = {feat_cols[int(k[1:])]: v
                 for k, v in imp_raw.items()
                 if k[1:].isdigit() and int(k[1:]) < len(feat_cols)}
    imp_sorted = dict(sorted(imp_named.items(),
                             key=lambda x: x[1], reverse=True))

    print(f"\n  Top-10 feature importances (gain):")
    for i, (feat, gain) in enumerate(list(imp_sorted.items())[:10]):
        print(f"    {i+1:2d}. {feat:<40s} {gain:.1f}")

    return model, iso, feat_cols, imp_sorted


# =============================================================================
# CALIBRATION DIAGNOSTIC PLOT
# =============================================================================

def plot_calibration(model, calibrator, feat_cols: list,
                     df: pd.DataFrame) -> None:
    """
    Reliability diagram comparing raw XGBoost vs. isotonic-calibrated
    probabilities against observed YRFI rates.  Saved to data/models/.
    """
    X = df[feat_cols].values
    y = df[TARGET].values

    raw_probs  = model.predict_proba(X)[:, 1]
    cal_probs  = calibrator.predict(raw_probs)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    for probs, label, color in [
        (raw_probs,  "XGBoost (uncalibrated)", "steelblue"),
        (cal_probs,  "Isotonic calibrated",    "tomato"),
    ]:
        frac_pos, mean_pred = calibration_curve(y, probs, n_bins=10)
        ax.plot(mean_pred, frac_pos, "o-", label=label, color=color)

    ax.set_xlabel("Mean predicted P(YRFI)")
    ax.set_ylabel("Fraction of YRFI outcomes")
    ax.set_title("NRFI/YRFI Model — Calibration Reliability Diagram")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out = os.path.join(MODEL_DIR, "nrfi_calibration_plot.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Calibration plot saved → {out}")


# =============================================================================
# SAVE ARTIFACTS
# =============================================================================

def save_artifacts(model, calibrator, feat_cols: list,
                   importances: dict, cv_metrics: dict) -> None:
    """Save all model artifacts to data/models/."""
    # XGBoost model
    model_path = os.path.join(MODEL_DIR, "nrfi_model.json")
    model.save_model(model_path)

    # Isotonic calibrator
    cal_path = os.path.join(MODEL_DIR, "nrfi_calibrator.pkl")
    with open(cal_path, "wb") as f:
        pickle.dump(calibrator, f)

    # Feature list
    feat_path = os.path.join(MODEL_DIR, "nrfi_features.json")
    with open(feat_path, "w") as f:
        json.dump({"features": feat_cols,
                   "target":   TARGET,
                   "n_features": len(feat_cols)}, f, indent=2)

    # CV metrics
    cv_path = os.path.join(MODEL_DIR, "nrfi_cv_metrics.json")
    with open(cv_path, "w") as f:
        json.dump(cv_metrics, f, indent=2)

    # Feature importances
    imp_path = os.path.join(MODEL_DIR, "nrfi_importances.json")
    with open(imp_path, "w") as f:
        json.dump(importances, f, indent=2)

    print(f"\n  Artifacts saved:")
    print(f"    {model_path}")
    print(f"    {cal_path}")
    print(f"    {feat_path}")
    print(f"    {cv_path}")
    print(f"    {imp_path}")


# =============================================================================
# MAIN
# =============================================================================

def train_nrfi_model() -> None:
    print("=" * 70)
    print("NRFI / YRFI MODEL — STEP 3: MODEL TRAINING")
    print("=" * 70)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n[ Load ] Dataset...")
    df = load_dataset()

    # ── Walk-forward CV ───────────────────────────────────────────────────────
    print("\n[ CV ] Walk-forward temporal cross-validation...")
    cv_metrics = walk_forward_cv(df)

    if cv_metrics:
        aucs = [m["auc_roc"] for m in cv_metrics.values()]
        print(f"\n  CV Summary: AUC-ROC = "
              f"{np.mean(aucs):.4f} ± {np.std(aucs):.4f} "
              f"across {len(aucs)} fold(s)")

    # ── Final model ───────────────────────────────────────────────────────────
    print("\n[ Train ] Final model (all seasons)...")
    model, calibrator, feat_cols, importances = train_final_model(df)

    # ── Calibration plot ──────────────────────────────────────────────────────
    print("\n[ Plot ] Calibration reliability diagram...")
    plot_calibration(model, calibrator, feat_cols, df)

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\n[ Save ] Writing artifacts to data/models/...")
    save_artifacts(model, calibrator, feat_cols, importances, cv_metrics)

    print("\n" + "=" * 70)
    print("STEP 3 COMPLETE — Run 04_export_nrfi.py next.")
    print("=" * 70)


if __name__ == "__main__":
    train_nrfi_model()
