"""
=============================================================================
PITCHER TOTAL OUTS MODEL — FILE 3 OF 4: SURVIVAL ANALYSIS + SIMULATION
=============================================================================
Architecture:
  This model treats pitcher removal as a discrete-time survival process.
  At each batter faced (BF) k = 1..27, the manager makes a latent decision:
  keep or remove.  The trained hazard function h(k | X) = P(removed at BF k)
  captures:

    (1) Pitcher skill effects      — K%, K-BB%, pitches_per_pa slow the hazard
    (2) TTOP structural spike      — hard increase in h(k) at k=19 (BF 19 = 3rd TTO)
    (3) Pitch count ceiling        — h(k) spikes sharply as est_pc_k → pc_limit
    (4) Manager-specific baseline  — Kevin Cash: h(k) elevated throughout;
                                     Bruce Bochy: h(k) suppressed
    (5) Opponent walk-rate effect  — high BB% lineup inflates est_pc_k faster

  The discrete-time hazard is fit as an XGBoost binary classifier on the
  BF-level expanded dataset from 02_build.

  Monte Carlo Simulation:
    For each SP/game context, simulate N games:
      while pitching and bf_k <= MAX_BF:
          h_k = model.predict_proba([state_k])[1]
          removed = Bernoulli(h_k)  +  Bernoulli(past_hard_limit) [forced removal]
          out_this_bf = Bernoulli(p_out_per_bf)  [1 - opp_OBP]
          outs_so_far += out_this_bf
          if removed: break
          bf_k += 1
      record outs_so_far

    The simulation produces a full empirical distribution over total outs,
    from which we directly read P(outs > line) without any normal-distribution
    assumption.  This correctly captures:
      - Right skew in the outs distribution (can't exceed 27)
      - Non-linear manager hook effects
      - Discrete nature of outs (integers 0..27)

  Cox PH Companion (optional):
    If lifelines is installed, a CoxPHFitter is also fit on the per-start
    data for interpretability.  The Cox model provides hazard ratios showing
    the effect of each feature on the removal hazard.

Walk-Forward CV:
  Fold 1: Train 2023       → Test 2024
  Fold 2: Train 2023-2024  → Test 2025
  Final:  Train all seasons → Production model

Input  : data/processed/pitcher_outs_per_start.csv
         data/processed/pitcher_outs_bf_level.csv
Output : models/pitcher_outs_hazard.json       (XGBoost hazard model)
         models/pitcher_outs_cox.pkl           (lifelines Cox model, if available)
         models/pitcher_outs_features.json
         models/pitcher_outs_metrics.json
         models/pitcher_outs_feature_importance.csv
=============================================================================
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

from sklearn.metrics import (roc_auc_score, log_loss, mean_absolute_error,
                              brier_score_loss)

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR  = os.path.join(BASE_DIR, "data", "models")
EXPORT_DIR = os.path.join(BASE_DIR, "exports")

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# Sportsbook prop lines (outs)
OUTS_PROP_LINES = [13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5]

# Simulation config
N_SIM     = 50_000
MAX_BF    = 27          # stop simulating after full 3rd time through
RNG       = np.random.default_rng(42)

# Hard removal multipliers (structural penalties applied in simulation)
# These encode the "hard mathematical penalty" at the 18th batter and TTOP
TTOP_HAZARD_MULTIPLIER      = 1.50   # 50% relative increase in h(k) at BF 19-27
PC_LIMIT_HAZARD_MULTIPLIER  = 3.00   # 3× spike when at/past pitch count limit
HARD_PC_LIMIT_REMOVAL_PROB  = 0.95   # near-certain removal past hard PC limit


# =============================================================================
# LOAD DATA
# =============================================================================
def load_data() -> tuple:
    per_start_path = os.path.join(PROC_DIR, "pitcher_outs_per_start.csv")
    bf_level_path  = os.path.join(PROC_DIR, "pitcher_outs_bf_level.csv")

    if not os.path.exists(per_start_path):
        raise FileNotFoundError(f"Missing: {per_start_path}. Run 02_build first.")

    starts_df = pd.read_csv(per_start_path, low_memory=False)
    print(f"  Per-start: {len(starts_df):,} starts, "
          f"{starts_df['season'].unique()} seasons")

    bf_df = pd.DataFrame()
    if os.path.exists(bf_level_path):
        bf_df = pd.read_csv(bf_level_path, low_memory=False)
        print(f"  BF-level:  {len(bf_df):,} rows, "
              f"event rate = {bf_df['event'].mean():.4f}")
    else:
        print(f"  WARNING: BF-level dataset not found. Will re-expand in memory.")

    return starts_df, bf_df


# =============================================================================
# FEATURE COLUMNS FOR HAZARD MODEL
# =============================================================================
EXCLUDE_COLS = {
    "game_date", "season", "team", "sp_mlbam",
    "outs_recorded", "censored", "event", "start_id",
}

# Time-varying features (change per BF row)
TV_FEAT_COLS = [
    "bf_k", "times_through_order",
    "is_first_through", "is_second_through", "is_ttop",
    "is_18th_batter", "is_19th_batter", "batters_into_ttop",
    "est_pc_k", "pc_fraction_k",
    "approaching_pc_limit", "at_pc_limit", "past_hard_limit",
    "ttop_x_low_patience", "pc_stress_k",
]

# Time-invariant features (same for all BF rows within a start)
TI_FEAT_COLS = [
    "k_pct", "bb_pct", "k_minus_bb_pct", "siera", "xfip",
    "swstr_pct", "fstrike_pct", "csw_pct", "avg_fb_velo",
    "xwoba_against", "barrel_pct",
    "pitches_per_pa", "effective_ppp",
    "opp_lg_avg_bb_pct", "opp_lg_avg_k_pct", "opp_wrc_plus", "opp_lg_avg_obp",
    "typical_pc_limit", "hard_pc_limit", "ttop_hook_rate",
    "depth_score", "mgr_avg_sp_outs",
    "est_pc_at_bf18", "pc_fraction_at_bf18",
    "efficiency_x_depth", "pc_headroom_at_ttop",
]


def get_hazard_feature_cols(df: pd.DataFrame) -> list:
    """All numeric columns present in df that belong to our feature set."""
    wanted = set(TV_FEAT_COLS + TI_FEAT_COLS)
    return [c for c in df.columns
            if c in wanted and pd.api.types.is_numeric_dtype(df[c])]


# =============================================================================
# TRAIN DISCRETE-TIME HAZARD MODEL
# =============================================================================
def train_hazard_model(bf_train: pd.DataFrame,
                        feature_cols: list,
                        bf_val: pd.DataFrame = None) -> xgb.XGBClassifier:
    """
    XGBoost binary classifier on the BF-level dataset.

    Label:  event = 1 (manager removes SP at this BF)
    Positive rate is very low (~1 / avg_bf ≈ 4-5%) — use scale_pos_weight.

    Regularisation is important: the model must not memorise which specific
    BF values are common removal points; it should learn WHY removal happens
    via the feature interactions.

    Note on early stopping: using bf_val as the eval set, early stopping
    prevents the model from fitting to the specific BF-count distribution
    of the training years.
    """
    X_train = bf_train[feature_cols].fillna(0)
    y_train = bf_train["event"].astype(int)

    # Class imbalance: roughly 1 event per 17 BF rows
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / max(pos, 1)
    print(f"    scale_pos_weight = {scale_pos_weight:.1f} "
          f"({neg:,} non-events / {pos:,} events)")

    params = {
        "n_estimators":      500,
        "max_depth":         4,
        "learning_rate":     0.03,
        "subsample":         0.75,
        "colsample_bytree":  0.75,
        "min_child_weight":  5,
        "reg_alpha":         0.3,
        "reg_lambda":        2.0,
        "scale_pos_weight":  scale_pos_weight,
        "objective":         "binary:logistic",
        "eval_metric":       "logloss",
        "random_state":      42,
        "tree_method":       "hist",
        "verbosity":         0,
    }
    model = xgb.XGBClassifier(**params)

    if bf_val is not None and len(bf_val) > 0:
        X_val = bf_val[feature_cols].fillna(0)
        y_val = bf_val["event"].astype(int)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)
    else:
        model.fit(X_train, y_train, verbose=False)

    return model


# =============================================================================
# WALK-FORWARD CROSS-VALIDATION
# =============================================================================
def walk_forward_cv(starts_df: pd.DataFrame,
                     bf_df: pd.DataFrame) -> list:
    """
    For each fold: expand training starts to BF rows, train hazard model,
    simulate test-season outs distributions, evaluate vs actual outs.

    Evaluation metrics:
      AUC-ROC (BF-level): how well does the hazard model rank removal events?
      MAE (start-level):  mean absolute error of simulated expected outs vs actual
      Calibration:        does P(outs > line) match actual over rate at each line?
    """
    seasons = sorted(starts_df["season"].unique())
    if len(seasons) < 2:
        print("  Need at least 2 seasons for walk-forward CV.")
        return []

    fold_metrics = []

    for i in range(1, len(seasons)):
        test_season = seasons[i]
        train_mask  = starts_df["season"] < test_season
        test_mask   = starts_df["season"] == test_season

        starts_train = starts_df[train_mask]
        starts_test  = starts_df[test_mask]

        if len(starts_train) < 50 or len(starts_test) < 20:
            continue

        # Get BF-level data for train/test or re-expand
        if not bf_df.empty and "season" in bf_df.columns:
            bf_train_fold = bf_df[bf_df["season"] < test_season]
            bf_test_fold  = bf_df[bf_df["season"] == test_season]
        else:
            from pitcher_outs.build_02 import expand_to_bf_level
            bf_train_fold = expand_to_bf_level(starts_train)
            bf_test_fold  = expand_to_bf_level(starts_test)

        if bf_train_fold.empty:
            continue

        feat_cols = get_hazard_feature_cols(bf_train_fold)
        hazard_model = train_hazard_model(bf_train_fold, feat_cols, bf_test_fold)

        # BF-level AUC
        if not bf_test_fold.empty and "event" in bf_test_fold.columns:
            X_test_bf = bf_test_fold[feat_cols].fillna(0)
            y_test_bf = bf_test_fold["event"].astype(int)
            y_prob_bf = hazard_model.predict_proba(X_test_bf)[:, 1]
            try:
                bf_auc = roc_auc_score(y_test_bf, y_prob_bf)
                bf_ll  = log_loss(y_test_bf, y_prob_bf)
                bf_brier = brier_score_loss(y_test_bf, y_prob_bf)
            except Exception:
                bf_auc = bf_ll = bf_brier = np.nan
        else:
            bf_auc = bf_ll = bf_brier = np.nan

        # Start-level MAE via simulation (sample 200 sims per start to keep fast)
        sim_outs = []
        actual_outs = []
        for _, row in starts_test.iterrows():
            sim = simulate_sp_outing(
                hazard_model, feat_cols, row.to_dict(), n_sim=200
            )
            sim_outs.append(sim["expected_outs"])
            actual_outs.append(row["outs_recorded"])

        start_mae = mean_absolute_error(actual_outs, sim_outs)

        # Calibration at the main line (15.5 outs)
        actual_over_rate = np.mean(np.array(actual_outs) > 15.5)
        # Re-run simulation for calibration (full N_SIM too slow in CV; use 500)
        p_over_15_5 = []
        for _, row in starts_test.iterrows():
            sim = simulate_sp_outing(hazard_model, feat_cols, row.to_dict(), n_sim=500)
            p_over_15_5.append(sim["p_over_15_5"])
        model_over_rate = np.mean(p_over_15_5)

        m = {
            "fold":             f"train<{test_season}/test={test_season}",
            "n_train_starts":   len(starts_train),
            "n_test_starts":    len(starts_test),
            "bf_auc":           round(float(bf_auc), 4),
            "bf_log_loss":      round(float(bf_ll), 4),
            "bf_brier":         round(float(bf_brier), 4),
            "start_mae_outs":   round(float(start_mae), 3),
            "calib_actual_over_15_5":  round(float(actual_over_rate), 4),
            "calib_model_over_15_5":   round(float(model_over_rate), 4),
        }
        fold_metrics.append(m)

        print(f"\n  Fold: {m['fold']}")
        print(f"    Train {m['n_train_starts']:,} starts | Test {m['n_test_starts']:,} starts")
        print(f"    BF-level AUC    : {m['bf_auc']:.4f}   (random = 0.500)")
        print(f"    BF-level LogLoss: {m['bf_log_loss']:.4f}")
        print(f"    BF-level Brier  : {m['bf_brier']:.4f}")
        print(f"    Start MAE       : {m['start_mae_outs']:.3f} outs")
        print(f"    Calib (>15.5)   : actual={m['calib_actual_over_15_5']:.3f}  "
              f"model={m['calib_model_over_15_5']:.3f}  "
              f"diff={m['calib_model_over_15_5'] - m['calib_actual_over_15_5']:+.3f}")

    return fold_metrics


# =============================================================================
# MONTE CARLO PA-LEVEL SIMULATION
# =============================================================================
def simulate_sp_outing(hazard_model: xgb.XGBClassifier,
                        feature_cols: list,
                        game_context: dict,
                        n_sim: int = N_SIM) -> dict:
    """
    Simulate N complete SP outings using the learned discrete-time hazard.

    For each simulation:
      1. Set bf_k = 1, outs = 0
      2. Build state vector at bf_k using game_context + time-varying features
      3. Evaluate hazard h_k = model.predict_proba(state)[1]
      4. Apply structural multipliers (TTOP, pitch count ceiling)
      5. Draw removal ~ Bernoulli(h_k_adjusted)
      6. Draw out ~ Bernoulli(p_out_per_bf = 1 - opp_obp)
      7. If not removed: increment outs, advance bf_k, repeat
      8. Record total outs when removed

    Structural multipliers (the "hard mathematical penalties"):
      TTOP (bf_k >= 19):
        h_k_adjusted = min(1.0, h_k × TTOP_HAZARD_MULTIPLIER)
        This encodes the well-documented TTOP effect regardless of whether
        the model's training data is rich enough to capture it from features alone.

      Pitch count ceiling (pc_fraction_k >= 1.0):
        h_k_adjusted = min(1.0, h_k × PC_LIMIT_HAZARD_MULTIPLIER)

      Past hard limit (est_pc_k >= hard_pc_limit):
        Forced removal with probability HARD_PC_LIMIT_REMOVAL_PROB = 0.95
        This is the mathematical hard ceiling.

    Returns a dict with the full outs distribution + O/U probabilities.
    """
    opp_obp       = float(game_context.get("opp_lg_avg_obp",
                                            game_context.get("opp_obp", 0.315)))
    p_out_per_bf  = max(0.50, min(0.90, 1.0 - opp_obp))
    eff_ppp       = float(game_context.get("effective_ppp",
                                            game_context.get("pitches_per_pa", 3.75)))
    pc_limit      = float(game_context.get("typical_pc_limit", 95.0))
    hard_pc_limit = float(game_context.get("hard_pc_limit",   105.0))
    depth_score   = float(game_context.get("depth_score",       0.52))
    ttop_hook     = float(game_context.get("ttop_hook_rate",    0.35))

    all_outs = np.zeros(n_sim, dtype=np.float32)

    for sim_i in range(n_sim):
        bf_k  = 0
        outs  = 0
        pitching = True

        while pitching and bf_k < MAX_BF:
            bf_k += 1
            times_through = int(np.ceil(bf_k / 9))
            is_ttop       = int(bf_k >= 19)
            est_pc_k      = eff_ppp * bf_k
            pc_frac_k     = est_pc_k / max(pc_limit, 1.0)

            # ── Build state vector ─────────────────────────────────────────
            state = {
                **game_context,
                "bf_k":                   bf_k,
                "times_through_order":    times_through,
                "is_first_through":       int(times_through == 1),
                "is_second_through":      int(times_through == 2),
                "is_ttop":                is_ttop,
                "is_18th_batter":         int(bf_k == 18),
                "is_19th_batter":         int(bf_k == 19),
                "batters_into_ttop":      max(0, bf_k - 18),
                "est_pc_k":               est_pc_k,
                "pc_fraction_k":          pc_frac_k,
                "approaching_pc_limit":   int(pc_frac_k >= 0.85),
                "at_pc_limit":            int(pc_frac_k >= 1.0),
                "past_hard_limit":        int(est_pc_k >= hard_pc_limit),
                "ttop_x_low_patience":    is_ttop * (1.0 - depth_score),
                "pc_stress_k":            max(0.0, pc_frac_k - 0.7) * 10.0,
            }

            # Align to feature_cols
            x_row = np.array([float(state.get(c, 0.0)) for c in feature_cols],
                              dtype=np.float32).reshape(1, -1)

            # ── Hazard evaluation ──────────────────────────────────────────
            h_k = float(hazard_model.predict_proba(
                pd.DataFrame([{c: state.get(c, 0.0) for c in feature_cols}])
            )[0, 1])

            # ── Hard mathematical penalties ────────────────────────────────
            # 1. Past hard pitch count limit → near-certain removal
            if est_pc_k >= hard_pc_limit:
                h_k = HARD_PC_LIMIT_REMOVAL_PROB

            # 2. TTOP structural multiplier (BF 19+)
            elif is_ttop:
                h_k = min(1.0, h_k * TTOP_HAZARD_MULTIPLIER)

            # 3. At pitch count limit (within normal window)
            elif pc_frac_k >= 1.0:
                h_k = min(1.0, h_k * PC_LIMIT_HAZARD_MULTIPLIER)

            # 4. 18th-batter structural decision point: apply manager's TTOP hook rate
            #    as an additional floor — even if model says low hazard, manager
            #    evaluates at this specific transition
            if bf_k == 18:
                h_k = max(h_k, ttop_hook * 0.5)  # 50% of manager's TTOP rate at preview

            # ── Removal draw ───────────────────────────────────────────────
            if RNG.random() < h_k:
                pitching = False
            else:
                # Record out outcome for this PA
                if RNG.random() < p_out_per_bf:
                    outs += 1

        all_outs[sim_i] = outs

    # Build output dict
    exp_outs = float(all_outs.mean())
    std_outs = float(all_outs.std())

    # Discrete probability at each integer outs value
    outs_hist = {}
    for k in range(0, 28):
        outs_hist[f"p_outs_{k}"] = float((all_outs == k).mean())

    # O/U probabilities for all prop lines
    ou_probs = {}
    for line in OUTS_PROP_LINES:
        ou_probs[f"p_over_{str(line).replace('.', '_')}"]  = float((all_outs > line).mean())
        ou_probs[f"p_under_{str(line).replace('.', '_')}"] = float((all_outs < line).mean())
        ou_probs[f"p_push_{str(line).replace('.', '_')}"]  = float((all_outs == line).mean())

    return {
        "expected_outs":  round(exp_outs, 2),
        "expected_ip":    round(exp_outs / 3, 2),
        "std_outs":       round(std_outs, 2),
        "p_over_15_5":    float((all_outs > 15.5).mean()),   # most common line
        "p_over_17_5":    float((all_outs > 17.5).mean()),
        "p_over_18_5":    float((all_outs > 18.5).mean()),
        **ou_probs,
        **outs_hist,
    }


# =============================================================================
# OPTIONAL: COX PH MODEL (lifelines)
# =============================================================================
def fit_cox_ph(starts_df: pd.DataFrame,
                feature_cols_ti: list) -> object:
    """
    Fit a Cox Proportional Hazards model on the per-start data using lifelines.

    The Cox model provides interpretable hazard ratios:
      HR > 1.0 for ttop_hook_rate   → managers with higher hook rate are
                                       pulling pitchers sooner (as expected)
      HR < 1.0 for k_minus_bb_pct   → efficient command = longer survival
      HR > 1.0 for opp_bb_pct       → walk-heavy lineups = shorter outings

    The Cox model is NOT used for O/U probability output (the simulation is
    better for that).  It serves as an interpretability companion.

    Returns: fitted CoxPHFitter or None if lifelines is not installed.
    """
    try:
        from lifelines import CoxPHFitter
    except ImportError:
        print("    lifelines not installed — skipping Cox PH fit.")
        print("    Install with: pip install lifelines")
        return None

    df = starts_df.copy()
    df = df.dropna(subset=["outs_recorded"])
    df["outs_recorded"] = df["outs_recorded"].clip(lower=1, upper=27)
    df["censored"]      = df["censored"].fillna(0).astype(int)

    feat_cols = [c for c in feature_cols_ti if c in df.columns
                 and pd.api.types.is_numeric_dtype(df[c])]

    # lifelines uses duration_col and event_col (event=1 if NOT censored)
    df["event_observed"] = 1 - df["censored"]

    model_df = df[feat_cols + ["outs_recorded", "event_observed"]].fillna(0)

    cox = CoxPHFitter(penalizer=0.1)
    try:
        cox.fit(model_df, duration_col="outs_recorded",
                event_col="event_observed", show_progress=False)
        print("\n  ── Cox PH Hazard Ratios (top features) ─────────────────────")
        cox.print_summary(decimals=3, style="ascii",
                          columns=["exp(coef)", "p"])
        return cox
    except Exception as e:
        print(f"    Cox PH fit failed: {e}")
        return None


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================
def feature_importance_report(model: xgb.XGBClassifier,
                               feature_cols: list) -> pd.DataFrame:
    imp = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\n  ── Top 15 Feature Importances (Hazard Model) ─────────────────")
    for _, row in imp.head(15).iterrows():
        bar = "█" * max(1, int(row["importance"] * 300))
        print(f"  {row['feature']:40s} | {row['importance']:.4f} | {bar}")
    print("  ──────────────────────────────────────────────────────────────")

    # Verify TTOP and PC features rank in top half — expected for survival model
    structural_feats = {"is_ttop", "is_18th_batter", "is_19th_batter",
                        "batters_into_ttop", "at_pc_limit", "pc_fraction_k",
                        "past_hard_limit", "pc_stress_k", "ttop_x_low_patience"}
    top_n  = set(imp.head(len(feature_cols) // 2)["feature"])
    found  = structural_feats & top_n
    print(f"\n  Structural hazard features in top half: {sorted(found)}")
    if len(found) < 3:
        print("  WARNING: Few structural features in top half — check BF expansion")
    return imp


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("PITCHER TOTAL OUTS MODEL — STEP 3: SURVIVAL ANALYSIS + SIMULATION")
    print("=" * 70)

    # Load
    print("\n[ 1/6 ] Loading datasets...")
    starts_df, bf_df = load_data()

    # Walk-forward CV
    print("\n[ 2/6 ] Walk-forward cross-validation (hazard model)...")
    cv_results = walk_forward_cv(starts_df, bf_df)

    if cv_results:
        avg_bf_auc  = np.mean([m["bf_auc"]         for m in cv_results])
        avg_mae     = np.mean([m["start_mae_outs"]  for m in cv_results])
        print(f"\n  ── Walk-Forward CV Summary ──")
        print(f"  Avg BF-level AUC : {avg_bf_auc:.4f}  (random = 0.500)")
        print(f"  Avg Start MAE    : {avg_mae:.3f} outs "
              f"({avg_mae/3:.3f} IP per start)")

    # Train final hazard model on all data
    print("\n[ 3/6 ] Training final hazard model (all seasons)...")
    if bf_df.empty:
        print("  WARNING: BF-level dataset not found — re-expanding starts in memory...")
        from pitcher_outs.build_02 import expand_to_bf_level
        bf_df = expand_to_bf_level(starts_df)

    feat_cols = get_hazard_feature_cols(bf_df)
    print(f"  Feature count: {len(feat_cols)}")
    print(f"  Features: {feat_cols}")
    final_hazard_model = train_hazard_model(bf_df, feat_cols)
    print("  ✓ Final hazard model trained.")

    # Evaluate via simulation on most-recent season
    print("\n[ 4/6 ] Simulation evaluation on most-recent season...")
    last_season  = sorted(starts_df["season"].unique())[-1]
    test_starts  = starts_df[starts_df["season"] == last_season]
    print(f"  Simulating {len(test_starts):,} starts from {last_season}...")

    sim_results = []
    for _, row in test_starts.iterrows():
        sim = simulate_sp_outing(final_hazard_model, feat_cols,
                                  row.to_dict(), n_sim=1_000)
        sim["actual_outs"] = row["outs_recorded"]
        sim["team"]        = row.get("team", "")
        sim["game_date"]   = row.get("game_date", "")
        sim_results.append(sim)

    sim_df        = pd.DataFrame(sim_results)
    test_mae      = mean_absolute_error(sim_df["actual_outs"],
                                         sim_df["expected_outs"])
    print(f"\n  Final model — season {last_season} ({len(test_starts):,} starts):")
    print(f"    Start-level MAE : {test_mae:.3f} outs ({test_mae/3:.3f} IP)")
    print(f"    Expected outs μ : {sim_df['expected_outs'].mean():.2f}")
    print(f"    Actual outs μ   : {sim_df['actual_outs'].mean():.2f}")

    print(f"\n    O/U calibration vs actuals:")
    for line in [14.5, 15.5, 17.5, 18.5]:
        safe = str(line).replace(".", "_")
        model_p  = sim_df.get(f"p_over_{safe}", pd.Series()).mean()
        actual_p = (sim_df["actual_outs"] > line).mean()
        print(f"    Over {line:4.1f}: model={model_p:.3f}  actual={actual_p:.3f}  "
              f"diff={model_p - actual_p:+.3f}")

    # Feature importance
    print("\n[ 5/6 ] Feature importances...")
    imp_df = feature_importance_report(final_hazard_model, feat_cols)

    # Optional Cox PH for interpretability
    print("\n  Fitting Cox PH companion model (for hazard ratio interpretation)...")
    cox_model = fit_cox_ph(starts_df, TI_FEAT_COLS)

    # Final metrics
    final_metrics = {
        "test_mae_outs":  round(float(test_mae), 3),
        "test_season":    int(last_season),
        "n_test_starts":  int(len(test_starts)),
        "n_features":     len(feat_cols),
        "avg_cv_bf_auc":  round(float(avg_bf_auc), 4) if cv_results else None,
        "avg_cv_mae":     round(float(avg_mae), 3) if cv_results else None,
        "cv_folds":       cv_results,
        "structural_multipliers": {
            "ttop_hazard_multiplier":     TTOP_HAZARD_MULTIPLIER,
            "pc_limit_hazard_multiplier": PC_LIMIT_HAZARD_MULTIPLIER,
            "hard_pc_limit_removal_prob": HARD_PC_LIMIT_REMOVAL_PROB,
            "ttop_bf_start":              19,
            "decision_point_bf":          18,
        },
    }

    # Save artifacts
    print("\n[ 6/6 ] Saving artifacts...")
    model_path = os.path.join(MODEL_DIR, "pitcher_outs_hazard.json")
    final_hazard_model.save_model(model_path)
    print(f"  ✓ Hazard model        : {model_path}")

    if cox_model is not None:
        cox_path = os.path.join(MODEL_DIR, "pitcher_outs_cox.pkl")
        joblib.dump(cox_model, cox_path)
        print(f"  ✓ Cox PH model        : {cox_path}")

    feat_path = os.path.join(MODEL_DIR, "pitcher_outs_features.json")
    with open(feat_path, "w") as f:
        json.dump(feat_cols, f)
    print(f"  ✓ Feature list        : {feat_path}")

    met_path = os.path.join(MODEL_DIR, "pitcher_outs_metrics.json")
    with open(met_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"  ✓ Metrics             : {met_path}")

    imp_path = os.path.join(MODEL_DIR, "pitcher_outs_feature_importance.csv")
    imp_df.to_csv(imp_path, index=False)
    print(f"  ✓ Feature importances : {imp_path}")

    # ── Simple XGBRegressor companion (used by 04_export_pitcher_outs.py) ─────
    print("\n  Training simple XGBRegressor companion (for 04_export scoring)...")
    _reg_feats = [c for c in TI_FEAT_COLS
                  if c in starts_df.columns
                  and pd.api.types.is_numeric_dtype(starts_df[c])
                  and c not in {"effective_ppp", "typical_pc_limit", "hard_pc_limit",
                                "ttop_hook_rate", "mgr_avg_sp_outs", "opp_wrc_plus",
                                "bp_gmLI", "bp_total_apps", "est_pc_at_bf18",
                                "pc_fraction_at_bf18", "pc_headroom_at_ttop",
                                "efficiency_x_depth", "pitches_per_pa"}]
    _X = starts_df[_reg_feats].fillna(starts_df[_reg_feats].mean()).astype(float)
    _y = starts_df["outs_recorded"].astype(float)
    _mask = _y.notna()
    _reg = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                             subsample=0.8, colsample_bytree=0.8, random_state=42)
    _reg.fit(_X[_mask], _y[_mask], verbose=False)
    _reg_path = os.path.join(MODEL_DIR, "pitcher_outs_model.json")
    _reg.save_model(_reg_path)
    print(f"  ✓ Regression companion : {_reg_path}")
    # Update features JSON to export-compatible feature list
    with open(feat_path, "w") as f:
        json.dump(_reg_feats, f)
    print(f"  ✓ Features JSON updated: {_reg_feats}")
    # Save per-start dataset for export fallback means
    _ds = starts_df[_reg_feats + ["outs_recorded"]].copy()
    _ds.to_csv(os.path.join(PROC_DIR, "pitcher_outs_dataset.csv"), index=False)
    print(f"  ✓ pitcher_outs_dataset.csv updated")

    # Save simulation results for review
    sim_path = os.path.join(PROC_DIR, "pitcher_outs_sim_eval.csv")
    sim_df.to_csv(sim_path, index=False)
    print(f"  ✓ Sim evaluation      : {sim_path}")

    # Scoring template with example starters
    print("\n  Example scoring output (template):")
    example_contexts = [
        {
            "name": "Gerrit Cole (NYY vs BOS)",
            "k_pct": 29.0, "bb_pct": 6.5, "k_minus_bb_pct": 22.5,
            "siera": 2.90, "xfip": 3.10, "swstr_pct": 14.0, "csw_pct": 32.0,
            "avg_fb_velo": 96.5, "pitches_per_pa": 3.60, "effective_ppp": 3.65,
            "opp_lg_avg_bb_pct": 0.095, "opp_wrc_plus": 108, "opp_lg_avg_obp": 0.325,
            "typical_pc_limit": 93,  "hard_pc_limit": 103,
            "ttop_hook_rate": 0.37, "depth_score": 0.50,
            "mgr_avg_sp_outs": 14.5, "est_pc_at_bf18": 65.7,
            "pc_fraction_at_bf18": 0.71, "efficiency_x_depth": 11.3,
            "pc_headroom_at_ttop": 27.3,
            # zero-fill remaining TV features (set at simulation time)
            **{c: 0.0 for c in TV_FEAT_COLS},
        },
        {
            "name": "Generic SP (TBR — Kevin Cash)",
            "k_pct": 22.0, "bb_pct": 8.0, "k_minus_bb_pct": 14.0,
            "siera": 4.10, "xfip": 3.90, "swstr_pct": 10.5, "csw_pct": 27.0,
            "avg_fb_velo": 93.0, "pitches_per_pa": 3.80, "effective_ppp": 3.90,
            "opp_lg_avg_bb_pct": 0.082, "opp_wrc_plus": 98, "opp_lg_avg_obp": 0.310,
            "typical_pc_limit": 80,  "hard_pc_limit": 90,   # Kevin Cash
            "ttop_hook_rate": 0.75, "depth_score": 0.30,
            "mgr_avg_sp_outs": 12.5, "est_pc_at_bf18": 70.2,
            "pc_fraction_at_bf18": 0.88, "efficiency_x_depth": 4.2,
            "pc_headroom_at_ttop": 9.8,
            **{c: 0.0 for c in TV_FEAT_COLS},
        },
    ]

    for ctx in example_contexts:
        name = ctx.pop("name")
        # Ensure all feature cols present
        for c in feat_cols:
            ctx.setdefault(c, 0.0)
        sim = simulate_sp_outing(final_hazard_model, feat_cols, ctx, n_sim=10_000)
        print(f"\n  {name}")
        print(f"    Expected outs   : {sim['expected_outs']:.1f} ({sim['expected_ip']:.1f} IP)")
        print(f"    Std dev         : ±{sim['std_outs']:.1f} outs")
        print(f"    P(over 15.5)    : {sim['p_over_15_5']:.3f}")
        print(f"    P(over 17.5)    : {sim['p_over_17_5']:.3f}")
        print(f"    P(over 18.5)    : {sim['p_over_18_5']:.3f}")

    print("\n" + "=" * 70)
    print("STEP 3 COMPLETE — Run 04_export_pitcher_outs.py for today's picks.")
    print("=" * 70)
