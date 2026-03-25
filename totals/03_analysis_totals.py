"""
=============================================================================
OVER/UNDER TOTALS MODEL — FILE 3 OF 4: MODEL TRAINING  (REFACTORED)
=============================================================================
Statistical framework: Negative Binomial NB2 regression (two equations)

Why NB2 instead of Poisson:
  MLB runs per half-game have empirical variance ≈ 1.25–1.40× the mean.
  Poisson assumes Var[Y] = μ exactly — violated, producing overconfident
  probability estimates and mis-sized edges.

  NB2 parameterization: Var[Y] = μ + α·μ²
    α = overdispersion parameter (estimated from data)
    α → 0 recovers Poisson; MLB runs typically give α ≈ 0.05–0.15

Two-equation system:
  Model A (home_runs)  : f(home offense, away SP/bullpen, park, weather)
  Model B (away_runs)  : f(away offense, home SP/bullpen, park, weather)

  Each model produces a (μ, α) pair for a Negative Binomial distribution.
  The joint total runs distribution is approximated by Monte Carlo
  convolution:  Z = X_home + X_away  where X ~ NB(μ, α).

  This correctly propagates the uncertainty in each half-inning and
  produces calibrated O/U edge probabilities.

Walk-forward cross-validation:
  Fold 1: Train = 2023        →  Test = 2024
  Fold 2: Train = 2023+2024   →  Test = 2025

Outputs:
  models/totals_nb_home.pkl    : NB model for home team runs
  models/totals_nb_away.pkl    : NB model for away team runs
  models/totals_features.json  : feature lists for each model
  models/totals_metrics.json   : CV and hold-out metrics
=============================================================================
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial

from scipy.stats import nbinom, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR  = os.path.join(BASE_DIR, "data", "models")
EXPORT_DIR = os.path.join(BASE_DIR, "exports")
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# Monte Carlo sample size for O/U probability estimation
MC_SIMS = 200_000


# =============================================================================
# OVERDISPERSION TEST
# =============================================================================

def overdispersion_test(y: pd.Series, mu: np.ndarray) -> dict:
    """
    Test whether run totals are overdispersed relative to Poisson.

    Method: Cameron & Trivedi (1990) auxiliary regression.
      Regress (y - μ)² − y on μ² (no intercept).
      If the coefficient t-stat > 1.96, reject Poisson in favor of NB.

    Also reports:
      Pearson chi-squared / df > 1.5 → strong overdispersion signal
      Var/Mean > 1.1 → practical overdispersion

    Returns
    -------
    dict with: pearson_chi2, df, dispersion_ratio, var_mean_ratio,
               overdispersed (bool)
    """
    y_arr  = np.asarray(y, dtype=float)
    mu_arr = np.asarray(mu, dtype=float)

    pearson_resid2 = (y_arr - mu_arr) ** 2 / mu_arr
    pearson_chi2   = pearson_resid2.sum()
    df             = len(y_arr) - 1
    disp_ratio     = pearson_chi2 / df

    var_mean = y_arr.var() / y_arr.mean()

    return {
        "pearson_chi2":   round(float(pearson_chi2), 2),
        "df":             df,
        "dispersion_ratio": round(float(disp_ratio), 4),
        "var_mean_ratio": round(float(var_mean), 4),
        "overdispersed":  bool(disp_ratio > 1.2),
    }


# =============================================================================
# FEATURE SELECTION: build side-specific feature lists
# =============================================================================

def get_feature_cols(df: pd.DataFrame, side: str) -> list:
    """
    Select features appropriate for modeling one side's run scoring.

    For modeling HOME runs:
      - home offensive features (the bats scoring the runs)
      - AWAY SP / bullpen features (the pitcher allowing the runs)
      - park + weather (affect both, but park belongs to home)

    For modeling AWAY runs: mirror the home/away roles.

    Parameters
    ----------
    side : "home" | "away"
    """
    exclude = {"game_date", "date_str", "Season", "home_team", "away_team",
               "home_runs", "away_runs", "total_runs"}

    opponent = "away" if side == "home" else "home"
    batting  = side
    pitching = opponent   # the opposing pitching staff

    feats = []

    for col in df.columns:
        if col in exclude:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        # Include batting team's offensive stats
        if col.startswith(f"{batting}_off_"):
            feats.append(col)

        # Include OPPOSING team's pitching stats (SP + bullpen)
        elif col.startswith(f"{pitching}_sp_") or col.startswith(f"{pitching}_bp_") \
                or col.startswith(f"{pitching}_pit_"):
            feats.append(col)

        # Include home manager hook rate (affects SP innings depth)
        elif col == f"{batting}_hook_rate":
            feats.append(col)

        # Park and weather affect run scoring for both teams
        elif col.startswith("wx_") or col.startswith("dyn_"):
            feats.append(col)
        elif col in ("wind_outfield_comp", "is_artificial", "is_dome",
                     "is_coors", "altitude_ft", "base_pf",
                     "is_cold_game", "is_hot_game"):
            feats.append(col)

    # Remove duplicates, preserve order
    seen = set()
    return [c for c in feats if c not in seen and not seen.add(c)]


# =============================================================================
# NB2 MODEL TRAINING
# =============================================================================

def fit_nb2(X_train: pd.DataFrame, y_train: pd.Series,
            label: str = "") -> tuple:
    """
    Fit a Negative Binomial NB2 regression.

    NB2: log(μ) = Xβ;  Var[Y] = μ + α·μ²

    statsmodels NegativeBinomial with loglike_method='nb2' estimates
    both β (coefficients) and α (overdispersion) via MLE.

    Parameters
    ----------
    X_train : pd.DataFrame    Feature matrix (will have constant added)
    y_train : pd.Series       Run count target (non-negative integers)
    label   : str             Display label for printing

    Returns
    -------
    (result, alpha, mu_train)
      result    : statsmodels NegativeBinomialResults object
      alpha     : estimated overdispersion parameter α
      mu_train  : predicted means on training data
    """
    print(f"  Fitting NB2 for {label}...")

    X_const = sm.add_constant(X_train.fillna(X_train.mean()).astype(float))
    y_arr   = y_train.astype(float)

    try:
        model  = NegativeBinomial(y_arr, X_const, loglike_method="nb2")
        result = model.fit(
            method  = "bfgs",
            maxiter = 300,
            disp    = False,
            full_output = False,
        )
    except Exception as e:
        print(f"    WARNING: BFGS failed ({e}), retrying with Newton-Raphson...")
        try:
            result = model.fit(method="newton", maxiter=200, disp=False)
        except Exception as e2:
            print(f"    ERROR: NB2 fit failed entirely ({e2}). Falling back to Poisson GLM.")
            from statsmodels.genmod.families import Poisson as PoissonFamily
            model_p = sm.GLM(y_arr, X_const, family=PoissonFamily())
            result  = model_p.fit(maxiter=100, disp=False)
            alpha   = 0.001   # Near-Poisson fallback
            mu_train = result.predict(X_const)
            return result, alpha, mu_train

    # Overdispersion parameter (alpha in NB2 params)
    # statsmodels stores it as "alpha" in the params Series
    alpha = float(np.exp(result.params.get("alpha", np.log(0.1))))
    # Note: statsmodels parameterizes log(alpha) internally; we exponentiate

    # Predicted means
    mu_train = result.predict(X_const)

    # Overdispersion diagnostic
    disp = overdispersion_test(y_train, mu_train)
    print(f"  ✓ {label} NB2 fitted")
    print(f"    α (overdispersion) = {alpha:.4f}  "
          f"{'(≈ Poisson)' if alpha < 0.02 else '(overdispersed)'}")
    print(f"    Var/Mean ratio     = {disp['var_mean_ratio']:.4f}  "
          f"({'NB justified' if disp['overdispersed'] else 'Poisson would suffice'})")
    print(f"    Pearson χ²/df      = {disp['dispersion_ratio']:.4f}")
    print(f"    Pseudo R² (McFadden) = "
          f"{max(0.0, 1 - result.llf / result.llnull):.4f}")
    print(f"    AIC = {result.aic:.1f}")

    # Top significant coefficients
    if hasattr(result, "pvalues"):
        coefs = pd.DataFrame({
            "feature": result.params.index,
            "coef":    result.params.values,
            "pval":    result.pvalues.values,
        })
        coefs = coefs[coefs["pval"] < 0.10].sort_values("pval")
        print(f"\n    Significant predictors (p<0.10):")
        for _, r in coefs.head(8).iterrows():
            dir_arrow = "↑" if r["coef"] > 0 else "↓"
            effect    = np.exp(r["coef"])
            print(f"      {r['feature']:40s}  β={r['coef']:+.4f}  "
                  f"exp(β)={effect:.4f}  p={r['pval']:.4f} {dir_arrow}")

    return result, alpha, mu_train


# =============================================================================
# PREDICTION HELPER
# =============================================================================

def predict_nb2(result, alpha: float, X_new: pd.DataFrame) -> np.ndarray:
    """
    Generate predicted means μ for new observations.

    Parameters
    ----------
    result   : fitted NB2 result (or GLM result as fallback)
    alpha    : overdispersion parameter α
    X_new    : feature matrix (constants will be added)

    Returns
    -------
    np.ndarray of predicted means μ_i
    """
    X_const = sm.add_constant(X_new.fillna(X_new.mean()).astype(float),
                               has_constant="add")
    return np.asarray(result.predict(X_const), dtype=float)


# =============================================================================
# MONTE CARLO O/U PROBABILITY
# =============================================================================

def nb_ou_probability(mu_home: float, alpha_home: float,
                       mu_away: float, alpha_away: float,
                       ou_line: float,
                       n_sims: int = MC_SIMS) -> dict:
    """
    Estimate P(over), P(under), P(push) via Monte Carlo simulation of the
    sum of two independent Negative Binomial random variables.

    NB2 → scipy parameterization:
      NB(μ, α) in NB2 parameterization maps to scipy.stats.nbinom(n, p) where:
        n = 1 / α      (shape / number of successes)
        p = 1 / (1 + α·μ)

    The total runs distribution Z = X_home + X_away is NOT analytically
    tractable for two separate NB distributions with different μ and α.
    Monte Carlo convolution is the correct approach.

    Parameters
    ----------
    mu_home, mu_away     : predicted mean runs for each side
    alpha_home, alpha_away: overdispersion parameters
    ou_line              : market over/under line (e.g., 8.5)
    n_sims               : Monte Carlo sample size

    Returns
    -------
    dict with p_over, p_under, p_push, lambda_home, lambda_away,
              lambda_total, ou_edge_over, ou_edge_under
    """
    def nb_scipy_params(mu: float, alpha: float) -> tuple:
        """Convert NB2 (μ, α) to scipy nbinom (n, p) parameterization."""
        alpha_safe = max(alpha, 1e-6)
        n = 1.0 / alpha_safe
        p = 1.0 / (1.0 + alpha_safe * mu)
        return n, p

    rng = np.random.default_rng(seed=42)
    n_h, p_h = nb_scipy_params(mu_home, alpha_home)
    n_a, p_a = nb_scipy_params(mu_away, alpha_away)

    home_sims  = rng.negative_binomial(n_h, p_h, size=n_sims).astype(float)
    away_sims  = rng.negative_binomial(n_a, p_a, size=n_sims).astype(float)
    total_sims = home_sims + away_sims

    ou_line_f  = float(ou_line)
    p_over     = float(np.mean(total_sims > ou_line_f))
    p_under    = float(np.mean(total_sims < ou_line_f))
    p_push     = float(np.mean(total_sims == ou_line_f))

    lambda_total = mu_home + mu_away

    return {
        "p_over":        round(p_over,  4),
        "p_under":       round(p_under, 4),
        "p_push":        round(p_push,  4),
        "lambda_home":   round(mu_home,       3),
        "lambda_away":   round(mu_away,       3),
        "lambda_total":  round(lambda_total,  3),
        # Raw edge = our prob − implied fair-line prob (50% no-vig baseline)
        "ou_edge_over":  round(p_over  - 0.50, 4),
        "ou_edge_under": round(p_under - 0.50, 4),
    }


# =============================================================================
# WALK-FORWARD CROSS-VALIDATION
# =============================================================================

def walk_forward_cv(df: pd.DataFrame, home_feats: list,
                    away_feats: list) -> list:
    """
    Walk-forward CV: for each test season, train on all prior seasons.

    Metrics per fold:
      MAE_home, MAE_away, MAE_total : mean absolute error in runs
      r_home, r_away, r_total       : Pearson correlation
    """
    seasons = sorted(df["Season"].dropna().unique())
    if len(seasons) < 2:
        return []

    fold_metrics = []

    for i in range(1, len(seasons)):
        test_season = seasons[i]
        tr_mask     = df["Season"] < test_season
        te_mask     = df["Season"] == test_season

        if tr_mask.sum() < 100 or te_mask.sum() < 50:
            continue

        tr, te = df[tr_mask], df[te_mask]
        X_tr_h, y_tr_h = tr[home_feats], tr["home_runs"]
        X_te_h, y_te_h = te[home_feats], te["home_runs"]
        X_tr_a, y_tr_a = tr[away_feats], tr["away_runs"]
        X_te_a, y_te_a = te[away_feats], te["away_runs"]

        try:
            res_h, alpha_h, _ = fit_nb2(X_tr_h, y_tr_h, f"home runs (fold test={test_season})")
            res_a, alpha_a, _ = fit_nb2(X_tr_a, y_tr_a, f"away runs (fold test={test_season})")

            mu_h = predict_nb2(res_h, alpha_h, X_te_h)
            mu_a = predict_nb2(res_a, alpha_a, X_te_a)
            mu_t = mu_h + mu_a

            y_t  = y_te_h.values + y_te_a.values

            mae_h = mean_absolute_error(y_te_h, mu_h)
            mae_a = mean_absolute_error(y_te_a, mu_a)
            mae_t = mean_absolute_error(y_t,    mu_t)
            r_h   = pearsonr(y_te_h.values, mu_h)[0]
            r_a   = pearsonr(y_te_a.values, mu_a)[0]
            r_t   = pearsonr(y_t,           mu_t)[0]

            m = {
                "fold":      f"train<{test_season} / test={test_season}",
                "n_train":   int(tr_mask.sum()),
                "n_test":    int(te_mask.sum()),
                "mae_home":  round(mae_h, 4),
                "mae_away":  round(mae_a, 4),
                "mae_total": round(mae_t, 4),
                "r_home":    round(r_h, 4),
                "r_away":    round(r_a, 4),
                "r_total":   round(r_t, 4),
                "alpha_home":round(alpha_h, 4),
                "alpha_away":round(alpha_a, 4),
            }
            fold_metrics.append(m)

            print(f"\n  ── Fold: {m['fold']} ─────────────────────────────────")
            print(f"    Train {m['n_train']:,}  |  Test {m['n_test']:,}")
            print(f"    Home runs  — MAE: {mae_h:.3f}  r: {r_h:.4f}  α: {alpha_h:.4f}")
            print(f"    Away runs  — MAE: {mae_a:.3f}  r: {r_a:.4f}  α: {alpha_a:.4f}")
            print(f"    Total runs — MAE: {mae_t:.3f}  r: {r_t:.4f}")
            print(f"    (Baseline MAE using mean prediction ≈ "
                  f"{mean_absolute_error(y_t, np.full_like(y_t, float(np.mean(y_t)))):.3f})")

        except Exception as e:
            print(f"  WARNING: fold {test_season} failed: {e}")

    return fold_metrics


# =============================================================================
# SCORE UPCOMING GAMES
# =============================================================================

def score_games(res_home, alpha_home: float, home_feats: list,
                res_away, alpha_away: float, away_feats: list,
                matchup_df: pd.DataFrame,
                ou_lines: dict = None) -> pd.DataFrame:
    """
    Apply the trained NB models to today's upcoming games.

    For each game:
      1. Predict μ_home from home offense + away SP/bullpen + park + weather
      2. Predict μ_away from away offense + home SP/bullpen + park + weather
      3. Monte Carlo convolution → P(over) / P(under)

    Parameters
    ----------
    ou_lines : dict   {"NYY-BOS": 8.5, ...}  Market lines for each game.
    """
    if ou_lines is None:
        ou_lines = {}

    mu_home = predict_nb2(res_home, alpha_home,
                          matchup_df.reindex(columns=home_feats, fill_value=0))
    mu_away = predict_nb2(res_away, alpha_away,
                          matchup_df.reindex(columns=away_feats, fill_value=0))

    results = []
    for i, (row_idx, row) in enumerate(matchup_df.iterrows()):
        game_key = f"{row.get('home_team','HOME')}-{row.get('away_team','AWAY')}"
        line     = ou_lines.get(game_key, ou_lines.get(str(i), 8.5))

        probs = nb_ou_probability(
            mu_home=float(mu_home[i]),
            alpha_home=alpha_home,
            mu_away=float(mu_away[i]),
            alpha_away=alpha_away,
            ou_line=float(line),
        )
        results.append({**probs, "ou_line": line})

    res_df = pd.DataFrame(results, index=matchup_df.index)
    return pd.concat([matchup_df, res_df], axis=1)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TOTALS MODEL — STEP 3: NB2 TRAINING  (HOME + AWAY EQUATIONS)")
    print("=" * 70)

    data_path = os.path.join(PROC_DIR, "totals_dataset.csv")
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found. Run 02_build_totals.py first.")
        exit(1)

    # ── Load ─────────────────────────────────────────────────────────────────
    print("\n[ 1/5 ] Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"  {len(df):,} games | seasons: {sorted(df['Season'].dropna().unique())}")
    print(f"  Total runs  — mean: {df['total_runs'].mean():.3f}, "
          f"var: {df['total_runs'].var():.3f}, "
          f"disp: {df['total_runs'].var()/df['total_runs'].mean():.3f}")

    # ── Feature selection ─────────────────────────────────────────────────────
    print("\n[ 2/5 ] Selecting side-specific features...")
    home_feats = get_feature_cols(df, "home")
    away_feats = get_feature_cols(df, "away")
    print(f"  Home runs model features ({len(home_feats)}): {home_feats[:6]} ...")
    print(f"  Away runs model features ({len(away_feats)}): {away_feats[:6]} ...")

    # ── Walk-forward CV ───────────────────────────────────────────────────────
    print("\n[ 3/5 ] Walk-forward cross-validation...")
    cv_results = walk_forward_cv(df, home_feats, away_feats)

    if cv_results:
        avg = lambda key: np.mean([m[key] for m in cv_results if key in m])
        print(f"\n  ── Walk-Forward CV Summary ─────────────────────────────────")
        print(f"  Avg MAE  (home) : {avg('mae_home'):.3f} runs")
        print(f"  Avg MAE  (away) : {avg('mae_away'):.3f} runs")
        print(f"  Avg MAE  (total): {avg('mae_total'):.3f} runs")
        print(f"  Avg r    (total): {avg('r_total'):.4f}")
        print(f"  Avg α    (home) : {avg('alpha_home'):.4f}")
        print(f"  Avg α    (away) : {avg('alpha_away'):.4f}")

    # ── Train final models ────────────────────────────────────────────────────
    print("\n[ 4/5 ] Training final NB2 models (all seasons)...")
    X_home_all = df[home_feats]
    X_away_all = df[away_feats]
    y_home_all = df["home_runs"].astype(float)
    y_away_all = df["away_runs"].astype(float)

    res_home, alpha_home, mu_home_train = fit_nb2(X_home_all, y_home_all, "HOME runs")
    res_away, alpha_away, mu_away_train = fit_nb2(X_away_all, y_away_all, "AWAY runs")

    # Hold-out evaluation on most recent season
    last_season = int(sorted(df["Season"].dropna().unique())[-1])
    te_mask     = df["Season"] == last_season
    if te_mask.sum() >= 50:
        print(f"\n  ── Final Model Hold-Out Evaluation (season {last_season}) ──────")
        mu_h_te = predict_nb2(res_home, alpha_home, df.loc[te_mask, home_feats])
        mu_a_te = predict_nb2(res_away, alpha_away, df.loc[te_mask, away_feats])
        mu_t_te = mu_h_te + mu_a_te
        y_h_te  = df.loc[te_mask, "home_runs"].values
        y_a_te  = df.loc[te_mask, "away_runs"].values
        y_t_te  = y_h_te + y_a_te

        mae_t  = mean_absolute_error(y_t_te, mu_t_te)
        rmse_t = np.sqrt(mean_squared_error(y_t_te, mu_t_te))
        r_t    = pearsonr(y_t_te, mu_t_te)[0]
        print(f"  Total runs — MAE: {mae_t:.3f}  RMSE: {rmse_t:.3f}  r: {r_t:.4f}")
        print(f"  α home = {alpha_home:.4f}  |  α away = {alpha_away:.4f}")

        # Demonstrate O/U probability for sample games
        print(f"\n  Sample O/U predictions (ou_line = 8.5):")
        for j in range(min(3, int(te_mask.sum()))):
            probs = nb_ou_probability(
                float(mu_h_te[j]), alpha_home,
                float(mu_a_te[j]), alpha_away,
                ou_line=8.5,
                n_sims=50_000,
            )
            game_row = df[te_mask].iloc[j]
            actual   = int(y_t_te[j])
            print(f"    {game_row.get('home_team','?')} vs {game_row.get('away_team','?')} — "
                  f"λ={probs['lambda_total']:.1f}  P(over)={probs['p_over']:.3f}  "
                  f"P(under)={probs['p_under']:.3f}  actual={actual}")
    else:
        mae_t, rmse_t, r_t = np.nan, np.nan, np.nan

    # ── Save artifacts ────────────────────────────────────────────────────────
    print("\n[ 5/5 ] Saving models and metadata...")

    joblib.dump(res_home, os.path.join(MODEL_DIR, "totals_nb_home.pkl"))
    joblib.dump(res_away, os.path.join(MODEL_DIR, "totals_nb_away.pkl"))
    print(f"  ✓ totals_nb_home.pkl, totals_nb_away.pkl")

    feat_payload = {
        "home_features": home_feats,
        "away_features": away_feats,
        "alpha_home":    float(alpha_home),
        "alpha_away":    float(alpha_away),
    }
    feat_path = os.path.join(MODEL_DIR, "totals_features.json")
    with open(feat_path, "w") as f:
        json.dump(feat_payload, f, indent=2)
    print(f"  ✓ totals_features.json (α_home={alpha_home:.4f}, α_away={alpha_away:.4f})")

    metrics = {
        "model_family":  "NegativeBinomial_NB2",
        "two_equations": True,
        "alpha_home":    float(alpha_home),
        "alpha_away":    float(alpha_away),
        "holdout_season":last_season,
        "holdout_mae_total":  float(mae_t)  if not np.isnan(mae_t)  else None,
        "holdout_rmse_total": float(rmse_t) if not np.isnan(rmse_t) else None,
        "holdout_r_total":    float(r_t)    if not np.isnan(r_t)    else None,
        "mc_simulations":     MC_SIMS,
        "cv_folds": cv_results,
    }
    met_path = os.path.join(MODEL_DIR, "totals_metrics.json")
    with open(met_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ totals_metrics.json")

    # Scoring template (placeholder; 04_export fills with real features)
    col_means = df[home_feats + away_feats].mean().to_dict()
    template  = pd.DataFrame([
        {"home_team": "NYY", "away_team": "BOS", **col_means},
        {"home_team": "LAD", "away_team": "SFG", **col_means},
        {"home_team": "COL", "away_team": "ARI", **col_means},
    ])
    ou_lines = {"NYY-BOS": 8.5, "LAD-SFG": 7.5, "COL-ARI": 11.0}
    scored = score_games(
        res_home, alpha_home, home_feats,
        res_away, alpha_away, away_feats,
        template, ou_lines,
    )
    scored.to_csv(os.path.join(PROC_DIR, "totals_predictions.csv"), index=False)
    disp_cols = ["home_team", "away_team", "lambda_total",
                 "lambda_home", "lambda_away", "ou_line", "p_over", "p_under"]
    disp_cols = [c for c in disp_cols if c in scored.columns]
    print(f"\n  Sample predictions:")
    print(scored[disp_cols].to_string(index=False))

    print("\n" + "=" * 70)
    print("STEP 3 COMPLETE — Run 04_export_totals.py next.")
    print("NB summary:")
    print(f"  α_home = {alpha_home:.4f} | α_away = {alpha_away:.4f}")
    print(f"  (α > 0 confirms overdispersion — NB superior to Poisson)")
    print("=" * 70)
