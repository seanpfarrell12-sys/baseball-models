"""
=============================================================================
OVER/UNDER TOTALS MODEL — FILE 4 OF 4: EDGE SCORING AND CSV EXPORT
=============================================================================
Purpose : Compare Poisson model's O/U probabilities to market lines;
          calculate edge, EV%, Kelly Criterion, and export recommendations.
Input   : ../data/processed/totals_predictions.csv (from Step 3)
Output  : ../exports/totals_edges_YYYYMMDD.csv

Over/Under specific considerations:
─────────────────────────────────────────────────────────────────────────────
Unlike moneyline, totals bets have a symmetric structure:
  - Over bet: P(actual runs > line) vs market's implied P(over)
  - Under bet: P(actual runs < line) vs market's implied P(under)
  - Half-point lines: no push possible (8.5, 9.5, etc.)
  - Whole-number lines: push possible (8, 9, etc.)

Model's output (from Step 3):
  - lambda_hat : Expected total runs (e.g., 8.43 runs)
  - p_over     : P(over the market line) via Poisson distribution
  - p_under    : P(under the market line)

Market juice on totals:
  - Standard: both sides at -110 (4.55% vig per side)
  - Alternate lines offered at different juice levels

The key edge signal:
  - Our model's lambda disagrees with what the market line implies.
  - If our model says lambda=9.2 and the line is 8.5 → strong Over edge.
  - If our model says lambda=7.8 and the line is 9.0 → strong Under edge.

For R users:
  - The structure of this file mirrors the moneyline export file.
  - Totals-specific: we track lambda_hat as the central prediction.
  - Over/Under are treated as two separate bet opportunities per game.
=============================================================================
"""

import os
import sys
import json
import requests
from datetime import datetime, date
import pandas as pd
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.action_network import get_totals_odds
from utils.probable_starters import (get_games_with_sp_stats,
                                      get_lineup_batting_features,
                                      get_lineups,
                                      load_batting_stats)
from utils.bullpen import get_bullpen_availability

# --- Configuration ----------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR  = os.path.join(BASE_DIR, "data", "models")
EXPORT_DIR = os.path.join(BASE_DIR, "exports")

ODDS_API_KEY     = "fbc985ad430c95d6435cb75210f7b989"
ODDS_API_URL_TOT = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"

KELLY_FRACTION   = 0.25
MAX_BET_FRACTION = 0.05
MIN_EDGE         = 0.06


# =============================================================================
# TOTALS-SPECIFIC ODDS UTILITIES
# =============================================================================

def american_to_decimal(american_odds: float) -> float:
    """Convert American odds to decimal (same as moneyline export)."""
    if american_odds >= 0:
        return (american_odds / 100.0) + 1.0
    else:
        return (100.0 / abs(american_odds)) + 1.0


def juice_to_implied_prob(juice: float) -> float:
    """
    Convert American juice (vig) to implied probability for one side.

    For totals, both Over and Under are typically -110 (standard juice).
    -110 juice → decimal 1.909 → implied 52.4%
    After vig removal: 52.4% / (52.4%+52.4%) = 50.0% per side.

    If over is at -115 and under is at -105:
      Over implied:  1/(1+100/115) = 53.5%
      Under implied: 1/(1+100/105) = 51.2%
      Total: 104.7% → vig = 4.7%
      Over fair prob = 53.5% / 104.7% = 51.1%
      Under fair prob = 51.2% / 104.7% = 48.9%
    """
    dec = american_to_decimal(juice)
    return 1.0 / dec


def remove_vig_ou(over_juice: float, under_juice: float) -> tuple:
    """
    Remove vig from over/under juice to get fair probabilities.

    Returns
    -------
    tuple : (fair_p_over, fair_p_under)
    """
    p_over_raw  = juice_to_implied_prob(over_juice)
    p_under_raw = juice_to_implied_prob(under_juice)
    total       = p_over_raw + p_under_raw
    if total <= 0:
        return 0.5, 0.5
    return p_over_raw / total, p_under_raw / total


def calculate_ou_edge(model_p_over: float, model_p_under: float,
                       market_p_over: float, market_p_under: float) -> tuple:
    """
    Calculate edge for both sides of an over/under bet.

    Returns
    -------
    tuple : (over_edge, under_edge)
        Positive = model favors that side vs market.
    """
    over_edge  = round(model_p_over  - market_p_over,  4)
    under_edge = round(model_p_under - market_p_under, 4)
    return over_edge, under_edge


def calculate_ev_pct(model_prob: float, decimal_odds: float) -> float:
    """
    EV% = model_prob × (dec_odds - 1) - (1 - model_prob).
    Same formula as moneyline — applicable to any binary bet.
    """
    return round(model_prob * (decimal_odds - 1) - (1 - model_prob), 4)


def kelly_criterion(model_prob: float, decimal_odds: float,
                    fraction: float = KELLY_FRACTION,
                    max_bet: float = MAX_BET_FRACTION) -> float:
    """Standard Kelly formula — see moneyline export for full documentation."""
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - model_prob
    f_full = (b * model_prob - q) / b
    return round(min(max(f_full * fraction, 0.0), max_bet), 4)


def compute_edge_score(edge: float, ev_pct: float, kelly: float,
                        lambda_hat: float, ou_line: float) -> float:
    """
    Compute 0–10 edge score for totals bets.

    Additional signal for totals: the "gap" between lambda_hat and the line.
    A large gap (lambda=9.5 vs line=7.5) should score higher than a small gap.

    Lambda gap component:
      gap_pct = |lambda_hat - ou_line| / ou_line
      gap_score = 2.0 × clip(gap_pct, 0, 0.25) / 0.25
    """
    edge_component = 3.0 * np.clip(edge,   0, 0.15) / 0.15
    ev_component   = 3.0 * np.clip(ev_pct, 0, 0.12) / 0.12
    kelly_comp     = 2.0 * np.clip(kelly,  0, 0.05) / 0.05

    # Lambda gap: how different is our expected total from the line?
    if ou_line and ou_line > 0:
        gap_pct  = abs(lambda_hat - ou_line) / ou_line
        gap_comp = 2.0 * np.clip(gap_pct, 0, 0.25) / 0.25
    else:
        gap_comp = 0.0

    score = edge_component + ev_component + kelly_comp + gap_comp
    return round(min(score, 10.0), 2)


# =============================================================================
# ODDS LOADING
# =============================================================================

def fetch_totals_odds_api(api_key: str) -> pd.DataFrame:
    """
    Fetch today's MLB totals (over/under) odds from The Odds API.

    Market key for totals: 'totals' (not 'h2h')
    """
    if not api_key:
        return pd.DataFrame()

    params = {
        "apiKey":     api_key,
        "regions":    "us",
        "markets":    "totals",
        "oddsFormat": "american",
        "bookmakers": "fanduel,draftkings,betmgm",
    }

    try:
        response = requests.get(ODDS_API_URL_TOT, params=params, timeout=10)
        response.raise_for_status()
        games_data = response.json()

        records = []
        for game in games_data:
            home_team = game.get("home_team", "")
            away_team = game.get("away_team", "")

            over_list, under_list, line_list = [], [], []

            for bookmaker in game.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market["key"] == "totals":
                        for outcome in market.get("outcomes", []):
                            line_list.append(outcome.get("point", 8.5))
                            if outcome["name"] == "Over":
                                over_list.append(outcome["price"])
                            elif outcome["name"] == "Under":
                                under_list.append(outcome["price"])

            if over_list and under_list:
                records.append({
                    "home_team":    home_team,
                    "away_team":    away_team,
                    "ou_line":      np.mean(line_list),
                    "over_juice":   np.mean(over_list),
                    "under_juice":  np.mean(under_list),
                })

        odds_df = pd.DataFrame(records)
        print(f"  ✓ Fetched {len(odds_df)} totals lines from The Odds API.")
        return odds_df

    except Exception as e:
        print(f"  WARNING: Odds API call failed: {e}")
        return pd.DataFrame()


def load_manual_totals_odds(path: str = None) -> pd.DataFrame:
    """
    Load totals odds from a manually created CSV.

    Expected format:
      home_team,away_team,ou_line,over_juice,under_juice
      NYY,BOS,9.0,-110,-110
      LAD,SFG,7.5,-115,-105
    """
    if path and os.path.exists(path):
        return pd.read_csv(path)

    # Create template with standard -110/-110 juice (most common)
    template = pd.DataFrame({
        "home_team":   ["NYY",  "LAD",  "COL"],
        "away_team":   ["BOS",  "SFG",  "ARI"],
        "ou_line":     [9.0,    7.5,    11.5],
        "over_juice":  [-110.0, -115.0, -110.0],
        "under_juice": [-110.0, -105.0, -110.0],
    })
    template_path = os.path.join(PROC_DIR, "totals_odds_template.csv")
    template.to_csv(template_path, index=False)
    print(f"  Created odds template at: {template_path}")
    return template


# Park altitude (feet above sea level) — static per ballpark
_ALTITUDE_MAP = {
    "ARI": 1082, "ATL": 1050, "BAL": 25,  "BOS": 21,  "CHC": 595,
    "CWS": 595,  "CIN": 490,  "CLE": 660, "COL": 5280, "DET": 601,
    "HOU": 43,   "KCR": 750,  "LAA": 171, "LAD": 340,  "MIA": 13,
    "MIL": 634,  "MIN": 841,  "NYM": 21,  "NYY": 21,   "OAK": 30,
    "PHI": 20,   "PIT": 1050, "SDP": 16,  "SEA": 17,   "SFG": 10,
    "STL": 465,  "TBR": 28,   "TEX": 571, "TOR": 251,  "WSN": 25,
}

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")


def score_live_games_totals(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score today's games using trained Poisson GLM + individual SP stats.

    Returns DataFrame with home_team, away_team, game_date, lambda_hat.
    p_over/p_under are computed later in build_totals_edge_report() once
    the actual ou_line from odds is known.
    """
    import joblib
    import statsmodels.api as sm

    model_home = joblib.load(os.path.join(MODEL_DIR, "totals_nb_home.pkl"))
    model_away = joblib.load(os.path.join(MODEL_DIR, "totals_nb_away.pkl"))
    with open(os.path.join(MODEL_DIR, "totals_features.json")) as f:
        feat_cfg   = json.load(f)
    home_feats = feat_cfg["home_features"] if isinstance(feat_cfg, dict) else feat_cfg
    away_feats = feat_cfg.get("away_features", feat_cfg) if isinstance(feat_cfg, dict) else feat_cfg
    alpha_home = feat_cfg.get("alpha_home", 0.001) if isinstance(feat_cfg, dict) else 0.001
    alpha_away = feat_cfg.get("alpha_away", 0.001) if isinstance(feat_cfg, dict) else 0.001
    feature_cols = list(dict.fromkeys(home_feats + away_feats))  # all unique features

    # ── Fetch individual SP stats ──────────────────────────────────────────
    print("  Fetching today's probable starters (individual stats)...")
    game_date = date.today().strftime("%Y-%m-%d")
    sp_df = get_games_with_sp_stats(game_date)   # auto-selects most recent season
    sp_lookup = {}
    if not sp_df.empty:
        for _, sp_row in sp_df.iterrows():
            sp_lookup[(sp_row["home_team"], sp_row["away_team"])] = sp_row

    # ── Lineup-based batting features (most recent season) ──────────────────
    print("  Fetching today's lineups for lineup-based batting features...")
    lineups, lineup_sources = get_lineups(game_date, return_sources=True)
    batting_df = load_batting_stats()   # most recent season (2026 → 2025 fallback)
    lineup_bat = get_lineup_batting_features(lineups, batting_df)

    # Team batting fallback (used when lineups not posted)
    bat_season = int(batting_df["Season"].max()) if not batting_df.empty else 2025
    bat_csv = pd.read_csv(os.path.join(RAW_DIR, "raw_team_batting.csv"))
    bat_csv = bat_csv[bat_csv["Season"] == bat_season] if bat_season in bat_csv["Season"].values \
              else bat_csv[bat_csv["Season"] == bat_csv["Season"].max()]
    bat_map = {r["Team"]: r for _, r in bat_csv.iterrows()}

    # Team pitching (always from team CSV — not lineup-dependent)
    pit = pd.read_csv(os.path.join(RAW_DIR, "raw_team_pitching.csv"))
    pit_latest = int(pit["Season"].max())
    pit_season = pit_latest if len(pit[pit["Season"] == pit_latest]) >= 20 else pit_latest - 1
    pit = pit[pit["Season"] == pit_season]
    pit_map = {r["Team"]: r for _, r in pit.iterrows()}

    # Bullpen: compute per-team BP stats from raw_sp_stats.csv (GS < 5, IP >= 5)
    _FG_TO_STD = {
        "SFG": "SF", "WSN": "WSH", "SDP": "SD", "KCR": "KC",
        "TBR": "TB", "CHW": "CWS",
    }
    bp_map = {}
    try:
        sp_raw = pd.read_csv(os.path.join(RAW_DIR, "raw_sp_stats.csv"))
        sp_raw = sp_raw[sp_raw["Season"] == pit_season].copy()
        sp_raw["IP"] = pd.to_numeric(sp_raw["IP"], errors="coerce").fillna(0)
        if "Team" in sp_raw.columns:
            sp_raw["team_std"] = sp_raw["Team"].map(_FG_TO_STD).fillna(sp_raw["Team"])
        gs_col = sp_raw["GS"].fillna(0) if "GS" in sp_raw.columns else pd.Series(0, index=sp_raw.index)
        rp = sp_raw[(gs_col < 5) & (sp_raw["IP"] >= 5)]
        if not rp.empty and "team_std" in rp.columns:
            for team, grp in rp.groupby("team_std"):
                tw = grp["IP"].sum()
                if tw > 0:
                    bp_map[team] = {
                        "bp_era":   (grp["ERA"].fillna(4.5) * grp["IP"]).sum() / tw if "ERA" in grp.columns else np.nan,
                        "bp_k_pct": (grp["K%"].fillna(0.22) * grp["IP"]).sum() / tw if "K%" in grp.columns else np.nan,
                        "bp_fip":   (grp["FIP"].fillna(4.5) * grp["IP"]).sum() / tw if "FIP" in grp.columns else np.nan,
                    }
    except Exception as _e:
        print(f"  WARNING: Bullpen computation failed — {_e}")

    # Bullpen availability (recent workload from MLB Stats API)
    print("  Fetching bullpen availability (last 3 days)...")
    try:
        bp_avail = get_bullpen_availability(game_date)
    except Exception as _e:
        print(f"  WARNING: Bullpen availability fetch failed — {_e}")
        bp_avail = {}

    park = pd.read_csv(os.path.join(RAW_DIR, "raw_park_factors.csv"))
    park_map = {r["team"]: r for _, r in park.iterrows()}

    # Manager hook rates
    mgr_map = {}
    _mgr_path = os.path.join(RAW_DIR, "raw_manager_hook.csv")
    if os.path.exists(_mgr_path):
        try:
            _mgr = pd.read_csv(_mgr_path)
            for _, _mr in _mgr.iterrows():
                _team = _mr.get("Team", _mr.get("team", None))
                _hr   = _mr.get("hook_rate", _mr.get("HookRate", np.nan))
                if _team:
                    mgr_map[str(_team)] = float(_hr) if pd.notna(_hr) else np.nan
        except Exception as _me:
            print(f"  WARNING: Could not load manager hook rates — {_me}")

    _ds = pd.read_csv(os.path.join(PROC_DIR, "totals_dataset.csv"))
    _avail = [c for c in feature_cols if c in _ds.columns]
    train_means = _ds[_avail].mean()

    rows = []
    for _, game in odds_df.iterrows():
        home = game["home_team"]
        away = game["away_team"]

        # Skip games where either lineup is not yet confirmed
        if lineup_sources.get(home, "projected") != "confirmed" or \
           lineup_sources.get(away, "projected") != "confirmed":
            print(f"  Skipping {away} @ {home} — lineup(s) not confirmed yet.")
            continue

        sp_row = sp_lookup.get((home, away), None)

        # Skip games where either starting pitcher is not individually confirmed
        if sp_row is None or \
           sp_row.get("home_sp_source") == "team_avg" or \
           sp_row.get("away_sp_source") == "team_avg":
            print(f"  Skipping {away} @ {home} — starting pitcher(s) not confirmed yet.")
            continue

        if sp_row is not None:
            home_sp_siera = float(sp_row.get("home_sp_siera", np.nan))
            home_sp_k_pct = float(sp_row.get("home_sp_k_pct", np.nan))
            away_sp_siera = float(sp_row.get("away_sp_siera", np.nan))
            away_sp_k_pct = float(sp_row.get("away_sp_k_pct", np.nan))
            sp_source = sp_row.get("home_sp_source", "individual")
        else:
            home_sp_siera = home_sp_k_pct = away_sp_siera = away_sp_k_pct = np.nan
            sp_source = "team_avg_fallback"

        # Prefer lineup-based batting; fall back to team CSV averages
        hlb = lineup_bat.get(home, {})
        alb = lineup_bat.get(away, {})
        h_bat_fallback = bat_map.get(home, {})
        a_bat_fallback = bat_map.get(away, {})
        h_pit = pit_map.get(home, {})
        a_pit = pit_map.get(away, {})
        pk    = park_map.get(home, {})

        roof    = pk.get("roof", "open") if isinstance(pk, dict) else getattr(pk, "roof", "open")
        surface = pk.get("surface", "natural") if isinstance(pk, dict) else getattr(pk, "surface", "natural")
        home_covered    = 0.5 if roof == "retractable" else (1.0 if roof == "fixed" else 0.0)
        home_artificial = 1.0 if surface == "artificial" else 0.0

        both_sp = pd.notna(home_sp_siera) and pd.notna(away_sp_siera)
        bat_source_home = hlb.get("_lineup_source", "confirmed") if hlb else "team_avg"
        bat_source_away = alb.get("_lineup_source", "confirmed") if alb else "team_avg"
        row = {
            "game_date":             game.get("game_time", date.today().isoformat())[:10],
            "home_team":             home,
            "away_team":             away,
            "sp_source":             sp_source,
            "bat_source_home":       bat_source_home,
            "bat_source_away":       bat_source_away,
            # Backward-compat old column names
            "home_team_off_woba":    hlb.get("off_woba", h_bat_fallback.get("wOBA", np.nan)),
            "home_team_off_iso":     hlb.get("off_iso",  h_bat_fallback.get("ISO",  np.nan)),
            "home_team_pit_era":     h_pit.get("ERA",  np.nan) if isinstance(h_pit, dict) else getattr(h_pit, "ERA",  np.nan),
            "home_team_pit_xfip":    h_pit.get("xFIP", np.nan) if isinstance(h_pit, dict) else getattr(h_pit, "xFIP", np.nan),
            "away_team_off_woba":    alb.get("off_woba", a_bat_fallback.get("wOBA", np.nan)),
            "away_team_off_iso":     alb.get("off_iso",  a_bat_fallback.get("ISO",  np.nan)),
            "away_team_pit_era":     a_pit.get("ERA",  np.nan) if isinstance(a_pit, dict) else getattr(a_pit, "ERA",  np.nan),
            "away_team_pit_xfip":    a_pit.get("xFIP", np.nan) if isinstance(a_pit, dict) else getattr(a_pit, "xFIP", np.nan),
            # New batting columns
            "home_off_woba":         hlb.get("off_woba",    h_bat_fallback.get("wOBA", np.nan)),
            "home_off_iso":          hlb.get("off_iso",     h_bat_fallback.get("ISO",  np.nan)),
            "home_off_wrcplus":      float(hlb.get("off_wrc_plus", h_bat_fallback.get("wRC+", 100.0)) or 100.0),
            "home_off_k_pct":        float(hlb.get("off_k_pct",   h_bat_fallback.get("K%",   0.22))  or 0.22),
            "home_off_bb_pct":       float(hlb.get("off_bb_pct",  h_bat_fallback.get("BB%",  0.085)) or 0.085),
            "home_off_obp":          float(hlb.get("off_obp",     h_bat_fallback.get("OBP",  0.32))  or 0.32),
            "home_off_slg":          float(hlb.get("off_slg",     h_bat_fallback.get("SLG",  0.40))  or 0.40),
            "away_off_woba":         alb.get("off_woba",    a_bat_fallback.get("wOBA", np.nan)),
            "away_off_iso":          alb.get("off_iso",     a_bat_fallback.get("ISO",  np.nan)),
            "away_off_wrcplus":      float(alb.get("off_wrc_plus", a_bat_fallback.get("wRC+", 100.0)) or 100.0),
            "away_off_k_pct":        float(alb.get("off_k_pct",   a_bat_fallback.get("K%",   0.22))  or 0.22),
            "away_off_bb_pct":       float(alb.get("off_bb_pct",  a_bat_fallback.get("BB%",  0.085)) or 0.085),
            "away_off_obp":          float(alb.get("off_obp",     a_bat_fallback.get("OBP",  0.32))  or 0.32),
            "away_off_slg":          float(alb.get("off_slg",     a_bat_fallback.get("SLG",  0.40))  or 0.40),
            # New pitching columns (new naming)
            "home_pit_era":          float(h_pit.get("ERA",   np.nan) if isinstance(h_pit, dict) else getattr(h_pit, "ERA",   np.nan)),
            "home_pit_xfip":         float(h_pit.get("xFIP",  np.nan) if isinstance(h_pit, dict) else getattr(h_pit, "xFIP",  np.nan)),
            "home_pit_fip":          float(h_pit.get("FIP",   np.nan) if isinstance(h_pit, dict) else getattr(h_pit, "FIP",   np.nan)),
            "home_pit_siera":        float(h_pit.get("SIERA", np.nan) if isinstance(h_pit, dict) else getattr(h_pit, "SIERA", np.nan)),
            "home_pit_k_pct":        float(h_pit.get("K%",    np.nan) if isinstance(h_pit, dict) else getattr(h_pit, "K%",    np.nan)),
            "home_pit_bb_pct":       float(h_pit.get("BB%",   np.nan) if isinstance(h_pit, dict) else getattr(h_pit, "BB%",   np.nan)),
            "home_pit_k_bb_pct":     float(h_pit.get("K-BB%", np.nan) if isinstance(h_pit, dict) else getattr(h_pit, "K-BB%", np.nan)),
            "away_pit_era":          float(a_pit.get("ERA",   np.nan) if isinstance(a_pit, dict) else getattr(a_pit, "ERA",   np.nan)),
            "away_pit_xfip":         float(a_pit.get("xFIP",  np.nan) if isinstance(a_pit, dict) else getattr(a_pit, "xFIP",  np.nan)),
            "away_pit_fip":          float(a_pit.get("FIP",   np.nan) if isinstance(a_pit, dict) else getattr(a_pit, "FIP",   np.nan)),
            "away_pit_siera":        float(a_pit.get("SIERA", np.nan) if isinstance(a_pit, dict) else getattr(a_pit, "SIERA", np.nan)),
            "away_pit_k_pct":        float(a_pit.get("K%",    np.nan) if isinstance(a_pit, dict) else getattr(a_pit, "K%",    np.nan)),
            "away_pit_bb_pct":       float(a_pit.get("BB%",   np.nan) if isinstance(a_pit, dict) else getattr(a_pit, "BB%",   np.nan)),
            "away_pit_k_bb_pct":     float(a_pit.get("K-BB%", np.nan) if isinstance(a_pit, dict) else getattr(a_pit, "K-BB%", np.nan)),
            # SP columns
            "home_sp_siera":         home_sp_siera,
            "home_sp_k_pct":         home_sp_k_pct,
            "home_sp_xfip":          float(sp_row.get("home_sp_xfip",    np.nan)) if sp_row is not None else np.nan,
            "home_sp_fip":           float(sp_row.get("home_sp_fip",     np.nan)) if sp_row is not None else np.nan,
            "home_sp_era":           float(sp_row.get("home_sp_era",     np.nan)) if sp_row is not None else np.nan,
            "home_sp_bb_pct":        float(sp_row.get("home_sp_bb_pct",  np.nan)) if sp_row is not None else np.nan,
            "home_sp_k_bb_pct":      float(sp_row.get("home_sp_k_bb_pct", np.nan)) if sp_row is not None else np.nan,
            "home_sp_gb_pct":        float(sp_row.get("home_sp_gb_pct",  np.nan)) if sp_row is not None else np.nan,
            "away_sp_siera":         away_sp_siera,
            "away_sp_k_pct":         away_sp_k_pct,
            "away_sp_xfip":          float(sp_row.get("away_sp_xfip",    np.nan)) if sp_row is not None else np.nan,
            "away_sp_fip":           float(sp_row.get("away_sp_fip",     np.nan)) if sp_row is not None else np.nan,
            "away_sp_era":           float(sp_row.get("away_sp_era",     np.nan)) if sp_row is not None else np.nan,
            "away_sp_bb_pct":        float(sp_row.get("away_sp_bb_pct",  np.nan)) if sp_row is not None else np.nan,
            "away_sp_k_bb_pct":      float(sp_row.get("away_sp_k_bb_pct", np.nan)) if sp_row is not None else np.nan,
            "away_sp_gb_pct":        float(sp_row.get("away_sp_gb_pct",  np.nan)) if sp_row is not None else np.nan,
            # Bullpen features
            "home_bp_era":           bp_map.get(home, {}).get("bp_era",   train_means.get("home_bp_era",   4.30)),
            "home_bp_k_pct":         bp_map.get(home, {}).get("bp_k_pct", train_means.get("home_bp_k_pct", 0.245)),
            "home_bp_fip":           bp_map.get(home, {}).get("bp_fip",   train_means.get("home_bp_fip",   4.20)),
            "away_bp_era":           bp_map.get(away, {}).get("bp_era",   train_means.get("away_bp_era",   4.30)),
            "away_bp_k_pct":         bp_map.get(away, {}).get("bp_k_pct", train_means.get("away_bp_k_pct", 0.245)),
            "away_bp_fip":           bp_map.get(away, {}).get("bp_fip",   train_means.get("away_bp_fip",   4.20)),
            # Legacy combined/diff features
            "combined_sp_siera":     (home_sp_siera + away_sp_siera) / 2 if both_sp else np.nan,
            "diff_sp_siera":         home_sp_siera - away_sp_siera if both_sp else np.nan,
            # Stadium/park features (new naming)
            "home_park_factor":      pk.get("pf_runs", 100) if isinstance(pk, dict) else getattr(pk, "pf_runs", 100),
            "base_pf":               pk.get("pf_runs", 100) if isinstance(pk, dict) else getattr(pk, "pf_runs", 100),
            "altitude":              _ALTITUDE_MAP.get(home, 0),
            "altitude_ft":           _ALTITUDE_MAP.get(home, 0),
            "is_coors":              1 if home == "COL" else 0,
            "home_covered":          home_covered,
            "home_artificial":       home_artificial,
            "is_artificial":         home_artificial,
            "is_dome":               1.0 if (pk.get("roof", "") if isinstance(pk, dict) else getattr(pk, "roof", "")) == "fixed" else 0.0,
            # Manager hook rates
            "home_hook_rate":        mgr_map.get(home, np.nan),
            "away_hook_rate":        mgr_map.get(away, np.nan),
            # Weather defaults (no live weather feed)
            "temperature_f":         72,
            "wx_temperature_f":      72.0,
            "wx_humidity_pct":       60.0,
            "wx_wind_speed_mph":     8.0,
            "wx_wind_direction_deg": 180.0,
            "is_hot_game":           0,
            "is_cold_game":          0,
            # Dynamic park factors (default neutral)
            "dyn_pf":                1.0,
            "dyn_temp_factor":       1.0,
            "dyn_humidity_factor":   1.0,
            "dyn_wind_factor":       1.0,
            "wind_outfield_comp":    0.0,
        }
        rows.append(row)

    game_df = pd.DataFrame(rows)

    for col in feature_cols:
        if col in game_df.columns:
            game_df[col] = game_df[col].fillna(train_means.get(col, 0.0))
        else:
            game_df[col] = train_means.get(col, 0.0)

    def _aligned_predict(model, feats, df):
        """Predict with column alignment in case model dropped multicollinear features."""
        X = sm.add_constant(df[feats].astype(float), has_constant="add")
        if hasattr(model, "params") and len(model.params) != X.shape[1]:
            fitted_cols = list(model.model.exog_names) if hasattr(model.model, "exog_names") \
                          else list(model.params.index)
            for c in fitted_cols:
                if c not in X.columns:
                    X[c] = 0.0
            X = X[fitted_cols]
        bad_cols = [c for c in X.columns if not np.isfinite(X[c]).all()]
        if bad_cols:
            print(f"  WARNING: Non-finite values in feature columns: {bad_cols}")
            for c in bad_cols:
                X[c] = np.nan_to_num(X[c], nan=0.0, posinf=0.0, neginf=0.0)
        return np.asarray(model.predict(exog=X), dtype=float)

    mu_home = _aligned_predict(model_home, home_feats, game_df)
    mu_away = _aligned_predict(model_away, away_feats, game_df)
    game_df["lambda_home_runs"] = np.round(mu_home, 3)
    game_df["lambda_away_runs"] = np.round(mu_away, 3)
    game_df["lambda_hat"] = (mu_home + mu_away).round(3)

    # Apply bullpen availability adjustments (post-model lambda scaling)
    # A taxed home bullpen → away team may score more runs (and vice versa)
    # We track home_lambda / away_lambda separately then recombine as lambda_hat
    if bp_avail:
        game_df["home_lambda"] = game_df["lambda_hat"] / 2.0
        game_df["away_lambda"] = game_df["lambda_hat"] / 2.0
        for idx, row in game_df.iterrows():
            h = row["home_team"]
            a = row["away_team"]
            # Taxed home bullpen → away team scores more
            home_adj = bp_avail.get(h, {}).get("bp_adjustment", 1.0)
            # Taxed away bullpen → home team scores more
            away_adj = bp_avail.get(a, {}).get("bp_adjustment", 1.0)
            game_df.at[idx, "away_lambda"] = row["away_lambda"] * home_adj
            game_df.at[idx, "home_lambda"] = row["home_lambda"] * away_adj
            if home_adj > 1.0 or away_adj > 1.0:
                print(f"  BP adjustment: {a} @ {h} "
                      f"(home_bp_adj={home_adj:.2f}, away_bp_adj={away_adj:.2f})")
        game_df["lambda_hat"] = (game_df["home_lambda"] + game_df["away_lambda"]).round(3)
        game_df.drop(columns=["home_lambda", "away_lambda"], inplace=True)

    n_ind = (game_df.get("sp_source", pd.Series()) == "individual").sum()
    print(f"  ✓ Scored {len(game_df)} games ({n_ind} with individual SP stats). "
          f"Mean λ̂ = {game_df['lambda_hat'].mean():.2f} runs.")
    return game_df


# =============================================================================
# MAIN EDGE REPORT BUILDER
# =============================================================================

def build_totals_edge_report(predictions_df: pd.DataFrame,
                               odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join Poisson model predictions with market totals lines.

    Each game produces TWO bet opportunities (Over and Under).
    We evaluate both and include any that show positive edge.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        From Step 3: columns include lambda_hat, p_over, p_under, home_team, away_team
    odds_df : pd.DataFrame
        Odds data: columns include ou_line, over_juice, under_juice

    Returns
    -------
    pd.DataFrame
        One row per bet (Over OR Under) with all edge metrics.
    """
    # Merge on team names
    df = predictions_df.merge(odds_df, on=["home_team", "away_team"], how="left")

    rows = []

    for _, row in df.iterrows():
        lambda_hat  = row.get("lambda_hat")
        ou_line     = row.get("ou_line")

        if pd.isna(ou_line) or pd.isna(lambda_hat):
            home = row.get("home_team", "?")
            away = row.get("away_team", "?")
            print(f"  SKIP: No totals odds found for {away} @ {home} — skipping game.")
            continue

        lambda_hat = float(lambda_hat)
        ou_line    = float(ou_line)

        if not np.isfinite(lambda_hat) or lambda_hat <= 0:
            home = row.get("home_team", "?")
            away = row.get("away_team", "?")
            print(f"  SKIP: Invalid lambda_hat ({lambda_hat}) for {away} @ {home} — skipping game.")
            continue

        # Compute p_over/p_under from lambda_hat + Poisson distribution
        # This handles both the old predictions CSV path and the new live scoring path
        from scipy.stats import poisson as sp_poisson
        p_under_mod = float(sp_poisson.cdf(int(ou_line), lambda_hat))
        p_over_mod  = 1.0 - float(sp_poisson.cdf(int(np.ceil(ou_line) - 1), lambda_hat))
        # If half-point line (8.5), no push — use continuous boundary
        if ou_line != int(ou_line):
            p_under_mod = float(sp_poisson.cdf(int(ou_line), lambda_hat))
            p_over_mod  = 1.0 - p_under_mod

        # Use pre-computed p_over/p_under if available (from Step 3 CSV)
        if "p_over" in row.index and pd.notna(row.get("p_over")):
            p_over_mod  = row["p_over"]
            p_under_mod = row["p_under"]
        over_juice  = row.get("over_juice",  -110)
        under_juice = row.get("under_juice", -110)

        # Convert market juice to fair probabilities
        fair_p_over, fair_p_under = remove_vig_ou(over_juice, under_juice)

        # Decimal odds for each side
        over_dec  = american_to_decimal(over_juice)
        under_dec = american_to_decimal(under_juice)

        # Edges
        over_edge, under_edge = calculate_ou_edge(
            p_over_mod, p_under_mod, fair_p_over, fair_p_under
        )

        # Lambda gap (how far our prediction is from the line)
        lambda_gap    = lambda_hat - ou_line
        lambda_pct_gap = lambda_gap / ou_line if ou_line > 0 else 0

        for side, model_p, market_p, dec_odds, am_juice, edge in [
            ("OVER",  p_over_mod,  fair_p_over,  over_dec,  over_juice,  over_edge),
            ("UNDER", p_under_mod, fair_p_under, under_dec, under_juice, under_edge),
        ]:
            ev_pct = calculate_ev_pct(model_p, dec_odds)
            kelly  = kelly_criterion(model_p, dec_odds)
            score  = compute_edge_score(edge, ev_pct, kelly, lambda_hat, ou_line)

            rows.append({
                "game_date":       row.get("game_date", datetime.now().strftime("%Y-%m-%d")),
                "home_team":       row.get("home_team", ""),
                "away_team":       row.get("away_team", ""),
                "bet_type":        side,              # "OVER" or "UNDER"
                "ou_line":         ou_line,           # Market line (e.g., 8.5)
                "lambda_hat":      round(lambda_hat, 3), # Model expected total
                "lambda_gap":      round(lambda_gap, 3), # Positive = lean OVER
                "model_prob":      round(model_p, 4),
                "market_implied":  round(market_p, 4),
                "edge":            edge,
                "ev_pct":          ev_pct,
                "kelly_fraction":  kelly,
                "edge_score":      score,
                "juice":           am_juice,
                "decimal_odds":    round(dec_odds, 4),
                "vig_pct":         round((
                    juice_to_implied_prob(over_juice)
                    + juice_to_implied_prob(under_juice) - 1
                ) * 100, 2),
                "is_value_bet":    1 if edge >= MIN_EDGE and ev_pct > 0 else 0,
            })

    report_df = pd.DataFrame(rows)
    report_df = report_df.sort_values("edge_score", ascending=False).reset_index(drop=True)
    return report_df


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("TOTALS MODEL — STEP 4: EDGE SCORING AND EXPORT")
    print("=" * 70)
    today_str = datetime.now().strftime("%Y%m%d")

    print("\n[ 1/4 ] Fetching today's games and odds...")
    odds_df = get_totals_odds(date.today().strftime("%Y-%m-%d"))
    if odds_df.empty:
        if ODDS_API_KEY:
            odds_df = fetch_totals_odds_api(ODDS_API_KEY)
        else:
            manual_path = os.path.join(PROC_DIR, "totals_odds_today.csv")
            odds_df     = load_manual_totals_odds(manual_path)
    if odds_df.empty:
        print("  No games found for today. Exiting.")
        exit(0)
    print(f"  ✓ {len(odds_df)} games loaded.")

    print("\n[ 2/4 ] Scoring today's games (individual probable starters)...")
    predictions_df = score_live_games_totals(odds_df)
    print(f"  Mean lambda_hat: {predictions_df['lambda_hat'].mean():.2f} runs")

    print("\n[ 3/4 ] Computing edge scores...")
    # Merge ou_line into predictions before building edge report
    predictions_df = predictions_df.merge(
        odds_df[["home_team", "away_team", "ou_line"]].drop_duplicates(["home_team", "away_team"]),
        on=["home_team", "away_team"], how="left"
    )
    edge_report = build_totals_edge_report(predictions_df, odds_df)

    # Summary
    value_bets = edge_report[edge_report["is_value_bet"] == 1]
    strong     = edge_report[edge_report["edge_score"] >= 7]
    overs      = value_bets[value_bets["bet_type"] == "OVER"]
    unders     = value_bets[value_bets["bet_type"] == "UNDER"]

    print(f"\n  ── Summary ─────────────────────────────────────────────────")
    print(f"  Total bet opportunities: {len(edge_report)}")
    print(f"  Value bets:              {len(value_bets)} "
          f"({len(overs)} OVER, {len(unders)} UNDER)")
    print(f"  Strong plays (score≥7):  {len(strong)}")
    print(f"  ─────────────────────────────────────────────────────────────")

    # Top plays
    if not edge_report.empty:
        print(f"\n  ── Top Totals Plays ─────────────────────────────────────────")
        disp_cols = ["home_team", "away_team", "bet_type", "ou_line",
                     "lambda_hat", "model_prob", "edge", "ev_pct",
                     "kelly_fraction", "edge_score"]
        disp_cols = [c for c in disp_cols if c in edge_report.columns]
        print(edge_report[edge_report["is_value_bet"]==1][disp_cols]
              .head(10).to_string(index=False))

    print(f"\n[ 4/4 ] Exporting...")
    output_path = os.path.join(EXPORT_DIR, f"totals_edges_{today_str}.xlsx")
    edge_report.to_excel(output_path, index=False, engine='openpyxl')
    print(f"  ✓ Full report: {output_path}")

    if not value_bets.empty:
        plays_path = os.path.join(EXPORT_DIR, f"totals_plays_{today_str}.xlsx")
        value_bets.to_excel(plays_path, index=False, engine='openpyxl')
        print(f"  ✓ Value bets:  {plays_path}")

    print("\n" + "=" * 70)
    print("TOTALS MODEL — COMPLETE")
    print("=" * 70)
