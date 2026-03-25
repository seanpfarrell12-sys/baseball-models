"""
=============================================================================
MONEYLINE MODEL — FILE 4 OF 4: EDGE SCORING AND CSV EXPORT
=============================================================================
Purpose : Compare model probabilities to market odds; calculate edge scores
          and Kelly Criterion bet sizing; export final bet recommendations.
Input   : ../data/processed/moneyline_predictions.csv (from Step 3)
Output  : ../exports/moneyline_edges_YYYYMMDD.csv

Edge Score Calculations:
─────────────────────────────────────────────────────────────────────────────
1. IMPLIED PROBABILITY (from market odds):
   - American odds → decimal odds → implied probability
   - Remove the vig (overround) to get true market probability
   - +150 American = 2.50 decimal = 40.0% implied (before vig removal)

2. EDGE:
   edge = model_probability - market_implied_probability_no_vig
   Positive edge → model believes this team is more likely to win than priced
   Example: model says 55%, market implies 50% → edge = +5%

3. EXPECTED VALUE (EV%):
   EV% = (model_prob × (decimal_odds - 1)) - (1 - model_prob)
   EV% > 0 → positive expected value bet
   EV% of 3% → for every $100 bet, expect to profit $3 long-run

4. KELLY CRITERION:
   Full Kelly: f* = (b×p - q) / b
     where b = decimal_odds - 1, p = model_prob, q = 1 - model_prob
   Use FRACTIONAL Kelly (25%) to reduce variance:
     f_recommended = f* × 0.25
   Never bet more than 5% of bankroll on a single game (hard cap).

5. NUMERIC EDGE SCORE (1–10 scale):
   Combines edge magnitude, EV%, and confidence into a single ranking score.
   Score ≥ 7 = strong play | 5–6 = moderate play | < 5 = pass

Odds input options:
  Option A: Provide a CSV with today's odds (manual entry)
  Option B: Use The Odds API (free tier: 500 requests/month)
            Signup: https://the-odds-api.com/
  Option C: Enter odds interactively (for testing)

For R users:
  - f-strings: f"text {variable}" = paste0("text ", variable) in R
  - datetime.now() = Sys.time() in R; .strftime() = format() in R
  - df.to_csv() = write.csv() in R
=============================================================================
"""

import os
import sys
import json
import requests
from datetime import datetime, date
import pandas as pd
import numpy as np

# Add project root to path so we can import the action_network utility
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.action_network import get_moneyline_odds
from utils.probable_starters import (get_games_with_sp_stats,
                                      get_lineup_batting_features,
                                      get_lineups,
                                      load_batting_stats)

# --- Configuration ----------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR  = os.path.join(BASE_DIR, "data", "models")
EXPORT_DIR = os.path.join(BASE_DIR, "exports")

# The Odds API (free tier — set your key here or leave empty for manual mode)
# Get a free key at: https://the-odds-api.com/
ODDS_API_KEY = "fbc985ad430c95d6435cb75210f7b989"
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"

# Fractional Kelly — we use 25% Kelly to reduce variance
# Full Kelly is mathematically optimal but causes very large drawdowns
KELLY_FRACTION = 0.25

# Maximum single-game bankroll allocation (hard cap regardless of Kelly)
MAX_BET_FRACTION = 0.05   # Never bet more than 5% of bankroll

# Minimum edge (in probability units) to consider a bet
MIN_EDGE = 0.03           # 3% edge minimum — filters out noise

# Minimum numeric edge score to include in "strong plays" report
MIN_EDGE_SCORE = 5.0


# =============================================================================
# ODDS CONVERSION UTILITIES
# =============================================================================

def american_to_decimal(american_odds: float) -> float:
    """
    Convert American (moneyline) odds to decimal odds.

    American odds format:
      Positive (+150): profit on $100 bet → decimal = (odds/100) + 1
      Negative (-130): bet needed to profit $100 → decimal = (100/|odds|) + 1

    Examples:
      +150 → 2.50 (bet $100, win $150, return $250 total)
      -110 → 1.909 (bet $110, win $100, return $210 total)
      -150 → 1.667 (bet $150, win $100, return $250 total)

    R equivalent: function(odds) ifelse(odds > 0, odds/100+1, 100/abs(odds)+1)
    """
    if american_odds >= 0:
        return (american_odds / 100.0) + 1.0
    else:
        return (100.0 / abs(american_odds)) + 1.0


def decimal_to_implied_prob(decimal_odds: float) -> float:
    """
    Convert decimal odds to raw implied probability.

    Raw implied probability INCLUDES the vig (sportsbook margin).
    Two-sided market: home implied + away implied > 1.0 (the excess is vig).

    Example: Home -110, Away -110
      Home implied: 1/1.909 = 52.4%
      Away implied: 1/1.909 = 52.4%
      Total: 104.8% → vig = 4.8%

    R equivalent: function(dec_odds) 1 / dec_odds
    """
    if decimal_odds <= 0:
        return 0.5  # Fallback for bad data
    return 1.0 / decimal_odds


def remove_vig(implied_home: float, implied_away: float) -> tuple:
    """
    Remove the vig (overround) from implied probabilities.

    The "fair" probability is each side's share of the total implied probability.
    This gives us the market's TRUE assessment of each team's win probability,
    stripped of the sportsbook's margin.

    Method: Divide each side by the sum of both sides.
      P_home_no_vig = implied_home / (implied_home + implied_away)
      P_away_no_vig = implied_away / (implied_home + implied_away)

    Example: Home -110, Away -110
      implied_home = implied_away = 0.524
      P_home_no_vig = 0.524 / 1.048 = 0.500 (fair 50/50 game)

    R equivalent:
      remove_vig <- function(h, a) {
        total <- h + a
        c(h/total, a/total)
      }
    """
    total = implied_home + implied_away
    if total <= 0:
        return 0.5, 0.5
    return implied_home / total, implied_away / total


# =============================================================================
# EDGE AND KELLY CALCULATIONS
# =============================================================================

def calculate_edge(model_prob: float, market_prob_no_vig: float) -> float:
    """
    Calculate the raw edge: how much better our model is than the market.

    edge > 0 → we think this team is more likely to win than the market prices
    edge < 0 → market is pricing this team higher than our model thinks

    This is the primary signal for bet selection. We only bet positive edges.

    Example:
      model_prob = 0.56 (model says 56% to win)
      market_prob = 0.52 (market implies 52% to win)
      edge = 0.56 - 0.52 = +0.04 (+4 percentage points)
    """
    return round(model_prob - market_prob_no_vig, 4)


def calculate_ev_percent(model_prob: float, decimal_odds: float) -> float:
    """
    Calculate Expected Value as a percentage of the amount wagered.

    EV% = (P_win × profit_per_unit) - (P_lose × 1 unit)
         = model_prob × (decimal_odds - 1) - (1 - model_prob) × 1

    If EV% = 0.04 (4%), betting $100 returns $4 expected profit per bet.

    Example:
      model_prob = 0.56, decimal_odds = 1.95 (roughly -105 American)
      EV% = 0.56 × (1.95-1) - 0.44 × 1 = 0.532 - 0.44 = 0.092 = 9.2%

    R equivalent:
      ev_pct <- function(p, dec) p * (dec - 1) - (1 - p)
    """
    ev = model_prob * (decimal_odds - 1) - (1 - model_prob)
    return round(ev, 4)


def kelly_criterion(model_prob: float, decimal_odds: float,
                    fraction: float = KELLY_FRACTION,
                    max_bet: float = MAX_BET_FRACTION) -> float:
    """
    Calculate recommended bet size as a fraction of bankroll.

    Full Kelly formula:
      f* = (b×p - q) / b
      where: b = decimal_odds - 1 (net profit per unit bet)
             p = model probability of winning
             q = 1 - p = probability of losing

    Fractional Kelly:
      f_recommended = f* × KELLY_FRACTION
      Default: 25% Kelly (quarter-Kelly)

    Hard cap:
      f_recommended = min(f_recommended, MAX_BET_FRACTION)

    Why fractional Kelly?
      - Full Kelly is mathematically optimal for bankroll growth over time
      - But it's extremely volatile — drawdowns of 30-40% are common
      - Quarter-Kelly gives ~87% of the growth with much lower variance
      - Most professional sports bettors use 10-33% Kelly

    Example:
      model_prob=0.56, decimal_odds=1.95
      b = 0.95, p = 0.56, q = 0.44
      f* = (0.95×0.56 - 0.44) / 0.95 = (0.532 - 0.44) / 0.95 = 0.097
      f_recommended = 0.097 × 0.25 = 0.0242 → bet 2.42% of bankroll

    R equivalent:
      kelly <- function(p, dec, frac=0.25, cap=0.05) {
        b <- dec - 1; q <- 1 - p
        f_full <- (b*p - q) / b
        min(max(f_full * frac, 0), cap)
      }
    """
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0

    q = 1.0 - model_prob
    f_full = (b * model_prob - q) / b

    # Apply fractional Kelly
    f_fractional = f_full * fraction

    # Apply hard cap and floor at 0 (never recommend negative bet)
    f_recommended = min(max(f_fractional, 0.0), max_bet)
    return round(f_recommended, 4)


def compute_edge_score(edge: float, ev_pct: float, kelly: float,
                        model_prob: float) -> float:
    """
    Compute a single numeric edge score (0–10 scale) for ranking bets.

    Combines multiple signals into one number for easy sorting:
      - edge magnitude (raw probability edge)
      - EV% (economic value)
      - Kelly size (implied bet confidence)
      - Model probability extremity (confident predictions score higher)

    Score interpretation:
      8–10 : Strong bet — high confidence, meaningful edge
      6–7  : Moderate bet — worth considering
      4–5  : Weak bet — marginal edge, use caution
      0–3  : Pass — edge not significant

    Formula (simplified scoring):
      score = 3.0 × clip(edge, 0, 0.15)/0.15      # Edge component (0–3)
             + 3.0 × clip(ev_pct, 0, 0.15)/0.15    # EV component (0–3)
             + 2.0 × clip(kelly, 0, 0.05)/0.05     # Kelly component (0–2)
             + 2.0 × abs(model_prob - 0.5)/0.5     # Conviction component (0–2)

    R equivalent:
      edge_score <- function(e, ev, k, p) {
        3*(min(max(e,0), 0.15)/0.15) + 3*(min(max(ev,0), 0.15)/0.15) +
        2*(min(max(k,0), 0.05)/0.05) + 2*(abs(p-0.5)/0.5)
      }
    """
    # clip(x, min, max) = constrain x to [min, max] range
    # In R: pmin(pmax(x, min_val), max_val)
    edge_component  = 3.0 * np.clip(edge,    0, 0.15) / 0.15
    ev_component    = 3.0 * np.clip(ev_pct,  0, 0.15) / 0.15
    kelly_component = 2.0 * np.clip(kelly,   0, 0.05) / 0.05
    conv_component  = 2.0 * abs(model_prob - 0.5) / 0.5

    score = edge_component + ev_component + kelly_component + conv_component
    return round(min(score, 10.0), 2)


# =============================================================================
# ODDS DATA LOADING
# =============================================================================

def fetch_odds_from_api(api_key: str) -> pd.DataFrame:
    """
    Fetch today's MLB moneyline odds from The Odds API (free tier).

    Free tier: 500 requests/month
    Documentation: https://the-odds-api.com/liveapi/guides/v4/

    Returns
    -------
    pd.DataFrame
        One row per game with home_team, away_team, home_odds_american,
        away_odds_american columns.
    """
    if not api_key:
        print("  No API key provided. Skipping live odds fetch.")
        return pd.DataFrame()

    params = {
        "apiKey":  api_key,
        "regions": "us",
        "markets": "h2h",         # h2h = moneyline (head-to-head)
        "oddsFormat": "american", # Return odds in American format
        "bookmakers": "fanduel,draftkings,betmgm",  # Consensus from multiple books
    }

    try:
        response = requests.get(ODDS_API_URL, params=params, timeout=10)
        response.raise_for_status()
        games_data = response.json()

        records = []
        for game in games_data:
            home_team = game.get("home_team", "")
            away_team = game.get("away_team", "")

            # Average odds across bookmakers for consensus line
            home_odds_list, away_odds_list = [], []

            for bookmaker in game.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market["key"] == "h2h":
                        for outcome in market.get("outcomes", []):
                            if outcome["name"] == home_team:
                                home_odds_list.append(outcome["price"])
                            elif outcome["name"] == away_team:
                                away_odds_list.append(outcome["price"])

            if home_odds_list and away_odds_list:
                records.append({
                    "home_team":         home_team,
                    "away_team":         away_team,
                    "home_odds_american": np.mean(home_odds_list),
                    "away_odds_american": np.mean(away_odds_list),
                    "game_time":          game.get("commence_time", ""),
                })

        odds_df = pd.DataFrame(records)
        print(f"  ✓ Fetched {len(odds_df)} games from The Odds API.")
        return odds_df

    except Exception as e:
        print(f"  WARNING: Odds API call failed: {e}")
        return pd.DataFrame()


def load_manual_odds(odds_csv_path: str = None) -> pd.DataFrame:
    """
    Load odds from a manually created CSV file.

    Expected CSV format:
      home_team,away_team,home_odds_american,away_odds_american
      NYY,BOS,-130,+110
      LAD,SFG,-180,+155
      HOU,TEX,-115,-105

    If no CSV path is provided, this function creates a template.
    """
    if odds_csv_path and os.path.exists(odds_csv_path):
        print(f"  Loading odds from: {odds_csv_path}")
        return pd.read_csv(odds_csv_path)

    # Return example template if no file provided
    print("  No odds file found. Creating example template.")
    template = pd.DataFrame({
        "home_team":          ["NYY",  "LAD",  "HOU"],
        "away_team":          ["BOS",  "SFG",  "TEX"],
        "home_odds_american": [-130.0, -180.0, -115.0],
        "away_odds_american": [+110.0, +155.0, -105.0],
    })
    template_path = os.path.join(PROC_DIR, "odds_input_template.csv")
    template.to_csv(template_path, index=False)
    print(f"  ✓ Saved odds template to: {template_path}")
    print("  Fill in actual odds and re-run, OR use The Odds API key.")
    return template


# =============================================================================
# LIVE GAME SCORING — Build features for today's games from 2025 stats
# =============================================================================

# Duplicate team abbreviation map here so export file is self-contained
_FG_TO_BREF = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHC", "CWS": "CWS", "CIN": "CIN", "CLE": "CLE",
    "COL": "COL", "DET": "DET", "HOU": "HOU", "KCR": "KCR",
    "LAA": "LAA", "LAD": "LAD", "MIA": "MIA", "MIL": "MIL",
    "MIN": "MIN", "NYM": "NYM", "NYY": "NYY", "OAK": "OAK",
    "PHI": "PHI", "PIT": "PIT", "SDP": "SDP", "SEA": "SEA",
    "SFG": "SFG", "STL": "STL", "TBR": "TBR", "TEX": "TEX",
    "TOR": "TOR", "WSN": "WSN",
    "CHW": "CWS", "SD": "SDP", "SF": "SFG", "TB": "TBR",
    "KC": "KCR", "WAS": "WSN",
}

_PARK_FACTORS = {
    "ARI": 97,  "ATL": 102, "BAL": 104, "BOS": 105, "CHC": 101,
    "CWS": 96,  "CIN": 103, "CLE": 98,  "COL": 116, "DET": 96,
    "HOU": 97,  "KCR": 97,  "LAA": 97,  "LAD": 95,  "MIA": 97,
    "MIL": 99,  "MIN": 101, "NYM": 97,  "NYY": 105, "OAK": 96,
    "PHI": 102, "PIT": 97,  "SDP": 96,  "SEA": 94,  "SFG": 93,
    "STL": 99,  "TBR": 97,  "TEX": 101, "TOR": 102, "WSN": 100,
}


def _build_team_features_2025() -> dict:
    """
    Build a dict mapping team abbreviation → 2025 feature values.

    Uses the most recent complete season (2025) to score 2026 games.
    Returns a flat dict: {team: {feature_name: value}}
    """
    raw_dir = os.path.join(BASE_DIR, "data", "raw")

    # ── Pitching: IP-weighted mean of starters (GS >= 5) ──────────────────
    pit = pd.read_csv(os.path.join(raw_dir, "raw_pitching_stats.csv"))
    pit["team_std"] = pit["Team"].map(_FG_TO_BREF).fillna(pit["Team"])
    pit_latest = int(pit["Season"].max())
    pit_season = pit_latest if len(pit[pit["Season"] == pit_latest]) >= 20 else pit_latest - 1
    starters = pit[(pit["Season"] == pit_season) & (pit["GS"] >= 5)].copy()

    pit_cols = ["SIERA", "xFIP", "FIP", "K%", "BB%", "K-BB%"]
    pit_cols = [c for c in pit_cols if c in starters.columns]

    sp_by_team = {}
    for team, grp in starters.groupby("team_std"):
        weights = grp["IP"].fillna(0)
        total_w = weights.sum()
        if total_w == 0:
            vals = grp[pit_cols].mean()
        else:
            vals = {c: (grp[c].fillna(grp[c].mean()) * weights).sum() / total_w
                    for c in pit_cols}
        # Rename to match training feature names
        sp_by_team[team] = {
            "sp_siera":   vals.get("SIERA",  vals["SIERA"]  if "SIERA"  in vals else np.nan),
            "sp_xfip":    vals.get("xFIP",   vals["xFIP"]   if "xFIP"   in vals else np.nan),
            "sp_fip":     vals.get("FIP",    vals["FIP"]    if "FIP"    in vals else np.nan),
            "sp_k_pct":   vals.get("K%",     vals["K%"]     if "K%"     in vals else np.nan),
            "sp_bb_pct":  vals.get("BB%",    vals["BB%"]    if "BB%"    in vals else np.nan),
            "sp_k_bb_pct":vals.get("K-BB%",  vals["K-BB%"]  if "K-BB%" in vals else np.nan),
        }

    # ── Batting: wOBA, ISO, BABIP + BaseRuns from team totals ─────────────
    bat = pd.read_csv(os.path.join(raw_dir, "raw_team_batting.csv"))
    bat["team_std"] = bat["Team"].map(_FG_TO_BREF).fillna(bat["Team"])
    bat_latest = int(bat["Season"].max())
    bat_season = bat_latest if len(bat[bat["Season"] == bat_latest]) >= 20 else bat_latest - 1
    bat25 = bat[bat["Season"] == bat_season].copy()

    off_by_team = {}
    for _, row in bat25.iterrows():
        team = row["team_std"]
        # BaseRuns: (A×B)/(B+C)+D
        h  = row.get("H",  0)
        b2 = row.get("2B", 0)
        b3 = row.get("3B", 0)
        hr = row.get("HR", 0)
        bb = row.get("BB", 0)
        ab = row.get("AB", 1)
        g  = max(row.get("G", 162), 1)
        singles = h - b2 - b3 - hr
        A = h + bb - hr
        B = 0.8*singles + 2.1*b2 + 3.4*b3 + 1.8*hr + 0.1*bb
        C = ab - h
        D = hr
        base_runs = (A * B) / (B + C + 1e-9) + D
        off_by_team[team] = {
            "off_woba":           row.get("wOBA",  np.nan),
            "off_iso":            row.get("ISO",   np.nan),
            "off_babip":          row.get("BABIP", np.nan),
            "base_runs_per_game": base_runs / g,
        }

    # ── Merge into single dict per team ───────────────────────────────────
    all_teams = set(list(sp_by_team.keys()) + list(off_by_team.keys()))
    team_features = {}
    for team in all_teams:
        feats = {}
        feats.update(sp_by_team.get(team, {}))
        feats.update(off_by_team.get(team, {}))
        feats["park_factor"] = _PARK_FACTORS.get(team, 100)
        team_features[team] = feats

    return team_features


def score_live_games(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score today's games from Action Network using the trained model.

    SP stats come from today's probable starters (individual 2025 stats).
    Offensive stats come from 2025 team totals.

    Parameters
    ----------
    odds_df : pd.DataFrame
        From get_moneyline_odds() — must have home_team, away_team columns.

    Returns
    -------
    pd.DataFrame
        odds_df with p_home_win and p_away_win columns added.
    """
    import xgboost as xgb

    # Load trained model
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(MODEL_DIR, "moneyline_model.json"))

    with open(os.path.join(MODEL_DIR, "moneyline_features.json")) as f:
        feature_cols = json.load(f)

    # ── Fetch individual probable starter stats from MLB Stats API ──────────
    print("  Fetching today's probable starters (individual stats)...")
    game_date = date.today().strftime("%Y-%m-%d")
    sp_df = get_games_with_sp_stats(game_date)   # auto-selects most recent season

    # ── Build lineup-based batting features (most recent season) ────────────
    print("  Fetching today's lineups for lineup-based batting features...")
    lineups, lineup_sources = get_lineups(game_date, return_sources=True)
    batting_df = load_batting_stats()   # loads most recent season (2026 → 2025 fallback)
    lineup_bat = get_lineup_batting_features(lineups, batting_df)

    # Fall back to team CSV averages for teams whose lineup isn't posted yet
    print("  Building team batting fallback (for teams without confirmed lineups)...")
    team_feats = _build_team_features_2025()

    # Load training means as fallback for missing values
    train_means = pd.read_csv(os.path.join(PROC_DIR, "moneyline_dataset.csv"))[feature_cols].mean()

    # Build a lookup from sp_df: (home_team, away_team) → SP stat row
    sp_lookup = {}
    if not sp_df.empty:
        for _, sp_row in sp_df.iterrows():
            key = (sp_row["home_team"], sp_row["away_team"])
            sp_lookup[key] = sp_row

    rows = []
    for _, game in odds_df.iterrows():
        home = game["home_team"]
        away = game["away_team"]

        # Skip games where either lineup is not yet confirmed
        if lineup_sources.get(home, "projected") != "confirmed" or \
           lineup_sources.get(away, "projected") != "confirmed":
            print(f"  Skipping {away} @ {home} — lineup(s) not confirmed yet.")
            continue

        hf = team_feats.get(home, {})
        af = team_feats.get(away, {})

        # Prefer individual probable starter stats; fall back to team averages
        _sp = sp_lookup.get((home, away), {})
        sp_row = _sp.to_dict() if isinstance(_sp, pd.Series) else _sp
        if sp_row:
            h_sp = {k.replace("home_", ""): v for k, v in sp_row.items()
                    if k.startswith("home_sp_")}
            a_sp = {k.replace("away_", ""): v for k, v in sp_row.items()
                    if k.startswith("away_sp_")}
            sp_source = sp_row.get("home_sp_source", "individual")
        else:
            h_sp = {k: hf.get(k, np.nan) for k in
                    ["sp_siera", "sp_xfip", "sp_fip", "sp_k_pct", "sp_bb_pct", "sp_k_bb_pct"]}
            a_sp = {k: af.get(k, np.nan) for k in
                    ["sp_siera", "sp_xfip", "sp_fip", "sp_k_pct", "sp_bb_pct", "sp_k_bb_pct"]}
            sp_source = "team_avg_fallback"

        # Prefer lineup-based batting (today's actual players, most recent season)
        # Fall back to team-average CSV stats when lineup not posted
        hlb = lineup_bat.get(home, {})
        alb = lineup_bat.get(away, {})
        bat_source_home = hlb.get("_lineup_source", "confirmed") if hlb else "team_avg"
        bat_source_away = alb.get("_lineup_source", "confirmed") if alb else "team_avg"

        home_off_woba  = hlb.get("off_woba",            hf.get("off_woba",           np.nan))
        home_off_iso   = hlb.get("off_iso",              hf.get("off_iso",            np.nan))
        home_off_babip = hlb.get("off_babip",            hf.get("off_babip",          np.nan))
        home_base_runs = hlb.get("base_runs_per_game",   hf.get("base_runs_per_game", np.nan))
        away_off_woba  = alb.get("off_woba",             af.get("off_woba",           np.nan))
        away_off_iso   = alb.get("off_iso",              af.get("off_iso",            np.nan))
        away_off_babip = alb.get("off_babip",            af.get("off_babip",          np.nan))
        away_base_runs = alb.get("base_runs_per_game",   af.get("base_runs_per_game", np.nan))

        row = {
            "game_date":       game.get("game_time", date.today().isoformat())[:10],
            "home_team":       home,
            "away_team":       away,
            "home_sp_name":    sp_row.get("home_sp_name", "") if sp_row else "",
            "away_sp_name":    sp_row.get("away_sp_name", "") if sp_row else "",
            "sp_source":       sp_source,
            "bat_source_home": bat_source_home,
            "bat_source_away": bat_source_away,
            # Home pitching (individual starter)
            "home_sp_siera":    h_sp.get("sp_siera",    np.nan),
            "home_sp_xfip":     h_sp.get("sp_xfip",     np.nan),
            "home_sp_fip":      h_sp.get("sp_fip",      np.nan),
            "home_sp_k_pct":    h_sp.get("sp_k_pct",    np.nan),
            "home_sp_bb_pct":   h_sp.get("sp_bb_pct",   np.nan),
            "home_sp_k_bb_pct": h_sp.get("sp_k_bb_pct", np.nan),
            # Away pitching (individual starter)
            "away_sp_siera":    a_sp.get("sp_siera",    np.nan),
            "away_sp_xfip":     a_sp.get("sp_xfip",     np.nan),
            "away_sp_fip":      a_sp.get("sp_fip",      np.nan),
            "away_sp_k_pct":    a_sp.get("sp_k_pct",    np.nan),
            "away_sp_bb_pct":   a_sp.get("sp_bb_pct",   np.nan),
            "away_sp_k_bb_pct": a_sp.get("sp_k_bb_pct", np.nan),
            # Home offense (lineup-based if available, else team avg)
            "home_off_woba":           home_off_woba,
            "home_off_iso":            home_off_iso,
            "home_off_babip":          home_off_babip,
            "home_base_runs_per_game": home_base_runs,
            "home_park_factor":        hf.get("park_factor", 100),
            # Away offense (lineup-based if available, else team avg)
            "away_off_woba":           away_off_woba,
            "away_off_iso":            away_off_iso,
            "away_off_babip":          away_off_babip,
            "away_base_runs_per_game": away_base_runs,
            # Differentials
            "diff_sp_siera":           h_sp.get("sp_siera", np.nan) - a_sp.get("sp_siera", np.nan),
            "diff_sp_xfip":            h_sp.get("sp_xfip",  np.nan) - a_sp.get("sp_xfip",  np.nan),
            "diff_base_runs_per_game": home_base_runs - away_base_runs,
            "home_field": 1,
            "home_odds_american": game.get("home_odds_american", np.nan),
            "away_odds_american": game.get("away_odds_american", np.nan),
        }
        rows.append(row)

    game_df = pd.DataFrame(rows)

    # Fill any NaN with training set means
    for col in feature_cols:
        if col in game_df.columns:
            game_df[col] = game_df[col].fillna(train_means.get(col, 0.0))
        else:
            game_df[col] = train_means.get(col, 0.0)

    # Score with model
    X = game_df[feature_cols]
    probs = model.predict_proba(X)
    game_df["p_home_win"] = probs[:, 1].round(4)
    game_df["p_away_win"] = probs[:, 0].round(4)

    n_ind_sp  = (game_df.get("sp_source",       pd.Series()) == "individual").sum()
    n_lineup  = (game_df.get("bat_source_home", pd.Series()) == "lineup").sum()
    print(f"  ✓ Scored {len(game_df)} games | "
          f"{n_ind_sp} individual SP | {n_lineup} lineup-based batting.")
    return game_df


# =============================================================================
# MAIN EXPORT FUNCTION
# =============================================================================

def build_edge_report(predictions_df: pd.DataFrame,
                       odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join model predictions with market odds and compute all edge metrics.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        From Step 3 — contains p_home_win, p_away_win, home_team, away_team
    odds_df : pd.DataFrame
        Moneyline odds — home_odds_american, away_odds_american per game

    Returns
    -------
    pd.DataFrame
        Full edge report with all metrics for each side of each game.
    """
    # predictions_df was built directly from odds_df in score_live_games(),
    # so odds columns are already present — merge only when odds aren't there yet.
    odds_cols = ["home_odds_american", "away_odds_american"]
    if all(c in predictions_df.columns for c in odds_cols):
        df = predictions_df.copy()
    else:
        df = predictions_df.merge(
            odds_df[["home_team", "away_team"] + odds_cols].drop_duplicates(
                subset=["home_team", "away_team"]
            ),
            on=["home_team", "away_team"], how="left"
        )

    rows = []

    for _, row in df.iterrows():
        home_odds_am = row.get("home_odds_american", -110)
        away_odds_am = row.get("away_odds_american", -110)

        # Convert to decimal odds
        home_dec = american_to_decimal(home_odds_am)
        away_dec = american_to_decimal(away_odds_am)

        # Raw implied probabilities (include vig)
        home_implied_raw = decimal_to_implied_prob(home_dec)
        away_implied_raw = decimal_to_implied_prob(away_dec)

        # Remove vig (no-vig market probabilities)
        home_market, away_market = remove_vig(home_implied_raw, away_implied_raw)

        # Model probabilities
        p_home = row.get("p_home_win", 0.5)
        p_away = row.get("p_away_win", 0.5)

        # Edge calculations for each side
        for side, model_p, market_p, dec_odds, am_odds in [
            ("home", p_home, home_market, home_dec, home_odds_am),
            ("away", p_away, away_market, away_dec, away_odds_am),
        ]:
            edge   = calculate_edge(model_p, market_p)
            ev     = calculate_ev_percent(model_p, dec_odds)
            kelly  = kelly_criterion(model_p, dec_odds)
            score  = compute_edge_score(edge, ev, kelly, model_p)

            rows.append({
                "game_date":          row.get("game_date", datetime.now().strftime("%Y-%m-%d")),
                "home_team":          row.get("home_team", ""),
                "away_team":          row.get("away_team", ""),
                "bet_side":           side.upper(),  # "HOME" or "AWAY"
                "model_prob":         round(model_p,    4),
                "market_implied":     round(market_p,   4),
                "edge":               edge,           # Raw edge (+ = value)
                "ev_pct":             ev,             # Expected value %
                "kelly_fraction":     kelly,          # Recommended bet size
                "edge_score":         score,          # 0–10 composite score
                "american_odds":      am_odds,
                "decimal_odds":       round(dec_odds, 4),
                "vig_pct":           round((home_implied_raw + away_implied_raw - 1) * 100, 2),
                "is_value_bet":      1 if edge >= MIN_EDGE and ev > 0 else 0,
            })

    report_df = pd.DataFrame(rows)

    # Sort by edge_score descending (best bets at top)
    # In R: report_df[order(-report_df$edge_score), ]
    report_df = report_df.sort_values("edge_score", ascending=False).reset_index(drop=True)

    return report_df


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("MONEYLINE MODEL — STEP 4: EDGE SCORING AND EXPORT")
    print("=" * 70)
    today_str = datetime.now().strftime("%Y%m%d")

    # Fetch today's games and odds from Action Network
    print("\n[ 1/4 ] Fetching today's games and odds...")
    odds_df = get_moneyline_odds(date.today().strftime("%Y-%m-%d"))
    if odds_df.empty:
        if ODDS_API_KEY:
            odds_df = fetch_odds_from_api(ODDS_API_KEY)
        else:
            manual_path = os.path.join(PROC_DIR, "moneyline_odds_today.csv")
            odds_df     = load_manual_odds(manual_path)
    if odds_df.empty:
        print("  No games found for today. Exiting.")
        exit(0)
    print(f"  ✓ {len(odds_df)} games loaded from Action Network.")

    # Score today's games using trained model + individual probable starter stats
    print("\n[ 2/4 ] Scoring today's games (individual probable starters)...")
    predictions_df = score_live_games(odds_df)

    # Build edge report
    print("\n[ 3/4 ] Computing edge scores...")
    edge_report = build_edge_report(predictions_df, odds_df)
    print(f"  ✓ Edge report built: {len(edge_report)} bet opportunities assessed.")

    # Summary statistics
    value_bets = edge_report[edge_report["is_value_bet"] == 1]
    strong     = edge_report[edge_report["edge_score"] >= 7]
    print(f"\n  ── Summary ─────────────────────────────────────────────────")
    print(f"  Total bet opportunities: {len(edge_report)}")
    print(f"  Value bets (edge ≥ {MIN_EDGE:.0%}): {len(value_bets)}")
    print(f"  Strong plays (score ≥ 7): {len(strong)}")
    if len(value_bets):
        print(f"  Avg Kelly on value bets:  {value_bets['kelly_fraction'].mean():.3f} "
              f"({value_bets['kelly_fraction'].mean()*100:.1f}% of bankroll)")
    print(f"  ─────────────────────────────────────────────────────────────")

    # Top plays preview
    if not edge_report.empty:
        print(f"\n  ── Top Moneyline Plays ──────────────────────────────────────")
        top_cols = ["home_team", "away_team", "bet_side", "model_prob",
                    "market_implied", "edge", "ev_pct", "kelly_fraction",
                    "edge_score", "american_odds"]
        top_cols = [c for c in top_cols if c in edge_report.columns]
        print(edge_report[edge_report["is_value_bet"] == 1][top_cols]
              .head(10).to_string(index=False))

    # Export
    print(f"\n[ 4/4 ] Exporting CSV...")
    output_path = os.path.join(EXPORT_DIR, f"moneyline_edges_{today_str}.xlsx")
    edge_report.to_excel(output_path, index=False, engine='openpyxl')
    print(f"  ✓ Saved: {output_path}")

    # Also save just the value bets for quick reference
    if not value_bets.empty:
        value_path = os.path.join(EXPORT_DIR, f"moneyline_plays_{today_str}.xlsx")
        value_bets.to_excel(value_path, index=False, engine='openpyxl')
        print(f"  ✓ Value bets only: {value_path}")

    print("\n" + "=" * 70)
    print("MONEYLINE MODEL — COMPLETE")
    print(f"Output: {output_path}")
    print("=" * 70)
