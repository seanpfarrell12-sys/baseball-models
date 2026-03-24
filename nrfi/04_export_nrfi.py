"""
=============================================================================
NRFI / YRFI MODEL — FILE 4 OF 4: DAILY EDGE SCORING AND EXPORT
=============================================================================
Purpose : For today's games with a confirmed starting lineup and both SPs
          announced, score the NRFI/YRFI market and identify games where
          our model's P(YRFI) diverges from the book's implied probability
          by at least MIN_EDGE.

Market context:
  NRFI/YRFI is a game-level binary market, not a player prop.
  There are exactly two sides per game:
    YRFI (Yes Run First Inning) — at least one run by either team
    NRFI (No Run First Inning)  — zero runs in the entire 1st inning

  Books price this with American odds on each side.  The NRFI side is
  typically -115 to -125 (favorite); YRFI is +100 to +110.  In recent
  seasons the market YRFI rate has been ~55-57%, pricing has tightened,
  and there is genuine alpha in identifying mis-priced games.

Daily scoring pipeline:
  1. Fetch today's probable starters (via probable_starters.py).
  2. Fetch confirmed top-3 batting lineups (via probable_starters.py).
  3. Score each game: build a feature vector identical to the training set,
     call model.predict_proba() → raw P(YRFI), then apply isotonic
     calibration to get calibrated P(YRFI).
  4. Fetch live NRFI/YRFI odds from The Odds API.
  5. Compute edge = model P(YRFI) − book implied P(YRFI).
  6. Compute EV% and Kelly criterion bet size.
  7. Export to CSV and console.

Input:
  data/models/nrfi_model.json
  data/models/nrfi_calibrator.pkl
  data/models/nrfi_features.json
  data/raw/raw_nrfi_park_meta.json    (park meta for environmental features)
  data/raw/raw_fg_pitching_nrfi.csv   (Stuff+/Location+ for today's SPs)
  data/raw/raw_batting_splits_lhp.csv / raw_batting_splits_rhp.csv

Output:
  exports/nrfi_edges_YYYYMMDD.csv
=============================================================================
"""

import os
import sys
import json
import pickle
import requests
import warnings
import math
from datetime  import datetime, date
from math      import radians, cos

import numpy  as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.action_network    import get_nrfi_odds
from utils.probable_starters import (get_probable_starters,
                                     normalize_name,
                                     get_lineups,
                                     get_lineup_batting_features,
                                     load_batting_stats)

# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR    = os.path.join(BASE_DIR, "data", "raw")
MODEL_DIR  = os.path.join(BASE_DIR, "data", "models")
EXPORT_DIR = os.path.join(BASE_DIR, "exports")
os.makedirs(EXPORT_DIR, exist_ok=True)

ODDS_API_KEY      = "fbc985ad430c95d6435cb75210f7b989"
KELLY_FRACTION    = 0.25    # fractional Kelly for game-level markets
MAX_BET_FRACTION  = 0.04    # cap per game at 4% of bankroll
MIN_EDGE          = 0.03    # minimum edge to flag as a pick (3%)
CURRENT_SEASON    = 2025

# Open-Meteo forecast URL
OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


# =============================================================================
# LOAD MODEL ARTIFACTS
# =============================================================================

def load_model():
    """Load trained XGBoost + isotonic calibrator + feature list."""
    model_path = os.path.join(MODEL_DIR, "nrfi_model.json")
    cal_path   = os.path.join(MODEL_DIR, "nrfi_calibrator.pkl")
    feat_path  = os.path.join(MODEL_DIR, "nrfi_features.json")

    for p in (model_path, cal_path, feat_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found — run 03_analysis_nrfi.py first")

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    with open(cal_path, "rb") as f:
        calibrator = pickle.load(f)

    with open(feat_path) as f:
        feat_data = json.load(f)
    feat_cols = feat_data["features"]

    return model, calibrator, feat_cols


def load_park_meta():
    path = os.path.join(RAW_DIR, "raw_nrfi_park_meta.json")
    if not os.path.exists(path):
        return {}, {}
    with open(path) as f:
        data = json.load(f)
    return data.get("stadium_meta", {}), data.get("park_hr_factors", {})


def load_sp_lookup():
    """
    Load SP first-inning stats + FG Stuff+/Location+ keyed by
    (pitcher_name_normalized, season).  Returns a dict for fast lookup.
    """
    # First-inning stats from training data
    nrfi_path = os.path.join(BASE_DIR, "data", "processed", "nrfi_dataset.csv")
    sp_records = {}

    if os.path.exists(nrfi_path):
        df = pd.read_csv(nrfi_path, low_memory=False)
        # We'll use season=CURRENT_SEASON stats where available; prior year as fallback
        for _, row in df.iterrows():
            pass  # aggregate approach below is more efficient

        # Build SP feature means from the processed dataset
        sp_cols = [c for c in df.columns if c.startswith("home_sp_") or
                   c.startswith("away_sp_")]
        # Aggregate mean per (sp_mlbam, season) — just for reference
        # The main SP lookup is built from raw files below

    # FG pitching (Stuff+/Location+)
    fg_path = os.path.join(RAW_DIR, "raw_fg_pitching_nrfi.csv")
    fg = pd.DataFrame()
    if os.path.exists(fg_path):
        fg = pd.read_csv(fg_path, low_memory=False)
        if "Name" in fg.columns:
            fg["name_norm"] = fg["Name"].apply(normalize_name)
        if "Season" in fg.columns:
            fg["season"] = pd.to_numeric(fg["Season"], errors="coerce")

    return fg


def load_batting_splits_for_scoring():
    """Load LHP/RHP batting splits for top-3 lineup scoring."""
    lhp = pd.DataFrame()
    rhp = pd.DataFrame()
    lhp_path = os.path.join(RAW_DIR, "raw_batting_splits_lhp.csv")
    rhp_path = os.path.join(RAW_DIR, "raw_batting_splits_rhp.csv")
    if os.path.exists(lhp_path):
        lhp = pd.read_csv(lhp_path, low_memory=False)
    if os.path.exists(rhp_path):
        rhp = pd.read_csv(rhp_path, low_memory=False)
    return lhp, rhp


# =============================================================================
# REAL-TIME WEATHER (OPEN-METEO FORECAST)
# =============================================================================

def get_forecast_weather(home_team: str, game_dt: datetime,
                          stadium_meta: dict) -> dict:
    """
    Pull Open-Meteo hourly forecast for the stadium on game day.
    Returns dict: temperature_f, wind_speed_mph, wind_dir_deg, humidity_pct
    """
    meta = stadium_meta.get(home_team, {})
    if not meta:
        return {}

    roof = None
    try:
        with open(os.path.join(RAW_DIR, "raw_nrfi_park_meta.json")) as f:
            park_meta = json.load(f)
        roof = park_meta.get("park_hr_factors", {}).get(home_team, {}).get("roof")
    except Exception:
        pass

    # Dome/retractable parks: weather is controlled
    if roof in ("dome", "retractable"):
        return {"is_dome": 1, "temperature_f": 72.0,
                "wind_speed_mph": 0.0, "wind_dir_deg": 0.0,
                "humidity_pct": 50.0}

    try:
        params = {
            "latitude":           meta["lat"],
            "longitude":          meta["lon"],
            "hourly":             "temperature_2m,relativehumidity_2m,"
                                  "windspeed_10m,winddirection_10m",
            "temperature_unit":   "fahrenheit",
            "windspeed_unit":     "mph",
            "timezone":           "auto",
            "start_date":         game_dt.strftime("%Y-%m-%d"),
            "end_date":           game_dt.strftime("%Y-%m-%d"),
        }
        r = requests.get(OPENMETEO_FORECAST_URL, params=params, timeout=10)
        r.raise_for_status()
        data  = r.json()
        hours = data.get("hourly", {})
        times = hours.get("time", [])

        # Pick hour closest to first pitch (typ. 7 PM ± 2 h local)
        game_hour = game_dt.hour if game_dt.hour != 0 else 19
        best_i    = min(range(len(times)),
                        key=lambda i: abs(int(times[i][11:13]) - game_hour),
                        default=0)

        def _get(key):
            vals = hours.get(key, [])
            return float(vals[best_i]) if best_i < len(vals) else np.nan

        return {
            "temperature_f":  _get("temperature_2m"),
            "humidity_pct":   _get("relativehumidity_2m"),
            "wind_speed_mph": _get("windspeed_10m"),
            "wind_dir_deg":   _get("winddirection_10m"),
            "is_dome":        0,
        }
    except Exception as e:
        print(f"    WARNING: weather fetch failed for {home_team} — {e}")
        return {}


def compute_wind_toward_cf(wind_speed: float, wind_dir: float,
                            cf_bearing: float, is_dome: int) -> float:
    """Wind component toward CF. Positive = tailwind = HR-friendly."""
    if is_dome or np.isnan(wind_speed) or np.isnan(wind_dir):
        return 0.0
    wind_toward_deg = (wind_dir + 180) % 360
    angle_diff      = radians((wind_toward_deg - cf_bearing + 360) % 360)
    return float(wind_speed) * cos(angle_diff)


# =============================================================================
# SP FEATURE VECTOR (from FG + Statcast stats)
# =============================================================================

def get_sp_features(sp_name: str, sp_hand: str,
                    fg_df: pd.DataFrame, season: int) -> dict:
    """
    Build SP feature dict for scoring.

    Priority:
      1. FanGraphs current season stats (Stuff+, Location+, K%, BB%, SwStr%)
      2. Prior season stats if current season unavailable
      3. League median fallback (populated by imputation later)
    """
    feats = {
        "fi_era":         None,
        "fi_k_pct":       None,
        "fi_bb_pct":      None,
        "fi_hr_per_9":    None,
        "fi_whiff_pct":   None,
        "stuff_plus":     None,
        "location_plus":  None,
        "swstr_pct":      None,
        "f_strike_pct":   None,
        "is_lhp":         1.0 if sp_hand == "L" else 0.0,
    }

    if fg_df.empty or "name_norm" not in fg_df.columns:
        return feats

    name_norm = normalize_name(sp_name)

    # Try current season first, then prior year
    for yr in (season, season - 1):
        mask = (fg_df["name_norm"] == name_norm) & (fg_df["season"] == yr)
        rows = fg_df[mask]
        if rows.empty:
            continue
        row = rows.iloc[-1]   # take most recent row if multiple

        COLUMN_MAP = {
            "stuff_plus":    ["Stuff+", "stuff_plus"],
            "location_plus": ["Location+", "location_plus"],
            "swstr_pct":     ["SwStr%", "swstr_pct"],
            "f_strike_pct":  ["F-Strike%", "fstrike_pct"],
            "fi_k_pct":      ["K%", "SO%"],
            "fi_bb_pct":     ["BB%", "uBB%"],
        }
        for feat, candidates in COLUMN_MAP.items():
            if feats[feat] is not None:
                continue
            for col in candidates:
                if col in row.index and pd.notna(row[col]):
                    val = row[col]
                    # Convert "12.5%" string format
                    if isinstance(val, str) and val.endswith("%"):
                        val = float(val.replace("%", "")) / 100
                    feats[feat] = float(val)
                    break
        break

    return feats


# =============================================================================
# TOP-3 LINEUP FEATURE VECTOR
# =============================================================================

def get_top3_features(top3_names: list, sp_hand: str,
                      df_lhp: pd.DataFrame, df_rhp: pd.DataFrame,
                      chad: pd.DataFrame = None) -> dict:
    """
    Build average platoon batting stats for the top-3 lineup slots.

    sp_hand: 'L' or 'R' — determines which splits file to use.
    """
    splits_df = df_lhp if sp_hand == "L" else df_rhp
    FEAT_COLS  = {
        "wrc_plus": ["wRC+", "wRC"],
        "obp":      ["OBP", "On-Base%"],
        "iso":      ["ISO"],
        "k_pct":    ["K%", "SO%"],
        "bb_pct":   ["BB%", "uBB%"],
    }

    vals = {f: [] for f in FEAT_COLS}

    if splits_df.empty:
        return {f: None for f in FEAT_COLS}

    # Normalise column names once
    if "name_norm" not in splits_df.columns:
        name_col = "Name" if "Name" in splits_df.columns else \
                   "PlayerName" if "PlayerName" in splits_df.columns else None
        if name_col:
            splits_df = splits_df.copy()
            splits_df["name_norm"] = splits_df[name_col].apply(normalize_name)

    for name in top3_names:
        if not name:
            continue
        norm = normalize_name(name)
        # Get prior-year stats
        for yr in (CURRENT_SEASON - 1, CURRENT_SEASON):
            yr_col = "season" if "season" in splits_df.columns else "Season"
            mask   = (splits_df.get("name_norm", pd.Series()) == norm)
            if yr_col in splits_df.columns:
                mask = mask & (splits_df[yr_col] == yr)
            rows = splits_df[mask]
            if rows.empty:
                continue
            row = rows.iloc[-1]
            for feat, candidates in FEAT_COLS.items():
                for col in candidates:
                    if col in row.index and pd.notna(row[col]):
                        val = row[col]
                        if isinstance(val, str) and val.endswith("%"):
                            val = float(val.replace("%", "")) / 100
                        vals[feat].append(float(val))
                        break
            break

    return {f: float(np.mean(v)) if v else None for f, v in vals.items()}


# =============================================================================
# ODDS UTILITIES
# =============================================================================

def american_to_decimal(odds: float) -> float:
    return (odds / 100 + 1) if odds >= 0 else (100 / abs(odds) + 1)


def juice_to_implied_prob(juice: float) -> float:
    dec = american_to_decimal(juice)
    return 1.0 / dec


def remove_vig(yes_juice: float, no_juice: float) -> tuple:
    """Return (true_p_yes, true_p_no) with vig removed."""
    p_yes_raw = juice_to_implied_prob(yes_juice)
    p_no_raw  = juice_to_implied_prob(no_juice)
    total     = p_yes_raw + p_no_raw
    return p_yes_raw / total, p_no_raw / total


def kelly_bet(edge: float, win_prob: float, decimal_odds: float,
              fraction: float = KELLY_FRACTION,
              cap: float = MAX_BET_FRACTION) -> float:
    """Fractional Kelly criterion bet size as fraction of bankroll."""
    b = decimal_odds - 1
    if b <= 0 or win_prob <= 0:
        return 0.0
    kelly_full = (b * win_prob - (1 - win_prob)) / b
    return round(min(max(kelly_full * fraction, 0.0), cap), 4)


def ev_percent(model_prob: float, book_prob: float,
               decimal_odds: float) -> float:
    """Expected value as percent of bet: EV% = (model_p × decimal_odds - 1)."""
    return round((model_prob * decimal_odds - 1) * 100, 2)


# =============================================================================
# GAME SCORING PIPELINE
# =============================================================================

def score_game(game: dict, model, calibrator, feat_cols: list,
               fg_df: pd.DataFrame,
               df_lhp: pd.DataFrame, df_rhp: pd.DataFrame,
               stadium_meta: dict, park_hr_factors: dict,
               game_dt: datetime) -> dict:
    """
    Build feature vector for one game and return scored result dict.

    `game` dict keys expected:
      home_team, away_team,
      home_sp_name, home_sp_hand,
      away_sp_name, away_sp_hand,
      home_top3_names (list of 3 player name strings),
      away_top3_names (list of 3 player name strings),
    """
    home  = game.get("home_team", "")
    away  = game.get("away_team", "")
    season = CURRENT_SEASON

    # ── SP features ──────────────────────────────────────────────────────────
    home_sp_feats = get_sp_features(
        game.get("home_sp_name", ""), game.get("home_sp_hand", "R"),
        fg_df, season,
    )
    away_sp_feats = get_sp_features(
        game.get("away_sp_name", ""), game.get("away_sp_hand", "R"),
        fg_df, season,
    )

    # ── Lineup features ───────────────────────────────────────────────────────
    # Home top-3 face the AWAY SP hand
    home_top3 = get_top3_features(
        game.get("home_top3_names", []), game.get("away_sp_hand", "R"),
        df_lhp, df_rhp,
    )
    # Away top-3 face the HOME SP hand
    away_top3 = get_top3_features(
        game.get("away_top3_names", []), game.get("home_sp_hand", "R"),
        df_lhp, df_rhp,
    )

    # ── Environmental features ────────────────────────────────────────────────
    wx         = get_forecast_weather(home, game_dt, stadium_meta)
    cf_bearing = stadium_meta.get(home, {}).get("cf_bearing", 0)
    altitude   = stadium_meta.get(home, {}).get("alt_ft", 0)
    hr_factor  = park_hr_factors.get(home, {}).get("hr_factor", 100)
    roof       = park_hr_factors.get(home, {}).get("roof", "open")
    is_dome    = int(roof in ("dome", "retractable"))

    temp         = wx.get("temperature_f", 72.0) or 72.0
    wind_spd     = wx.get("wind_speed_mph", 0.0) or 0.0
    wind_dir     = wx.get("wind_dir_deg",   0.0) or 0.0
    humidity     = wx.get("humidity_pct",   50.0) or 50.0

    wind_toward  = compute_wind_toward_cf(wind_spd, wind_dir, cf_bearing, is_dome)
    temp_carry   = max(0.0, (temp - 70.0) * 0.002) if not is_dome else 0.0
    alt_carry    = (altitude / 1000.0) * 0.01
    hr_env       = (hr_factor / 100.0
                    * (1.0 + temp_carry)
                    * (1.0 + alt_carry)
                    * (1.0 + np.clip(wind_toward, -20, 20) * 0.005))

    # ── Interaction features ──────────────────────────────────────────────────
    home_sp_k    = home_sp_feats.get("fi_k_pct") or 0.22
    away_sp_k    = away_sp_feats.get("fi_k_pct") or 0.22
    home_sp_bb   = home_sp_feats.get("fi_bb_pct") or 0.085
    away_sp_bb   = away_sp_feats.get("fi_bb_pct") or 0.085

    home_wrc     = home_top3.get("wrc_plus") or 100.0
    away_wrc     = away_top3.get("wrc_plus") or 100.0
    home_iso     = home_top3.get("iso") or 0.155
    away_iso     = away_top3.get("iso") or 0.155

    # ── Assemble feature row ──────────────────────────────────────────────────
    def _sp(side, feats, key):
        col = f"{side}_sp_{key}"
        return feats.get(key)

    row = {
        # Home SP
        "home_sp_fi_era":        _sp("home", home_sp_feats, "fi_era"),
        "home_sp_fi_k_pct":      _sp("home", home_sp_feats, "fi_k_pct"),
        "home_sp_fi_bb_pct":     _sp("home", home_sp_feats, "fi_bb_pct"),
        "home_sp_fi_hr_per_9":   _sp("home", home_sp_feats, "fi_hr_per_9"),
        "home_sp_fi_whiff_pct":  _sp("home", home_sp_feats, "fi_whiff_pct"),
        "home_sp_stuff_plus":    _sp("home", home_sp_feats, "stuff_plus"),
        "home_sp_location_plus": _sp("home", home_sp_feats, "location_plus"),
        "home_sp_swstr_pct":     _sp("home", home_sp_feats, "swstr_pct"),
        "home_sp_f_strike_pct":  _sp("home", home_sp_feats, "f_strike_pct"),
        "home_sp_is_lhp":        _sp("home", home_sp_feats, "is_lhp"),
        # Away SP
        "away_sp_fi_era":        _sp("away", away_sp_feats, "fi_era"),
        "away_sp_fi_k_pct":      _sp("away", away_sp_feats, "fi_k_pct"),
        "away_sp_fi_bb_pct":     _sp("away", away_sp_feats, "fi_bb_pct"),
        "away_sp_fi_hr_per_9":   _sp("away", away_sp_feats, "fi_hr_per_9"),
        "away_sp_fi_whiff_pct":  _sp("away", away_sp_feats, "fi_whiff_pct"),
        "away_sp_stuff_plus":    _sp("away", away_sp_feats, "stuff_plus"),
        "away_sp_location_plus": _sp("away", away_sp_feats, "location_plus"),
        "away_sp_swstr_pct":     _sp("away", away_sp_feats, "swstr_pct"),
        "away_sp_f_strike_pct":  _sp("away", away_sp_feats, "f_strike_pct"),
        "away_sp_is_lhp":        _sp("away", away_sp_feats, "is_lhp"),
        # Lineup
        "home_top3_wrc_plus":    home_top3.get("wrc_plus"),
        "home_top3_obp":         home_top3.get("obp"),
        "home_top3_iso":         home_top3.get("iso"),
        "home_top3_k_pct":       home_top3.get("k_pct"),
        "home_top3_bb_pct":      home_top3.get("bb_pct"),
        "away_top3_wrc_plus":    away_top3.get("wrc_plus"),
        "away_top3_obp":         away_top3.get("obp"),
        "away_top3_iso":         away_top3.get("iso"),
        "away_top3_k_pct":       away_top3.get("k_pct"),
        "away_top3_bb_pct":      away_top3.get("bb_pct"),
        # Environmental
        "temperature_f":         temp,
        "wind_toward_cf":        wind_toward,
        "humidity_pct":          humidity,
        "hr_park_factor":        hr_factor,
        "hr_environment":        hr_env,
        "temp_carry_factor":     temp_carry,
        "alt_carry_factor":      alt_carry,
        "altitude_ft":           altitude,
        "is_dome":               is_dome,
        # Interactions
        "combined_fi_bb_pct":    (home_sp_bb + away_sp_bb) / 2,
        "home_lineup_vs_away_sp": home_wrc * (1 - away_sp_k),
        "away_lineup_vs_home_sp": away_wrc * (1 - home_sp_k),
        "home_hr_threat":         hr_env * max(home_iso, 0),
        "away_hr_threat":         hr_env * max(away_iso, 0),
    }

    # ── Build feature array in the correct order ──────────────────────────────
    # Fill missing features with NaN (will be imputed by XGBoost's tree logic)
    feat_arr = np.array(
        [row.get(c, np.nan) if row.get(c) is not None else np.nan
         for c in feat_cols],
        dtype=float,
    ).reshape(1, -1)

    # ── Predict ───────────────────────────────────────────────────────────────
    raw_prob  = float(model.predict_proba(feat_arr)[0, 1])
    cal_prob  = float(calibrator.predict([raw_prob])[0])

    return {
        "home_team":         home,
        "away_team":         away,
        "home_sp":           game.get("home_sp_name", ""),
        "away_sp":           game.get("away_sp_name", ""),
        "p_yrfi_raw":        round(raw_prob, 4),
        "p_yrfi":            round(cal_prob, 4),
        "p_nrfi":            round(1 - cal_prob, 4),
        "temperature_f":     round(temp, 1),
        "wind_toward_cf":    round(wind_toward, 1),
        "hr_environment":    round(hr_env, 3),
    }


# =============================================================================
# MAIN DAILY RUN
# =============================================================================

def run_nrfi_export(target_date: str = None, verbose: bool = True) -> pd.DataFrame:
    """
    Full daily scoring pipeline.  `target_date` format: 'YYYY-MM-DD'.
    If None, uses today.
    """
    today_str  = target_date or date.today().strftime("%Y-%m-%d")
    today_dt   = datetime.strptime(today_str, "%Y-%m-%d").replace(hour=19)

    print("=" * 70)
    print(f"NRFI / YRFI MODEL — STEP 4: DAILY EXPORT  ({today_str})")
    print("=" * 70)

    # ── Load model + data ─────────────────────────────────────────────────────
    print("\n[ Load ] Model artifacts...")
    model, calibrator, feat_cols = load_model()

    print("[ Load ] Park metadata...")
    stadium_meta, park_hr_factors = load_park_meta()

    print("[ Load ] SP stats (FG)...")
    fg_df = load_sp_lookup()

    print("[ Load ] Batting splits...")
    df_lhp, df_rhp = load_batting_splits_for_scoring()

    # ── Probable starters ─────────────────────────────────────────────────────
    print(f"\n[ Starters ] Fetching probable starters for {today_str}...")
    try:
        starters = get_probable_starters(today_str)
    except Exception as e:
        print(f"  WARNING: probable starters failed — {e}")
        starters = []

    if not starters:
        print("  No probable starters available — cannot score NRFI today")
        return pd.DataFrame()

    print(f"  {len(starters)} probable SP matchups found")

    # ── Lineups ───────────────────────────────────────────────────────────────
    print(f"\n[ Lineups ] Fetching confirmed lineups...")
    try:
        lineups = get_lineups(today_str)
    except Exception as e:
        print(f"  WARNING: lineup fetch failed — {e}")
        lineups = {}

    # ── NRFI/YRFI Odds ────────────────────────────────────────────────────────
    print(f"\n[ Odds ] Fetching NRFI/YRFI market odds...")
    try:
        odds_df = get_nrfi_odds(game_date=today_str, token=ODDS_API_KEY)
    except Exception as e:
        print(f"  WARNING: odds fetch failed — {e}")
        odds_df = pd.DataFrame()

    if not odds_df.empty:
        print(f"  {len(odds_df):,} NRFI/YRFI markets found")
    else:
        print("  No NRFI/YRFI odds available — will export model probs only")

    # ── Score each game ───────────────────────────────────────────────────────
    print(f"\n[ Score ] Scoring {len(starters)} games...")
    results = []

    for matchup in starters:
        home_team     = matchup.get("home_team", "")
        away_team     = matchup.get("away_team", "")
        home_sp_name  = matchup.get("home_pitcher", "")
        away_sp_name  = matchup.get("away_pitcher", "")
        home_sp_hand  = matchup.get("home_pitcher_hand", "R")
        away_sp_hand  = matchup.get("away_pitcher_hand", "R")

        # Top-3 lineup slots (from confirmed lineups if available)
        def _top3(team_key):
            team_lu = lineups.get(team_key, {})
            slots   = team_lu.get("batting_order", [])
            names   = [s.get("name", "") for s in slots[:3]]
            while len(names) < 3:
                names.append("")
            return names

        home_top3 = _top3(home_team)
        away_top3 = _top3(away_team)

        game_input = {
            "home_team":       home_team,
            "away_team":       away_team,
            "home_sp_name":    home_sp_name,
            "home_sp_hand":    home_sp_hand,
            "away_sp_name":    away_sp_name,
            "away_sp_hand":    away_sp_hand,
            "home_top3_names": home_top3,
            "away_top3_names": away_top3,
        }

        try:
            scored = score_game(
                game_input, model, calibrator, feat_cols,
                fg_df, df_lhp, df_rhp,
                stadium_meta, park_hr_factors, today_dt,
            )
        except Exception as e:
            print(f"  WARNING: scoring failed for {away_team}@{home_team} — {e}")
            continue

        results.append(scored)

    if not results:
        print("  No games scored successfully.")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # ── Join odds ─────────────────────────────────────────────────────────────
    if not odds_df.empty:
        odds_cols = ["home_team", "away_team",
                     "nrfi_over_juice", "nrfi_under_juice", "nrfi_implied_prob"]
        odds_sub  = odds_df[[c for c in odds_cols if c in odds_df.columns]].copy()
        df = df.merge(odds_sub, on=["home_team", "away_team"], how="left")
    else:
        df["nrfi_over_juice"]    = np.nan
        df["nrfi_under_juice"]   = np.nan
        df["nrfi_implied_prob"]  = np.nan

    # ── Edge calculation ──────────────────────────────────────────────────────
    df["edge_yrfi"] = np.nan
    df["edge_nrfi"] = np.nan
    df["ev_pct"]    = np.nan
    df["kelly_bet"] = np.nan
    df["bet_side"]  = ""
    df["bet_odds"]  = np.nan

    for idx, row in df.iterrows():
        p_yrfi   = row["p_yrfi"]
        p_nrfi   = 1 - p_yrfi
        book_imp = row.get("nrfi_implied_prob", np.nan)
        yrfi_j   = row.get("nrfi_over_juice",  np.nan)
        nrfi_j   = row.get("nrfi_under_juice", np.nan)

        if pd.isna(book_imp) or pd.isna(yrfi_j) or pd.isna(nrfi_j):
            continue

        # De-vigged true book probs
        true_p_yrfi, true_p_nrfi = remove_vig(float(yrfi_j), float(nrfi_j))

        edge_yrfi = p_yrfi - true_p_yrfi
        edge_nrfi = p_nrfi - true_p_nrfi
        df.at[idx, "edge_yrfi"] = round(edge_yrfi, 4)
        df.at[idx, "edge_nrfi"] = round(edge_nrfi, 4)

        # Best side
        if abs(edge_yrfi) >= abs(edge_nrfi) and abs(edge_yrfi) >= MIN_EDGE:
            bet_side  = "YRFI"
            bet_juice = float(yrfi_j)
            bet_prob  = p_yrfi
            dec_odds  = american_to_decimal(bet_juice)
            df.at[idx, "bet_side"]  = bet_side
            df.at[idx, "bet_odds"]  = bet_juice
            df.at[idx, "ev_pct"]    = ev_percent(bet_prob, true_p_yrfi, dec_odds)
            df.at[idx, "kelly_bet"] = kelly_bet(edge_yrfi, bet_prob, dec_odds)
        elif abs(edge_nrfi) >= MIN_EDGE:
            bet_side  = "NRFI"
            bet_juice = float(nrfi_j)
            bet_prob  = p_nrfi
            dec_odds  = american_to_decimal(bet_juice)
            df.at[idx, "bet_side"]  = bet_side
            df.at[idx, "bet_odds"]  = bet_juice
            df.at[idx, "ev_pct"]    = ev_percent(bet_prob, true_p_nrfi, dec_odds)
            df.at[idx, "kelly_bet"] = kelly_bet(edge_nrfi, bet_prob, dec_odds)

    # ── Unified edge + is_value_bet columns (required by notifier.py) ────────
    df["edge"] = df.apply(
        lambda r: r["edge_yrfi"] if r["bet_side"] == "YRFI"
                  else r["edge_nrfi"] if r["bet_side"] == "NRFI"
                  else np.nan,
        axis=1,
    )
    df["is_value_bet"] = (df["bet_side"] != "").astype(int)

    # ── Sort and display ──────────────────────────────────────────────────────
    df["abs_edge"] = df[["edge_yrfi", "edge_nrfi"]].abs().max(axis=1)
    df = df.sort_values("abs_edge", ascending=False).drop(columns=["abs_edge"])

    picks = df[df["bet_side"] != ""].copy()
    print(f"\n  Games scored:    {len(df)}")
    print(f"  Picks (≥{MIN_EDGE:.0%} edge): {len(picks)}")

    if verbose and not picks.empty:
        print("\n" + "=" * 70)
        print(f"  NRFI/YRFI PICKS — {today_str}")
        print("=" * 70)
        for _, row in picks.iterrows():
            print(f"  {row['away_team']:3s} @ {row['home_team']:3s}  |  "
                  f"{row['bet_side']:4s}  |  "
                  f"Model P(YRFI)={row['p_yrfi']:.3f}  |  "
                  f"Odds {int(row['bet_odds']):+d}  |  "
                  f"Edge {row['edge_yrfi'] if row['bet_side']=='YRFI' else row['edge_nrfi']:+.1%}  |  "
                  f"EV {row['ev_pct']:+.1f}%  |  "
                  f"Kelly {row['kelly_bet']:.1%}  |  "
                  f"Temp {row['temperature_f']:.0f}°F  Wind {row['wind_toward_cf']:+.0f}mph CF")
        print("=" * 70)

    # ── Export ────────────────────────────────────────────────────────────────
    col_order = [
        "away_team", "home_team", "away_sp", "home_sp",
        "p_yrfi", "p_nrfi", "p_yrfi_raw",
        "nrfi_over_juice", "nrfi_under_juice", "nrfi_implied_prob",
        "edge_yrfi", "edge_nrfi",
        "bet_side", "bet_odds", "ev_pct", "kelly_bet",
        "temperature_f", "wind_toward_cf", "hr_environment",
    ]
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order + [c for c in df.columns if c not in col_order]]

    out_path = os.path.join(EXPORT_DIR, f"nrfi_edges_{today_str}.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  ✓ Exported {len(df)} games → {out_path}")

    print("\n" + "=" * 70)
    print("STEP 4 COMPLETE")
    print("=" * 70)
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NRFI/YRFI daily export")
    parser.add_argument("--date", type=str, default=None,
                        help="Target date YYYY-MM-DD (default: today)")
    args = parser.parse_args()
    run_nrfi_export(target_date=args.date)
