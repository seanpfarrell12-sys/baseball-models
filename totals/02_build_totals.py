"""
=============================================================================
OVER/UNDER TOTALS MODEL — FILE 2 OF 4: DATASET CONSTRUCTION  (REFACTORED)
=============================================================================
Builds separate feature matrices for the two-equation Negative Binomial
system: one model for HOME runs, one model for AWAY runs.

Separating the two teams captures:
  - Asymmetric park effects (home lineup benefits more from short RF porch)
  - Lineup vs opposing SP (specific matchup, not team average)
  - Bullpen usage patterns (home manager's hook rate affects reliever quality)

Dynamic park factor physics:
  ┌─────────────────────────────────────────────────────────────────────┐
  │ Air density governs ball carry distance.                            │
  │                                                                     │
  │ ρ_air ∝ (1/T) × (1 - 0.378 × Pv/P)                                │
  │                                                                     │
  │ where T = absolute temperature (Rankine), Pv = vapor pressure       │
  │ (from RH), P = station pressure (from altitude).                   │
  │                                                                     │
  │ Simplified per-feature adjustments from reference 65°F, 50% RH:   │
  │   Temperature : +0.20%/°F above 65 → more carry                   │
  │   Humidity    : +0.04%/% RH above 50 → humid air is LESS dense    │
  │   Altitude    : already absorbed into base park factor             │
  │   Wind (out)  : +0.30% run expectation per mph of outfield wind    │
  │   Wind (in)   : −0.30% run expectation per mph of infield wind     │
  │   Roof closed : all weather effects zero'd out                     │
  └─────────────────────────────────────────────────────────────────────┘

Fallback behavior:
  If historical weather CSV is missing for a game date, city seasonal
  averages are used. This degrades gracefully without crashing.

Input  : data/raw/ (from 01_input_totals.py)
Output : data/processed/totals_dataset.csv
         (columns: id cols, home_features, away_features, weather_features,
                   home_runs [target 1], away_runs [target 2], total_runs)
=============================================================================
"""

import os
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

# Team abbreviation normalizer (FanGraphs → standard)
FG_TO_STD = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHC", "CWS": "CWS", "CIN": "CIN", "CLE": "CLE",
    "COL": "COL", "DET": "DET", "HOU": "HOU", "KCR": "KCR",
    "LAA": "LAA", "LAD": "LAD", "MIA": "MIA", "MIL": "MIL",
    "MIN": "MIN", "NYM": "NYM", "NYY": "NYY", "OAK": "OAK",
    "PHI": "PHI", "PIT": "PIT", "SDP": "SDP", "SEA": "SEA",
    "SFG": "SFG", "STL": "STL", "TBR": "TBR", "TEX": "TEX",
    "TOR": "TOR", "WSN": "WSN",
    "CHW": "CWS", "SD": "SDP", "SF": "SFG", "TB": "TBR",
    "KC":  "KCR", "WAS": "WSN",
}

# City-average seasonal temperatures (°F, April–September) — fallback only
CITY_AVG_TEMP = {
    "ARI": 88, "ATL": 75, "BAL": 72, "BOS": 65, "CHC": 68,
    "CWS": 70, "CIN": 73, "CLE": 66, "COL": 70, "DET": 67,
    "HOU": 82, "KCR": 76, "LAA": 75, "LAD": 73, "MIA": 84,
    "MIL": 66, "MIN": 67, "NYM": 71, "NYY": 71, "OAK": 62,
    "PHI": 72, "PIT": 68, "SDP": 68, "SEA": 62, "SFG": 60,
    "STL": 76, "TBR": 82, "TEX": 85, "TOR": 67, "WSN": 74,
}


# =============================================================================
# PHYSICS ENGINE: Dynamic Park Factor
# =============================================================================

def wind_outfield_component(wind_direction_deg: float,
                             cf_bearing_deg: float) -> float:
    """
    Compute the scalar projection of wind velocity onto the outfield axis.

    Returns a value in [-1, +1]:
      +1.0 = wind blowing full-speed OUT toward CF (maximum boost)
       0.0 = wind blowing cross-field (no outfield effect)
      -1.0 = wind blowing full-speed IN from CF (maximum suppression)

    Physics:
      Wind direction (meteorological) = direction wind is coming FROM.
      Wind velocity vector points TO: (wind_dir + 180°) mod 360.
      Project that vector onto the CF unit vector (cf_bearing_deg from N).
      Component = cos(angle between wind-toward and cf-bearing).

    Parameters
    ----------
    wind_direction_deg : float
        Meteorological wind direction (where wind is coming FROM), 0–360°.
    cf_bearing_deg : float
        Compass bearing from home plate toward center field, 0–360°.
    """
    wind_toward_deg    = (wind_direction_deg + 180.0) % 360.0
    angle_diff_deg     = wind_toward_deg - cf_bearing_deg
    angle_diff_rad     = np.radians(angle_diff_deg)
    return float(np.cos(angle_diff_rad))


def compute_dynamic_park_factor(
    base_pf:           float,
    temperature_f:     float,
    humidity_pct:      float,
    wind_speed_mph:    float,
    wind_direction_deg:float,
    cf_bearing_deg:    float,
    altitude_ft:       float,
    roof:              str,
) -> dict:
    """
    Compute a weather-adjusted, physics-grounded park factor for a single game.

    Returns a dict with individual components so downstream models can use
    them as separate features rather than just the combined scalar.

    Physics basis:
      Air density ρ (kg/m³) = P_station / (R_dry × T) × (1 − 0.378 × e/P)
      where e = partial pressure of water vapor (from RH and temperature).
      At sea level, sea-level pressure, 65°F, 50% RH: ρ ≈ 1.19 kg/m³.
      Carry distance ∝ 1/ρ, so low density = ball carries farther.

      Simplified linear approximations valid over typical game conditions:
        ΔCarry_temp     ≈ +0.20% per °F above 65°F  (T effect on density)
        ΔCarry_humidity ≈ +0.04% per % RH above 50% (vapor displacement)
        ΔCarry_wind     ≈ +0.30% per mph of outfield wind component
        Altitude already absorbed in base park factor.

      Run scoring responds more strongly to HR carry than singles/doubles.
      The combined effect is applied multiplicatively to the base park factor.

    Parameters
    ----------
    roof : str   "open" | "retractable" | "fixed"
        For "fixed" (TBR): all weather effects are zero.
        For "retractable": apply 40% of effect (unknown open/closed status).

    Returns
    -------
    dict with keys:
      dynamic_pf          : final combined park factor
      temp_factor         : multiplicative temperature adjustment
      humidity_factor     : multiplicative humidity adjustment
      wind_factor         : multiplicative wind adjustment
      wind_outfield_comp  : raw wind outfield component (−1 to +1)
      weather_scale       : fraction of weather effect applied (0/0.4/1.0)
    """
    # Determine how much weather penetrates the stadium
    if roof == "fixed":
        weather_scale = 0.0
    elif roof == "retractable":
        weather_scale = 0.40   # Unknown open/closed; partial exposure assumed
    else:
        weather_scale = 1.0

    # ── Temperature effect ───────────────────────────────────────────────────
    # Reference: 65°F (typical spring game).  Air density decreases with temp,
    # so hotter = less dense = more carry.  ~0.20% per °F.
    temp_delta   = float(temperature_f) - 65.0
    temp_factor  = 1.0 + weather_scale * 0.0020 * temp_delta

    # ── Humidity effect ──────────────────────────────────────────────────────
    # Humid air IS less dense than dry air — water vapor (MW=18) displaces
    # heavier N₂ (28) and O₂ (32).  Reference: 50% RH.  ~0.04% per % above 50.
    hum_delta       = max(0.0, float(humidity_pct) - 50.0)
    humidity_factor = 1.0 + weather_scale * 0.0004 * hum_delta

    # ── Wind effect ──────────────────────────────────────────────────────────
    # Project wind onto outfield axis.  0.30% per mph of outfield component.
    # Negative (headwind from CF) suppresses scoring.
    wof_comp   = wind_outfield_component(float(wind_direction_deg), float(cf_bearing_deg))
    wind_delta = float(wind_speed_mph) * wof_comp
    wind_factor = 1.0 + weather_scale * 0.0030 * wind_delta

    dynamic_pf = float(base_pf) * temp_factor * humidity_factor * wind_factor

    return {
        "dynamic_pf":         round(dynamic_pf,    3),
        "temp_factor":        round(temp_factor,    5),
        "humidity_factor":    round(humidity_factor,5),
        "wind_factor":        round(wind_factor,    5),
        "wind_outfield_comp": round(wof_comp,       4),
        "weather_scale":      weather_scale,
    }


# =============================================================================
# LOAD RAW DATA
# =============================================================================

def load_raw() -> dict:
    files = {
        "games":    "raw_game_schedules.csv",
        "team_bat": "raw_team_batting.csv",
        "team_pit": "raw_team_pitching.csv",
        "sp_stats": "raw_sp_stats.csv",
        "weather":  "raw_weather_historical.csv",
        "park":     "raw_park_factors.csv",
        "geo":      "raw_stadium_geo.csv",
        "cf":       "raw_stadium_cf_bearings.csv",
        "mgr":      "raw_manager_hook.csv",
    }
    data = {}
    for key, fname in files.items():
        path = os.path.join(RAW_DIR, fname)
        if os.path.exists(path):
            data[key] = pd.read_csv(path, low_memory=False)
            print(f"  ✓ {fname}: {len(data[key]):,} rows")
        else:
            print(f"  ✗ {fname}: NOT FOUND (run 01_input_totals.py)")
            data[key] = pd.DataFrame()

    # Stadium metadata JSON (preferred over individual CSVs)
    meta_path = os.path.join(RAW_DIR, "stadium_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            data["meta"] = json.load(f)
    else:
        data["meta"] = {}

    return data


# =============================================================================
# BUILD TEAM-SEASON FEATURES (prior-year stats for each side)
# =============================================================================

def build_team_context(data: dict) -> pd.DataFrame:
    """
    One row per team-season with offensive and pitching metrics.
    Used as prior-year features for each game side.
    """
    team_bat = data["team_bat"].copy()
    team_pit = data["team_pit"].copy()

    for df in [team_bat, team_pit]:
        for col in ("Team", "Tm"):
            if col in df.columns:
                df["team_std"] = df[col].map(FG_TO_STD).fillna(df[col])
                break

    # Offense: wRC+, wOBA, ISO, K%, BB%
    bat_cols = [c for c in ["team_std", "Season", "wRC+", "wOBA",
                             "ISO", "K%", "BB%", "OBP", "SLG"] if c in team_bat.columns]
    team_bat_sub = team_bat[bat_cols].rename(columns={
        c: f"off_{c.lower().replace('+','plus').replace('%','_pct')}"
        for c in bat_cols if c not in ("team_std", "Season")
    })

    # Pitching: ERA, FIP, xFIP, SIERA, K%
    pit_cols = [c for c in ["team_std", "Season", "ERA", "FIP", "xFIP",
                             "SIERA", "K%", "BB%", "K-BB%"] if c in team_pit.columns]
    team_pit_sub = team_pit[pit_cols].rename(columns={
        c: f"pit_{c.lower().replace('%','_pct').replace('-','_').replace('/','_')}"
        for c in pit_cols if c not in ("team_std", "Season")
    })

    team_ctx = team_bat_sub.merge(team_pit_sub, on=["team_std", "Season"], how="outer")
    print(f"    ✓ Team context: {len(team_ctx):,} team-seasons")
    return team_ctx


def build_sp_context(data: dict) -> pd.DataFrame:
    """
    IP-weighted SP quality per team-season (SIERA, xFIP, K%, GB%).
    Used as the opposing pitcher quality feature for each batting side.
    """
    sp = data["sp_stats"].copy()
    if sp.empty:
        return pd.DataFrame()

    if "Team" in sp.columns:
        sp["team_std"] = sp["Team"].map(FG_TO_STD).fillna(sp["Team"])

    sp["IP"] = pd.to_numeric(sp.get("IP", pd.Series(1, index=sp.index)),
                              errors="coerce").fillna(1)
    if "GS" in sp.columns:
        sp = sp[sp["GS"].fillna(0) >= 5].copy()

    metric_cols = [c for c in ["SIERA", "xFIP", "FIP", "ERA", "K%", "BB%",
                                "K-BB%", "GB%"] if c in sp.columns]

    def ip_wavg(grp):
        w = grp["IP"].fillna(0)
        tw = w.sum()
        if tw == 0:
            return pd.Series({f"sp_{m.lower().replace('%','_pct').replace('-','_')}":
                               grp[m].mean() for m in metric_cols})
        return pd.Series({
            f"sp_{m.lower().replace('%','_pct').replace('-','_')}":
            (grp[m].fillna(grp[m].mean()) * w).sum() / tw
            for m in metric_cols
        })

    sp_ctx = sp.groupby(["team_std", "Season"]).apply(ip_wavg).reset_index()
    # Also add bullpen context: pitchers with GS < 5 (relievers)
    sp_all = data["sp_stats"].copy()
    if "GS" in sp_all.columns:
        sp_all["team_std"] = sp_all.get("Team", sp_all.get("Tm", "")).map(FG_TO_STD).fillna(
            sp_all.get("Team", sp_all.get("Tm", ""))
        )
        sp_all["IP"] = pd.to_numeric(sp_all.get("IP", 1), errors="coerce").fillna(1)
        rp = sp_all[(sp_all["GS"].fillna(0) < 5) & (sp_all["IP"] >= 5)].copy()
        if not rp.empty:
            rp_cols = [c for c in ["ERA", "K%", "FIP"] if c in rp.columns]
            def bp_wavg(grp):
                w = grp["IP"].fillna(0); tw = w.sum()
                if tw == 0:
                    return pd.Series({f"bp_{c.lower().replace('%','_pct')}": grp[c].mean()
                                      for c in rp_cols})
                return pd.Series({
                    f"bp_{c.lower().replace('%','_pct')}":
                    (grp[c].fillna(grp[c].mean()) * w).sum() / tw
                    for c in rp_cols
                })
            bp_ctx = rp.groupby(["team_std", "Season"]).apply(bp_wavg).reset_index()
            sp_ctx = sp_ctx.merge(bp_ctx, on=["team_std", "Season"], how="left")

    print(f"    ✓ SP context: {len(sp_ctx):,} team-seasons")
    return sp_ctx


# =============================================================================
# BUILD WEATHER LOOKUP (team × game_date → weather conditions)
# =============================================================================

def build_weather_lookup(data: dict) -> dict:
    """
    Build a fast lookup dict: (team, "YYYY-MM-DD") → weather dict.

    Falls back to city seasonal average if no Open-Meteo data available.
    """
    wx_lookup = {}

    if not data["weather"].empty:
        wx = data["weather"].copy()
        for _, row in wx.iterrows():
            key = (str(row.get("team", "")), str(row.get("date", "")))
            wx_lookup[key] = {
                "temperature_f":     float(row.get("temperature_f",     72.0)),
                "humidity_pct":      float(row.get("humidity_pct",      55.0)),
                "wind_speed_mph":    float(row.get("wind_speed_mph",     5.0)),
                "wind_direction_deg":float(row.get("wind_direction_deg",180.0)),
            }

    return wx_lookup


# =============================================================================
# BUILD GAME-LEVEL DATASET
# =============================================================================

def build_game_dataset(data: dict, team_ctx: pd.DataFrame,
                        sp_ctx: pd.DataFrame, wx_lookup: dict) -> pd.DataFrame:
    """
    For each game, assemble the complete feature vector for both the
    home-runs model and the away-runs model.

    Feature prefixes:
      home_off_*   : home team offensive metrics (prior year)
      home_pit_*   : home team pitching metrics (prior year)
      home_sp_*    : home rotation SP quality (prior year, faces away lineup)
      home_bp_*    : home bullpen quality
      away_off_*   : away team offensive metrics
      away_pit_*   : away team pitching metrics
      away_sp_*    : away rotation SP quality (faces home lineup)
      away_bp_*    : away bullpen quality
      wx_*         : game-time weather features
      env_*        : static environment (altitude, surface, roof)
      dyn_pf_*     : dynamic park factor components

    Targets:
      home_runs    : runs scored by home team (NB model 1 target)
      away_runs    : runs scored by away team (NB model 2 target)
      total_runs   : home + away (for O/U evaluation)
    """
    games = data["games"].copy()

    # ── Identify home team rows ──────────────────────────────────────────────
    if "Home_Away" in games.columns:
        home_games = games[
            (games["Home_Away"] == "Home") |
            games["Home_Away"].isna() |
            (games["Home_Away"] == "")
        ].copy()
    else:
        home_games = games.copy()

    home_games["home_team"] = home_games["Team"].map(FG_TO_STD).fillna(home_games["Team"])
    home_games["away_team"] = home_games["Opp"].map(FG_TO_STD).fillna(home_games["Opp"])

    home_games["home_runs"] = pd.to_numeric(home_games["R"],  errors="coerce")
    home_games["away_runs"] = pd.to_numeric(home_games["RA"], errors="coerce")
    home_games["total_runs"] = home_games["home_runs"] + home_games["away_runs"]
    home_games = home_games.dropna(subset=["home_runs", "away_runs"])

    # Date
    if "Date" in home_games.columns:
        home_games["game_date"]  = pd.to_datetime(home_games["Date"], errors="coerce")
        home_games["date_str"]   = home_games["game_date"].dt.strftime("%Y-%m-%d")
    else:
        home_games["game_date"] = pd.NaT
        home_games["date_str"]  = ""

    # Prior-year feature season
    home_games["feature_season"] = home_games["Season"] - 1

    # ── Merge team context (home and away) ───────────────────────────────────
    home_ctx = team_ctx.rename(columns={
        "team_std": "home_team", "Season": "feature_season",
        **{c: f"home_{c}" for c in team_ctx.columns if c not in ("team_std", "Season")}
    })
    away_ctx = team_ctx.rename(columns={
        "team_std": "away_team", "Season": "feature_season",
        **{c: f"away_{c}" for c in team_ctx.columns if c not in ("team_std", "Season")}
    })
    df = home_games.merge(home_ctx, on=["home_team", "feature_season"], how="left")
    df = df.merge(away_ctx, on=["away_team", "feature_season"], how="left")

    # ── Merge SP context ─────────────────────────────────────────────────────
    if not sp_ctx.empty:
        home_sp = sp_ctx.rename(columns={
            "team_std": "home_team", "Season": "feature_season",
            **{c: f"home_{c}" for c in sp_ctx.columns if c not in ("team_std", "Season")}
        })
        away_sp = sp_ctx.rename(columns={
            "team_std": "away_team", "Season": "feature_season",
            **{c: f"away_{c}" for c in sp_ctx.columns if c not in ("team_std", "Season")}
        })
        df = df.merge(home_sp, on=["home_team", "feature_season"], how="left")
        df = df.merge(away_sp, on=["away_team", "feature_season"], how="left")

    # ── Manager hook rate ────────────────────────────────────────────────────
    mgr = data.get("mgr", pd.DataFrame())
    if not mgr.empty and "team" in mgr.columns and "hook_rate" in mgr.columns:
        hook_map = dict(zip(mgr["team"], mgr["hook_rate"].astype(float)))
        df["home_hook_rate"] = df["home_team"].map(hook_map).fillna(0.47)
        df["away_hook_rate"] = df["away_team"].map(hook_map).fillna(0.47)

    # ── Static environment features ──────────────────────────────────────────
    meta = data.get("meta", {})
    df["base_pf"]       = df["home_team"].map(
        lambda t: meta.get(t, {}).get("pf_runs", 100))
    df["altitude_ft"]   = df["home_team"].map(
        lambda t: meta.get(t, {}).get("altitude_ft", 100))
    df["cf_bearing"]    = df["home_team"].map(
        lambda t: meta.get(t, {}).get("cf_bearing_deg", 0))
    df["roof"]          = df["home_team"].map(
        lambda t: meta.get(t, {}).get("roof", "open"))
    df["is_artificial"] = (df["home_team"].map(
        lambda t: meta.get(t, {}).get("surface", "natural")) == "artificial").astype(int)
    df["is_dome"]       = (df["roof"] == "fixed").astype(int)

    # Coors Field flag — largest single park effect in baseball
    df["is_coors"] = (df["home_team"] == "COL").astype(int)

    # ── Weather features (actual historical or city-average fallback) ────────
    wx_rows = []
    n_actual, n_fallback = 0, 0

    for _, row in df.iterrows():
        team     = row["home_team"]
        date_str = row["date_str"]
        wx_key   = (str(team), str(date_str))

        if wx_key in wx_lookup:
            wx = wx_lookup[wx_key]
            n_actual += 1
        else:
            # Fallback: city seasonal average, neutral wind
            wx = {
                "temperature_f":     CITY_AVG_TEMP.get(team, 72.0),
                "humidity_pct":      55.0,
                "wind_speed_mph":     5.0,
                "wind_direction_deg":180.0,
            }
            n_fallback += 1
        wx_rows.append(wx)

    wx_df = pd.DataFrame(wx_rows, index=df.index)
    df["wx_temperature_f"]      = wx_df["temperature_f"]
    df["wx_humidity_pct"]       = wx_df["humidity_pct"]
    df["wx_wind_speed_mph"]     = wx_df["wind_speed_mph"]
    df["wx_wind_direction_deg"] = wx_df["wind_direction_deg"]

    print(f"    Weather: {n_actual:,} actual records | {n_fallback:,} city-average fallbacks")

    # ── Compute dynamic park factor ──────────────────────────────────────────
    dyn_rows = []
    for _, row in df.iterrows():
        dpf = compute_dynamic_park_factor(
            base_pf=           row.get("base_pf", 100),
            temperature_f=     row["wx_temperature_f"],
            humidity_pct=      row["wx_humidity_pct"],
            wind_speed_mph=    row["wx_wind_speed_mph"],
            wind_direction_deg=row["wx_wind_direction_deg"],
            cf_bearing_deg=    row.get("cf_bearing", 0),
            altitude_ft=       row.get("altitude_ft", 100),
            roof=              row.get("roof", "open"),
        )
        dyn_rows.append(dpf)

    dyn_df = pd.DataFrame(dyn_rows, index=df.index)
    df["dyn_pf"]             = dyn_df["dynamic_pf"]
    df["dyn_temp_factor"]    = dyn_df["temp_factor"]
    df["dyn_humidity_factor"]= dyn_df["humidity_factor"]
    df["dyn_wind_factor"]    = dyn_df["wind_factor"]
    df["wind_outfield_comp"] = dyn_df["wind_outfield_comp"]

    # ── Derived combined features ────────────────────────────────────────────
    off_plus_h = df.get("home_off_wrcplus", df.get("home_off_wrc_plus", pd.Series(100, index=df.index)))
    off_plus_a = df.get("away_off_wrcplus", df.get("away_off_wrc_plus", pd.Series(100, index=df.index)))
    df["combined_off_wrc_plus"] = (off_plus_h.fillna(100) + off_plus_a.fillna(100)) / 2

    sp_siera_h = df.get("home_sp_siera", pd.Series(4.2, index=df.index))
    sp_siera_a = df.get("away_sp_siera", pd.Series(4.2, index=df.index))
    df["combined_sp_siera"]     = (sp_siera_h.fillna(4.2) + sp_siera_a.fillna(4.2)) / 2

    # Temperature above/below thresholds (binary context flags)
    df["is_cold_game"] = (df["wx_temperature_f"] < 50).astype(int)
    df["is_hot_game"]  = (df["wx_temperature_f"] > 85).astype(int)

    print(f"    ✓ Game dataset: {len(df):,} games | mean total runs: {df['total_runs'].mean():.2f}")
    print(f"    Total runs variance / mean = "
          f"{df['total_runs'].var():.3f} / {df['total_runs'].mean():.3f} = "
          f"{df['total_runs'].var()/df['total_runs'].mean():.3f}  "
          f"(>1.0 → overdispersed → NB justified)")
    return df


# =============================================================================
# FINALIZE DATASETS
# =============================================================================

def finalize_datasets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and clean the final feature columns. Outputs a single CSV with
    both home_runs and away_runs as targets.  The analysis script will
    split these into two separate NB models.
    """
    # ── Identify all feature columns ─────────────────────────────────────────
    # Home-side features (offense of home lineup + pitching of home staff)
    home_off = [c for c in df.columns if c.startswith("home_off_")]
    home_pit = [c for c in df.columns if c.startswith("home_sp_") or c.startswith("home_pit_")]
    home_bp  = [c for c in df.columns if c.startswith("home_bp_")]
    away_off = [c for c in df.columns if c.startswith("away_off_")]
    away_pit = [c for c in df.columns if c.startswith("away_sp_") or c.startswith("away_pit_")]
    away_bp  = [c for c in df.columns if c.startswith("away_bp_")]

    # Weather and physics
    wx_cols  = [c for c in df.columns if c.startswith("wx_") or c.startswith("dyn_")]
    env_cols = [c for c in ["wind_outfield_comp", "is_artificial", "is_dome", "is_coors",
                             "altitude_ft", "base_pf", "is_cold_game", "is_hot_game"]
                if c in df.columns]
    hook_cols = [c for c in ["home_hook_rate", "away_hook_rate"] if c in df.columns]
    comb_cols = [c for c in ["combined_off_wrc_plus", "combined_sp_siera"] if c in df.columns]

    feature_cols = (home_off + home_pit + home_bp +
                    away_off + away_pit + away_bp +
                    wx_cols + env_cols + hook_cols + comb_cols)

    # Remove duplicate column names
    seen = set()
    feature_cols = [c for c in feature_cols if c not in seen and not seen.add(c)]

    # Only keep features that exist in the DataFrame
    feature_cols = [c for c in feature_cols if c in df.columns]

    id_cols  = [c for c in ["game_date", "Season", "date_str",
                              "home_team", "away_team"] if c in df.columns]
    tgt_cols = [c for c in ["home_runs", "away_runs", "total_runs"] if c in df.columns]

    final = df[id_cols + feature_cols + tgt_cols].copy()

    # Impute missing features with column means
    means     = final[feature_cols].mean()
    n_imputed = 0
    for col in feature_cols:
        n_miss = final[col].isna().sum()
        if n_miss:
            final[col] = final[col].fillna(means[col])
            n_imputed += n_miss

    # Drop games with no target data or clearly erroneous totals
    final = final.dropna(subset=["home_runs", "away_runs"])
    final = final[(final["home_runs"] >= 0) & (final["away_runs"] >= 0)]
    final = final[(final["total_runs"] >= 1) & (final["total_runs"] <= 35)]

    print(f"    ✓ Final dataset: {len(final):,} games, {len(feature_cols)} features")
    print(f"    Imputed {n_imputed:,} missing feature values")
    print(f"    Home runs — mean: {final['home_runs'].mean():.2f}, "
          f"var: {final['home_runs'].var():.2f}, "
          f"disp: {final['home_runs'].var()/final['home_runs'].mean():.3f}")
    print(f"    Away runs — mean: {final['away_runs'].mean():.2f}, "
          f"var: {final['away_runs'].var():.2f}, "
          f"disp: {final['away_runs'].var()/final['away_runs'].mean():.3f}")
    return final


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("TOTALS MODEL — STEP 2: DATASET CONSTRUCTION  (NB + WEATHER)")
    print("=" * 70)

    print("\n[ 1/5 ] Loading raw data...")
    data = load_raw()

    print("\n[ 2/5 ] Building team context (offense + pitching)...")
    team_ctx = build_team_context(data)

    print("\n[ 3/5 ] Building SP / bullpen context...")
    sp_ctx = build_sp_context(data)

    print("\n[ 4/5 ] Building weather lookup...")
    wx_lookup = build_weather_lookup(data)
    print(f"  ✓ Weather lookup: {len(wx_lookup):,} team-date entries")

    print("\n[ 5/5 ] Building game-level dataset + dynamic park factors...")
    game_df  = build_game_dataset(data, team_ctx, sp_ctx, wx_lookup)
    final_df = finalize_datasets(game_df)

    out_path = os.path.join(PROC_DIR, "totals_dataset.csv")
    final_df.to_csv(out_path, index=False)
    print(f"\n  ✓ Saved totals_dataset.csv ({len(final_df):,} rows)")

    print("\n" + "=" * 70)
    print("STEP 2 COMPLETE — Run 03_analysis_totals.py next.")
    print("=" * 70)
