"""
=============================================================================
NRFI / YRFI MODEL — FILE 1 OF 4: DATA INPUT
=============================================================================
Market:
  NRFI = No Run First Inning — zero runs scored by EITHER team in the 1st
  YRFI = Yes Run First Inning — at least one run scored by either team

Why this is NOT a full-game stats problem:
  - Starting pitchers are at peak effectiveness in the 1st inning (fresh arm,
    hitters are seeing them for the first time through the order)
  - The full-game ERA of a 6.00 ERA pitcher does not predict his 1st inning
    performance; his 1st-inning ERA might be 2.00 because he front-loads effort
  - The 1-2-3 hitters are the ONLY hitters who are guaranteed to bat in the
    first inning — lineup quality must be isolated to these three slots
  - Weather / park factors disproportionately affect early-inning home runs
    (cold temperature suppresses carry, wind into outfield boosts it)

Data sources (NRFI-specific):
  1. Statcast first-inning pitch data  — YRFI labels + 1st-inn SP stats
     Pulled monthly and filtered to inning=1 to keep size manageable.
     Provides: K, BB, HR, runs-scored, p_throws per SP per start.

  2. FanGraphs pitching stats + Stuff+ / Location+
     Stuff+ / Location+ are pre-game pitch-quality models that predict
     early-inning effectiveness better than ERA.

  3. FanGraphs batting platoon splits (vs LHP / vs RHP)
     Isolated to the top-3 spots in the order — the only batters
     guaranteed to appear in the first inning.

  4. Open-Meteo historical weather  (reuse approach from totals model)

  5. Chadwick Bureau register  (reuse existing raw_chadwick.csv)

  6. Retrosheet game logs  (reuse existing raw_retrosheet_{yr}.csv for
     batting order slots 1-2-3 and confirmed SP identity per game)

Output: data/raw/
  raw_nrfi_statcast_{yr}.csv    (first-inning only, ~40-50K rows/season)
  raw_fg_pitching_nrfi.csv      (FG stats including Stuff+, Location+)
  raw_nrfi_weather_hist.csv     (Open-Meteo historical, if not already pulled)
=============================================================================
"""

import os
import time
import calendar
import warnings
import requests
import numpy as np
import pandas as pd
import pybaseball as pyb
from datetime import date as _date

warnings.filterwarnings("ignore")
pyb.cache.enable()

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR   = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

STAT_YEARS = [2022, 2023, 2024, 2025]
GAME_YEARS = [2023, 2024, 2025]

# MLB season months (April=4 through October=10)
SEASON_MONTHS = list(range(4, 11))

# Stadium metadata: lat, lon, CF bearing from home plate (degrees N), altitude
# Used for Open-Meteo weather pull and wind decomposition.
# CF bearing = compass direction from home plate toward center field.
STADIUM_META = {
    "ARI": {"lat": 33.446,  "lon": -112.067, "cf_bearing": 345, "alt_ft": 1082},
    "ATL": {"lat": 33.891,  "lon":  -84.468, "cf_bearing": 25,  "alt_ft": 1050},
    "BAL": {"lat": 39.284,  "lon":  -76.622, "cf_bearing": 40,  "alt_ft": 50},
    "BOS": {"lat": 42.347,  "lon":  -71.097, "cf_bearing": 90,  "alt_ft": 20},
    "CHC": {"lat": 41.948,  "lon":  -87.656, "cf_bearing": 50,  "alt_ft": 595},
    "CWS": {"lat": 41.830,  "lon":  -87.634, "cf_bearing": 5,   "alt_ft": 595},
    "CIN": {"lat": 39.097,  "lon":  -84.507, "cf_bearing": 20,  "alt_ft": 495},
    "CLE": {"lat": 41.496,  "lon":  -81.685, "cf_bearing": 5,   "alt_ft": 640},
    "COL": {"lat": 39.756,  "lon": -104.994, "cf_bearing": 340, "alt_ft": 5200},
    "DET": {"lat": 42.339,  "lon":  -83.049, "cf_bearing": 340, "alt_ft": 600},
    "HOU": {"lat": 29.757,  "lon":  -95.356, "cf_bearing": 10,  "alt_ft": 50},
    "KCR": {"lat": 39.051,  "lon":  -94.480, "cf_bearing": 5,   "alt_ft": 750},
    "LAA": {"lat": 33.800,  "lon": -117.883, "cf_bearing": 325, "alt_ft": 160},
    "LAD": {"lat": 34.074,  "lon": -118.240, "cf_bearing": 25,  "alt_ft": 512},
    "MIA": {"lat": 25.778,  "lon":  -80.220, "cf_bearing": 350, "alt_ft": 10},
    "MIL": {"lat": 43.029,  "lon":  -87.971, "cf_bearing": 5,   "alt_ft": 635},
    "MIN": {"lat": 44.982,  "lon":  -93.278, "cf_bearing": 10,  "alt_ft": 830},
    "NYM": {"lat": 40.757,  "lon":  -73.846, "cf_bearing": 5,   "alt_ft": 20},
    "NYY": {"lat": 40.830,  "lon":  -73.928, "cf_bearing": 20,  "alt_ft": 55},
    "OAK": {"lat": 37.751,  "lon": -122.200, "cf_bearing": 355, "alt_ft": 25},
    "PHI": {"lat": 39.906,  "lon":  -75.166, "cf_bearing": 10,  "alt_ft": 20},
    "PIT": {"lat": 40.447,  "lon":  -80.006, "cf_bearing": 10,  "alt_ft": 730},
    "SDP": {"lat": 32.707,  "lon": -117.157, "cf_bearing": 340, "alt_ft": 20},
    "SEA": {"lat": 47.591,  "lon": -122.332, "cf_bearing": 350, "alt_ft": 20},
    "SFG": {"lat": 37.779,  "lon": -122.389, "cf_bearing": 60,  "alt_ft": 10},
    "STL": {"lat": 38.623,  "lon":  -90.193, "cf_bearing": 15,  "alt_ft": 465},
    "TBR": {"lat": 27.768,  "lon":  -82.653, "cf_bearing": 350, "alt_ft": 10},
    "TEX": {"lat": 32.751,  "lon":  -97.083, "cf_bearing": 5,   "alt_ft": 551},
    "TOR": {"lat": 43.642,  "lon":  -79.389, "cf_bearing": 10,  "alt_ft": 300},
    "WSN": {"lat": 38.873,  "lon":  -77.008, "cf_bearing": 5,   "alt_ft": 20},
}

# HR park factor (runs index, 100 = average) and roof type
# roof: "open" | "retractable" | "dome"
PARK_HR_FACTORS = {
    "ARI": {"hr_factor": 107, "roof": "retractable"},
    "ATL": {"hr_factor": 105, "roof": "open"},
    "BAL": {"hr_factor": 112, "roof": "open"},
    "BOS": {"hr_factor": 92,  "roof": "open"},
    "CHC": {"hr_factor": 110, "roof": "open"},
    "CWS": {"hr_factor": 103, "roof": "open"},
    "CIN": {"hr_factor": 108, "roof": "open"},
    "CLE": {"hr_factor": 98,  "roof": "open"},
    "COL": {"hr_factor": 118, "roof": "open"},
    "DET": {"hr_factor": 97,  "roof": "open"},
    "HOU": {"hr_factor": 98,  "roof": "retractable"},
    "KCR": {"hr_factor": 101, "roof": "open"},
    "LAA": {"hr_factor": 97,  "roof": "open"},
    "LAD": {"hr_factor": 109, "roof": "open"},
    "MIA": {"hr_factor": 96,  "roof": "retractable"},
    "MIL": {"hr_factor": 103, "roof": "retractable"},
    "MIN": {"hr_factor": 104, "roof": "retractable"},
    "NYM": {"hr_factor": 103, "roof": "open"},
    "NYY": {"hr_factor": 116, "roof": "open"},
    "OAK": {"hr_factor": 89,  "roof": "open"},
    "PHI": {"hr_factor": 104, "roof": "open"},
    "PIT": {"hr_factor": 101, "roof": "open"},
    "SDP": {"hr_factor": 96,  "roof": "open"},
    "SEA": {"hr_factor": 97,  "roof": "retractable"},
    "SFG": {"hr_factor": 93,  "roof": "open"},
    "STL": {"hr_factor": 100, "roof": "open"},
    "TBR": {"hr_factor": 95,  "roof": "dome"},
    "TEX": {"hr_factor": 108, "roof": "retractable"},
    "TOR": {"hr_factor": 106, "roof": "retractable"},
    "WSN": {"hr_factor": 105, "roof": "open"},
}


# =============================================================================
# 1. STATCAST FIRST-INNING DATA  (YRFI labels + first-inning SP stats)
# =============================================================================
def pull_first_inning_statcast(game_years: list) -> pd.DataFrame:
    """
    Pull Statcast pitch-by-pitch data for each season, filtered immediately
    to inning=1.  This is the primary training-data source for:
      a) YRFI outcome labels (did a run score in the first inning?)
      b) Per-pitcher first-inning performance stats (K%, BB%, HR, runs allowed)

    Monthly pulls reduce HTTP timeout risk and memory pressure.
    After filtering to inning=1, each season's data is ~40-50K rows.

    Key Statcast columns used:
      pitcher (MLBAM ID)    — the pitching player in this half-inning
      p_throws ('R'/'L')    — pitcher handedness
      events                — outcome of each plate appearance (null for non-terminal pitches)
      inning_topbot         — 'Top' (away bats, home SP pitches) or 'Bot' (vice versa)
      away_score, home_score — cumulative scores after this pitch
      home_team, away_team  — abbreviated team names
      game_pk               — unique game identifier
    """
    all_frames = []

    for yr in game_years:
        out_path = os.path.join(RAW_DIR, f"raw_nrfi_statcast_{yr}.csv")
        if os.path.exists(out_path):
            df = pd.read_csv(out_path, low_memory=False)
            print(f"    Statcast 1st-inn {yr}: loading {len(df):,} rows from cache")
            all_frames.append(df)
            continue

        print(f"  Pulling Statcast first-inning data for {yr} (monthly)...")
        yr_frames = []

        for month in SEASON_MONTHS:
            last_day  = calendar.monthrange(yr, month)[1]
            start_dt  = f"{yr}-{month:02d}-01"
            end_dt    = f"{yr}-{month:02d}-{last_day:02d}"
            try:
                df = pyb.statcast(start_dt=start_dt, end_dt=end_dt)
                # Filter to 1st inning immediately — reduces stored data by ~90%
                df = df[df["inning"] == 1].copy()
                df["season"] = yr
                yr_frames.append(df)
                print(f"    {yr}-{month:02d}: {len(df):,} 1st-inning pitches")
                time.sleep(2)
            except Exception as e:
                print(f"    WARNING: Statcast {yr}-{month:02d} failed — {e}")

        if yr_frames:
            yr_df = pd.concat(yr_frames, ignore_index=True)
            yr_df.to_csv(out_path, index=False)
            print(f"    ✓ {len(yr_df):,} 1st-inning pitches for {yr} → {out_path}")
            all_frames.append(yr_df)

    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True)


# =============================================================================
# 2. FANGRAPHS PITCHING STATS (including Stuff+ and Location+)
# =============================================================================
def pull_fg_pitching_nrfi(stat_years: list) -> pd.DataFrame:
    """
    Pull FanGraphs pitcher stats optimised for NRFI feature engineering.

    Priority columns:
      Stuff+     — pre-game pitch quality model (velocity, movement, extension)
                   Higher Stuff+ → better pitch characteristics in early innings
      Location+  — pitch command/location model
                   High Location+ → fewer walks and weak first-pitch counts
      Pitching+  — composite Stuff + Location + Usage
      K%, BB%, K-BB%, SwStr%, F-Strike%, CSW%

    Note: Stuff+ and Location+ are available from 2020 onward on FanGraphs.
    They may appear in the API response as 'Stuff+', 'Location+', 'Pitching+'
    but column availability varies by pybaseball version.
    """
    frames = []
    for yr in stat_years:
        print(f"  Pulling FanGraphs pitching stats {yr} (for Stuff+/Location+)...")
        try:
            df = pyb.pitching_stats(yr, yr, qual=0, ind=1)
            df["Season"] = yr
            frames.append(df)
            time.sleep(1.5)
        except Exception as e:
            print(f"    WARNING: FG pitching {yr} failed — {e}")

    if not frames:
        return pd.DataFrame()

    out_df = pd.concat(frames, ignore_index=True)

    # Filter to SPs (GS >= 3 for partial seasons)
    if "GS" in out_df.columns:
        out_df["GS"] = pd.to_numeric(out_df["GS"], errors="coerce").fillna(0)
        out_df = out_df[out_df["GS"] >= 3].copy()

    # Log which Stuff+ columns are present
    stuff_cols = [c for c in out_df.columns
                  if any(k in c for k in ["Stuff", "Location", "Pitching+"])]
    if stuff_cols:
        print(f"    Stuff+/Location+ columns found: {stuff_cols}")
    else:
        print("    NOTE: Stuff+/Location+ not in FG response — "
              "will use SwStr%/CSW% as proxies")

    out = os.path.join(RAW_DIR, "raw_fg_pitching_nrfi.csv")
    out_df.to_csv(out, index=False)
    print(f"    {len(out_df):,} pitcher-seasons → {out}")
    return out_df


# =============================================================================
# 3. FANGRAPHS BATTING SPLITS (vs LHP / RHP) — same API used by other models
# =============================================================================
def pull_batting_splits_nrfi(stat_years: list) -> tuple:
    """
    Compute first-inning batting platoon splits directly from Statcast data.

    Rather than relying on the FanGraphs splits API (which returns 500 errors),
    we aggregate wOBA, K%, BB%, ISO, and OBP from the already-pulled inning=1
    Statcast files.  Per-season splits for batters with < MIN_SPLIT_PA are
    supplemented with their pooled career average across all available seasons.

    Output columns (matching what 02_build_nrfi.py expects):
      IDfg, key_mlbam, season, wRC+, OBP, ISO, K%, BB%

    Note: wRC+ column is populated with first-inning wOBA (0.200–0.450 scale).
    Both training and scoring use the same scale so model weights learn correctly.
    """
    MIN_SPLIT_PA = 10

    lhp_path = os.path.join(RAW_DIR, "raw_batting_splits_lhp.csv")
    rhp_path = os.path.join(RAW_DIR, "raw_batting_splits_rhp.csv")

    # ── Load existing inning=1 Statcast files ────────────────────────────────
    sc_frames = []
    for yr in stat_years:
        p = os.path.join(RAW_DIR, f"raw_nrfi_statcast_{yr}.csv")
        if os.path.exists(p):
            sc_frames.append(pd.read_csv(p, low_memory=False))
    if not sc_frames:
        print("  WARNING: no Statcast files found — platoon splits will be null")
        return pd.DataFrame(), pd.DataFrame()

    sc = pd.concat(sc_frames, ignore_index=True)

    # Keep only rows that are plate-appearance endings (woba_denom == 1)
    pa = sc[sc["woba_denom"] == 1].copy()
    pa["is_k"]  = (pa["events"] == "strikeout").astype(int)
    pa["is_bb"] = pa["events"].isin(["walk", "intent_walk"]).astype(int)
    pa["is_hbp"]= (pa["events"] == "hit_by_pitch").astype(int)
    pa["is_hit"]= pa["events"].isin(
        ["single", "double", "triple", "home_run"]).astype(int)

    # Compute per (batter, p_throws, season) aggregates
    grp = pa.groupby(["batter", "p_throws", "season"])
    agg = grp.agg(
        pa_count  = ("woba_denom", "sum"),
        woba_sum  = ("woba_value", "sum"),
        iso_sum   = ("iso_value",  "sum"),
        k_count   = ("is_k",       "sum"),
        bb_count  = ("is_bb",      "sum"),
        hbp_count = ("is_hbp",     "sum"),
        hit_count = ("is_hit",     "sum"),
    ).reset_index()

    agg["wOBA"]  = agg["woba_sum"]  / agg["pa_count"]
    agg["ISO"]   = agg["iso_sum"]   / agg["pa_count"]
    agg["K%"]    = agg["k_count"]   / agg["pa_count"]
    agg["BB%"]   = agg["bb_count"]  / agg["pa_count"]
    agg["OBP"]   = (agg["hit_count"] + agg["bb_count"] + agg["hbp_count"]) \
                   / agg["pa_count"]
    # Expose wOBA as "wRC+" (build script looks for "wRC+" first; same scale used
    # in both training and live scoring so model weights are internally consistent)
    agg["wRC+"]  = agg["wOBA"]

    # ── Supplement thin seasons with career-pooled average ───────────────────
    career = pa.groupby(["batter", "p_throws"]).agg(
        pa_count  = ("woba_denom", "sum"),
        woba_sum  = ("woba_value", "sum"),
        iso_sum   = ("iso_value",  "sum"),
        k_count   = ("is_k",       "sum"),
        bb_count  = ("is_bb",      "sum"),
        hbp_count = ("is_hbp",     "sum"),
        hit_count = ("is_hit",     "sum"),
    ).reset_index()
    career["wOBA"] = career["woba_sum"]  / career["pa_count"]
    career["ISO"]  = career["iso_sum"]   / career["pa_count"]
    career["K%"]   = career["k_count"]   / career["pa_count"]
    career["BB%"]  = career["bb_count"]  / career["pa_count"]
    career["OBP"]  = (career["hit_count"] + career["bb_count"] + career["hbp_count"]) \
                     / career["pa_count"]
    career["wRC+"] = career["wOBA"]

    # For each season-year combo, fill thin rows with career values
    rows_out = []
    for yr in stat_years:
        yr_agg = agg[agg["season"] == yr].copy()
        thin   = yr_agg[yr_agg["pa_count"] < MIN_SPLIT_PA]["batter"].unique()
        # Supplement thin batters with career row stamped with this season
        if len(thin):
            fill = career[career["batter"].isin(thin)].copy()
            fill["season"] = yr
            yr_agg = pd.concat(
                [yr_agg[yr_agg["pa_count"] >= MIN_SPLIT_PA], fill],
                ignore_index=True
            )
        rows_out.append(yr_agg)

    splits = pd.concat(rows_out, ignore_index=True)

    # ── Bridge MLBAM → FGid via Chadwick register ────────────────────────────
    chadwick_path = os.path.join(RAW_DIR, "raw_chadwick.csv")
    if not os.path.exists(chadwick_path):
        print("  Pulling Chadwick Bureau register...")
        try:
            chad = pyb.chadwick_register()
            keep = [c for c in ["key_mlbam", "key_fangraphs", "name_first", "name_last"]
                    if c in chad.columns]
            chad = chad[keep].dropna(subset=["key_mlbam", "key_fangraphs"])
            chad["key_mlbam"]     = chad["key_mlbam"].astype(int)
            chad["key_fangraphs"] = chad["key_fangraphs"].astype(int)
            chad.to_csv(chadwick_path, index=False)
            print(f"    Saved {len(chad):,} rows → {chadwick_path}")
        except Exception as e:
            print(f"    WARNING: Chadwick pull failed — {e}; using MLBAM as IDfg fallback")
            chad = pd.DataFrame(columns=["key_mlbam", "key_fangraphs", "name_first", "name_last"])
    else:
        chad = pd.read_csv(chadwick_path, low_memory=False)
        chad["key_mlbam"]     = pd.to_numeric(chad["key_mlbam"],     errors="coerce")
        chad["key_fangraphs"] = pd.to_numeric(chad["key_fangraphs"], errors="coerce")
        chad = chad.dropna(subset=["key_mlbam", "key_fangraphs"])

    mlbam_to_fg = (chad.drop_duplicates("key_mlbam")
                       .set_index("key_mlbam")["key_fangraphs"]
                       .to_dict())

    splits["key_mlbam"] = splits["batter"].astype(int)
    splits["IDfg"] = splits["key_mlbam"].map(mlbam_to_fg)
    # Fallback: use MLBAM ID as IDfg when FGid unknown (keeps batter in training)
    splits["IDfg"] = splits["IDfg"].fillna(splits["key_mlbam"])
    splits["IDfg"] = splits["IDfg"].astype(int)

    out_cols = ["IDfg", "key_mlbam", "season", "wRC+", "OBP", "ISO", "K%", "BB%"]

    df_lhp = splits[splits["p_throws"] == "L"][out_cols].reset_index(drop=True)
    df_rhp = splits[splits["p_throws"] == "R"][out_cols].reset_index(drop=True)

    df_lhp.to_csv(lhp_path, index=False)
    df_rhp.to_csv(rhp_path, index=False)
    print(f"    LHP splits: {len(df_lhp):,} batters | RHP splits: {len(df_rhp):,} batters")
    return df_lhp, df_rhp


# =============================================================================
# 4. HISTORICAL WEATHER (Open-Meteo archive)
# =============================================================================
def pull_historical_weather_nrfi(game_years: list) -> pd.DataFrame:
    """
    Pull game-time weather conditions for each stadium and season.
    Reuses the Open-Meteo archive API (same approach as totals model).
    If raw_weather_historical.csv already exists, loads it directly.
    """
    existing = os.path.join(RAW_DIR, "raw_weather_historical.csv")
    if os.path.exists(existing):
        print("    Weather: loading from existing totals model raw file")
        return pd.read_csv(existing)

    print("  Pulling historical weather from Open-Meteo...")
    ARCHIVE_URL = "https://archive.open-meteo.com/v1/archive"
    records = []

    for team, meta in STADIUM_META.items():
        for yr in game_years:
            params = {
                "latitude":   meta["lat"],
                "longitude":  meta["lon"],
                "start_date": f"{yr}-04-01",
                "end_date":   f"{yr}-10-31",
                "hourly":     "temperature_2m,relativehumidity_2m,"
                              "windspeed_10m,winddirection_10m",
                "temperature_unit": "fahrenheit",
                "windspeed_unit":   "mph",
                "timezone":         "auto",
            }
            try:
                r = requests.get(ARCHIVE_URL, params=params, timeout=30)
                r.raise_for_status()
                data  = r.json()
                hours = data.get("hourly", {})
                times = hours.get("time", [])

                for i, t in enumerate(times):
                    if not (18 <= int(t[11:13]) <= 20):  # 6pm-8pm local
                        continue
                    records.append({
                        "team":          team,
                        "season":        yr,
                        "date":          t[:10],
                        "temperature_f": hours.get("temperature_2m", [None]*len(times))[i],
                        "humidity_pct":  hours.get("relativehumidity_2m", [None]*len(times))[i],
                        "wind_speed_mph":hours.get("windspeed_10m", [None]*len(times))[i],
                        "wind_dir_deg":  hours.get("winddirection_10m", [None]*len(times))[i],
                    })
                time.sleep(0.5)
            except Exception as e:
                print(f"    WARNING: weather {team} {yr} — {e}")

    wx_df = pd.DataFrame(records)
    if wx_df.empty:
        print("  WARNING: No weather records collected — "
              "weather features will be null (check network connectivity)")
        empty = pd.DataFrame(columns=["team", "season", "date",
                                      "temperature_f", "humidity_pct",
                                      "wind_speed_mph", "wind_dir_deg"])
        empty.to_csv(existing, index=False)
        return empty
    # Average over the 6-8pm window per (team, date)
    wx_agg = (wx_df.groupby(["team", "season", "date"])
              [["temperature_f", "humidity_pct",
                "wind_speed_mph", "wind_dir_deg"]]
              .mean().reset_index())
    wx_agg.to_csv(existing, index=False)
    print(f"    ✓ {len(wx_agg):,} team-date weather records → {existing}")
    return wx_agg


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("NRFI / YRFI MODEL — STEP 1: DATA INPUT")
    print("=" * 70)

    print("\n[ 1/4 ] Statcast first-inning data (YRFI labels + 1st-inn SP stats)...")
    pull_first_inning_statcast(GAME_YEARS)

    print("\n[ 2/4 ] FanGraphs pitching stats (Stuff+, Location+, K%, BB%)...")
    pull_fg_pitching_nrfi(STAT_YEARS)

    print("\n[ 3/4 ] Batting platoon splits (vs LHP / vs RHP)...")
    pull_batting_splits_nrfi(STAT_YEARS)

    print("\n[ 4/4 ] Historical weather (Open-Meteo)...")
    pull_historical_weather_nrfi(GAME_YEARS)

    import json
    meta_path = os.path.join(RAW_DIR, "raw_nrfi_park_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "stadium_meta":    STADIUM_META,
            "park_hr_factors": PARK_HR_FACTORS,
        }, f, indent=2)
    print(f"\n  ✓ Park/stadium metadata → {meta_path}")

    print("\n" + "=" * 70)
    print("STEP 1 COMPLETE — Run 02_build_nrfi.py next.")
    print("=" * 70)
