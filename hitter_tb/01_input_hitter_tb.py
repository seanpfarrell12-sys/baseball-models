"""
=============================================================================
HITTER TOTAL BASES MODEL — FILE 1 OF 4: DATA INPUT  (REFACTORED)
=============================================================================
Purpose : Pull all raw data needed for the hitter total bases model.

Key design changes from prior version:
  - ALL joins use MLBAM (key_mlbam) numeric IDs, not name strings.
    This eliminates the "0 hitters matched" merge failure in the old code.
  - Chadwick Bureau register bridges Savant MLBAM ↔ FanGraphs IDfg.
  - Added retrosheet game logs for confirmed SP identity each game.
  - Added batting-order / lineup-position data from game logs.
  - Pitcher arsenal data structured per-PA simulation (by MLBAM, season).

Data sources:
  1. Chadwick Bureau register         — ID crosswalk (MLBAM ↔ FGid)
  2. Retrosheet game logs             — confirmed SP + batting order slot
  3. Statcast batter expected stats   — xBA, xSLG, xwOBA per batter/season
  4. Statcast batter EV / barrel      — exit_velocity_avg, barrel_batted_rate,
                                        launch_angle_avg, hard_hit_percent
  5. FanGraphs batting splits         — wRC+, wOBA, ISO, K%, BB% vs LHP/RHP
  6. Statcast pitcher arsenal stats   — pitch-type velo, break, whiff% per SP
  7. Statcast pitcher expected stats  — xwOBA against, barrel%, hard_hit% per SP

Input  : none (pulls from pybaseball / APIs)
Output : data/raw/
          raw_chadwick.csv
          raw_retrosheet_*.csv        (one per game year)
          raw_batter_expected.csv
          raw_batter_ev_barrels.csv
          raw_batting_splits_lhp.csv
          raw_batting_splits_rhp.csv
          raw_pitcher_arsenal.csv
          raw_pitcher_expected.csv
=============================================================================
"""

import os
import time
import warnings
import requests
import pandas as pd
import numpy as np
import pybaseball as pyb
from datetime import date as _date

warnings.filterwarnings("ignore")
pyb.cache.enable()

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR     = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

STAT_YEARS  = [2022, 2023, 2024, 2025]   # feature years (prior-year features)
GAME_YEARS  = [2023, 2024, 2025]          # years with confirmed game outcomes
MIN_PA      = 100                         # minimum PA to include a batter


# =============================================================================
# 1. CHADWICK BUREAU REGISTER  (MLBAM ↔ FanGraphs ID bridge)
# =============================================================================
def pull_chadwick_register() -> pd.DataFrame:
    print("  Pulling Chadwick Bureau register...")
    chad = pyb.chadwick_register()
    chad = chad[["key_mlbam", "key_fangraphs", "name_first", "name_last"]].copy()
    chad = chad.dropna(subset=["key_mlbam", "key_fangraphs"])
    chad["key_mlbam"]    = chad["key_mlbam"].astype(int)
    chad["key_fangraphs"] = chad["key_fangraphs"].astype(int)
    out = os.path.join(RAW_DIR, "raw_chadwick.csv")
    chad.to_csv(out, index=False)
    print(f"    Saved {len(chad):,} rows → {out}")
    return chad


# =============================================================================
# 2. RETROSHEET GAME LOGS  (SP identity + batting order)
# =============================================================================
def pull_retrosheet_logs(game_years: list) -> pd.DataFrame:
    """
    Retrosheet game logs contain:
      h_starting_pitcher_id / v_starting_pitcher_id  — retrosheet pitcher IDs
      h_bat_1_id … h_bat_9_id                         — home batting order
      v_bat_1_id … v_bat_9_id                         — away batting order

    We save one CSV per year then concatenate.
    """
    frames = []
    for yr in game_years:
        print(f"  Pulling retrosheet game log {yr}...")
        try:
            gl = pyb.retrosheet_game_log(yr)
            out = os.path.join(RAW_DIR, f"raw_retrosheet_{yr}.csv")
            gl.to_csv(out, index=False)
            print(f"    {len(gl):,} games saved → {out}")
            frames.append(gl)
            time.sleep(1)
        except Exception as e:
            print(f"    WARNING: retrosheet {yr} failed — {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# =============================================================================
# 3. STATCAST BATTER EXPECTED STATS  (xBA, xSLG, xwOBA per MLBAM/season)
# =============================================================================
def pull_batter_expected_stats(stat_years: list) -> pd.DataFrame:
    frames = []
    for yr in stat_years:
        print(f"  Pulling batter expected stats {yr}...")
        try:
            df = pyb.statcast_batter_expected_stats(yr, minPA=MIN_PA)
            df["season"] = yr
            frames.append(df)
            time.sleep(1)
        except Exception as e:
            print(f"    WARNING: batter expected {yr} failed — {e}")
    if not frames:
        return pd.DataFrame()
    out_df = pd.concat(frames, ignore_index=True)
    # Savant uses 'player_id' as the MLBAM identifier
    if "player_id" in out_df.columns:
        out_df.rename(columns={"player_id": "key_mlbam"}, inplace=True)
    out_df["key_mlbam"] = pd.to_numeric(out_df["key_mlbam"], errors="coerce")
    out = os.path.join(RAW_DIR, "raw_batter_expected.csv")
    out_df.to_csv(out, index=False)
    print(f"    Saved {len(out_df):,} rows → {out}")
    return out_df


# =============================================================================
# 4. STATCAST BATTER EV / BARREL  (exit_velocity_avg, barrel_batted_rate, etc.)
# =============================================================================
def pull_batter_ev_barrels(stat_years: list) -> pd.DataFrame:
    """
    statcast_batter_exitvelo_barrels() returns MLBAM 'player_id'.
    We rename to key_mlbam for consistency — no name-based matching needed.
    """
    frames = []
    for yr in stat_years:
        print(f"  Pulling batter EV/barrel stats {yr}...")
        try:
            df = pyb.statcast_batter_exitvelo_barrels(yr, minBBE=50)
            df["season"] = yr
            frames.append(df)
            time.sleep(1)
        except Exception as e:
            print(f"    WARNING: batter EV/barrel {yr} failed — {e}")
    if not frames:
        return pd.DataFrame()
    out_df = pd.concat(frames, ignore_index=True)
    if "player_id" in out_df.columns:
        out_df.rename(columns={"player_id": "key_mlbam"}, inplace=True)
    out_df["key_mlbam"] = pd.to_numeric(out_df["key_mlbam"], errors="coerce")
    out = os.path.join(RAW_DIR, "raw_batter_ev_barrels.csv")
    out_df.to_csv(out, index=False)
    print(f"    Saved {len(out_df):,} rows → {out}")
    return out_df


# =============================================================================
# 5. FANGRAPHS BATTING SPLITS  (wRC+, wOBA, ISO, K%, BB% vs LHP and RHP)
# =============================================================================
SPLIT_CODES = {
    "vs_lhp": 117,   # vs Left-Handed Pitcher
    "vs_rhp": 118,   # vs Right-Handed Pitcher
}

def pull_batting_splits(stat_years: list) -> tuple:
    """
    FanGraphs splits API returns season-level splits by PA split code.
    Returns (df_lhp, df_rhp) DataFrames with IDfg for joining.
    """
    base_url = "https://www.fangraphs.com/api/leaders/splits/splits-leaders"

    def _fetch_split(split_code: int, split_label: str,
                     year: int) -> pd.DataFrame:
        params = {
            "strPos":       "all",
            "season":       year,
            "season1":      year,
            "mingames":     0,
            "minpa":        MIN_PA,
            "split":        split_code,
            "splitTeams":   False,
            "statType":     "player",
            "statgroup":    "dashboard",
            "startDate":    f"{year}-01-01",
            "endDate":      f"{year}-12-31",
            "players":      "",
            "z_players":    "",
        }
        headers = {"User-Agent": "baseball-model-research/1.0"}
        try:
            r = requests.get(base_url, params=params, headers=headers,
                             timeout=30)
            r.raise_for_status()
            data = r.json()
            rows = data.get("data", data) if isinstance(data, dict) else data
            df   = pd.DataFrame(rows)
            df["season"]      = year
            df["split_label"] = split_label
            return df
        except Exception as e:
            print(f"      WARNING: FG splits {split_label} {year} — {e}")
            return pd.DataFrame()

    lhp_frames, rhp_frames = [], []
    for yr in stat_years:
        print(f"  Pulling FG batting splits {yr}...")
        lhp_frames.append(_fetch_split(SPLIT_CODES["vs_lhp"], "vs_lhp", yr))
        rhp_frames.append(_fetch_split(SPLIT_CODES["vs_rhp"], "vs_rhp", yr))
        time.sleep(1.5)

    df_lhp = pd.concat([f for f in lhp_frames if not f.empty], ignore_index=True)
    df_rhp = pd.concat([f for f in rhp_frames if not f.empty], ignore_index=True)

    # FanGraphs batting data uses "playerid" or "IDfg"
    for df in [df_lhp, df_rhp]:
        for col in ["playerid", "IDfg", "PlayerID"]:
            if col in df.columns:
                df.rename(columns={col: "IDfg"}, inplace=True)
                break
        if "IDfg" in df.columns:
            df["IDfg"] = pd.to_numeric(df["IDfg"], errors="coerce")

    out_lhp = os.path.join(RAW_DIR, "raw_batting_splits_lhp.csv")
    out_rhp = os.path.join(RAW_DIR, "raw_batting_splits_rhp.csv")
    df_lhp.to_csv(out_lhp, index=False)
    df_rhp.to_csv(out_rhp, index=False)
    print(f"    LHP splits: {len(df_lhp):,} rows → {out_lhp}")
    print(f"    RHP splits: {len(df_rhp):,} rows → {out_rhp}")
    return df_lhp, df_rhp


# =============================================================================
# 6. STATCAST PITCHER ARSENAL STATS  (pitch-type level data per SP, per season)
# =============================================================================
FASTBALL_TYPES  = {"FF", "SI", "FT", "FC"}
OFFSPEED_TYPES  = {"SL", "CU", "CH", "KC", "FS", "EP", "KN", "SC"}

def pull_pitcher_arsenal(stat_years: list) -> pd.DataFrame:
    """
    Returns one row per (pitcher_id, pitch_type, season).
    Columns include: avg_speed, spin_rate, pfx_x, pfx_z, whiff_percent,
                     pitch_percent (usage rate).
    MLBAM pitcher_id retained for joins.
    """
    frames = []
    for yr in stat_years:
        print(f"  Pulling pitcher arsenal stats {yr}...")
        try:
            df = pyb.statcast_pitcher_arsenal_stats(yr, minP=100)
            df["season"] = yr
            frames.append(df)
            time.sleep(1)
        except Exception as e:
            print(f"    WARNING: pitcher arsenal {yr} failed — {e}")
    if not frames:
        return pd.DataFrame()
    out_df = pd.concat(frames, ignore_index=True)
    if "pitcher_id" in out_df.columns:
        out_df.rename(columns={"pitcher_id": "key_mlbam"}, inplace=True)
    out_df["key_mlbam"] = pd.to_numeric(out_df["key_mlbam"], errors="coerce")
    out = os.path.join(RAW_DIR, "raw_pitcher_arsenal.csv")
    out_df.to_csv(out, index=False)
    print(f"    Saved {len(out_df):,} rows → {out}")
    return out_df


# =============================================================================
# 7. STATCAST PITCHER EXPECTED STATS  (xwOBA against, barrel%, hard_hit%)
# =============================================================================
def pull_pitcher_expected_stats(stat_years: list) -> pd.DataFrame:
    frames = []
    for yr in stat_years:
        print(f"  Pulling pitcher expected stats {yr}...")
        try:
            df = pyb.statcast_pitcher_expected_stats(yr, minIP=10)
            df["season"] = yr
            frames.append(df)
            time.sleep(1)
        except Exception as e:
            print(f"    WARNING: pitcher expected {yr} failed — {e}")
    if not frames:
        return pd.DataFrame()
    out_df = pd.concat(frames, ignore_index=True)
    if "player_id" in out_df.columns:
        out_df.rename(columns={"player_id": "key_mlbam"}, inplace=True)
    out_df["key_mlbam"] = pd.to_numeric(out_df["key_mlbam"], errors="coerce")
    out = os.path.join(RAW_DIR, "raw_pitcher_expected.csv")
    out_df.to_csv(out, index=False)
    print(f"    Saved {len(out_df):,} rows → {out}")
    return out_df


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("HITTER TB MODEL — STEP 1: DATA INPUT (REFACTORED)")
    print("=" * 70)

    print("\n[ 1/7 ] Chadwick Bureau register (MLBAM ↔ FanGraphs ID bridge)...")
    pull_chadwick_register()

    print("\n[ 2/7 ] Retrosheet game logs (confirmed SP + batting order)...")
    pull_retrosheet_logs(GAME_YEARS)

    print("\n[ 3/7 ] Batter expected stats (xBA, xSLG, xwOBA)...")
    pull_batter_expected_stats(STAT_YEARS)

    print("\n[ 4/7 ] Batter exit velocity / barrel data...")
    pull_batter_ev_barrels(STAT_YEARS)

    print("\n[ 5/7 ] FanGraphs batting splits (vs LHP / vs RHP)...")
    pull_batting_splits(STAT_YEARS)

    print("\n[ 6/7 ] Pitcher arsenal stats (pitch-type level)...")
    pull_pitcher_arsenal(STAT_YEARS)

    print("\n[ 7/7 ] Pitcher expected stats (xwOBA against, barrel%, HH%)...")
    pull_pitcher_expected_stats(STAT_YEARS)

    print("\n" + "=" * 70)
    print("STEP 1 COMPLETE — Run 02_build_hitter_tb.py next.")
    print("=" * 70)
