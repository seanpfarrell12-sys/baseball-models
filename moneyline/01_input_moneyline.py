"""
=============================================================================
MONEYLINE MODEL — FILE 1 OF 4: DATA INPUT  (REFACTORED)
=============================================================================
Pulls granular, matchup-specific data for the refactored moneyline model.

Data sources:
  - Retrosheet game logs  : SP identification + game results
  - Chadwick register     : retroID ↔ MLBAM ↔ FanGraphs ID cross-walk
  - Statcast expected     : xwOBA against, barrel%, hard_hit% per pitcher
  - Statcast arsenal      : velocity, spin, movement, whiff% by pitch type
  - FanGraphs pitching    : SIERA, xFIP, K%, BB%, handedness, GS (SP/RP flag)
  - FanGraphs platoon     : batter wRC+/wOBA/K%/ISO vs LHP and vs RHP

Run once per season (or weekly in-season):
    python3 moneyline/01_input_moneyline.py
=============================================================================
"""

import os
import time
import requests
import pandas as pd
import numpy as np
import pybaseball as pyb
from datetime import date as _date

pyb.cache.enable()

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR     = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# Years to pull stats for (game features use prior-year stats, so we need
# one extra year at the front: 2022 stats → features for 2023 games)
STAT_YEARS  = [2022, 2023, 2024, 2025]
# Years to pull retrosheet game logs (training outcomes + SP IDs)
GAME_YEARS  = [2023, 2024, 2025]
CURRENT_YEAR = _date.today().year   # 2026 — live stats for in-season scoring


# =============================================================================
# 1. Retrosheet game logs — definitive SP identification + game results
# =============================================================================
def pull_retrosheet_logs(years: list) -> pd.DataFrame:
    """
    Pull retrosheet game logs for game results and confirmed SP IDs.

    Key columns used downstream:
      date                        : YYYYMMDD int
      v_name / h_name             : retrosheet team codes (e.g. "LAN", "NYA")
      v_score / h_score           : final scores  →  home_win target
      v_starting_pitcher_id       : retrosheet player ID for visiting SP
      h_starting_pitcher_id       : retrosheet player ID for home SP
      v_starting_pitcher_name     : name string (fallback for name matching)
      h_starting_pitcher_name     : name string (fallback for name matching)
    """
    all_logs = []
    for year in years:
        print(f"  Pulling retrosheet game log for {year}...")
        try:
            df = pyb.retrosheet_game_log(year)
            df["season"] = year
            all_logs.append(df)
            time.sleep(1)
        except Exception as e:
            print(f"    WARNING: Retrosheet {year} failed: {e}")
    if not all_logs:
        return pd.DataFrame()
    return pd.concat(all_logs, ignore_index=True)


# =============================================================================
# 2. Chadwick Bureau register — cross-walk between ID systems
# =============================================================================
def pull_chadwick_register() -> pd.DataFrame:
    """
    Pull Chadwick Bureau player ID register.

    Used to map:
      key_retro      →  key_mlbam      (retrosheet ID → Statcast MLBAM ID)
      key_retro      →  key_fangraphs  (retrosheet ID → FanGraphs ID)

    This is the bridge between retrosheet SP IDs and Statcast/FG stats.
    """
    print("  Pulling Chadwick player ID register...")
    try:
        chad = pyb.chadwick_register()
        # Standardize column names (pybaseball occasionally renames these)
        rename = {}
        for col in chad.columns:
            if col.lower() in ("key_retro", "retro_id"):
                rename[col] = "key_retro"
            elif col.lower() in ("key_mlbam", "mlbam_id", "key_mlbam_id"):
                rename[col] = "key_mlbam"
            elif col.lower() in ("key_fangraphs", "fangraphs_id"):
                rename[col] = "key_fangraphs"
        chad = chad.rename(columns=rename)
        keep = [c for c in ["key_retro", "key_mlbam", "key_fangraphs",
                             "name_first", "name_last"] if c in chad.columns]
        chad = chad[keep].dropna(subset=["key_retro"])
        print(f"  ✓ Chadwick register: {len(chad):,} players with retro IDs")
        return chad
    except Exception as e:
        print(f"    WARNING: Chadwick register failed: {e}")
        return pd.DataFrame()


# =============================================================================
# 3. Statcast pitcher expected stats — quality-of-contact prevention
# =============================================================================
def pull_statcast_expected(years: list) -> pd.DataFrame:
    """
    Pull Statcast expected stats per pitcher-season.

    Key metrics:
      xwoba              : expected wOBA against (quality of contact allowed)
      barrel_batted_rate : barrel% allowed
      hard_hit_percent   : hard-hit% allowed (exit velo ≥ 95 mph)
      whiff_percent      : overall swing-and-miss rate
      k_percent          : K% (strikeout rate)
      bb_percent         : BB% (walk rate)
    """
    all_stats = []
    for year in years:
        print(f"  Pulling Statcast expected stats for {year}...")
        try:
            df = pyb.statcast_pitcher_expected_stats(year, minPA=50)
            df["season"] = year
            all_stats.append(df)
            time.sleep(2)
        except Exception as e:
            print(f"    WARNING: Statcast expected {year}: {e}")
    if not all_stats:
        return pd.DataFrame()
    return pd.concat(all_stats, ignore_index=True)


# =============================================================================
# 4. Statcast arsenal stats — pitch-level velocity, movement, sequencing
# =============================================================================
def pull_statcast_arsenal(years: list) -> pd.DataFrame:
    """
    Pull Statcast arsenal stats (one row per pitcher per pitch type per year).

    Key columns:
      pitch_type     : FF (4-seam), SI (sinker), SL (slider), CU (curve),
                       CH (changeup), KC (knuckle-curve), FS (splitter)
      pitches        : pitch count for this type (used to weight averages)
      avg_speed      : average velocity in mph
      avg_spin_rate  : average spin rate in RPM
      avg_break_x    : horizontal break in inches (positive = arm side)
      avg_break_z    : vertical break in inches (positive = more rise)
      avg_whiff_pct  : swing-and-miss rate for this pitch type
      avg_k_pct      : K% when this pitch type is in play
    """
    all_stats = []
    for year in years:
        print(f"  Pulling Statcast arsenal stats for {year}...")
        try:
            df = pyb.statcast_pitcher_arsenal_stats(year, minP=100)
            df["season"] = year
            all_stats.append(df)
            time.sleep(2)
        except Exception as e:
            print(f"    WARNING: Arsenal stats {year}: {e}")
    if not all_stats:
        return pd.DataFrame()
    return pd.concat(all_stats, ignore_index=True)


# =============================================================================
# 5. FanGraphs pitching — SIERA, xFIP, K%, BB%, GS flag, handedness
# =============================================================================
def pull_fg_pitching(years: list) -> pd.DataFrame:
    """
    Pull FanGraphs advanced pitching stats.

    Used for:
      - SP vs RP classification (GS column: SP has GS / G > 0.5)
      - Bullpen pool ERA, K%, FIP (pitchers with GS < 5)
      - Fallback SIERA/xFIP for SPs without enough Statcast PAs
      - Pitcher handedness (if 'Throws' column is present)
    """
    all_dfs = []
    keep_cols = [
        "Name", "Team", "Season", "IDfg",
        "G", "GS", "IP", "ERA", "FIP", "xFIP", "SIERA",
        "K%", "BB%", "K-BB%", "GB%",
        "WAR",
    ]
    for year in years:
        print(f"  Pulling FanGraphs pitching stats for {year}...")
        try:
            df = pyb.pitching_stats(year, year, qual=0, ind=1)
            df["Season"] = year
            # Include handedness if available
            if "Throws" in df.columns:
                keep_cols_yr = [c for c in keep_cols + ["Throws"] if c in df.columns]
            else:
                keep_cols_yr = [c for c in keep_cols if c in df.columns]
            all_dfs.append(df[keep_cols_yr])
            time.sleep(2)
        except Exception as e:
            print(f"    WARNING: FG pitching {year}: {e}")
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


# =============================================================================
# 6. FanGraphs platoon batting splits — wRC+ vs LHP and vs RHP
# =============================================================================
def pull_platoon_splits(years: list, min_pa: int = 50) -> tuple:
    """
    Pull batter platoon splits from FanGraphs splits leaderboard API.

    Returns two DataFrames:
      lhp_splits : batter stats when facing LHP (split code 117)
      rhp_splits : batter stats when facing RHP (split code 118)

    Key columns used downstream:
      Name / playerid / Team : player/team identification
      PA                     : plate appearances (used for weighting)
      wRC+                   : park/league-adjusted run creation
      wOBA                   : weighted on-base average
      OBP                    : on-base percentage
      ISO                    : isolated power (SLG - AVG)
      K%                     : strikeout rate
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; baseball-models/1.0)",
        "Referer":    "https://www.fangraphs.com/leaders/splits-leaderboards",
    }

    split_codes = {"L": "117", "R": "118"}
    results = {}

    for hand, split_code in split_codes.items():
        all_dfs = []
        for year in years:
            print(f"  Pulling platoon splits vs {hand}HP for {year}...")
            try:
                url = "https://www.fangraphs.com/api/leaders/splits/data"
                params = {
                    "stats":    "bat",
                    "position": "B",
                    "season":   year,
                    "split":    split_code,
                    "groupby":  "season",
                    "qual":     "y",
                    "count":    min_pa,
                    "type":     "0",       # dashboard (includes wRC+, wOBA)
                    "players":  "",
                    "team":     "0",
                    "lg":       "2,3",
                }
                resp = requests.get(url, params=params, headers=headers, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                rows = data.get("data", [])
                if rows:
                    df = pd.DataFrame(rows)
                    df["season"]  = year
                    df["vs_hand"] = hand
                    all_dfs.append(df)
                    print(f"    ✓ {len(df):,} batters (vs {hand}HP, {year})")
                else:
                    print(f"    WARNING: No rows returned for vs {hand}HP {year}")
                time.sleep(2)
            except Exception as e:
                print(f"    WARNING: Platoon splits vs {hand}HP {year}: {e}")

        results[hand] = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    return results["L"], results["R"]


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("MONEYLINE MODEL — STEP 1: DATA INPUT (REFACTORED)")
    print(f"Stat years : {STAT_YEARS}")
    print(f"Game years : {GAME_YEARS}")
    print("=" * 70)

    # ── 1. Retrosheet game logs ──────────────────────────────────────────────
    print("\n[ 1/6 ] Retrosheet game logs (SP IDs + results)...")
    retro_df = pull_retrosheet_logs(GAME_YEARS)
    if not retro_df.empty:
        retro_df.to_csv(os.path.join(RAW_DIR, "raw_retrosheet.csv"), index=False)
        print(f"  ✓ Saved raw_retrosheet.csv  ({len(retro_df):,} game-rows)")
    else:
        print("  ✗ No retrosheet data — check pybaseball version.")

    # ── 2. Chadwick register ─────────────────────────────────────────────────
    print("\n[ 2/6 ] Chadwick player ID register...")
    chad_df = pull_chadwick_register()
    if not chad_df.empty:
        chad_df.to_csv(os.path.join(RAW_DIR, "raw_chadwick.csv"), index=False)
        print(f"  ✓ Saved raw_chadwick.csv    ({len(chad_df):,} players)")

    # ── 3. Statcast expected stats ───────────────────────────────────────────
    print("\n[ 3/6 ] Statcast pitcher expected stats (xwOBA, barrel%, hard_hit%)...")
    sc_expected = pull_statcast_expected(STAT_YEARS)
    if not sc_expected.empty:
        sc_expected.to_csv(os.path.join(RAW_DIR, "raw_statcast_expected.csv"), index=False)
        print(f"  ✓ Saved raw_statcast_expected.csv ({len(sc_expected):,} pitcher-seasons)")

    # ── 4. Statcast arsenal stats ────────────────────────────────────────────
    print("\n[ 4/6 ] Statcast pitcher arsenal stats (velocity, spin, movement, whiff%)...")
    sc_arsenal = pull_statcast_arsenal(STAT_YEARS)
    if not sc_arsenal.empty:
        sc_arsenal.to_csv(os.path.join(RAW_DIR, "raw_statcast_arsenal.csv"), index=False)
        print(f"  ✓ Saved raw_statcast_arsenal.csv  ({len(sc_arsenal):,} pitcher-pitch-type rows)")

    # ── 5. FanGraphs pitching ────────────────────────────────────────────────
    print("\n[ 5/6 ] FanGraphs pitching stats (SIERA, xFIP, GS, handedness)...")
    fg_pit = pull_fg_pitching(STAT_YEARS)
    if not fg_pit.empty:
        fg_pit.to_csv(os.path.join(RAW_DIR, "raw_fg_pitching.csv"), index=False)
        print(f"  ✓ Saved raw_fg_pitching.csv       ({len(fg_pit):,} pitcher-seasons)")

    # ── 6. Platoon batting splits ────────────────────────────────────────────
    print("\n[ 6/6 ] FanGraphs platoon batting splits (wRC+ vs LHP / vs RHP)...")
    lhp_df, rhp_df = pull_platoon_splits(STAT_YEARS)
    if not lhp_df.empty:
        lhp_df.to_csv(os.path.join(RAW_DIR, "raw_platoon_vs_lhp.csv"), index=False)
        print(f"  ✓ Saved raw_platoon_vs_lhp.csv    ({len(lhp_df):,} batter-seasons)")
    if not rhp_df.empty:
        rhp_df.to_csv(os.path.join(RAW_DIR, "raw_platoon_vs_rhp.csv"), index=False)
        print(f"  ✓ Saved raw_platoon_vs_rhp.csv    ({len(rhp_df):,} batter-seasons)")

    # ── Live current-year stats (append as they accumulate in-season) ────────
    if CURRENT_YEAR not in STAT_YEARS:
        print(f"\n[ LIVE ] Pulling {CURRENT_YEAR} in-season stats...")
        for puller, name, dest in [
            (lambda: pull_statcast_expected([CURRENT_YEAR]),
             "Statcast expected", "raw_statcast_expected.csv"),
            (lambda: pull_statcast_arsenal([CURRENT_YEAR]),
             "Statcast arsenal", "raw_statcast_arsenal.csv"),
            (lambda: pull_fg_pitching([CURRENT_YEAR]),
             "FG pitching", "raw_fg_pitching.csv"),
        ]:
            try:
                new_df = puller()
                if not new_df.empty:
                    dest_path = os.path.join(RAW_DIR, dest)
                    if os.path.exists(dest_path):
                        existing = pd.read_csv(dest_path)
                        # Drop any stale current-year rows, replace with fresh pull
                        existing = existing[existing.get("season", existing.get("Season", 0)) != CURRENT_YEAR]
                        new_df = pd.concat([existing, new_df], ignore_index=True)
                    new_df.to_csv(dest_path, index=False)
                    print(f"  ✓ {name} {CURRENT_YEAR}: {len(new_df)} rows updated")
            except Exception as e:
                print(f"  WARNING: {name} {CURRENT_YEAR} failed: {e}")

    print("\n" + "=" * 70)
    print("STEP 1 COMPLETE — Run 02_build_moneyline.py next.")
    print("=" * 70)
