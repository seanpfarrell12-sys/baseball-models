"""
=============================================================================
HITTER TOTAL BASES MODEL — FILE 1 OF 4: DATA INPUT
=============================================================================
Purpose : Pull all raw data needed for the hitter total bases XGBoost model.
Sources : FanGraphs (via pybaseball), Baseball Savant/Statcast (via pybaseball)
Output  : CSV files saved to ../data/raw/

Total Bases background:
  - Total Bases = 1B×1 + 2B×2 + 3B×3 + HR×4
  - Market: "Will Player X get over/under 1.5 total bases today?"
  - Key signals: Barrel%, EV, Hard-Hit%, launch angle consistency,
    pitcher matchup (handedness, pitch mix, K/BB tendency), park factor.

Why Statcast matters here:
  - Traditional stats (AVG, SLG) are outcome-based and noisy.
  - Statcast metrics measure the QUALITY of contact — a leading indicator.
  - A hitter with high Barrel% but low AVG is due for positive regression.
  - We're modeling the underlying process, not just past results.

Data sources used:
  1. FanGraphs batting stats — wOBA, ISO, BB%, K%, platoon splits
  2. Baseball Savant exit velocity/barrel stats — Barrel%, HardHit%, EV, LA
  3. Baseball Savant expected stats — xBA, xSLG, xwOBA
  4. FanGraphs pitching stats — for matchup features (SP faced each game)
  5. Baseball Savant pitch arsenal data — SP pitch mix by type

For R users:
  - Statcast data comes from Baseball Savant (savant.baseball)
  - pybaseball wraps the Savant API so we don't need to scrape HTML
  - `statcast_batter_exitvelo_barrels()` = season-level EV/barrel aggregates
=============================================================================
"""

import os
import time
import pandas as pd
import numpy as np
import pybaseball as pyb
from datetime import date as _date

pyb.cache.enable()

# --- Configuration ----------------------------------------------------------
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR      = os.path.join(BASE_DIR, "data", "raw")
TRAIN_YEARS  = [2023, 2024, 2025]
CURRENT_YEAR = _date.today().year   # 2026 — refreshed each run

# Minimum plate appearances to include a batter (filters small samples)
MIN_PA = 150

# Minimum batted ball events for Statcast barrel/EV data
MIN_BBE = 75


# =============================================================================
# FUNCTION 1: Pull Statcast Exit Velocity and Barrel Data
# =============================================================================
def pull_statcast_ev_barrels(years: list) -> pd.DataFrame:
    """
    Pull season-level exit velocity and barrel statistics per batter.

    This is the heart of the total bases model — Statcast quality-of-contact
    metrics are far more predictive than traditional batting stats.

    Key columns returned:
      - avg_hit_speed    : Mean exit velocity (mph) — higher = harder contact
      - ev95plus         : Number of batted balls hit 95+ mph (Hard-Hit count)
      - ev95percent      : Hard-Hit rate (% of BBE at 95+ mph)
      - barrels          : Count of barrels (optimal EV + launch angle combos)
      - brl_percent      : Barrel % (barrels per PA) — strongest HR/XBH predictor
      - anglesweetspotpct: % of batted balls in "sweet spot" (8–32° launch angle)
      - avg_hit_angle    : Mean launch angle

    In R, you'd access this data through a manual Savant export CSV.
    pybaseball automates this download.

    Parameters
    ----------
    years : list of int

    Returns
    -------
    pd.DataFrame
        One row per batter per season with exit velocity stats.
    """
    all_ev = []

    for year in years:
        print(f"  Pulling Statcast EV/barrel data for {year}...")
        try:
            # minBBE: minimum batted ball events for inclusion
            # Filters out pitchers and very part-time players
            df = pyb.statcast_batter_exitvelo_barrels(year, minBBE=MIN_BBE)
            df["Season"] = year
            all_ev.append(df)
            time.sleep(2)
        except Exception as e:
            print(f"    WARNING: EV/barrel pull failed for {year}: {e}")

    ev_df = pd.concat(all_ev, ignore_index=True)

    # Rename columns to be more descriptive (Savant uses abbreviated names)
    # In R: names(df)[names(df) == "old"] <- "new"
    rename_map = {
        "last_name, first_name": "player_name",  # Savant format: "Smith, John"
        "avg_hit_speed":         "avg_exit_velo",
        "ev95percent":           "hard_hit_pct",
        "ev95plus":              "hard_hit_count",
        "brl_percent":           "barrel_pct",
        "brl_pa":                "barrel_per_pa",
        "anglesweetspotpercent": "sweet_spot_pct",
        "avg_hit_angle":         "avg_launch_angle",
    }
    # Only rename columns that actually exist
    rename_map = {k: v for k, v in rename_map.items() if k in ev_df.columns}
    ev_df = ev_df.rename(columns=rename_map)

    print(f"  ✓ EV/barrel data: {len(ev_df):,} batter-seasons pulled.")
    return ev_df


# =============================================================================
# FUNCTION 2: Pull Statcast Expected Stats (xBA, xSLG, xwOBA)
# =============================================================================
def pull_statcast_expected_stats(years: list) -> pd.DataFrame:
    """
    Pull Statcast 'expected' outcome statistics per batter.

    Expected stats use exit velocity and launch angle to predict what a
    player's batting average, slugging, and wOBA SHOULD be, removing
    the influence of defense, park, and batted ball luck.

    Key columns:
      - est_ba     (xBA)   : Expected batting average
      - est_slg    (xSLG)  : Expected slugging percentage
      - est_woba   (xwOBA) : Expected weighted on-base average
      - est_woba_minus_woba_diff : Positive = hitter got unlucky (due for upswing)
                                   Negative = hitter got lucky (due for regression)

    The difference between actual and expected is a powerful luck indicator.
    """
    all_xstats = []

    for year in years:
        print(f"  Pulling Statcast expected stats for {year}...")
        try:
            df = pyb.statcast_batter_expected_stats(year, minPA=MIN_PA)
            df["Season"] = year
            all_xstats.append(df)
            time.sleep(2)
        except Exception as e:
            print(f"    WARNING: Expected stats pull failed for {year}: {e}")

    if all_xstats:
        xstats_df = pd.concat(all_xstats, ignore_index=True)
        print(f"  ✓ Expected stats: {len(xstats_df):,} batter-seasons pulled.")
        return xstats_df
    return pd.DataFrame()


# =============================================================================
# FUNCTION 3: Pull FanGraphs Batting Stats (Platoon + Advanced)
# =============================================================================
def pull_batting_stats(years: list) -> pd.DataFrame:
    """
    Pull batter-season stats from FanGraphs for modeling features.

    Key metrics for total bases modeling:
      - wOBA / xwOBA  : Overall hitting quality (context-neutral)
      - ISO           : Isolated power (SLG - AVG), pure extra-base hit rate
      - wRC+          : Park-adjusted offensive value
      - K%, BB%       : Strikeout and walk rates (plate discipline)
      - Pull%, Cent%  : Batted ball direction tendencies

    We also derive:
      - 1B = H - 2B - 3B - HR  (singles)
      - TB per G = (1B + 2×2B + 3×3B + 4×HR) / G  (our training target)
    """
    all_bat = []

    for year in years:
        print(f"  Pulling FanGraphs batting stats for {year}...")
        df = pyb.batting_stats(year, year, qual=MIN_PA, ind=1)
        df["Season"] = year
        all_bat.append(df)
        time.sleep(2)

    batting = pd.concat(all_bat, ignore_index=True)

    # Compute total bases per game — this is our model's TARGET variable
    # TB = 1B + 2×2B + 3×3B + 4×HR
    # In R: batting$singles <- batting$H - batting$`2B` - batting$`3B` - batting$HR
    if all(c in batting.columns for c in ["H", "2B", "3B", "HR", "G"]):
        batting["singles"] = batting["H"] - batting["2B"] - batting["3B"] - batting["HR"]
        batting["total_bases"]       = (
            batting["singles"]
            + 2 * batting["2B"]
            + 3 * batting["3B"]
            + 4 * batting["HR"]
        )
        batting["tb_per_game"]       = batting["total_bases"] / batting["G"]
        # For prop bets, we need TB per game as our label
        # Market typically sets line at 1.5 TB (over/under 1.5)
        batting["over_1_5_tb_rate"]  = (batting["tb_per_game"] >= 1.5).astype(float)
        batting["over_0_5_tb_rate"]  = (batting["tb_per_game"] >= 0.5).astype(float)

    print(f"  ✓ Batting stats: {len(batting):,} batter-seasons pulled.")
    return batting


# =============================================================================
# FUNCTION 4: Pull Pitcher Pitch Arsenal Stats (Matchup Features)
# =============================================================================
def pull_pitcher_arsenal(years: list) -> pd.DataFrame:
    """
    Pull pitcher pitch-type arsenal data from Baseball Savant.

    For the hitter total bases model, matchup quality depends heavily on
    what pitches the opposing starter THROWS — not just their overall ERA.

    A hitter who crushes fastballs but struggles against sliders will have
    lower expected TB against a slider-heavy starter.

    Key columns:
      - pitch_type     : FA (4-seam fastball), SI (sinker), SL (slider), etc.
      - avg_speed      : Mean velocity for that pitch type
      - usage_pct      : How often they throw that pitch (0–100%)
      - avg_spin_rate  : RPM (higher spin = more movement)

    This is joined to each game's starting pitcher in the build step.
    """
    all_arsenal = []

    for year in years:
        print(f"  Pulling pitcher pitch arsenal for {year}...")
        try:
            # arsenal_type options: 'avg_speed', 'n_', 'usage', etc.
            df = pyb.statcast_pitcher_pitch_arsenal(
                year, minP=100, arsenal_type="avg_speed"
            )
            df["Season"] = year
            all_arsenal.append(df)
            time.sleep(2)
        except Exception as e:
            print(f"    WARNING: Pitch arsenal pull failed for {year}: {e}")

    if all_arsenal:
        arsenal = pd.concat(all_arsenal, ignore_index=True)
        print(f"  ✓ Pitcher arsenal: {len(arsenal):,} pitcher-season-pitch-type rows.")
        return arsenal
    return pd.DataFrame()


# =============================================================================
# FUNCTION 5: Pull Pitcher Expected Stats (for Matchup Quality)
# =============================================================================
def pull_pitcher_xstats(years: list) -> pd.DataFrame:
    """
    Pull Statcast expected stats for pitchers (xwOBA allowed, xERA).

    These tell us the QUALITY of contact a pitcher allows.
    A pitcher with high ERA but low xERA is getting unlucky — hitters
    are actually NOT making great contact, so they'd be a poor matchup
    for a "Over" total bases bet.
    """
    all_pit = []

    for year in years:
        print(f"  Pulling pitcher Statcast xstats for {year}...")
        try:
            df = pyb.statcast_pitcher_expected_stats(year, minPA=50)
            df["Season"] = year
            all_pit.append(df)
            time.sleep(2)
        except Exception as e:
            print(f"    WARNING: Pitcher xstats failed for {year}: {e}")

    if all_pit:
        pit_df = pd.concat(all_pit, ignore_index=True)
        print(f"  ✓ Pitcher xstats: {len(pit_df):,} pitcher-seasons pulled.")
        return pit_df
    return pd.DataFrame()


# =============================================================================
# FUNCTION 6: Pull FanGraphs Pitcher Stats (K%, BB%, handedness)
# =============================================================================
def pull_fg_pitcher_stats(years: list) -> pd.DataFrame:
    """
    Pull FanGraphs pitcher stats needed for matchup features.

    Key features for the hitter TB matchup:
      - K%          : Higher K% = worse matchup for hitters (fewer balls in play)
      - BB%         : Higher BB% = more base runners (indirect boost to TB value)
      - SIERA       : Overall quality — better pitchers = lower expected TB
      - SwStr%      : Swing-and-miss rate — hard to make contact at all
      - Left/Right  : Pitcher handedness (platoon advantage/disadvantage)
    """
    all_pit = []

    for year in years:
        print(f"  Pulling FanGraphs pitcher stats for {year}...")
        df = pyb.pitching_stats(year, year, qual=0, ind=1)
        df["Season"] = year
        if "GS" in df.columns:
            df = df[df["GS"] >= 5]  # Starters only
        all_pit.append(df)
        time.sleep(2)

    pitching = pd.concat(all_pit, ignore_index=True)
    print(f"  ✓ FG pitcher stats: {len(pitching):,} starter-seasons pulled.")
    return pitching


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("HITTER TOTAL BASES MODEL — STEP 1: DATA INPUT")
    print("=" * 70)
    print(f"Pulling data for seasons: {TRAIN_YEARS}")
    print(f"Minimum PA threshold: {MIN_PA}")
    print()

    print("[ 1/6 ] Pulling Statcast exit velocity / barrel data...")
    ev_df = pull_statcast_ev_barrels(TRAIN_YEARS)

    print("\n[ 2/6 ] Pulling Statcast batter expected stats (xBA, xSLG, xwOBA)...")
    xstats_df = pull_statcast_expected_stats(TRAIN_YEARS)

    print("\n[ 3/6 ] Pulling FanGraphs batting stats...")
    batting_df = pull_batting_stats(TRAIN_YEARS)

    print("\n[ 4/6 ] Pulling pitcher pitch arsenal data...")
    arsenal_df = pull_pitcher_arsenal(TRAIN_YEARS)

    print("\n[ 5/6 ] Pulling pitcher Statcast expected stats...")
    pitcher_xstats_df = pull_pitcher_xstats(TRAIN_YEARS)

    print("\n[ 6/6 ] Pulling FanGraphs pitcher stats (matchup features)...")
    fg_pitcher_df = pull_fg_pitcher_stats(TRAIN_YEARS)

    # --- Save files ----------------------------------------------------------
    print("\n[ SAVING ] Writing raw data files...")

    save_map = {
        "raw_hitter_ev_barrels.csv":     ev_df,
        "raw_hitter_xstats.csv":         xstats_df,
        "raw_hitter_batting.csv":        batting_df,
        "raw_pitcher_arsenal.csv":       arsenal_df,
        "raw_pitcher_xstats.csv":        pitcher_xstats_df,
        "raw_fg_pitcher_stats.csv":      fg_pitcher_df,
    }

    # In Python, .items() iterates over key-value pairs in a dict
    # In R: for (name in names(save_map)) { write.csv(save_map[[name]], ...) }
    for filename, df in save_map.items():
        if df is not None and not df.empty:
            path = os.path.join(RAW_DIR, filename)
            df.to_csv(path, index=False)
            print(f"  ✓ {filename:45s} ({len(df):,} rows)")

    # -------------------------------------------------------------------------
    # REFRESH CURRENT SEASON DATA (re-run weekly to pick up 2026 stats)
    # -------------------------------------------------------------------------
    if CURRENT_YEAR not in TRAIN_YEARS:
        print(f"\n[ REFRESH ] Pulling {CURRENT_YEAR} current-season data...")

        try:
            cur_bat = pull_batting_stats([CURRENT_YEAR])
            if not cur_bat.empty:
                batting_df = pd.concat(
                    [batting_df[batting_df["Season"] != CURRENT_YEAR], cur_bat],
                    ignore_index=True,
                )
                batting_df.to_csv(os.path.join(RAW_DIR, "raw_hitter_batting.csv"), index=False)
                print(f"  ✓ Merged {CURRENT_YEAR} batting into raw_hitter_batting.csv")
        except Exception as e:
            print(f"  WARNING: Could not pull {CURRENT_YEAR} batting stats — {e}")

        try:
            cur_ev = pull_statcast_ev_barrels([CURRENT_YEAR])
            if not cur_ev.empty:
                ev_df = pd.concat(
                    [ev_df[ev_df["Season"] != CURRENT_YEAR], cur_ev],
                    ignore_index=True,
                )
                ev_df.to_csv(os.path.join(RAW_DIR, "raw_hitter_ev_barrels.csv"), index=False)
                print(f"  ✓ Merged {CURRENT_YEAR} EV/barrel data into raw_hitter_ev_barrels.csv")
        except Exception as e:
            print(f"  WARNING: Could not pull {CURRENT_YEAR} EV/barrel data — {e}")

        try:
            cur_xstats = pull_statcast_expected_stats([CURRENT_YEAR])
            if not cur_xstats.empty:
                xstats_df = pd.concat(
                    [xstats_df[xstats_df["Season"] != CURRENT_YEAR], cur_xstats],
                    ignore_index=True,
                )
                xstats_df.to_csv(os.path.join(RAW_DIR, "raw_hitter_xstats.csv"), index=False)
                print(f"  ✓ Merged {CURRENT_YEAR} xstats into raw_hitter_xstats.csv")
        except Exception as e:
            print(f"  WARNING: Could not pull {CURRENT_YEAR} hitter xstats — {e}")

        try:
            cur_fg_pit = pull_fg_pitcher_stats([CURRENT_YEAR])
            if not cur_fg_pit.empty:
                fg_pitcher_df = pd.concat(
                    [fg_pitcher_df[fg_pitcher_df["Season"] != CURRENT_YEAR], cur_fg_pit],
                    ignore_index=True,
                )
                fg_pitcher_df.to_csv(os.path.join(RAW_DIR, "raw_fg_pitcher_stats.csv"), index=False)
                print(f"  ✓ Merged {CURRENT_YEAR} FG pitcher stats into raw_fg_pitcher_stats.csv")
        except Exception as e:
            print(f"  WARNING: Could not pull {CURRENT_YEAR} FG pitcher stats — {e}")

    print(f"\n  TIP: Re-run this file weekly during the {CURRENT_YEAR} season to refresh live stats.")

    print("\n" + "=" * 70)
    print("STEP 1 COMPLETE — Run 02_build_hitter_tb.py next.")
    print("=" * 70)
