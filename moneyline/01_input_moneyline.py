"""
=============================================================================
MONEYLINE MODEL — FILE 1 OF 4: DATA INPUT
=============================================================================
Purpose : Pull all raw data needed for the moneyline XGBoost model.
Sources : FanGraphs (via pybaseball), Baseball Reference (via pybaseball)
Output  : CSV files saved to ../data/raw/

Moneyline background:
  - We want to predict the probability that Team A beats Team B.
  - Key signals: starting pitcher quality (SIERA, xFIP), team offensive
    strength (wRC+, wOBA), and park/environmental context.
  - We use 3 seasons of historical data (2022–2024) as training data.

For R users:
  - Python's `import` = R's `library()`
  - `pd.DataFrame` = R's `data.frame`
  - `os.path.join()` = R's `file.path()`
  - `print()` works just like R's `print()` or `cat()`
=============================================================================
"""

# --- Imports ----------------------------------------------------------------
# In Python you must explicitly import every library you want to use.
# The `as` keyword creates an alias, so `pd` is short for `pandas`.

import os                          # File path operations
import time                        # Pausing between API calls to avoid rate limits
import pandas as pd                # Core data manipulation (like dplyr + data.frame)
import numpy as np                 # Numerical operations (like base R math functions)
import pybaseball as pyb           # Baseball data from Savant, FanGraphs, B-Ref

# Suppress pybaseball's progress bars for cleaner output (optional)
pyb.cache.enable()                 # Cache downloads so re-runs are faster

# --- Configuration ----------------------------------------------------------
# Define file paths using os.path.join so the code works on Mac/Windows/Linux.
# In R you'd use file.path("data", "raw", "filename.csv")

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR   = os.path.join(BASE_DIR, "data", "raw")

# Years to pull — 3 seasons of historical data for model training.
TRAIN_YEARS = [2023, 2024, 2025]

# Current season — pulled separately and merged in so scoring uses live stats.
# Once the season starts, re-run this file weekly to keep current-year data fresh.
from datetime import date as _date
CURRENT_YEAR = _date.today().year   # 2026

# All 30 MLB team abbreviations used by Baseball Reference.
# These match what schedule_and_record() expects as input.
ALL_TEAMS_BREF = [
    "ARI", "ATL", "BAL", "BOS", "CHC", "CWS", "CIN", "CLE",
    "COL", "DET", "HOU", "KCR", "LAA", "LAD", "MIA", "MIL",
    "MIN", "NYM", "NYY", "OAK", "PHI", "PIT", "SDP", "SEA",
    "SFG", "STL", "TBR", "TEX", "TOR", "WSN"
]


# =============================================================================
# FUNCTION 1: Pull FanGraphs Pitching Stats
# =============================================================================
def pull_pitching_stats(years: list) -> pd.DataFrame:
    """
    Pull pitcher-season stats from FanGraphs for each year in `years`.

    Key metrics for moneyline modeling:
      - SIERA  : Skill-Interactive ERA — best predictor of future ERA
      - xFIP   : Expected FIP, normalizes HR/FB to league average
      - FIP    : Fielding Independent Pitching — removes defense influence
      - K%     : Strikeout rate (outcomes the pitcher fully controls)
      - BB%    : Walk rate (outcomes the pitcher fully controls)
      - K-BB%  : Command metric — high K, low BB = dominant pitcher

    Parameters
    ----------
    years : list of int
        Seasons to pull (e.g., [2022, 2023, 2024])

    Returns
    -------
    pd.DataFrame
        One row per pitcher per season with FanGraphs advanced stats.

    R equivalent:
        # No direct R equivalent, but similar to scraping a website table
        # with rvest::read_html() + html_table()
    """
    all_pitching = []  # Empty list — in R this would be: all_pitching <- list()

    for year in years:
        print(f"  Pulling FanGraphs pitching stats for {year}...")
        # pybaseball.pitching_stats(start, end, qual, ind)
        #   qual=0 means include pitchers with any number of IP (no minimum)
        #   ind=1 means return data by individual season (not aggregated)
        #   GS (games started) >= 5 filter is applied later in the build step
        df = pyb.pitching_stats(year, year, qual=0, ind=1)
        df["Season"] = year          # Ensure season column is populated
        all_pitching.append(df)      # Append to list (like rbind accumulation in R)
        time.sleep(2)                # Wait 2 seconds — polite to FanGraphs servers

    # pd.concat() stacks DataFrames vertically = rbind() in R
    # ignore_index=True resets the row index (R doesn't have this concept by default)
    pitching = pd.concat(all_pitching, ignore_index=True)

    # Select columns relevant to moneyline modeling.
    # In R: pitching <- pitching[, c("Name", "Team", ...)]
    keep_cols = [
        "Name", "Team", "Season", "IDfg", "G", "GS", "IP",
        "ERA", "FIP", "xFIP", "SIERA",
        "K%", "BB%", "K-BB%",        # Command metrics
        "HR/9", "H/9", "BABIP",       # Contact quality allowed
        "LOB%",                        # Strand rate (luck indicator)
        "GB%", "FB%", "LD%",          # Batted ball profile
        "WAR",                         # Overall value
    ]
    # Only keep columns that actually exist in the data (API responses can vary by year)
    # In R: keep_cols[keep_cols %in% names(pitching)]
    keep_cols = [c for c in keep_cols if c in pitching.columns]
    pitching = pitching[keep_cols].copy()

    print(f"  ✓ Pitching stats: {len(pitching):,} pitcher-seasons pulled.")
    return pitching


# =============================================================================
# FUNCTION 2: Pull FanGraphs Batting Stats
# =============================================================================
def pull_batting_stats(years: list) -> pd.DataFrame:
    """
    Pull batter-season stats from FanGraphs for each year in `years`.

    Key metrics for moneyline modeling (team offensive strength):
      - wRC+  : Weighted Runs Created Plus — park/league adjusted offense.
                100 = league average, 120 = 20% above average.
                Most comprehensive single offensive metric.
      - wOBA  : Weighted On-Base Average — weights hits by run value.
                Similar to OBP but gives credit for extra-base hits.
      - ISO   : Isolated Power (SLG - AVG) — pure power metric.
      - BABIP : Batting Average on Balls in Play — luck indicator.
                Team BABIP far from .300 suggests regression coming.

    Parameters
    ----------
    years : list of int

    Returns
    -------
    pd.DataFrame
        One row per batter per season with FanGraphs advanced stats.
    """
    all_batting = []

    for year in years:
        print(f"  Pulling FanGraphs batting stats for {year}...")
        # qual=100 means minimum 100 PA — filters out cup-of-coffee players
        df = pyb.batting_stats(year, year, qual=100, ind=1)
        df["Season"] = year
        all_batting.append(df)
        time.sleep(2)

    batting = pd.concat(all_batting, ignore_index=True)

    keep_cols = [
        "Name", "Team", "Season", "IDfg",
        "G", "PA", "AB", "H", "1B", "2B", "3B", "HR",
        "R", "RBI", "BB", "SO", "SB", "CS",
        "AVG", "OBP", "SLG", "OPS",
        "wOBA", "wRAA", "wRC+",        # Context-neutral offensive value
        "ISO", "BABIP",                # Power + luck
        "BB%", "K%",                   # Plate discipline
        "WAR",
    ]
    keep_cols = [c for c in keep_cols if c in batting.columns]
    batting = batting[keep_cols].copy()

    print(f"  ✓ Batting stats: {len(batting):,} batter-seasons pulled.")
    return batting


# =============================================================================
# FUNCTION 3: Pull Team-Level Stats
# =============================================================================
def pull_team_stats(years: list) -> tuple:
    """
    Pull team-level batting and pitching aggregates from FanGraphs.

    Team stats are used to build overall team strength features for each game.
    Individual pitcher stats (SP quality) are layered on top in build step.

    Returns
    -------
    tuple : (team_batting_df, team_pitching_df)
        Two DataFrames with team-level stats, one row per team per season.
    """
    all_team_bat = []
    all_team_pit = []

    for year in years:
        print(f"  Pulling team batting stats for {year}...")
        tb = pyb.team_batting(year, year)
        tb["Season"] = year
        all_team_bat.append(tb)
        time.sleep(1)

        print(f"  Pulling team pitching stats for {year}...")
        tp = pyb.team_pitching(year, year)
        tp["Season"] = year
        all_team_pit.append(tp)
        time.sleep(1)

    team_batting  = pd.concat(all_team_bat,  ignore_index=True)
    team_pitching = pd.concat(all_team_pit,  ignore_index=True)

    print(f"  ✓ Team batting:  {len(team_batting):,} team-seasons pulled.")
    print(f"  ✓ Team pitching: {len(team_pitching):,} team-seasons pulled.")
    return team_batting, team_pitching


# =============================================================================
# FUNCTION 4: Pull Game-by-Game Schedule and Results
# =============================================================================
def pull_game_schedules(years: list, teams: list) -> pd.DataFrame:
    """
    Pull the game-by-game schedule and results for all 30 teams.

    This gives us our training OUTCOMES — did the home team win?
    We join this with pitcher/team stats to create feature rows.

    Baseball Reference's schedule_and_record() returns:
      - Date, Tm (team), Home_Away (@=away, blank=home), Opp (opponent)
      - W/L (result), R (runs scored), RA (runs allowed)
      - Win/Loss/Save (the pitchers)

    Note: Each game appears TWICE — once for each team. We deduplicate
    to home team perspective in the build step.

    Parameters
    ----------
    years : list of int
    teams : list of str
        B-Ref team abbreviations for all 30 teams.

    Returns
    -------
    pd.DataFrame
        All games for all teams across specified years.
    """
    all_games = []

    for year in years:
        print(f"  Pulling game schedules for {year}...")
        for team in teams:
            try:
                df = pyb.schedule_and_record(year, team)
                df["Season"] = year
                df["Team"]   = team
                all_games.append(df)
                time.sleep(0.5)  # Gentle rate limiting for B-Ref
            except Exception as e:
                # In Python, try/except = tryCatch() in R
                print(f"    WARNING: Could not pull {team} {year}: {e}")

    games = pd.concat(all_games, ignore_index=True)

    # Filter to completed regular season games only.
    # B-Ref includes postseason and has NaN rows for future games.
    # In R: games <- games[!is.na(games$R) & games$Inn == 9, ]
    games = games[games["R"].notna()].copy()

    print(f"  ✓ Game schedules: {len(games):,} team-game rows pulled.")
    return games


# =============================================================================
# FUNCTION 5: Pull Statcast Pitcher Expected Stats
# =============================================================================
def pull_statcast_pitcher_stats(years: list) -> pd.DataFrame:
    """
    Pull Statcast-based expected stats for pitchers from Baseball Savant.

    Statcast 'expected' metrics remove luck from outcomes:
      - est_era (xERA)  : Expected ERA based on quality of contact allowed
      - est_woba (xwOBA): Expected wOBA against — better than ERA for projection

    These supplement FanGraphs SIERA/xFIP for the moneyline model.

    Note: These are season-level aggregates, not pitch-level data.
    This is much smaller than pulling raw pitch-level Statcast data.
    """
    all_stats = []

    for year in years:
        print(f"  Pulling Statcast pitcher expected stats for {year}...")
        try:
            # minPA = minimum plate appearances (filters out low-sample pitchers)
            df = pyb.statcast_pitcher_expected_stats(year, minPA=50)
            df["Season"] = year
            all_stats.append(df)
            time.sleep(2)
        except Exception as e:
            print(f"    WARNING: Statcast pitcher stats failed for {year}: {e}")

    if all_stats:
        stats = pd.concat(all_stats, ignore_index=True)
        print(f"  ✓ Statcast pitcher stats: {len(stats):,} pitcher-seasons pulled.")
        return stats
    else:
        print("  WARNING: No Statcast pitcher data pulled. Returning empty DataFrame.")
        return pd.DataFrame()


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    """
    Python's `if __name__ == "__main__":` block runs only when you execute
    this file directly (python 01_input_moneyline.py). It does NOT run when
    another file imports functions from this file.

    In R, you don't need this — all code in a script runs top to bottom.
    In Python, this pattern separates reusable functions from execution logic.
    """
    print("=" * 70)
    print("MONEYLINE MODEL — STEP 1: DATA INPUT")
    print("=" * 70)
    print(f"Pulling historical data for seasons: {TRAIN_YEARS}")
    print(f"Pulling current-season data for:     {CURRENT_YEAR}")
    print(f"Saving raw data to: {RAW_DIR}")
    print()

    # --- Pull historical training data ----------------------------------------
    print("[ 1/5 ] Pulling FanGraphs pitching stats (historical)...")
    pitching_df = pull_pitching_stats(TRAIN_YEARS)

    print("\n[ 2/5 ] Pulling FanGraphs batting stats (historical)...")
    batting_df = pull_batting_stats(TRAIN_YEARS)

    print("\n[ 3/5 ] Pulling team-level stats (historical)...")
    team_bat_df, team_pit_df = pull_team_stats(TRAIN_YEARS)

    print("\n[ 4/5 ] Pulling game schedules and results (historical)...")
    games_df = pull_game_schedules(TRAIN_YEARS, ALL_TEAMS_BREF)

    print("\n[ 5/5 ] Pulling Statcast pitcher expected stats (historical)...")
    statcast_pit_df = pull_statcast_pitcher_stats(TRAIN_YEARS)

    # --- Pull current-season stats and merge in --------------------------------
    # These are appended to the raw CSVs so scoring functions automatically
    # pick up 2026 data as the season progresses (load_batting_stats() uses
    # the most recent season available).
    print(f"\n[ CURRENT SEASON ] Pulling {CURRENT_YEAR} stats for live scoring...")
    try:
        print(f"  Pulling {CURRENT_YEAR} pitching stats...")
        cur_pit = pull_pitching_stats([CURRENT_YEAR])
        if not cur_pit.empty:
            pitching_df = pd.concat(
                [pitching_df[pitching_df["Season"] != CURRENT_YEAR], cur_pit],
                ignore_index=True
            )
            print(f"  ✓ {len(cur_pit)} {CURRENT_YEAR} pitcher-seasons merged.")
    except Exception as e:
        print(f"  WARNING: Could not pull {CURRENT_YEAR} pitching — {e}")

    try:
        print(f"  Pulling {CURRENT_YEAR} batting stats...")
        cur_bat = pull_batting_stats([CURRENT_YEAR])
        if not cur_bat.empty:
            batting_df = pd.concat(
                [batting_df[batting_df["Season"] != CURRENT_YEAR], cur_bat],
                ignore_index=True
            )
            print(f"  ✓ {len(cur_bat)} {CURRENT_YEAR} batter-seasons merged.")
    except Exception as e:
        print(f"  WARNING: Could not pull {CURRENT_YEAR} batting — {e}")

    try:
        print(f"  Pulling {CURRENT_YEAR} team stats...")
        cur_tbat, cur_tpit = pull_team_stats([CURRENT_YEAR])
        if not cur_tbat.empty:
            team_bat_df = pd.concat(
                [team_bat_df[team_bat_df["Season"] != CURRENT_YEAR], cur_tbat],
                ignore_index=True
            )
        if not cur_tpit.empty:
            team_pit_df = pd.concat(
                [team_pit_df[team_pit_df["Season"] != CURRENT_YEAR], cur_tpit],
                ignore_index=True
            )
        print(f"  ✓ {CURRENT_YEAR} team stats merged.")
    except Exception as e:
        print(f"  WARNING: Could not pull {CURRENT_YEAR} team stats — {e}")

    # --- Save to CSV ----------------------------------------------------------
    print("\n[ SAVING ] Writing raw data to CSV files...")

    pitching_df.to_csv(os.path.join(RAW_DIR, "raw_pitching_stats.csv"), index=False)
    print(f"  ✓ Saved raw_pitching_stats.csv  ({len(pitching_df):,} rows)")

    batting_df.to_csv(os.path.join(RAW_DIR, "raw_batting_stats.csv"), index=False)
    print(f"  ✓ Saved raw_batting_stats.csv   ({len(batting_df):,} rows)")

    team_bat_df.to_csv(os.path.join(RAW_DIR, "raw_team_batting.csv"), index=False)
    print(f"  ✓ Saved raw_team_batting.csv    ({len(team_bat_df):,} rows)")

    team_pit_df.to_csv(os.path.join(RAW_DIR, "raw_team_pitching.csv"), index=False)
    print(f"  ✓ Saved raw_team_pitching.csv   ({len(team_pit_df):,} rows)")

    games_df.to_csv(os.path.join(RAW_DIR, "raw_game_schedules.csv"), index=False)
    print(f"  ✓ Saved raw_game_schedules.csv  ({len(games_df):,} rows)")

    if not statcast_pit_df.empty:
        statcast_pit_df.to_csv(os.path.join(RAW_DIR, "raw_statcast_pitcher.csv"), index=False)
        print(f"  ✓ Saved raw_statcast_pitcher.csv ({len(statcast_pit_df):,} rows)")

    print()
    print("=" * 70)
    print("STEP 1 COMPLETE — Run 02_build_moneyline.py next.")
    print(f"TIP: Re-run this file weekly during the {CURRENT_YEAR} season to refresh live stats.")
    print("=" * 70)
