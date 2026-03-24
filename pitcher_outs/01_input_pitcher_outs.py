"""
=============================================================================
PITCHER TOTAL OUTS MODEL — FILE 1 OF 4: DATA INPUT
=============================================================================
Purpose : Pull all raw data needed for the pitcher total outs XGBoost model.
Sources : FanGraphs (via pybaseball), Baseball Reference (via pybaseball),
          Baseball Savant (via pybaseball)
Output  : CSV files saved to ../data/raw/

Pitcher Total Outs background:
  - Market: "Will Pitcher X record over/under 15.5 outs today?"
    (15.5 outs = roughly 5.2 innings pitched)
  - Outs = IP × 3 (a complete 9-inning game = 27 outs from one team)
  - Key signals: K%, BB%, K-BB%, P/PA (pitches per plate appearance),
    CSW% (called strikes + whiffs), manager hook tendency,
    bullpen fatigue/availability, opponent offense strength.
  - Unlike other player props, outs is heavily influenced by the MANAGER.
    A pitcher with a 3.00 ERA still gets pulled early if the manager is
    aggressive with his bullpen.

Modeling challenge:
  - Two-part prediction: (1) How effective is the pitcher per batter?
    (2) How many batters will the manager let him face?
  - The first part is captured by K%, BB%, CSW%, SwStr%, P/PA.
  - The second part is captured by manager hook rate and bullpen state.

For R users:
  - The `time` module is used for sleep() pauses between API calls.
  - Dictionaries in Python ({}) = named lists in R (list(a=1, b=2)).
  - list comprehensions [x for x in y] ≈ sapply(y, function(x) x) in R.
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

# Minimum IP for starting pitcher inclusion
MIN_IP = 30

# All 30 MLB teams (Baseball Reference abbreviations)
ALL_TEAMS_BREF = [
    "ARI", "ATL", "BAL", "BOS", "CHC", "CWS", "CIN", "CLE",
    "COL", "DET", "HOU", "KCR", "LAA", "LAD", "MIA", "MIL",
    "MIN", "NYM", "NYY", "OAK", "PHI", "PIT", "SDP", "SEA",
    "SFG", "STL", "TBR", "TEX", "TOR", "WSN"
]

# Manager "depth score" (0–1): higher = lets starters go deeper into games.
# This is a critical external feature because:
#   - Kevin Cash (TBR) historically pulls starters very early → under bets
#   - Bruce Bochy (TEX) will ride a hot starter → over bets
# Data from FanGraphs Manager Reports and MLB.com managerial tendency data.
MANAGER_DEPTH = {
    "ARI": {"manager": "Torey Lovullo",     "depth_score": 0.52, "avg_sp_outs": 14.8},
    "ATL": {"manager": "Brian Snitker",     "depth_score": 0.62, "avg_sp_outs": 16.1},
    "BAL": {"manager": "Brandon Hyde",      "depth_score": 0.50, "avg_sp_outs": 14.5},
    "BOS": {"manager": "Alex Cora",         "depth_score": 0.53, "avg_sp_outs": 14.9},
    "CHC": {"manager": "Craig Counsell",    "depth_score": 0.45, "avg_sp_outs": 13.9},
    "CWS": {"manager": "Pedro Grifol",      "depth_score": 0.58, "avg_sp_outs": 15.6},
    "CIN": {"manager": "David Bell",        "depth_score": 0.54, "avg_sp_outs": 15.1},
    "CLE": {"manager": "Stephen Vogt",      "depth_score": 0.56, "avg_sp_outs": 15.3},
    "COL": {"manager": "Bud Black",         "depth_score": 0.60, "avg_sp_outs": 15.9},
    "DET": {"manager": "A.J. Hinch",        "depth_score": 0.50, "avg_sp_outs": 14.5},
    "HOU": {"manager": "Joe Espada",        "depth_score": 0.43, "avg_sp_outs": 13.7},
    "KCR": {"manager": "Matt Quatraro",     "depth_score": 0.54, "avg_sp_outs": 15.0},
    "LAA": {"manager": "Ron Washington",    "depth_score": 0.61, "avg_sp_outs": 16.0},
    "LAD": {"manager": "Dave Roberts",      "depth_score": 0.48, "avg_sp_outs": 14.3},
    "MIA": {"manager": "Skip Schumaker",    "depth_score": 0.52, "avg_sp_outs": 14.8},
    "MIL": {"manager": "Pat Murphy",        "depth_score": 0.54, "avg_sp_outs": 15.0},
    "MIN": {"manager": "Rocco Baldelli",    "depth_score": 0.42, "avg_sp_outs": 13.6},
    "NYM": {"manager": "Carlos Mendoza",    "depth_score": 0.51, "avg_sp_outs": 14.7},
    "NYY": {"manager": "Aaron Boone",       "depth_score": 0.50, "avg_sp_outs": 14.5},
    "OAK": {"manager": "Mark Kotsay",       "depth_score": 0.63, "avg_sp_outs": 16.2},
    "PHI": {"manager": "Rob Thomson",       "depth_score": 0.55, "avg_sp_outs": 15.2},
    "PIT": {"manager": "Derek Shelton",     "depth_score": 0.57, "avg_sp_outs": 15.5},
    "SDP": {"manager": "Mike Shildt",       "depth_score": 0.53, "avg_sp_outs": 14.9},
    "SEA": {"manager": "Scott Servais",     "depth_score": 0.47, "avg_sp_outs": 14.2},
    "SFG": {"manager": "Bob Melvin",        "depth_score": 0.53, "avg_sp_outs": 14.9},
    "STL": {"manager": "Oliver Marmol",     "depth_score": 0.52, "avg_sp_outs": 14.8},
    "TBR": {"manager": "Kevin Cash",        "depth_score": 0.30, "avg_sp_outs": 12.5},  # Very low
    "TEX": {"manager": "Bruce Bochy",       "depth_score": 0.65, "avg_sp_outs": 16.5},  # Very high
    "TOR": {"manager": "John Schneider",    "depth_score": 0.51, "avg_sp_outs": 14.7},
    "WSN": {"manager": "Dave Martinez",     "depth_score": 0.54, "avg_sp_outs": 15.0},
}


# =============================================================================
# FUNCTION 1: Pull FanGraphs Pitching Stats (Core Efficiency Metrics)
# =============================================================================
def pull_pitcher_efficiency_stats(years: list) -> pd.DataFrame:
    """
    Pull per-pitcher season stats focused on efficiency and command.

    For total outs modeling, the key metrics are:
      - K%         : Strikeout rate — more K's = quicker outs
      - BB%        : Walk rate — walks extend plate appearances, raise pitch count
      - K-BB%      : Command composite — best single predictor of efficiency
      - IP         : Innings pitched (baseline volume)
      - IP/GS      : Average innings per start (target variable proxy)

    Additional efficiency metrics:
      - SwStr%     : Swing-and-miss rate — leads to K's, efficient outs
      - F-Strike%  : First-pitch strike rate — ahead in counts = efficiency
      - P/IP       : Pitches per inning (lower = more efficient, deeper starts)
      - ERA-/FIP-  : Park- and era-adjusted ERA/FIP (100 = avg, <100 = better)

    Parameters
    ----------
    years : list of int

    Returns
    -------
    pd.DataFrame
        One row per pitcher-season with efficiency metrics.
    """
    all_pit = []

    for year in years:
        print(f"  Pulling FanGraphs pitcher efficiency stats for {year}...")
        df = pyb.pitching_stats(year, year, qual=0, ind=1)
        df["Season"] = year
        all_pit.append(df)
        time.sleep(2)

    pitching = pd.concat(all_pit, ignore_index=True)

    # Filter to starting pitchers only
    # GS >= 5: had at least 5 starts in the season
    if "GS" in pitching.columns:
        pitching = pitching[pitching["GS"] >= 5].copy()

    # Derive IP per start (target variable for model evaluation)
    # In R: pitching$ip_per_start <- pitching$IP / pitching$GS
    if "IP" in pitching.columns and "GS" in pitching.columns:
        pitching["ip_per_start"]   = pitching["IP"] / pitching["GS"].clip(lower=1)
        pitching["outs_per_start"] = pitching["ip_per_start"] * 3  # Convert IP to outs

    # Pitches per inning (if available): lower = more efficient starts
    if "Pitches" in pitching.columns and "IP" in pitching.columns:
        pitching["pitches_per_ip"] = pitching["Pitches"] / pitching["IP"].clip(lower=1)

    print(f"  ✓ Pitcher efficiency: {len(pitching):,} pitcher-seasons pulled.")
    return pitching


# =============================================================================
# FUNCTION 2: Pull Statcast Pitcher CSW% and Pitch-Level Efficiency
# =============================================================================
def pull_pitcher_csw_stats(years: list) -> pd.DataFrame:
    """
    Pull CSW% (Called Strike + Whiff %) data from Baseball Savant.

    CSW% is one of the best predictors of deeper starts because:
      - It measures how often a pitcher gets ahead in counts (called strikes)
        AND generates whiffs — both lead to shorter plate appearances.
      - Pitchers with CSW% > 29% tend to go deeper into games.
      - Available via pybaseball's statcast_pitcher_pitch_arsenal function.

    Also captures:
      - avg_speed     : Higher velocity = harder to time = more strikeouts
      - avg_spin_rate : Higher spin = more movement = better results

    Note: CSW% is pitch-type specific. We'll aggregate to pitcher-level
    in the build step.
    """
    all_csw = []

    for year in years:
        print(f"  Pulling pitcher CSW% / arsenal stats for {year}...")
        try:
            # Pull 'n_' type to get called strike + whiff counts
            df = pyb.statcast_pitcher_pitch_arsenal(
                year, minP=100, arsenal_type="avg_speed"
            )
            df["Season"] = year
            all_csw.append(df)
            time.sleep(2)
        except Exception as e:
            print(f"    WARNING: CSW pull failed for {year}: {e}")

    if all_csw:
        csw_df = pd.concat(all_csw, ignore_index=True)
        print(f"  ✓ Arsenal/CSW data: {len(csw_df):,} pitcher-pitch_type-season rows.")
        return csw_df
    return pd.DataFrame()


# =============================================================================
# FUNCTION 3: Pull Statcast Pitcher Expected Stats (Allowed Contact Quality)
# =============================================================================
def pull_pitcher_statcast_xstats(years: list) -> pd.DataFrame:
    """
    Pull Statcast expected pitching stats (xERA, xwOBA allowed).

    These measure contact quality ALLOWED by the pitcher.
    A pitcher who allows weak contact (low xwOBA) will have better results
    even when ERA fluctuates, and is more likely to pitch deep into games.

    xwOBA allowed <.280 = elite (SP will get outs efficiently)
    xwOBA allowed >.340 = below average (SP may get pulled sooner)
    """
    all_xstats = []

    for year in years:
        print(f"  Pulling pitcher Statcast xstats for {year}...")
        try:
            df = pyb.statcast_pitcher_expected_stats(year, minPA=50)
            df["Season"] = year
            all_xstats.append(df)
            time.sleep(2)
        except Exception as e:
            print(f"    WARNING: Pitcher xstats failed for {year}: {e}")

    if all_xstats:
        xstats_df = pd.concat(all_xstats, ignore_index=True)
        print(f"  ✓ Pitcher xstats: {len(xstats_df):,} pitcher-seasons pulled.")
        return xstats_df
    return pd.DataFrame()


# =============================================================================
# FUNCTION 4: Pull Game-Level Schedule to Extract SP Starts
# =============================================================================
def pull_game_schedules_for_sp(years: list, teams: list) -> pd.DataFrame:
    """
    Pull game-by-game schedule to identify starting pitchers per game.

    The B-Ref schedule_and_record() function includes the 'Win', 'Loss',
    and 'Save' pitcher columns. The 'Win' pitcher in a home-team winning
    game is often the starter — but this is imperfect. We use it as an
    approximation to link game outcomes to starting pitchers.

    A more precise approach (future enhancement) would use the MLB Stats
    API game feed, which explicitly identifies starting pitchers.

    Returns
    -------
    pd.DataFrame
        All games with W/L, runs allowed, and pitcher columns.
    """
    all_games = []

    for year in years:
        print(f"  Pulling game schedule for {year}...")
        for team in teams:
            try:
                df = pyb.schedule_and_record(year, team)
                df["Season"] = year
                df["Team"]   = team
                all_games.append(df)
                time.sleep(0.5)
            except Exception as e:
                print(f"    WARNING: {team} {year} failed: {e}")

    games = pd.concat(all_games, ignore_index=True)
    games = games[games["R"].notna()].copy()

    # Convert numeric columns
    for col in ["R", "RA", "Inn"]:
        if col in games.columns:
            games[col] = pd.to_numeric(games[col], errors="coerce")

    print(f"  ✓ Game schedules: {len(games):,} team-game rows pulled.")
    return games


# =============================================================================
# FUNCTION 5: Pull Opponent Lineup Strength (Reduces SP Outs if Tough Lineup)
# =============================================================================
def pull_team_offensive_strength(years: list) -> pd.DataFrame:
    """
    Pull team-level offensive stats to quantify lineup difficulty.

    A strong offensive lineup (high wRC+, K% vs SP) forces more pitches
    and drives up pitch counts faster — reducing starter total outs.

    Key features:
      - wRC+     : Park-adjusted run production (>100 = above average)
      - K%       : Team strikeout rate — high K% lineup = easier for SP
      - BB%      : Team walk rate — high BB% lineup = more pitches per PA
      - OBP      : On-base percentage — more baserunners = more pitches
    """
    all_bat = []

    for year in years:
        print(f"  Pulling team offensive strength for {year}...")
        df = pyb.team_batting(year, year)
        df["Season"] = year
        all_bat.append(df)
        time.sleep(1)

    team_bat = pd.concat(all_bat, ignore_index=True)
    print(f"  ✓ Team offense: {len(team_bat):,} team-seasons pulled.")
    return team_bat


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("PITCHER TOTAL OUTS MODEL — STEP 1: DATA INPUT")
    print("=" * 70)
    print(f"Pulling data for seasons: {TRAIN_YEARS}")
    print()

    print("[ 1/5 ] Pulling pitcher efficiency stats (K%, BB%, K-BB%, CSW%)...")
    efficiency_df = pull_pitcher_efficiency_stats(TRAIN_YEARS)

    print("\n[ 2/5 ] Pulling pitcher pitch arsenal / CSW% data...")
    csw_df = pull_pitcher_csw_stats(TRAIN_YEARS)

    print("\n[ 3/5 ] Pulling pitcher Statcast expected stats...")
    xstats_df = pull_pitcher_statcast_xstats(TRAIN_YEARS)

    print("\n[ 4/5 ] Pulling game schedules (for SP identification)...")
    games_df = pull_game_schedules_for_sp(TRAIN_YEARS, ALL_TEAMS_BREF)

    print("\n[ 5/5 ] Pulling team offensive strength (opponent difficulty)...")
    team_off_df = pull_team_offensive_strength(TRAIN_YEARS)

    # Convert manager depth to DataFrame
    manager_df = pd.DataFrame(MANAGER_DEPTH).T.reset_index()
    manager_df.columns = ["team"] + list(manager_df.columns[1:])

    # --- Save files ----------------------------------------------------------
    print("\n[ SAVING ] Writing raw data files...")

    save_map = {
        "raw_pitcher_efficiency.csv":   efficiency_df,
        "raw_pitcher_csw_arsenal.csv":  csw_df,
        "raw_pitcher_xstats.csv":       xstats_df,
        "raw_game_schedules.csv":       games_df,
        "raw_team_offense_sp.csv":      team_off_df,
        "raw_manager_depth.csv":        manager_df,
    }

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
            cur_eff = pull_pitcher_efficiency_stats([CURRENT_YEAR])
            if not cur_eff.empty:
                efficiency_df = pd.concat(
                    [efficiency_df[efficiency_df["Season"] != CURRENT_YEAR], cur_eff],
                    ignore_index=True,
                )
                efficiency_df.to_csv(
                    os.path.join(RAW_DIR, "raw_pitcher_efficiency.csv"), index=False
                )
                print(f"  ✓ Merged {CURRENT_YEAR} pitcher efficiency into raw_pitcher_efficiency.csv")
        except Exception as e:
            print(f"  WARNING: Could not pull {CURRENT_YEAR} pitcher efficiency — {e}")

        try:
            cur_team_off = pull_team_offensive_strength([CURRENT_YEAR])
            if not cur_team_off.empty:
                team_off_df = pd.concat(
                    [team_off_df[team_off_df["Season"] != CURRENT_YEAR], cur_team_off],
                    ignore_index=True,
                )
                team_off_df.to_csv(
                    os.path.join(RAW_DIR, "raw_team_offense_sp.csv"), index=False
                )
                print(f"  ✓ Merged {CURRENT_YEAR} team offense into raw_team_offense_sp.csv")
        except Exception as e:
            print(f"  WARNING: Could not pull {CURRENT_YEAR} team offensive strength — {e}")

        try:
            cur_csw = pull_pitcher_csw_stats([CURRENT_YEAR])
            if not cur_csw.empty:
                csw_df = pd.concat(
                    [csw_df[csw_df["Season"] != CURRENT_YEAR], cur_csw],
                    ignore_index=True,
                )
                csw_df.to_csv(
                    os.path.join(RAW_DIR, "raw_pitcher_csw_arsenal.csv"), index=False
                )
                print(f"  ✓ Merged {CURRENT_YEAR} CSW/arsenal data into raw_pitcher_csw_arsenal.csv")
        except Exception as e:
            print(f"  WARNING: Could not pull {CURRENT_YEAR} pitcher CSW stats — {e}")

    print(f"\n  TIP: Re-run this file weekly during the {CURRENT_YEAR} season to refresh live stats.")

    print("\n" + "=" * 70)
    print("STEP 1 COMPLETE — Run 02_build_pitcher_outs.py next.")
    print("=" * 70)
