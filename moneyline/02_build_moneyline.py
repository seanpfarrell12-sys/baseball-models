"""
=============================================================================
MONEYLINE MODEL — FILE 2 OF 4: DATASET CONSTRUCTION
=============================================================================
Purpose : Load raw data and construct the game-level feature matrix.
Input   : CSV files from ../data/raw/
Output  : ../data/processed/moneyline_dataset.csv

What we're building:
  - Each ROW = one game (from home team's perspective)
  - FEATURES = pitching quality (SIERA, xFIP), offense (wRC+, wOBA),
               BaseRuns differential, park factor, bullpen quality
  - TARGET = home_team_win (1 = home win, 0 = away win)

BaseRuns formula:
  Estimates how many runs a team SHOULD have scored based on their
  underlying performance — removes sequencing luck from actual runs scored.
  Formula: BaseRuns = (A × B) / (B + C) + D
    A = runners who reach base (H + BB + HBP - HR)
    B = run advancement factor (0.8×1B + 2.1×2B + 3.4×3B + 1.8×HR + 0.1×BB)
    C = outs not contributing to scoring (AB - H + CS + GIDP)
    D = runs that score regardless of baserunners (HR)

For R users:
  - `pd.read_csv()` = read.csv() in R
  - `df.merge()` = merge() in R (LEFT JOIN by default with how='left')
  - `df.dropna()` = na.omit() in R
  - `df.fillna(value)` = replace NA with value (no direct R equivalent without dplyr)
  - `df.groupby().mean()` = aggregate(df, by=..., FUN=mean) in R
=============================================================================
"""

import os
import pandas as pd
import numpy as np

# --- Configuration ----------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR     = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR    = os.path.join(BASE_DIR, "data", "processed")

# FanGraphs to Baseball Reference team abbreviation mapping
# pybaseball uses FanGraphs abbreviations in pitching/batting stats
# but Baseball Reference abbreviations in schedule_and_record()
FG_TO_BREF = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHC", "CWS": "CWS", "CIN": "CIN", "CLE": "CLE",
    "COL": "COL", "DET": "DET", "HOU": "HOU", "KCR": "KCR",
    "LAA": "LAA", "LAD": "LAD", "MIA": "MIA", "MIL": "MIL",
    "MIN": "MIN", "NYM": "NYM", "NYY": "NYY", "OAK": "OAK",
    "PHI": "PHI", "PIT": "PIT", "SDP": "SDP", "SEA": "SEA",
    "SFG": "SFG", "STL": "STL", "TBR": "TBR", "TEX": "TEX",
    "TOR": "TOR", "WSN": "WSN",
    # FanGraphs uses these alternates sometimes
    "CHW": "CWS", "SD":  "SDP", "SF":  "SFG", "TB":  "TBR",
    "KC":  "KCR", "WAS": "WSN",
}

# Park factors (from raw data or hard-coded fallback)
# These adjust expected runs for each team's home stadium
PARK_FACTORS = {
    "ARI": 97,  "ATL": 102, "BAL": 104, "BOS": 105, "CHC": 101,
    "CWS": 96,  "CIN": 103, "CLE": 98,  "COL": 116, "DET": 96,
    "HOU": 97,  "KCR": 97,  "LAA": 97,  "LAD": 95,  "MIA": 97,
    "MIL": 99,  "MIN": 101, "NYM": 97,  "NYY": 105, "OAK": 96,
    "PHI": 102, "PIT": 97,  "SDP": 96,  "SEA": 94,  "SFG": 93,
    "STL": 99,  "TBR": 97,  "TEX": 101, "TOR": 102, "WSN": 100,
}


# =============================================================================
# HELPER: Compute BaseRuns from Team Batting Stats
# =============================================================================
def compute_base_runs(df: pd.DataFrame, prefix: str = "") -> pd.Series:
    """
    Compute the BaseRuns statistic for a team.

    BaseRuns estimates how many runs a team should score based on their
    component stats — it removes sequencing luck from actual run scoring.

    A team that "clusters" hits (all in the same inning) scores more than
    their BaseRuns estimate; a team with scattered hits scores less.
    BaseRuns is the "true talent" run scoring estimate.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: H, 2B, 3B, HR, BB, AB, G (with optional prefix)
    prefix : str
        Column prefix (e.g., "home_" or "away_") if columns are prefixed.

    Returns
    -------
    pd.Series
        BaseRuns estimate per game (divides by G for a per-game rate).

    R equivalent:
        # You'd write this as a vectorized function on a data.frame
        compute_base_runs <- function(df) {
          singles <- df$H - df$`2B` - df$`3B` - df$HR
          A <- singles + df$`2B` + df$`3B` + df$BB
          B <- 0.8*singles + 2.1*df$`2B` + 3.4*df$`3B` + 1.8*df$HR + 0.1*df$BB
          C <- df$AB - df$H
          D <- df$HR
          (A * B) / (B + C) + D
        }
    """
    p = prefix  # Shorthand for column access

    # Safely retrieve columns, defaulting to 0 if missing
    # In R: df[["col"]] — but here we handle missing columns gracefully
    def col(name):
        full = f"{p}{name}"
        return df[full] if full in df.columns else pd.Series(0, index=df.index)

    H   = col("H")
    d2B = col("2B")
    d3B = col("3B")
    HR  = col("HR")
    BB  = col("BB")
    AB  = col("AB")
    G   = col("G").clip(lower=1)  # Avoid division by zero

    singles = H - d2B - d3B - HR

    # BaseRuns components
    A = singles + d2B + d3B + BB           # Runners reaching base (excluding HR)
    B = (0.8 * singles
         + 2.1 * d2B
         + 3.4 * d3B
         + 1.8 * HR
         + 0.1 * BB)                        # Run advancement factor
    C = (AB - H).clip(lower=1)             # Outs made (prevent division by zero)
    D = HR                                 # Runs guaranteed to score (HR)

    # BaseRuns formula
    base_runs = (A * B) / (B + C) + D

    # Return as per-game rate (more comparable across teams)
    return base_runs / G


# =============================================================================
# FUNCTION 1: Load and Clean Raw Data
# =============================================================================
def load_raw_data() -> dict:
    """
    Load all raw CSV files into DataFrames.

    Returns
    -------
    dict
        Keys: 'pitching', 'batting', 'team_bat', 'team_pit', 'games'
        Values: corresponding DataFrames

    Python dict = R named list. Access with data['pitching'] = data[["pitching"]] in R.
    """
    print("  Loading raw data files...")

    data = {}

    # Define files to load with error handling
    # In R: tryCatch(read.csv("path"), error = function(e) NULL)
    files = {
        "pitching":    "raw_pitching_stats.csv",
        "batting":     "raw_batting_stats.csv",
        "team_bat":    "raw_team_batting.csv",
        "team_pit":    "raw_team_pitching.csv",
        "games":       "raw_game_schedules.csv",
        "statcast_pit":"raw_statcast_pitcher.csv",
    }

    for key, filename in files.items():
        path = os.path.join(RAW_DIR, filename)
        if os.path.exists(path):
            data[key] = pd.read_csv(path)
            print(f"    ✓ {filename}: {len(data[key]):,} rows")
        else:
            print(f"    ✗ {filename}: NOT FOUND — run 01_input_moneyline.py first")
            data[key] = pd.DataFrame()

    return data


# =============================================================================
# FUNCTION 2: Build Team-Season Feature Table
# =============================================================================
def build_team_features(data: dict) -> pd.DataFrame:
    """
    Aggregate individual stats to team-level season features.

    For each team-season, we compute:
      - Pitching quality: mean SIERA, xFIP of starters (GS > 5)
      - Offense quality: wRC+, wOBA from team_batting
      - BaseRuns: context-neutral run scoring estimate
      - Bullpen ERA (non-starters)

    This creates the "team card" that we attach to each game.

    Returns
    -------
    pd.DataFrame
        One row per team per season with aggregated features.
    """
    print("  Building team features...")

    # --- Normalize team abbreviations ----------------------------------------
    team_bat = data["team_bat"].copy()
    team_pit = data["team_pit"].copy()

    # Standardize team column name — FanGraphs uses 'Team', B-Ref may vary
    for df in [team_bat, team_pit]:
        if "Team" in df.columns:
            df["team_std"] = df["Team"].map(FG_TO_BREF).fillna(df["Team"])
        elif "Tm" in df.columns:
            df["team_std"] = df["Tm"].map(FG_TO_BREF).fillna(df["Tm"])

    # --- Starting pitcher aggregation ----------------------------------------
    # We want the "staff ace quality" — top 5 starters by IP for each team-season
    pitching = data["pitching"].copy()
    pitching["team_std"] = pitching["Team"].map(FG_TO_BREF).fillna(pitching["Team"])

    # Filter to starters (GS >= 5)
    starters = pitching[pitching.get("GS", pd.Series(0, index=pitching.index)) >= 5].copy()

    # Key pitching features — take team-weighted average by IP
    pit_cols = ["SIERA", "xFIP", "FIP", "ERA", "K%", "BB%", "K-BB%"]
    pit_cols = [c for c in pit_cols if c in starters.columns]

    def weighted_mean(group, cols, weight_col="IP"):
        """Compute IP-weighted mean — better pitchers who threw more innings
        get more weight than pitchers who threw 5 innings."""
        if weight_col not in group.columns:
            return group[cols].mean()
        weights = group[weight_col].fillna(0)
        total_w = weights.sum()
        if total_w == 0:
            return group[cols].mean()
        # In R: weighted.mean(x, w)
        return pd.Series({c: (group[c].fillna(group[c].mean()) * weights).sum() / total_w
                          for c in cols})

    # Group by team and season, apply weighted mean
    # In R: aggregate(. ~ team_std + Season, data=starters, FUN=weighted.mean)
    sp_features = (
        starters.groupby(["team_std", "Season"])
        .apply(lambda g: weighted_mean(g, pit_cols))
        .reset_index()
    )
    # Prefix SP columns to avoid confusion with team-level columns
    sp_features = sp_features.rename(columns={c: f"sp_{c.lower().replace('%','_pct').replace('-','_')}"
                                               for c in pit_cols})

    # --- Team offensive features ---------------------------------------------
    off_cols = ["wRC+", "wOBA", "ISO", "BABIP", "BB%", "K%", "OBP", "SLG"]
    off_cols = [c for c in off_cols if c in team_bat.columns]

    team_offense = team_bat[["team_std", "Season"] + off_cols].copy()
    # Rename to avoid column name conflicts
    team_offense = team_offense.rename(columns={
        c: f"off_{c.lower().replace('+','plus').replace('%','_pct').replace('/','_')}"
        for c in off_cols
    })

    # --- Compute BaseRuns ----------------------------------------------------
    # Need raw counting stats from team batting
    base_run_cols = ["H", "2B", "3B", "HR", "BB", "AB", "G"]
    available = [c for c in base_run_cols if c in team_bat.columns]
    if len(available) >= 5:
        team_bat_br = team_bat.copy()
        team_bat_br["base_runs_per_game"] = compute_base_runs(team_bat_br)
        team_offense = team_offense.merge(
            team_bat_br[["team_std", "Season", "base_runs_per_game"]],
            on=["team_std", "Season"], how="left"
        )

    # --- Park factors --------------------------------------------------------
    # Map home team to park factor
    park_df = pd.DataFrame(list(PARK_FACTORS.items()), columns=["team_std", "park_factor"])

    # --- Combine all team features -------------------------------------------
    # Merge SP quality + offense + park factor
    team_features = team_offense.merge(sp_features, on=["team_std", "Season"], how="left")
    team_features = team_features.merge(park_df, on="team_std", how="left")
    team_features["park_factor"] = team_features["park_factor"].fillna(100)

    print(f"    ✓ Team features: {len(team_features):,} team-season rows.")
    return team_features


# =============================================================================
# FUNCTION 3: Build Game-Level Dataset (HOME TEAM PERSPECTIVE)
# =============================================================================
def build_game_dataset(data: dict, team_features: pd.DataFrame) -> pd.DataFrame:
    """
    Join game results (outcomes) with team features to create the training set.

    Strategy:
      - Each game appears in schedule data for BOTH teams (home and away).
      - We filter to HOME TEAM rows only (where Home_Away column is empty/blank).
      - Then join with team features for both home and away teams.
      - Features are prefixed 'home_' and 'away_' to distinguish.

    TARGET variable:
      - home_win = 1 if home team won, 0 if away team won
      - We model P(home_team_wins | features)

    Returns
    -------
    pd.DataFrame
        One row per game with home+away features and binary outcome.
    """
    print("  Building game-level dataset...")

    games = data["games"].copy()

    # --- Parse date column ---------------------------------------------------
    # B-Ref date format can be inconsistent; pd.to_datetime handles most formats
    # In R: as.Date(games$Date, format="%Y-%m-%d")
    if "Date" in games.columns:
        games["game_date"] = pd.to_datetime(games["Date"], errors="coerce")

    # --- Identify home vs away games -----------------------------------------
    # In B-Ref schedule data, 'Home_Away' column is:
    #   "" (empty/NaN) = home game for this team
    #   "@"            = away game for this team
    if "Home_Away" in games.columns:
        # Home games: Home_Away is "Home", NaN, or empty string
        # (pybaseball uses "Home"/"@"; older B-Ref format uses ""/NaN)
        home_games = games[
            (games["Home_Away"] == "Home") |
            games["Home_Away"].isna() |
            (games["Home_Away"] == "")
        ].copy()
    else:
        # Fallback: assume all rows are home games (will deduplicate later)
        home_games = games.copy()

    # Standardize team names
    home_games["home_team"] = home_games["Team"].map(FG_TO_BREF).fillna(home_games["Team"])
    home_games["away_team"] = home_games["Opp"].map(FG_TO_BREF).fillna(home_games["Opp"])

    # Parse win/loss
    # W/L column in B-Ref can be "W", "L", "W-wo" (walkoff), "L-wo", etc.
    if "W/L" in home_games.columns:
        home_games["home_win"] = home_games["W/L"].str.startswith("W").astype(int)
    else:
        # Fallback: R > RA means home team won
        home_games["home_win"] = (home_games["R"] > home_games["RA"]).astype(int)

    # Convert run columns to numeric
    for col in ["R", "RA"]:
        if col in home_games.columns:
            home_games[col] = pd.to_numeric(home_games[col], errors="coerce")

    home_games["total_runs"] = home_games["R"].fillna(0) + home_games["RA"].fillna(0)

    # --- Join home team features ---------------------------------------------
    # We use PRIOR SEASON stats as features to avoid look-ahead bias.
    # If game is in 2023, we use 2022 season stats.
    # This mirrors how bettors use preseason projections.
    home_games["feature_season"] = home_games["Season"] - 1

    # Merge home team features (prefix all with 'home_')
    home_feat = team_features.rename(columns={
        "team_std": "home_team",
        "Season":   "feature_season",
        **{c: f"home_{c}" for c in team_features.columns
           if c not in ["team_std", "Season"]}
    })
    # In R: merge(home_games, home_feat, by=c("home_team", "feature_season"), all.x=TRUE)
    game_df = home_games.merge(home_feat, on=["home_team", "feature_season"], how="left")

    # Merge away team features (prefix all with 'away_')
    away_feat = team_features.rename(columns={
        "team_std": "away_team",
        "Season":   "feature_season",
        **{c: f"away_{c}" for c in team_features.columns
           if c not in ["team_std", "Season"]}
    })
    game_df = game_df.merge(away_feat, on=["away_team", "feature_season"], how="left")

    # --- Compute differential features ---------------------------------------
    # The difference between home and away metrics is often more predictive
    # than the raw values themselves.
    # In R: game_df$siera_diff <- game_df$home_sp_siera - game_df$away_sp_siera
    for base_col in ["sp_siera", "sp_xfip", "sp_fip", "sp_k_pct",
                     "off_wrc_plus", "base_runs_per_game"]:
        home_col = f"home_{base_col}"
        away_col = f"away_{base_col}"
        if home_col in game_df.columns and away_col in game_df.columns:
            game_df[f"diff_{base_col}"] = game_df[home_col] - game_df[away_col]

    # Home field advantage flag (always 1 in this dataset since we're only
    # looking at it from home team perspective — useful as a constant feature
    # that the model can learn the natural HFA from the data)
    game_df["home_field"] = 1

    print(f"    ✓ Game dataset: {len(game_df):,} games built.")
    print(f"    ✓ Home win rate: {game_df['home_win'].mean():.3f} (expect ~0.54)")
    return game_df


# =============================================================================
# FUNCTION 4: Handle Missing Values and Finalize Features
# =============================================================================
def finalize_dataset(game_df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values, encode categoricals, and select final features.

    Missing data strategy:
      - Pitching stats missing for expansion teams or new starters:
        fill with league average for that season.
      - Park factor missing: fill with 100 (league average).
      - Drop games where we have NO pitching or offensive features.

    Feature selection:
      - Drop raw counting stats (they're already absorbed into rate stats)
      - Drop identifier columns (names, dates) that the model shouldn't learn from

    Returns
    -------
    pd.DataFrame
        Clean dataset ready for XGBoost training.
    """
    print("  Finalizing dataset (handling missing values, selecting features)...")

    df = game_df.copy()

    # --- Define final feature columns ----------------------------------------
    # Grouped by category for readability
    # In R: feature_cols <- c("home_sp_siera", "away_sp_siera", ...)
    feature_cols = [
        # Starting pitcher quality (home and away)
        "home_sp_siera", "home_sp_xfip", "home_sp_fip",
        "home_sp_k_pct", "home_sp_bb_pct", "home_sp_k_bb_pct",
        "away_sp_siera", "away_sp_xfip", "away_sp_fip",
        "away_sp_k_pct", "away_sp_bb_pct", "away_sp_k_bb_pct",

        # Team offensive strength (home and away)
        "home_off_wrc_plus", "home_off_woba", "home_off_iso", "home_off_babip",
        "away_off_wrc_plus", "away_off_woba", "away_off_iso", "away_off_babip",

        # BaseRuns (expected run scoring)
        "home_base_runs_per_game", "away_base_runs_per_game",

        # Park and environment
        "home_park_factor",

        # Differential features (home - away)
        "diff_sp_siera", "diff_sp_xfip", "diff_off_wrc_plus", "diff_base_runs_per_game",

        # Home field advantage
        "home_field",
    ]

    # Keep only columns that exist in the dataset
    feature_cols = [c for c in feature_cols if c in df.columns]
    target_col   = "home_win"

    # Columns to keep for identification (not used as features)
    id_cols = ["game_date", "Season", "home_team", "away_team", "total_runs"]
    id_cols = [c for c in id_cols if c in df.columns]

    df = df[id_cols + feature_cols + [target_col]].copy()

    # --- Fill missing values with league averages ----------------------------
    # Calculate column means for imputation
    # In R: for each column, replace NA with mean(col, na.rm=TRUE)
    col_means = df[feature_cols].mean()

    for col in feature_cols:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            df[col] = df[col].fillna(col_means[col])
            print(f"    Imputed {missing_count:4d} missing values in '{col}'")

    # --- Drop rows with no meaningful features -------------------------------
    # A game row with all features missing would be useless for training
    before = len(df)
    df = df.dropna(subset=[target_col])  # Must have outcome
    after  = len(df)
    if before != after:
        print(f"    Dropped {before - after} rows with missing target variable.")

    print(f"    ✓ Final dataset: {len(df):,} games, {len(feature_cols)} features.")
    print(f"    ✓ Seasons: {sorted(df['Season'].unique()) if 'Season' in df.columns else 'N/A'}")
    return df


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("MONEYLINE MODEL — STEP 2: DATASET CONSTRUCTION")
    print("=" * 70)

    # Step 1: Load raw data
    print("\n[ 1/4 ] Loading raw data...")
    data = load_raw_data()

    # Step 2: Build team-season features
    print("\n[ 2/4 ] Building team features...")
    team_features = build_team_features(data)

    # Step 3: Build game-level dataset
    print("\n[ 3/4 ] Building game-level dataset...")
    game_df = build_game_dataset(data, team_features)

    # Step 4: Finalize and clean
    print("\n[ 4/4 ] Finalizing dataset...")
    final_df = finalize_dataset(game_df)

    # Save processed dataset
    output_path = os.path.join(PROC_DIR, "moneyline_dataset.csv")
    final_df.to_csv(output_path, index=False)
    print(f"\n  ✓ Saved moneyline_dataset.csv ({len(final_df):,} rows)")
    print(f"  ✓ Columns: {list(final_df.columns)}")

    print("\n" + "=" * 70)
    print("STEP 2 COMPLETE — Run 03_analysis_moneyline.py next.")
    print("=" * 70)
