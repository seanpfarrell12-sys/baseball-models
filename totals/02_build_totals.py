"""
=============================================================================
OVER/UNDER TOTALS MODEL — FILE 2 OF 4: DATASET CONSTRUCTION
=============================================================================
Purpose : Build the game-level feature matrix for Poisson regression.
Input   : CSV files from ../data/raw/
Output  : ../data/processed/totals_dataset.csv

What we're building:
  - Each ROW = one game
  - FEATURES = team offense (wRC+, wOBA), SP quality (SIERA, xFIP),
               park factor (elevation, dimensions), weather (temp, wind),
               umpire tendencies, surface type
  - TARGET = total_runs (combined runs scored by both teams)

Why Poisson regression?
  - Runs are COUNT DATA: 0, 1, 2, 3, ... (non-negative integers)
  - Poisson regression models the RATE of event occurrence
  - Much more appropriate than linear regression for count outcomes
  - R equivalent: glm(total_runs ~ features, data=df, family=poisson(link="log"))
  - We model each team's runs separately, then sum for total

Weather treatment:
  - For HISTORICAL training data: we use seasonal temperature averages
    by city (a reasonable proxy when exact historical weather isn't available)
  - For SCORING upcoming games: we use the NWS weather API (01_input file)

For R users:
  - `pd.get_dummies()` = model.matrix() or one-hot encoding in R
  - `.astype(int)` = as.integer() in R
  - `df.assign()` = mutate() in dplyr
=============================================================================
"""

import os
import json
import pandas as pd
import numpy as np

# --- Configuration ----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")

# Team abbreviation mapping (FanGraphs → Baseball Reference)
FG_TO_BREF = {
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

# Historical average game-time temperatures (°F) by city — seasonal proxy.
# Used in training data when exact historical weather is unavailable.
# These are average April–September temperatures for each MLB city.
CITY_AVG_TEMP = {
    "ARI": 88, "ATL": 75, "BAL": 72, "BOS": 65, "CHC": 68,
    "CWS": 70, "CIN": 73, "CLE": 66, "COL": 70, "DET": 67,
    "HOU": 82, "KCR": 76, "LAA": 75, "LAD": 73, "MIA": 84,
    "MIL": 66, "MIN": 67, "NYM": 71, "NYY": 71, "OAK": 62,
    "PHI": 72, "PIT": 68, "SDP": 68, "SEA": 62, "SFG": 60,
    "STL": 76, "TBR": 82, "TEX": 85, "TOR": 67, "WSN": 74,
}

# Whether stadium is covered (roof) — affects weather impact
COVERED_STADIUMS = {"ARI", "HOU", "MIA", "MIL", "SEA", "TBR", "TOR"}

# Natural grass vs artificial turf — turf tends to increase batting avg on GB
ARTIFICIAL_TURF = {"ARI", "MIA", "TBR", "TOR"}

# Stadium altitude (feet above sea level) — higher altitude = thinner air = more HRs
STADIUM_ALTITUDE = {
    "ARI":  1082, "ATL": 1050, "BAL":  25, "BOS":  21, "CHC":  595,
    "CWS":  595,  "CIN":  490, "CLE":  660, "COL": 5280, "DET":  601,
    "HOU":   43,  "KCR": 908, "LAA":  160, "LAD":  512, "MIA":    6,
    "MIL":  635,  "MIN": 830, "NYM":   16, "NYY":   55, "OAK":   25,
    "PHI":   40,  "PIT": 730, "SDP":   17, "SEA":   11, "SFG":   10,
    "STL":  466,  "TBR":  28, "TEX":  551, "TOR":   76, "WSN":   25,
}


# =============================================================================
# FUNCTION 1: Load Raw Data
# =============================================================================
def load_raw_data() -> dict:
    """Load all raw CSVs needed for totals model construction."""
    print("  Loading raw data files...")
    data = {}
    files = {
        "games":    "raw_game_schedules.csv",
        "team_bat": "raw_team_batting.csv",
        "team_pit": "raw_team_pitching.csv",
        "sp_stats": "raw_sp_stats.csv",
        "park":     "raw_park_factors.csv",
    }
    for key, fname in files.items():
        path = os.path.join(RAW_DIR, fname)
        if os.path.exists(path):
            data[key] = pd.read_csv(path)
            print(f"    ✓ {fname}: {len(data[key]):,} rows")
        else:
            print(f"    ✗ {fname}: NOT FOUND — run 01_input_totals.py first")
            data[key] = pd.DataFrame()

    # Load park factors JSON if available (more detailed than CSV)
    json_path = os.path.join(RAW_DIR, "park_factors.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            data["park_json"] = json.load(f)
    return data


# =============================================================================
# FUNCTION 2: Build Team Offense/Defense Features per Season
# =============================================================================
def build_team_context(data: dict) -> pd.DataFrame:
    """
    Build a team-season lookup table with offensive and pitching metrics.

    This gets joined to each game row (home team and away team separately)
    to create the full feature set for each game.

    Key features:
      Offense: wRC+, wOBA, ISO — how many runs this team's lineup generates
      Defense/Pitching: SIERA, xFIP — how well this team's staff suppresses runs
      Environment: park factor, altitude, surface type

    Returns
    -------
    pd.DataFrame
        One row per team-season with offense + pitching + environment features.
    """
    print("  Building team context features...")

    team_bat = data["team_bat"].copy()
    team_pit = data["team_pit"].copy()

    # Standardize team column
    for df in [team_bat, team_pit]:
        if "Team" in df.columns:
            df["team_std"] = df["Team"].map(FG_TO_BREF).fillna(df["Team"])
        elif "Tm" in df.columns:
            df["team_std"] = df["Tm"].map(FG_TO_BREF).fillna(df["Tm"])
        else:
            df["team_std"] = "UNK"

    # Select offensive columns
    bat_cols = ["team_std", "Season", "wRC+", "wOBA", "ISO", "BABIP",
                "BB%", "K%", "OBP", "SLG", "AVG", "HR", "R", "G"]
    bat_cols = [c for c in bat_cols if c in team_bat.columns]
    team_bat_sub = team_bat[bat_cols].copy()

    # Rename with prefix to distinguish
    rename_bat = {c: f"team_off_{c.lower().replace('+','plus').replace('%','_pct').replace('/','_')}"
                  for c in bat_cols if c not in ["team_std", "Season"]}
    team_bat_sub = team_bat_sub.rename(columns=rename_bat)

    # Select pitching columns
    pit_cols = ["team_std", "Season", "ERA", "FIP", "xFIP", "SIERA",
                "K%", "BB%", "K-BB%", "HR/9", "H/9"]
    pit_cols = [c for c in pit_cols if c in team_pit.columns]
    team_pit_sub = team_pit[pit_cols].copy()

    rename_pit = {c: f"team_pit_{c.lower().replace('%','_pct').replace('/','_').replace('-','_')}"
                  for c in pit_cols if c not in ["team_std", "Season"]}
    team_pit_sub = team_pit_sub.rename(columns=rename_pit)

    # Merge offense + pitching for each team
    team_ctx = team_bat_sub.merge(
        team_pit_sub, on=["team_std", "Season"], how="outer"
    )

    # Add static environment features (these don't change by season)
    team_ctx["park_factor"] = team_ctx["team_std"].map(
        lambda t: data["park_json"].get(t, {}).get("pf_runs", 100)
        if "park_json" in data else 100
    )
    team_ctx["altitude"]     = team_ctx["team_std"].map(STADIUM_ALTITUDE).fillna(100)
    team_ctx["covered"]      = team_ctx["team_std"].isin(COVERED_STADIUMS).astype(int)
    team_ctx["artificial"]   = team_ctx["team_std"].isin(ARTIFICIAL_TURF).astype(int)
    team_ctx["avg_temp"]     = team_ctx["team_std"].map(CITY_AVG_TEMP).fillna(72)

    # Compute expected runs per game from park-adjusted offense
    # A quick estimate: wRC+ / 100 × league average runs per game (~4.5)
    if "team_off_wrc_plus" in team_ctx.columns:
        team_ctx["expected_rpg"] = (team_ctx["team_off_wrc_plus"].fillna(100) / 100) * 4.5

    print(f"    ✓ Team context: {len(team_ctx):,} team-season rows.")
    return team_ctx


# =============================================================================
# FUNCTION 3: Build Starting Pitcher Features per Game
# =============================================================================
def build_sp_features(data: dict) -> pd.DataFrame:
    """
    Build starting pitcher quality features for each team-season.

    For the totals model, the SP is the single most important factor.
    A dominant SP (low SIERA) dramatically lowers expected total runs.

    In daily scoring mode, you'd input the specific confirmed SP.
    In training mode, we use each team's top SP stats as a proxy.

    Returns
    -------
    pd.DataFrame
        One row per team-season with SP quality metrics.
    """
    print("  Building SP features...")

    sp = data["sp_stats"].copy()
    if sp.empty:
        print("    WARNING: SP stats empty — skipping SP features.")
        return pd.DataFrame()

    # Standardize team
    if "Team" in sp.columns:
        sp["team_std"] = sp["Team"].map(FG_TO_BREF).fillna(sp["Team"])

    # Filter to starters with meaningful IP
    if "GS" in sp.columns:
        sp = sp[sp["GS"] >= 5].copy()
    if "IP" in sp.columns:
        sp = sp[pd.to_numeric(sp["IP"], errors="coerce") >= 20].copy()

    # IP-weighted average of SP quality for each team-season
    sp["IP"] = pd.to_numeric(sp.get("IP", pd.Series(1, index=sp.index)), errors="coerce").fillna(1)

    metric_cols = ["SIERA", "xFIP", "FIP", "ERA", "K%", "BB%", "K-BB%", "SwStr%"]
    metric_cols = [c for c in metric_cols if c in sp.columns]

    def ip_weighted_avg(group):
        """IP-weighted mean of pitching metrics for a team's rotation."""
        w = group["IP"].fillna(0)
        total_w = w.sum()
        if total_w == 0:
            return pd.Series({f"sp_{m.lower().replace('%','_pct').replace('-','_')}":
                               group[m].mean() for m in metric_cols})
        return pd.Series({
            f"sp_{m.lower().replace('%','_pct').replace('-','_')}":
            (group[m].fillna(group[m].mean()) * w).sum() / total_w
            for m in metric_cols
        })

    sp_features = (
        sp.groupby(["team_std", "Season"])
        .apply(ip_weighted_avg)
        .reset_index()
    )

    print(f"    ✓ SP features: {len(sp_features):,} team-season rows.")
    return sp_features


# =============================================================================
# FUNCTION 4: Build Full Game-Level Training Dataset
# =============================================================================
def build_game_dataset(data: dict, team_ctx: pd.DataFrame, sp_features: pd.DataFrame) -> pd.DataFrame:
    """
    Join game results with team context and SP features to build training rows.

    Each game row gets:
      - Home team offense/pitching features
      - Away team offense/pitching features
      - Home team park/weather context
      - Combined total runs (TARGET variable for Poisson regression)

    Returns
    -------
    pd.DataFrame
        One row per game with all features and total_runs target.
    """
    print("  Building game-level training dataset...")

    games = data["games"].copy()

    # Keep only home team rows (each game listed twice in schedule data)
    if "Home_Away" in games.columns:
        home_games = games[
            (games["Home_Away"] == "Home") |
            games["Home_Away"].isna() |
            (games["Home_Away"] == "")
        ].copy()
    else:
        home_games = games.copy()

    # Standardize team names
    home_games["home_team"] = home_games["Team"].map(FG_TO_BREF).fillna(home_games["Team"])
    home_games["away_team"] = home_games["Opp"].map(FG_TO_BREF).fillna(home_games["Opp"])

    # Parse runs — these are our labels
    home_games["home_runs"] = pd.to_numeric(home_games["R"],  errors="coerce")
    home_games["away_runs"] = pd.to_numeric(home_games["RA"], errors="coerce")
    home_games["total_runs"] = home_games["home_runs"] + home_games["away_runs"]

    # Drop games with no run data
    home_games = home_games.dropna(subset=["total_runs"])

    # Use prior season features to avoid look-ahead bias
    home_games["feature_season"] = home_games["Season"] - 1

    # --- Merge home team context (prior year offense + pitching) ------------
    home_ctx = team_ctx.rename(columns={
        "team_std": "home_team",
        "Season":   "feature_season",
        **{c: f"home_{c}" for c in team_ctx.columns if c not in ["team_std", "Season"]}
    })
    game_df = home_games.merge(home_ctx, on=["home_team", "feature_season"], how="left")

    # --- Merge away team context (prior year offense) -----------------------
    away_ctx = team_ctx.rename(columns={
        "team_std": "away_team",
        "Season":   "feature_season",
        **{c: f"away_{c}" for c in team_ctx.columns if c not in ["team_std", "Season"]}
    })
    game_df = game_df.merge(away_ctx, on=["away_team", "feature_season"], how="left")

    # --- Merge SP features --------------------------------------------------
    if not sp_features.empty:
        home_sp = sp_features.rename(columns={
            "team_std": "home_team",
            "Season":   "feature_season",
            **{c: f"home_{c}" for c in sp_features.columns
               if c not in ["team_std", "Season"]}
        })
        game_df = game_df.merge(home_sp, on=["home_team", "feature_season"], how="left")

        away_sp = sp_features.rename(columns={
            "team_std": "away_team",
            "Season":   "feature_season",
            **{c: f"away_{c}" for c in sp_features.columns
               if c not in ["team_std", "Season"]}
        })
        game_df = game_df.merge(away_sp, on=["away_team", "feature_season"], how="left")

    # --- Add differential features ------------------------------------------
    diff_pairs = [
        ("home_team_off_wrc_plus",  "away_team_off_wrc_plus"),
        ("home_team_pit_siera",     "away_team_pit_siera"),
        ("home_sp_siera",           "away_sp_siera"),
        ("home_expected_rpg",       "away_expected_rpg"),
    ]
    for home_col, away_col in diff_pairs:
        if home_col in game_df.columns and away_col in game_df.columns:
            diff_name = f"diff_{home_col.replace('home_','')}"
            game_df[diff_name] = game_df[home_col] - game_df[away_col]

    # Combined offensive score = (home wRC+ + away wRC+) / 2
    if "home_team_off_wrc_plus" in game_df.columns and "away_team_off_wrc_plus" in game_df.columns:
        game_df["combined_wrc_plus"] = (
            game_df["home_team_off_wrc_plus"].fillna(100)
            + game_df["away_team_off_wrc_plus"].fillna(100)
        ) / 2

    # Combined SP quality (lower SIERA = better pitching matchup = lower total)
    if "home_sp_siera" in game_df.columns and "away_sp_siera" in game_df.columns:
        game_df["combined_sp_siera"] = (
            game_df["home_sp_siera"].fillna(4.20)
            + game_df["away_sp_siera"].fillna(4.20)
        ) / 2

    # Temperature effect on run scoring (for training, use city average temp)
    game_df["temperature_f"] = game_df["home_team"].map(CITY_AVG_TEMP).fillna(72)

    # Binary indicators for extreme weather conditions
    game_df["is_hot_game"]  = (game_df["temperature_f"] > 85).astype(int)
    game_df["is_cold_game"] = (game_df["temperature_f"] < 60).astype(int)

    # Altitude effect — Coors Field is the dominant factor here
    game_df["altitude"] = game_df["home_team"].map(STADIUM_ALTITUDE).fillna(100)
    game_df["is_coors"] = (game_df["home_team"] == "COL").astype(int)

    print(f"    ✓ Game dataset: {len(game_df):,} games.")
    print(f"    ✓ Mean total runs: {game_df['total_runs'].mean():.2f}")
    print(f"    ✓ Total runs range: {game_df['total_runs'].min():.0f}–{game_df['total_runs'].max():.0f}")
    return game_df


# =============================================================================
# FUNCTION 5: Finalize and Select Features
# =============================================================================
def finalize_dataset(game_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and finalize the dataset for Poisson regression.

    Final feature set is designed to capture:
      1. Combined offensive firepower (wRC+, wOBA, ISO)
      2. Combined pitching suppression (SIERA, xFIP)
      3. Environmental factors (park factor, altitude, temperature)
      4. Surface type (turf vs grass)
    """
    print("  Finalizing totals dataset...")

    feature_cols = [
        # Home team features
        "home_team_off_wrc_plus", "home_team_off_woba", "home_team_off_iso",
        "home_team_pit_era", "home_team_pit_xfip", "home_sp_siera", "home_sp_k_pct",

        # Away team features
        "away_team_off_wrc_plus", "away_team_off_woba", "away_team_off_iso",
        "away_team_pit_era", "away_team_pit_xfip", "away_sp_siera", "away_sp_k_pct",

        # Combined features (sum of both teams)
        "combined_wrc_plus", "combined_sp_siera",

        # Park and environment
        "home_park_factor", "altitude", "is_coors",
        "home_covered", "home_artificial",

        # Weather proxy
        "temperature_f", "is_hot_game", "is_cold_game",

        # Differential
        "diff_team_off_wrc_plus", "diff_sp_siera",
    ]

    target_col = "total_runs"
    id_cols    = ["game_date", "Season", "home_team", "away_team",
                  "home_runs", "away_runs"]
    id_cols    = [c for c in id_cols if c in game_df.columns]

    # Keep columns that exist
    feature_cols = [c for c in feature_cols if c in game_df.columns]
    keep = id_cols + feature_cols + [target_col]
    keep = [c for c in keep if c in game_df.columns]

    df = game_df[keep].copy()

    # Impute missing features with column means
    col_means = df[feature_cols].mean()
    for col in feature_cols:
        df[col] = df[col].fillna(col_means[col])

    # Remove games with extreme totals (likely data errors: 0 runs or 50+)
    df = df[(df[target_col] >= 1) & (df[target_col] <= 35)]

    # Remove games with missing target
    df = df.dropna(subset=[target_col])

    print(f"    ✓ Final dataset: {len(df):,} games, {len(feature_cols)} features.")
    return df


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("TOTALS MODEL — STEP 2: DATASET CONSTRUCTION")
    print("=" * 70)

    print("\n[ 1/4 ] Loading raw data...")
    data = load_raw_data()

    print("\n[ 2/4 ] Building team context features...")
    team_ctx = build_team_context(data)

    print("\n[ 3/4 ] Building SP features...")
    sp_features = build_sp_features(data)

    print("\n[ 4/4 ] Building game-level dataset...")
    game_df  = build_game_dataset(data, team_ctx, sp_features)
    final_df = finalize_dataset(game_df)

    output_path = os.path.join(PROC_DIR, "totals_dataset.csv")
    final_df.to_csv(output_path, index=False)
    print(f"\n  ✓ Saved totals_dataset.csv ({len(final_df):,} rows)")

    print("\n" + "=" * 70)
    print("STEP 2 COMPLETE — Run 03_analysis_totals.py next.")
    print("=" * 70)
