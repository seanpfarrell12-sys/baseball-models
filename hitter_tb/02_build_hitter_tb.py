"""
=============================================================================
HITTER TOTAL BASES MODEL — FILE 2 OF 4: DATASET CONSTRUCTION
=============================================================================
Purpose : Build the player-season feature matrix for hitter total bases.
Input   : CSV files from ../data/raw/
Output  : ../data/processed/hitter_tb_dataset.csv

What we're building:
  - Each ROW = one batter-season (aggregated to get per-game rates)
  - FEATURES = Statcast quality of contact (Barrel%, EV, HardHit%),
               traditional production (ISO, wOBA), plate discipline (BB%, K%),
               matchup context (SP handedness, SP K%, platoon advantage)
  - TARGET = tb_per_game (average total bases per game for that player-season)

Daily scoring mode:
  - For prop bets, we predict individual player TB in a specific game.
  - We use the season-level averages + today's SP matchup as features.
  - The model outputs E[TB] which we compare to the market prop line.

Key insight: "launch angle tightness" (SD of launch angle) matters.
  - A player with avg LA of 15° could achieve this via many grounders + flyouts.
  - A player who consistently hits at 15° is a much better TB bet.
  - We approximate this using sweet spot% (proxy for LA consistency).

For R users:
  - `pd.merge(..., how='left')` = merge(x, y, all.x=TRUE) in R
  - Lambda functions `lambda x: x*2` = anonymous functions function(x) x*2 in R
  - `df.apply(func, axis=1)` = apply(df, 1, func) in R (row-wise)
=============================================================================
"""

import os
import pandas as pd
import numpy as np

# --- Configuration ----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")

# FanGraphs → Baseball Reference team abbreviation mapping
FG_TO_BREF = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHC", "CWS": "CWS", "CIN": "CIN", "CLE": "CLE",
    "COL": "COL", "DET": "DET", "HOU": "HOU", "KCR": "KCR",
    "LAA": "LAA", "LAD": "LAD", "MIA": "MIA", "MIL": "MIL",
    "MIN": "MIN", "NYM": "NYM", "NYY": "NYY", "OAK": "OAK",
    "PHI": "PHI", "PIT": "PIT", "SDP": "SDP", "SEA": "SEA",
    "SFG": "SFG", "STL": "STL", "TBR": "TBR", "TEX": "TEX",
    "TOR": "TOR", "WSN": "WSN", "CHW": "CWS", "SD": "SDP",
    "SF": "SFG", "TB": "TBR", "KC": "KCR", "WAS": "WSN",
}

# Park factors for HR/TB context
PARK_HR_FACTOR = {
    "ARI": 101, "ATL": 104, "BAL": 110, "BOS": 95,  "CHC": 104,
    "CWS": 99,  "CIN": 106, "CLE": 96,  "COL": 116, "DET": 93,
    "HOU": 94,  "KCR": 97,  "LAA": 99,  "LAD": 91,  "MIA": 95,
    "MIL": 100, "MIN": 107, "NYM": 97,  "NYY": 114, "OAK": 96,
    "PHI": 106, "PIT": 97,  "SDP": 91,  "SEA": 89,  "SFG": 86,
    "STL": 101, "TBR": 98,  "TEX": 104, "TOR": 109, "WSN": 104,
}

# Exit velocity expected outcomes table (from document)
# Maps avg EV band → expected SLG (used for cross-validation context)
EV_TO_EXPECTED_SLG = {
    115: 1.800, 110: 1.600, 105: 1.300, 100: 0.750, 95: 0.450
}


# =============================================================================
# FUNCTION 1: Load and Align All Raw Files
# =============================================================================
def load_raw_data() -> dict:
    """
    Load all raw data files from the data/raw directory.

    Player ID alignment is a key challenge — FanGraphs and Statcast use
    different player IDs. We align on Name + Season as a fallback.
    """
    print("  Loading raw data files...")
    data = {}
    files = {
        "ev_barrels":    "raw_hitter_ev_barrels.csv",
        "xstats":        "raw_hitter_xstats.csv",
        "batting":       "raw_hitter_batting.csv",
        "pitcher_xstats": "raw_pitcher_xstats.csv",
        "fg_pitcher":    "raw_fg_pitcher_stats.csv",
        "arsenal":       "raw_pitcher_arsenal.csv",
    }
    for key, fname in files.items():
        path = os.path.join(RAW_DIR, fname)
        if os.path.exists(path):
            data[key] = pd.read_csv(path)
            print(f"    ✓ {fname}: {len(data[key]):,} rows")
        else:
            print(f"    ✗ {fname}: NOT FOUND")
            data[key] = pd.DataFrame()
    return data


# =============================================================================
# FUNCTION 2: Build Batter-Season Feature Table
# =============================================================================
def build_batter_features(data: dict) -> pd.DataFrame:
    """
    Merge FanGraphs batting stats with Statcast EV/barrel and expected stats.

    The merge strategy:
      1. Start with FanGraphs batting stats (wide coverage, many metrics)
      2. Left-join Statcast EV/barrel data on player_id + Season
      3. Left-join Statcast expected stats on player_id + Season
      4. Compute derived features (TB per game, platoon indicator, etc.)

    Player ID alignment:
      - FanGraphs uses 'IDfg' (numeric)
      - Statcast uses 'player_id' (numeric, same as MLB MLBAM ID)
      - These are DIFFERENT ID systems. We use playerid_lookup() to map them,
        or fall back to name matching. The build file uses name matching
        as a pragmatic approach.

    Returns
    -------
    pd.DataFrame
        One row per batter-season with all features and TB target.
    """
    print("  Building batter features...")

    batting = data["batting"].copy()
    if batting.empty:
        print("    ERROR: batting data is empty. Cannot build batter features.")
        return pd.DataFrame()

    # Standardize team
    if "Team" in batting.columns:
        batting["team_std"] = batting["Team"].map(FG_TO_BREF).fillna(batting["Team"])

    # Ensure we have a clean name for joining
    if "Name" in batting.columns:
        # Normalize name format: remove accents etc. for consistent matching
        batting["name_clean"] = batting["Name"].str.lower().str.strip()

    # Compute or verify total bases and per-game rate
    if "total_bases" not in batting.columns:
        if all(c in batting.columns for c in ["H", "2B", "3B", "HR", "G"]):
            batting["singles"]      = batting["H"] - batting["2B"] - batting["3B"] - batting["HR"]
            batting["total_bases"]  = (batting["singles"]
                                       + 2 * batting["2B"]
                                       + 3 * batting["3B"]
                                       + 4 * batting["HR"])
            batting["tb_per_game"]  = batting["total_bases"] / batting["G"].clip(lower=1)
        else:
            batting["tb_per_game"]  = np.nan

    # Power metrics: ISO = SLG - AVG (isolated extra-base hit power)
    if "SLG" in batting.columns and "AVG" in batting.columns:
        batting["ISO"] = batting["SLG"] - batting["AVG"]

    # HR rate per game (key power indicator)
    if "HR" in batting.columns and "G" in batting.columns:
        batting["hr_per_game"]  = batting["HR"] / batting["G"].clip(lower=1)
        batting["xbh_per_game"] = (batting["2B"] + batting["3B"] + batting["HR"]) \
                                   / batting["G"].clip(lower=1) if "2B" in batting.columns else np.nan

    # --- Merge Statcast EV/barrel data ---------------------------------------
    ev_df = data["ev_barrels"].copy()
    if not ev_df.empty:
        # Clean up player name column (Statcast format: "last_name, first_name")
        if "last_name, first_name" in ev_df.columns:
            ev_df["name_clean"] = (
                ev_df["last_name, first_name"]
                .str.lower().str.strip()
                .str.replace(", ", " ", regex=False)
            )
        elif "player_name" in ev_df.columns:
            ev_df["name_clean"] = ev_df["player_name"].str.lower().str.strip()

        # Try to merge on player_id first, then fall back to name
        ev_cols = ["Season", "avg_exit_velo", "hard_hit_pct", "barrel_pct",
                   "barrel_per_pa", "sweet_spot_pct", "avg_launch_angle",
                   "hard_hit_count", "barrels"]
        # Also check original column names in case rename wasn't applied
        ev_cols_raw = ["Season", "avg_hit_speed", "ev95percent", "brl_percent",
                       "brl_pa", "anglesweetspotpercent", "avg_hit_angle"]

        merge_cols  = [c for c in ev_cols if c in ev_df.columns]
        merge_cols += [c for c in ev_cols_raw if c in ev_df.columns and c not in merge_cols]

        if "player_id" in ev_df.columns and "IDfg" in batting.columns:
            # Ideal: merge on numeric ID (most reliable)
            # But FG IDfg ≠ Savant player_id — need MLBAM ID mapping
            # Fallback to name merge below
            pass

        if "name_clean" in ev_df.columns and "name_clean" in batting.columns:
            ev_subset = ev_df[["name_clean", "Season"] + [c for c in merge_cols
                                                            if c != "Season"]].copy()
            batting = batting.merge(ev_subset, on=["name_clean", "Season"], how="left")
            print(f"    Merged EV/barrel data: {batting['barrel_pct' if 'barrel_pct' in batting.columns else 'brl_percent'].notna().sum()} hitters matched")

    # --- Merge expected stats (xBA, xSLG, xwOBA) ----------------------------
    xstats = data["xstats"].copy()
    if not xstats.empty:
        if "last_name, first_name" in xstats.columns:
            xstats["name_clean"] = (
                xstats["last_name, first_name"]
                .str.lower().str.strip()
                .str.replace(", ", " ", regex=False)
            )

        xcols = [c for c in ["est_ba", "est_slg", "est_woba",
                              "est_woba_minus_woba_diff", "woba",
                              "est_slg_minus_slg_diff"] if c in xstats.columns]
        if xcols and "name_clean" in xstats.columns and "name_clean" in batting.columns:
            xstats_sub = xstats[["name_clean", "Season"] + xcols].copy()
            batting = batting.merge(xstats_sub, on=["name_clean", "Season"], how="left")
            print(f"    Merged expected stats: {xstats_sub.shape[0]} rows.")

    # --- Add park factor for home team --------------------------------------
    if "team_std" in batting.columns:
        batting["home_park_hr_factor"] = batting["team_std"].map(PARK_HR_FACTOR).fillna(100)

    # --- Create platoon indicator (LHH vs RHP → advantage) -----------------
    # Most hitters bat better against opposite-hand pitchers.
    # Without individual handedness data in FG batting_stats, we flag
    # the matchup at scoring time when SP handedness is known.
    # For training, we include a placeholder.
    batting["platoon_advantage"] = 0  # Updated at scoring time

    print(f"    ✓ Batter features: {len(batting):,} batter-season rows.")
    return batting


# =============================================================================
# FUNCTION 3: Build Pitcher Matchup Features
# =============================================================================
def build_pitcher_matchup_features(data: dict) -> pd.DataFrame:
    """
    Build a pitcher-season lookup for use in daily matchup scoring.

    When scoring today's games, we look up the opposing SP's stats here
    to create matchup-specific features:
      - sp_k_pct        : Higher K% = harder for hitter to reach base
      - sp_xwoba_allowed: Contact quality — low xwOBA = pitcher keeps hitters off base
      - sp_siera        : Overall quality
      - sp_swstr_pct    : Swing-and-miss rate — higher = harder to make contact at all

    Returns
    -------
    pd.DataFrame
        One row per pitcher-season for matchup lookups.
    """
    print("  Building pitcher matchup features...")

    fg_pit = data["fg_pitcher"].copy()
    if fg_pit.empty:
        return pd.DataFrame()

    if "Team" in fg_pit.columns:
        fg_pit["team_std"] = fg_pit["Team"].map(FG_TO_BREF).fillna(fg_pit["Team"])

    # Merge with Statcast pitcher xwOBA allowed
    pit_xstats = data["pitcher_xstats"].copy()
    if not pit_xstats.empty:
        if "last_name, first_name" in pit_xstats.columns:
            pit_xstats["name_clean"] = (
                pit_xstats["last_name, first_name"]
                .str.lower().str.strip()
                .str.replace(", ", " ", regex=False)
            )
        if "Name" in fg_pit.columns:
            fg_pit["name_clean"] = fg_pit["Name"].str.lower().str.strip()

        xcols_pit = [c for c in ["est_woba", "est_era", "woba"] if c in pit_xstats.columns]
        if xcols_pit and "name_clean" in pit_xstats.columns:
            pit_xstats_sub = pit_xstats[["name_clean", "Season"] + xcols_pit].copy()
            pit_xstats_sub = pit_xstats_sub.rename(
                columns={c: f"sp_{c}_allowed" for c in xcols_pit}
            )
            fg_pit = fg_pit.merge(pit_xstats_sub, on=["name_clean", "Season"], how="left")

    # Select key matchup features
    keep_cols = ["Name", "name_clean", "team_std", "Season", "IDfg",
                 "GS", "IP", "K%", "BB%", "K-BB%", "SIERA", "xFIP", "FIP",
                 "SwStr%", "F-Strike%"]
    # Add Statcast-derived columns
    keep_cols += [c for c in fg_pit.columns if c.startswith("sp_")]
    keep_cols  = [c for c in keep_cols if c in fg_pit.columns]

    pitcher_features = fg_pit[keep_cols].copy()

    # Rename for clarity in downstream joins
    rename_map = {
        "K%":       "sp_k_pct",
        "BB%":      "sp_bb_pct",
        "K-BB%":    "sp_k_minus_bb_pct",
        "SIERA":    "sp_siera",
        "xFIP":     "sp_xfip",
        "FIP":      "sp_fip",
        "SwStr%":   "sp_swstr_pct",
        "F-Strike%":"sp_fstrike_pct",
    }
    rename_map = {k: v for k, v in rename_map.items() if k in pitcher_features.columns}
    pitcher_features = pitcher_features.rename(columns=rename_map)

    print(f"    ✓ Pitcher matchup features: {len(pitcher_features):,} pitcher-season rows.")
    return pitcher_features


# =============================================================================
# FUNCTION 4: Create Matchup-Adjusted Hitter Dataset
# =============================================================================
def build_matchup_dataset(batter_df: pd.DataFrame, pitcher_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine batter features with a representative SP matchup.

    For TRAINING, we create synthetic matchup rows by pairing each
    hitter-season with the average opposing SP quality they faced.
    (Exact game-level matchup data would require pitch-by-pitch processing.)

    For SCORING, the user provides today's SP and we do a point lookup.

    Training target: tb_per_game (continuous) + over_1_5_tb_rate (binary)

    Returns
    -------
    pd.DataFrame
        Training dataset with hitter features + avg matchup context.
    """
    print("  Building matchup-adjusted dataset...")

    df = batter_df.copy()

    # For training: use league-average SP quality as the matchup baseline.
    # Each hitter faced a distribution of SPs — the average is a reasonable
    # training label. This avoids the complexity of exact game-level matching.
    if not pitcher_df.empty:
        # Compute league average SP stats by season
        sp_avg = pitcher_df.groupby("Season")[
            [c for c in ["sp_k_pct", "sp_bb_pct", "sp_siera", "sp_xfip", "sp_swstr_pct"]
             if c in pitcher_df.columns]
        ].mean().reset_index()

        # Rename to indicate these are "average opponent" features
        avg_rename = {c: f"opp_avg_{c}" for c in sp_avg.columns if c != "Season"}
        sp_avg = sp_avg.rename(columns=avg_rename)

        # Join season average SP stats to each batter
        # In R: merge(df, sp_avg, by="Season", all.x=TRUE)
        df = df.merge(sp_avg, on="Season", how="left")

    # Compute over/under threshold rates for different prop lines
    # In R: df$over_1_5_tb <- as.integer(df$tb_per_game >= 1.5)
    if "tb_per_game" in df.columns:
        df["over_0_5_tb_rate"] = (df["tb_per_game"] >= 0.5).astype(float)
        df["over_1_5_tb_rate"] = (df["tb_per_game"] >= 1.5).astype(float)
        df["over_2_5_tb_rate"] = (df["tb_per_game"] >= 2.5).astype(float)

    print(f"    ✓ Matchup dataset: {len(df):,} batter-season rows.")
    return df


# =============================================================================
# FUNCTION 5: Finalize Feature Selection
# =============================================================================
def finalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and clean the final feature set for XGBoost training.

    XGBoost benefits from:
      - Numeric features (no need for one-hot encoding, handles NaN internally)
      - Removing redundant correlated features
      - Sensible missing value handling

    Returns
    -------
    pd.DataFrame
        Clean dataset with selected features and targets.
    """
    print("  Finalizing hitter TB dataset...")

    # Define feature groups for readability
    quality_of_contact = [
        "barrel_pct", "brl_percent",      # Barrel% (two possible column names)
        "hard_hit_pct", "ev95percent",     # Hard-Hit% (95+ mph)
        "avg_exit_velo", "avg_hit_speed",  # Average exit velocity
        "sweet_spot_pct",                  # Launch angle in sweet spot (8–32°)
        "avg_launch_angle", "avg_hit_angle",
    ]

    expected_stats = [
        "est_ba", "est_slg", "est_woba",   # xBA, xSLG, xwOBA
        "est_woba_minus_woba_diff",         # Luck indicator (positive = due for uptick)
        "est_slg_minus_slg_diff",
    ]

    traditional_power = [
        "ISO", "SLG", "wOBA", "wRC+",      # Power and overall production
        "hr_per_game", "xbh_per_game",
        "BB%", "K%",                        # Plate discipline
        "BABIP",                            # Luck on balls in play
    ]

    context = [
        "home_park_hr_factor",             # Park boosts/suppresses TB
        "platoon_advantage",               # LHH vs RHP advantage (set at scoring time)
    ]

    sp_matchup = [
        "opp_avg_sp_k_pct", "opp_avg_sp_siera", "opp_avg_sp_xfip",
        "opp_avg_sp_swstr_pct", "opp_avg_sp_bb_pct",
    ]

    # Combine all feature groups
    all_features = quality_of_contact + expected_stats + traditional_power + context + sp_matchup

    # Filter to columns that actually exist (API responses vary)
    all_features = [c for c in all_features if c in df.columns]

    # Remove duplicates (can happen if both original and renamed columns exist)
    # In Python, dict.fromkeys preserves order while removing dupes (like unique() in R)
    all_features = list(dict.fromkeys(all_features))

    targets = ["tb_per_game", "over_0_5_tb_rate", "over_1_5_tb_rate", "over_2_5_tb_rate"]
    targets = [t for t in targets if t in df.columns]

    id_cols = ["Name", "Team", "Season", "G", "PA"]
    id_cols = [c for c in id_cols if c in df.columns]

    final_cols = id_cols + all_features + targets
    final_cols = [c for c in dict.fromkeys(final_cols) if c in df.columns]

    result = df[final_cols].copy()

    # Impute missing values with feature means
    feat_means = result[all_features].mean()
    for col in all_features:
        n_missing = result[col].isna().sum()
        if n_missing > 0:
            result[col] = result[col].fillna(feat_means[col])

    # Filter: need at least 100 PA to be useful
    if "PA" in result.columns:
        result = result[pd.to_numeric(result["PA"], errors="coerce") >= 100].copy()

    # Remove rows with no target variable
    if "tb_per_game" in result.columns:
        result = result.dropna(subset=["tb_per_game"])
        # Sanity check: TB per game should be between 0 and 6
        result = result[(result["tb_per_game"] >= 0) & (result["tb_per_game"] <= 6)]

    print(f"    ✓ Final dataset: {len(result):,} batter-seasons, {len(all_features)} features.")
    print(f"    ✓ Mean TB/game: {result['tb_per_game'].mean():.3f} (expect ~1.2–1.4)")
    return result


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("HITTER TOTAL BASES MODEL — STEP 2: DATASET CONSTRUCTION")
    print("=" * 70)

    print("\n[ 1/4 ] Loading raw data...")
    data = load_raw_data()

    print("\n[ 2/4 ] Building batter features...")
    batter_df = build_batter_features(data)

    print("\n[ 3/4 ] Building pitcher matchup features...")
    pitcher_df = build_pitcher_matchup_features(data)

    print("\n[ 4/4 ] Building final dataset...")
    matchup_df = build_matchup_dataset(batter_df, pitcher_df)
    final_df   = finalize_dataset(matchup_df)

    # Save processed dataset
    output_path = os.path.join(PROC_DIR, "hitter_tb_dataset.csv")
    final_df.to_csv(output_path, index=False)
    print(f"\n  ✓ Saved hitter_tb_dataset.csv ({len(final_df):,} rows)")

    # Also save the pitcher matchup lookup separately for use in scoring
    if not pitcher_df.empty:
        pit_path = os.path.join(PROC_DIR, "pitcher_matchup_lookup.csv")
        pitcher_df.to_csv(pit_path, index=False)
        print(f"  ✓ Saved pitcher_matchup_lookup.csv ({len(pitcher_df):,} rows)")

    print("\n" + "=" * 70)
    print("STEP 2 COMPLETE — Run 03_analysis_hitter_tb.py next.")
    print("=" * 70)
