"""
=============================================================================
PITCHER TOTAL OUTS MODEL — FILE 2 OF 4: DATASET CONSTRUCTION
=============================================================================
Purpose : Build the pitcher-season feature matrix for total outs modeling.
Input   : CSV files from ../data/raw/
Output  : ../data/processed/pitcher_outs_dataset.csv

What we're building:
  - Each ROW = one pitcher-season (aggregated across all starts)
  - FEATURES = K%, BB%, K-BB%, P/PA (efficiency), CSW%, manager hook rate,
               SP xwOBA allowed, opponent lineup strength
  - TARGET = outs_per_start (mean outs recorded per game start)
    (Outs = IP × 3, e.g., 6 IP = 18 outs)

Two-tier model structure:
  1. PITCHER SKILL TIER: K%, BB%, CSW%, SwStr% — how well the pitcher executes
  2. MANAGER DECISION TIER: hook_rate, bullpen_state — when does the manager pull him?

The manager tier creates a ceiling on outs even for dominant pitchers.
Example: Jacob deGrom with Tampa Bay would see fewer outs than with Texas,
purely due to Kevin Cash's historically aggressive hook tendencies.

Daily scoring:
  - Input: confirmed SP + manager + opponent lineup strength + bullpen state
  - Output: predicted outs, compared to market prop line, Kelly sizing

For R users:
  - `df.query("col > value")` = df[df$col > value, ] in R
  - `pd.to_numeric(x, errors='coerce')` = suppressWarnings(as.numeric(x)) in R
  - `df.rename(columns={old: new})` = names(df)[names(df) == old] <- new in R
=============================================================================
"""

import os
import pandas as pd
import numpy as np

# --- Configuration ----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")

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


# =============================================================================
# FUNCTION 1: Load Raw Data
# =============================================================================
def load_raw_data() -> dict:
    """Load all raw CSVs needed for pitcher outs model construction."""
    print("  Loading raw data files...")
    data = {}
    files = {
        "efficiency":  "raw_pitcher_efficiency.csv",
        "csw":         "raw_pitcher_csw_arsenal.csv",
        "xstats":      "raw_pitcher_xstats.csv",
        "games":       "raw_game_schedules.csv",
        "team_off":    "raw_team_offense_sp.csv",
        "manager":     "raw_manager_depth.csv",
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
# FUNCTION 2: Build Pitcher Efficiency Features
# =============================================================================
def build_pitcher_features(data: dict) -> pd.DataFrame:
    """
    Build per-pitcher, per-season feature table focused on efficiency metrics.

    Key efficiency metrics and their meaning:
      - K%     : Strikeout rate — more K's = faster, more predictable outs
      - BB%    : Walk rate — walks increase pitch count rapidly
      - K-BB%  : Net command — combines strikeout upside with walk downside
      - IP/GS  : Innings per start (the training target, computed here)
      - SwStr% : Swinging strike rate — leads to K's and shorter PAs
      - F-Strike%: First-pitch strike rate — getting ahead keeps PAs short
      - P/IP   : Pitches per inning (lower = more efficient, deeper starts)

    CSW% (Called Strike + Whiff %):
      - Strong predictor of deep starts
      - >29% = elite depth candidate
      - <23% = short outing risk even vs weak lineups

    Returns
    -------
    pd.DataFrame
        One row per pitcher-season with efficiency features + target.
    """
    print("  Building pitcher efficiency features...")

    eff = data["efficiency"].copy()
    if eff.empty:
        print("    ERROR: efficiency data empty.")
        return pd.DataFrame()

    # Standardize team
    if "Team" in eff.columns:
        eff["team_std"] = eff["Team"].map(FG_TO_BREF).fillna(eff["Team"])

    # Ensure IP and GS are numeric
    for col in ["IP", "GS", "G"]:
        if col in eff.columns:
            eff[col] = pd.to_numeric(eff[col], errors="coerce")

    # Filter to starting pitchers (GS ≥ 5)
    if "GS" in eff.columns:
        eff = eff[eff["GS"] >= 5].copy()

    # Compute TARGET: average outs per start
    # IP per start = total IP / number of starts
    if "IP" in eff.columns and "GS" in eff.columns:
        eff["ip_per_start"]   = eff["IP"] / eff["GS"].clip(lower=1)
        eff["outs_per_start"] = eff["ip_per_start"] * 3  # Convert IP to outs

    # Compute pitches per IP (proxy for efficiency)
    # This tells us how quickly the pitcher records outs
    if "Pitches" in eff.columns and "IP" in eff.columns:
        eff["pitches_per_ip"] = eff["Pitches"] / eff["IP"].clip(lower=0.1)
    elif "P/IP" in eff.columns:
        eff["pitches_per_ip"] = pd.to_numeric(eff["P/IP"], errors="coerce")

    # For over/under prop thresholds:
    # 15.5 outs ≈ 5.17 IP; 17.5 outs ≈ 5.83 IP; 18.5 outs = exactly 6.17 IP
    if "outs_per_start" in eff.columns:
        eff["over_15_5_outs_rate"] = (eff["outs_per_start"] >= 15.5).astype(float)
        eff["over_17_5_outs_rate"] = (eff["outs_per_start"] >= 17.5).astype(float)
        eff["over_18_5_outs_rate"] = (eff["outs_per_start"] >= 18.5).astype(float)

    # Standardize key metric names for consistent column naming
    rename_map = {
        "K%":      "k_pct",
        "BB%":     "bb_pct",
        "K-BB%":   "k_minus_bb_pct",
        "SIERA":   "siera",
        "xFIP":    "xfip",
        "FIP":     "fip",
        "ERA":     "era",
        "SwStr%":  "swstr_pct",
        "F-Strike%": "fstrike_pct",
        "CSW%":    "csw_pct",
    }
    rename_map = {k: v for k, v in rename_map.items() if k in eff.columns}
    eff = eff.rename(columns=rename_map)

    # Convert percentage columns from decimal to percent if needed
    # FanGraphs sometimes returns 0.254 instead of 25.4 for K%
    for pct_col in ["k_pct", "bb_pct", "k_minus_bb_pct", "swstr_pct", "fstrike_pct"]:
        if pct_col in eff.columns:
            col_data = pd.to_numeric(eff[pct_col], errors="coerce")
            # If max value < 1, data is in decimal form (0.254) — multiply by 100
            if col_data.max() < 1.5:
                eff[pct_col] = col_data * 100
                print(f"    Converted {pct_col} from decimal to percentage form.")

    print(f"    ✓ Pitcher features: {len(eff):,} pitcher-season rows.")
    print(f"    ✓ Mean outs/start: {eff.get('outs_per_start', pd.Series([0])).mean():.2f}")
    return eff


# =============================================================================
# FUNCTION 3: Build CSW% and Arsenal-Level Efficiency Features
# =============================================================================
def build_csw_features(data: dict) -> pd.DataFrame:
    """
    Build pitcher-level CSW% and arsenal velocity features.

    CSW% (Called Strike + Whiff %) is pulled at the pitch-type level
    from Baseball Savant. We aggregate to pitcher level here.

    Key insight from the document:
      - CSW% > 29% = high-depth candidate (more efficient outs)
      - Low CSW% but high K% = swing-heavy approach (works, but takes more pitches)

    Returns
    -------
    pd.DataFrame
        One row per pitcher-season with aggregated arsenal metrics.
    """
    print("  Building CSW% / arsenal features...")

    csw = data["csw"].copy()
    if csw.empty:
        print("    WARNING: No CSW data — skipping arsenal features.")
        return pd.DataFrame()

    # Standardize name for joining
    if "last_name, first_name" in csw.columns:
        csw["name_clean"] = (
            csw["last_name, first_name"]
            .str.lower().str.strip()
            .str.replace(", ", " ", regex=False)
        )
    elif "pitcher_name" in csw.columns:
        csw["name_clean"] = csw["pitcher_name"].str.lower().str.strip()

    # The arsenal data has one row per pitcher per pitch type.
    # Aggregate to pitcher level: take usage-weighted average velocity,
    # and flag whether pitcher has a strong breaking ball (slider/curve).
    pitch_type_col = "pitch_type" if "pitch_type" in csw.columns else None
    speed_col      = [c for c in ["avg_speed", "avg_velocity"] if c in csw.columns]
    speed_col      = speed_col[0] if speed_col else None

    if speed_col and "Season" in csw.columns:
        gb_cols = ["name_clean", "Season"] if "name_clean" in csw.columns else ["Season"]

        # Average fastball velocity (4-seam or 2-seam)
        fastballs = csw[csw.get("pitch_type", pd.Series(["FA"]*len(csw))).isin(
            ["FF", "FA", "SI", "FC"]
        )] if pitch_type_col else csw

        fb_velo = fastballs.groupby(gb_cols)[speed_col].mean().reset_index()
        fb_velo = fb_velo.rename(columns={speed_col: "avg_fb_velo"})

        # Does the pitcher have a plus breaking ball? (slider/curveball)
        if pitch_type_col:
            breakers = csw[csw["pitch_type"].isin(["SL", "CU", "KC", "SV", "ST"])]
            if not breakers.empty:
                has_breaker = (
                    breakers.groupby(gb_cols)["pitch_type"]
                    .count()
                    .reset_index()
                    .rename(columns={"pitch_type": "has_plus_breaker"})
                )
                has_breaker["has_plus_breaker"] = 1  # Present = 1

                fb_velo = fb_velo.merge(has_breaker, on=gb_cols, how="left")
                fb_velo["has_plus_breaker"] = fb_velo["has_plus_breaker"].fillna(0)

        print(f"    ✓ Arsenal features: {len(fb_velo):,} pitcher-season rows.")
        return fb_velo

    return pd.DataFrame()


# =============================================================================
# FUNCTION 4: Add Manager Hook and Opponent Offense Features
# =============================================================================
def add_manager_and_opponent_features(pitcher_df: pd.DataFrame, data: dict) -> pd.DataFrame:
    """
    Add manager hook tendency and opponent offensive strength to each pitcher row.

    This is the "decision layer" of the model:
      - A great pitcher with a pull-happy manager gets fewer outs.
      - A mediocre pitcher against a weak lineup might go deeper than expected.

    Manager features (from raw_manager_depth.csv):
      - depth_score : 0–1, higher = lets starters go deeper
      - avg_sp_outs : Historical average SP outs for that manager's team

    Opponent features:
      - opp_k_pct   : How often the opponent lineup strikes out (higher = easier for SP)
      - opp_wrc_plus: How good the opponent lineup is (higher = harder for SP)

    Returns
    -------
    pd.DataFrame
        Input DataFrame with manager and opponent columns added.
    """
    print("  Adding manager hook and opponent context...")

    df = pitcher_df.copy()

    # --- Manager hook data --------------------------------------------------
    manager_df = data["manager"].copy()
    if not manager_df.empty:
        # Merge on team_std to get each team's manager tendency
        if "team" in manager_df.columns and "team_std" in df.columns:
            manager_df = manager_df.rename(columns={"team": "team_std"})

        # Convert numeric columns
        for col in ["depth_score", "avg_sp_outs"]:
            if col in manager_df.columns:
                manager_df[col] = pd.to_numeric(manager_df[col], errors="coerce")

        merge_cols = ["team_std"] + [c for c in ["depth_score", "avg_sp_outs", "manager"]
                                      if c in manager_df.columns]
        manager_sub = manager_df[merge_cols].copy()
        df = df.merge(manager_sub, on="team_std", how="left")
        print(f"    Merged manager data: {df['depth_score'].notna().sum()} pitchers matched.")
    else:
        # Default to league-average manager tendency
        df["depth_score"] = 0.50
        df["avg_sp_outs"] = 15.0

    # --- Opponent offensive strength -----------------------------------------
    team_off = data["team_off"].copy()
    if not team_off.empty:
        if "Team" in team_off.columns:
            team_off["team_std"] = team_off["Team"].map(FG_TO_BREF).fillna(team_off["Team"])
        elif "Tm" in team_off.columns:
            team_off["team_std"] = team_off["Tm"].map(FG_TO_BREF).fillna(team_off["Tm"])

        # For each season, compute league-average opponent stats
        # (each team faces roughly the same distribution of opponents over a full season)
        opp_cols = [c for c in ["wRC+", "wOBA", "K%", "BB%", "OBP"] if c in team_off.columns]
        if opp_cols and "Season" in team_off.columns:
            league_avg_opp = team_off.groupby("Season")[opp_cols].mean().reset_index()
            league_avg_opp = league_avg_opp.rename(
                columns={c: f"opp_lg_avg_{c.lower().replace('+','plus').replace('%','_pct')}"
                          for c in opp_cols}
            )
            if "Season" in df.columns:
                df = df.merge(league_avg_opp, on="Season", how="left")
                print(f"    Merged opponent context: league averages by season.")

    return df


# =============================================================================
# FUNCTION 5: Finalize Dataset
# =============================================================================
def finalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select final features and clean the training dataset.

    Feature groups:
      1. Pitcher skill: k_pct, bb_pct, k_minus_bb_pct, siera, xfip
      2. Pitch efficiency: pitches_per_ip, swstr_pct, fstrike_pct, csw_pct
      3. Arsenal: avg_fb_velo, has_plus_breaker
      4. Manager: depth_score, avg_sp_outs
      5. Opponent context: opp_lg_avg_wrc_plus, opp_lg_avg_k_pct

    Targets:
      - outs_per_start (continuous regression target)
      - over_15_5_outs_rate, over_17_5_outs_rate, over_18_5_outs_rate (classification)

    Returns
    -------
    pd.DataFrame
        Clean, finalized training dataset.
    """
    print("  Finalizing pitcher outs dataset...")

    feature_cols = [
        # Pitcher skill (DIPS-focused metrics)
        "k_pct", "bb_pct", "k_minus_bb_pct",
        "siera", "xfip", "fip",

        # Pitch efficiency
        "pitches_per_ip", "swstr_pct", "fstrike_pct", "csw_pct",

        # Arsenal quality
        "avg_fb_velo", "has_plus_breaker",

        # Manager tendency
        "depth_score", "avg_sp_outs",

        # Opponent difficulty (league average for training)
        "opp_lg_avg_wrc_plus", "opp_lg_avg_k_pct",
        "opp_lg_avg_woba", "opp_lg_avg_obp",
    ]

    targets = ["outs_per_start", "ip_per_start",
               "over_15_5_outs_rate", "over_17_5_outs_rate", "over_18_5_outs_rate"]

    id_cols = ["Name", "team_std", "Season", "GS", "IP"]

    # Keep columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    targets      = [t for t in targets if t in df.columns]
    id_cols      = [c for c in id_cols if c in df.columns]

    keep = list(dict.fromkeys(id_cols + feature_cols + targets))
    result = df[[c for c in keep if c in df.columns]].copy()

    # Impute missing features with column means
    col_means = result[feature_cols].mean()
    for col in feature_cols:
        n_missing = result[col].isna().sum()
        if n_missing > 0:
            result[col] = result[col].fillna(col_means[col])
            if n_missing > 5:
                print(f"    Imputed {n_missing:3d} missing values in '{col}'")

    # Remove rows with missing target
    if "outs_per_start" in result.columns:
        result = result.dropna(subset=["outs_per_start"])
        # Sanity check: outs per start should be between 3 (1 IP) and 27 (9 IP CG)
        result = result[(result["outs_per_start"] >= 3) & (result["outs_per_start"] <= 27)]

    print(f"    ✓ Final dataset: {len(result):,} pitcher-seasons, {len(feature_cols)} features.")
    if "outs_per_start" in result.columns:
        print(f"    ✓ Mean outs/start: {result['outs_per_start'].mean():.2f}")
        print(f"    ✓ Over-15.5 rate:  {result['over_15_5_outs_rate'].mean():.3f}")
    return result


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("PITCHER TOTAL OUTS MODEL — STEP 2: DATASET CONSTRUCTION")
    print("=" * 70)

    print("\n[ 1/4 ] Loading raw data...")
    data = load_raw_data()

    print("\n[ 2/4 ] Building pitcher efficiency features...")
    pitcher_df = build_pitcher_features(data)

    print("\n[ 3/4 ] Building CSW% / arsenal features...")
    csw_df = build_csw_features(data)

    # Merge CSW/arsenal features if available
    if not csw_df.empty and not pitcher_df.empty:
        # Try to merge on name_clean + Season
        if "name_clean" in csw_df.columns and "Name" in pitcher_df.columns:
            pitcher_df["name_clean"] = pitcher_df["Name"].str.lower().str.strip()
            pitcher_df = pitcher_df.merge(
                csw_df, on=["name_clean", "Season"], how="left"
            )
            print(f"    Merged CSW data into pitcher features.")

    print("\n[ 4/4 ] Adding manager hook and opponent context...")
    pitcher_df = add_manager_and_opponent_features(pitcher_df, data)
    final_df   = finalize_dataset(pitcher_df)

    # Save outputs
    output_path = os.path.join(PROC_DIR, "pitcher_outs_dataset.csv")
    final_df.to_csv(output_path, index=False)
    print(f"\n  ✓ Saved pitcher_outs_dataset.csv ({len(final_df):,} rows)")
    print(f"  ✓ Columns: {list(final_df.columns)}")

    print("\n" + "=" * 70)
    print("STEP 2 COMPLETE — Run 03_analysis_pitcher_outs.py next.")
    print("=" * 70)
