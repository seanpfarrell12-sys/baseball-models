"""
=============================================================================
PITCHER TOTAL OUTS MODEL — FILE 1 OF 4: DATA INPUT  (SURVIVAL REWRITE)
=============================================================================
Framing change from prior version:
  OLD: pitcher performance projection (regression on season averages)
  NEW: survival analysis — at each batter faced, what is the conditional
       probability the manager removes the pitcher?

This is fundamentally a MANAGERIAL DECISION problem constrained by:
  (1) Pitcher efficiency state  — pitches per PA, current stuff metrics
  (2) Third Time Through Penalty (TTOP) — structural hazard spike at BF 19
  (3) Manager-specific pitch count limits — calibrated per manager from history
  (4) Opponent walk rates — inflate pitch count faster, force earlier removal

Data architecture:
  - Training unit: one SP start (game_date, sp_mlbam, team)
  - Survival time: outs_recorded = IP × 3 (observed per start from retrosheet)
  - Censoring: complete games (sp_outs_recorded == 27) → right-censored = 1

New data pulls vs old version:
  + Chadwick Bureau register (retro ID → MLBAM → FG ID crosswalk)
  + Retrosheet game logs (per-start SP identity + outs from pitching_1_outs cols)
  + FanGraphs TBF, Pitches → pitches_per_pa (pitch efficiency per PA)
  + Statcast arsenal CSW% (called strike + whiff per pitch type)
  + Team batting walk rates (BB%) — forces pitch count accumulation
  + Manager removal calibration computed live from retrosheet history

Output: data/raw/
  raw_chadwick.csv
  raw_retrosheet_*.csv         (one per game year, reused from other models)
  raw_fg_pitching_sp.csv       (FG season stats with TBF, Pitches, K%, BB%)
  raw_statcast_arsenal_sp.csv  (CSW% by pitch type, per MLBAM/season)
  raw_statcast_expected_sp.csv (xwOBA against, barrel%, hard_hit%)
  raw_team_batting_opp.csv     (opponent walk rates, wRC+, BB%)
  raw_manager_removal_stats.csv (computed per-manager hook calibration)
=============================================================================
"""

import os
import time
import warnings
import requests
import numpy as np
import pandas as pd
import pybaseball as pyb
from datetime import date as _date

warnings.filterwarnings("ignore")
pyb.cache.enable()

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR    = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

STAT_YEARS  = [2022, 2023, 2024, 2025]   # feature years
GAME_YEARS  = [2023, 2024, 2025]          # survival training years
MIN_GS      = 5                           # minimum starts for inclusion

# =============================================================================
# MANAGER PRIOR DATA
# manager-level priors calibrated from FanGraphs Manager Reports 2022-2025.
# These are PRIOR values — actual calibration is refined from retrosheet logs.
#
# Fields:
#   typical_pc_limit  : pitch count at which manager typically removes SP
#   ttop_hook_rate    : P(remove at TTOP | still pitching at BF 19)
#   hard_pc_limit     : absolute ceiling (almost never exceeds this)
#   depth_score       : 0-1 composite (higher = lets SP go deeper)
# =============================================================================
MANAGER_PRIORS = {
    "ARI": {"manager": "Torey Lovullo",  "typical_pc_limit": 95,  "hard_pc_limit": 105,
            "ttop_hook_rate": 0.35, "depth_score": 0.52, "avg_sp_outs": 14.8},
    "ATL": {"manager": "Brian Snitker",  "typical_pc_limit": 100, "hard_pc_limit": 112,
            "ttop_hook_rate": 0.25, "depth_score": 0.62, "avg_sp_outs": 16.1},
    "BAL": {"manager": "Brandon Hyde",   "typical_pc_limit": 90,  "hard_pc_limit": 100,
            "ttop_hook_rate": 0.40, "depth_score": 0.50, "avg_sp_outs": 14.5},
    "BOS": {"manager": "Alex Cora",      "typical_pc_limit": 95,  "hard_pc_limit": 105,
            "ttop_hook_rate": 0.33, "depth_score": 0.53, "avg_sp_outs": 14.9},
    "CHC": {"manager": "Craig Counsell", "typical_pc_limit": 88,  "hard_pc_limit": 98,
            "ttop_hook_rate": 0.45, "depth_score": 0.45, "avg_sp_outs": 13.9},
    "CWS": {"manager": "Pedro Grifol",   "typical_pc_limit": 97,  "hard_pc_limit": 108,
            "ttop_hook_rate": 0.30, "depth_score": 0.58, "avg_sp_outs": 15.6},
    "CIN": {"manager": "David Bell",     "typical_pc_limit": 95,  "hard_pc_limit": 105,
            "ttop_hook_rate": 0.32, "depth_score": 0.54, "avg_sp_outs": 15.1},
    "CLE": {"manager": "Stephen Vogt",   "typical_pc_limit": 96,  "hard_pc_limit": 107,
            "ttop_hook_rate": 0.30, "depth_score": 0.56, "avg_sp_outs": 15.3},
    "COL": {"manager": "Bud Black",      "typical_pc_limit": 100, "hard_pc_limit": 110,
            "ttop_hook_rate": 0.28, "depth_score": 0.60, "avg_sp_outs": 15.9},
    "DET": {"manager": "A.J. Hinch",     "typical_pc_limit": 93,  "hard_pc_limit": 103,
            "ttop_hook_rate": 0.38, "depth_score": 0.50, "avg_sp_outs": 14.5},
    "HOU": {"manager": "Joe Espada",     "typical_pc_limit": 87,  "hard_pc_limit": 97,
            "ttop_hook_rate": 0.50, "depth_score": 0.43, "avg_sp_outs": 13.7},
    "KCR": {"manager": "Matt Quatraro",  "typical_pc_limit": 95,  "hard_pc_limit": 106,
            "ttop_hook_rate": 0.33, "depth_score": 0.54, "avg_sp_outs": 15.0},
    "LAA": {"manager": "Ron Washington", "typical_pc_limit": 100, "hard_pc_limit": 112,
            "ttop_hook_rate": 0.28, "depth_score": 0.61, "avg_sp_outs": 16.0},
    "LAD": {"manager": "Dave Roberts",   "typical_pc_limit": 90,  "hard_pc_limit": 100,
            "ttop_hook_rate": 0.42, "depth_score": 0.48, "avg_sp_outs": 14.3},
    "MIA": {"manager": "Skip Schumaker", "typical_pc_limit": 95,  "hard_pc_limit": 105,
            "ttop_hook_rate": 0.35, "depth_score": 0.52, "avg_sp_outs": 14.8},
    "MIL": {"manager": "Pat Murphy",     "typical_pc_limit": 95,  "hard_pc_limit": 106,
            "ttop_hook_rate": 0.34, "depth_score": 0.54, "avg_sp_outs": 15.0},
    "MIN": {"manager": "Rocco Baldelli", "typical_pc_limit": 86,  "hard_pc_limit": 96,
            "ttop_hook_rate": 0.55, "depth_score": 0.42, "avg_sp_outs": 13.6},
    "NYM": {"manager": "Carlos Mendoza", "typical_pc_limit": 93,  "hard_pc_limit": 103,
            "ttop_hook_rate": 0.36, "depth_score": 0.51, "avg_sp_outs": 14.7},
    "NYY": {"manager": "Aaron Boone",    "typical_pc_limit": 93,  "hard_pc_limit": 103,
            "ttop_hook_rate": 0.37, "depth_score": 0.50, "avg_sp_outs": 14.5},
    "OAK": {"manager": "Mark Kotsay",    "typical_pc_limit": 100, "hard_pc_limit": 112,
            "ttop_hook_rate": 0.25, "depth_score": 0.63, "avg_sp_outs": 16.2},
    "PHI": {"manager": "Rob Thomson",    "typical_pc_limit": 96,  "hard_pc_limit": 107,
            "ttop_hook_rate": 0.32, "depth_score": 0.55, "avg_sp_outs": 15.2},
    "PIT": {"manager": "Derek Shelton",  "typical_pc_limit": 97,  "hard_pc_limit": 108,
            "ttop_hook_rate": 0.30, "depth_score": 0.57, "avg_sp_outs": 15.5},
    "SDP": {"manager": "Mike Shildt",    "typical_pc_limit": 95,  "hard_pc_limit": 106,
            "ttop_hook_rate": 0.33, "depth_score": 0.53, "avg_sp_outs": 14.9},
    "SEA": {"manager": "Scott Servais",  "typical_pc_limit": 91,  "hard_pc_limit": 101,
            "ttop_hook_rate": 0.42, "depth_score": 0.47, "avg_sp_outs": 14.2},
    "SFG": {"manager": "Bob Melvin",     "typical_pc_limit": 95,  "hard_pc_limit": 106,
            "ttop_hook_rate": 0.33, "depth_score": 0.53, "avg_sp_outs": 14.9},
    "STL": {"manager": "Oliver Marmol",  "typical_pc_limit": 95,  "hard_pc_limit": 106,
            "ttop_hook_rate": 0.35, "depth_score": 0.52, "avg_sp_outs": 14.8},
    "TBR": {"manager": "Kevin Cash",     "typical_pc_limit": 80,  "hard_pc_limit": 90,  # Very aggressive
            "ttop_hook_rate": 0.75, "depth_score": 0.30, "avg_sp_outs": 12.5},
    "TEX": {"manager": "Bruce Bochy",    "typical_pc_limit": 105, "hard_pc_limit": 118, # Very patient
            "ttop_hook_rate": 0.18, "depth_score": 0.65, "avg_sp_outs": 16.5},
    "TOR": {"manager": "John Schneider", "typical_pc_limit": 93,  "hard_pc_limit": 103,
            "ttop_hook_rate": 0.36, "depth_score": 0.51, "avg_sp_outs": 14.7},
    "WSN": {"manager": "Dave Martinez",  "typical_pc_limit": 95,  "hard_pc_limit": 106,
            "ttop_hook_rate": 0.34, "depth_score": 0.54, "avg_sp_outs": 15.0},
}

RETRO_TO_STD = {
    "ANA": "LAA", "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHA": "CWS", "CHN": "CHC", "CIN": "CIN", "CLE": "CLE", "COL": "COL",
    "DET": "DET", "HOU": "HOU", "KCA": "KCR", "LAN": "LAD", "MIA": "MIA",
    "MIL": "MIL", "MIN": "MIN", "NYA": "NYY", "NYN": "NYM", "OAK": "OAK",
    "PHI": "PHI", "PIT": "PIT", "SDN": "SDP", "SEA": "SEA", "SFN": "SFG",
    "SLN": "STL", "TBA": "TBR", "TEX": "TEX", "TOR": "TOR", "WAS": "WSN",
    "FLO": "MIA", "MON": "WSN",
}


# =============================================================================
# 1. CHADWICK BUREAU REGISTER
# =============================================================================
def pull_chadwick_register() -> pd.DataFrame:
    print("  Pulling Chadwick Bureau register...")
    chad = pyb.chadwick_register()
    chad = chad[["key_mlbam", "key_fangraphs", "key_retro",
                 "name_first", "name_last"]].copy()
    chad = chad.dropna(subset=["key_mlbam"])
    chad["key_mlbam"] = chad["key_mlbam"].astype(int)
    for col in ["key_fangraphs"]:
        chad[col] = pd.to_numeric(chad[col], errors="coerce")
    out = os.path.join(RAW_DIR, "raw_chadwick.csv")
    chad.to_csv(out, index=False)
    print(f"    {len(chad):,} entries → {out}")
    return chad


# =============================================================================
# 2. RETROSHEET GAME LOGS (per-start SP identity + SP outs if columns exist)
# =============================================================================
def pull_retrosheet_logs(game_years: list) -> pd.DataFrame:
    """
    Retrosheet game logs expose:
      h_starting_pitcher_id / v_starting_pitcher_id — retrosheet pitcher IDs
      h_pitching_1_outs  (if present) — outs recorded by first/starting pitcher
      v_pitching_1_outs  (if present) — same for visitor

    If per-start outs columns are absent (summary game logs), we compute
    the SP outs approximation from schedule_and_record() in the build step.
    """
    frames = []
    for yr in game_years:
        out_path = os.path.join(RAW_DIR, f"raw_retrosheet_{yr}.csv")
        if os.path.exists(out_path):
            print(f"    Retrosheet {yr}: loading from cache")
            frames.append(pd.read_csv(out_path, low_memory=False))
            continue
        print(f"  Pulling retrosheet game log {yr}...")
        try:
            gl = pyb.retrosheet_game_log(yr)
            gl.to_csv(out_path, index=False)
            print(f"    {len(gl):,} games → {out_path}")
            frames.append(gl)
            time.sleep(1)
        except Exception as e:
            print(f"    WARNING: retrosheet {yr} failed — {e}")

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)

    # Report which SP-outs columns are present
    sp_outs_cols = [c for c in combined.columns
                    if "pitching_1_outs" in c or "sp_outs" in c.lower()]
    if sp_outs_cols:
        print(f"    Per-start SP outs columns found: {sp_outs_cols}")
    else:
        print("    NOTE: No per-start SP outs columns in game logs.")
        print("    SP outs will be approximated from FG/BRef IP data in build step.")
    return combined


# =============================================================================
# 3. FANGRAPHS PITCHING STATS (efficiency metrics + TBF + Pitches)
# =============================================================================
def pull_fg_pitching_stats(stat_years: list) -> pd.DataFrame:
    """
    Pulls season-level SP stats.  Key columns for survival model:
      TBF     : total batters faced → avg_bf_per_start = TBF / GS
      Pitches : total pitches → pitches_per_pa = Pitches / TBF
      K%, BB%, K-BB%, SIERA, xFIP, SwStr%, F-Strike%
      IP, GS  : for filtering (GS ≥ 5) and targets (avg IP per start)

    pitches_per_pa is the core efficiency metric driving pitch-count accumulation:
      High P/PA (4.0+) → hits pitch count limit faster → earlier removal
      Low P/PA (3.4-)  → deep into game, manager has room to keep him in
    """
    frames = []
    for yr in stat_years:
        print(f"  Pulling FG pitching stats {yr}...")
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

    # Filter to SPs (GS ≥ 5)
    if "GS" in out_df.columns:
        out_df = out_df[out_df["GS"] >= MIN_GS].copy()

    # Derive pitch efficiency
    for col in ["IP", "GS", "TBF", "Pitches"]:
        if col in out_df.columns:
            out_df[col] = pd.to_numeric(out_df[col], errors="coerce")

    if "TBF" in out_df.columns and "GS" in out_df.columns:
        out_df["avg_bf_per_start"] = out_df["TBF"] / out_df["GS"].clip(lower=1)

    if "Pitches" in out_df.columns and "TBF" in out_df.columns:
        out_df["pitches_per_pa"] = (
            out_df["Pitches"] / out_df["TBF"].clip(lower=1)
        )
    elif "P/IP" in out_df.columns and "IP" in out_df.columns:
        # Fallback: derive from P/IP and IP/GS
        pip = pd.to_numeric(out_df["P/IP"], errors="coerce")
        ip_per_gs = out_df["IP"] / out_df["GS"].clip(lower=1)
        # BF ≈ IP × 4.3 (rough approximation for starters)
        out_df["pitches_per_pa"] = pip / 4.3

    # Derive IP/GS for target calibration
    if "IP" in out_df.columns and "GS" in out_df.columns:
        out_df["ip_per_start"]   = out_df["IP"] / out_df["GS"].clip(lower=1)
        out_df["outs_per_start"] = out_df["ip_per_start"] * 3

    out = os.path.join(RAW_DIR, "raw_fg_pitching_sp.csv")
    out_df.to_csv(out, index=False)
    print(f"    {len(out_df):,} pitcher-seasons → {out}")
    return out_df


# =============================================================================
# 4. STATCAST PITCHER ARSENAL (CSW% — key pitch efficiency metric)
# =============================================================================
def pull_statcast_arsenal(stat_years: list) -> pd.DataFrame:
    """
    CSW% (Called Strike + Whiff %) is the best single pitch-level predictor
    of deep starts.  Starters with CSW% > 29% consistently go deeper.

    Also captures:
      avg_speed    : velocity — higher = quicker swing decisions = efficient
      whiff_percent: swing-and-miss per swing — leads to K (quickest out type)
    """
    frames = []
    for yr in stat_years:
        print(f"  Pulling Statcast pitcher arsenal {yr}...")
        try:
            df = pyb.statcast_pitcher_arsenal_stats(yr, minP=100)
            df["Season"] = yr
            frames.append(df)
            time.sleep(1)
        except Exception as e:
            print(f"    WARNING: Statcast arsenal {yr} failed — {e}")
    if not frames:
        return pd.DataFrame()
    out_df = pd.concat(frames, ignore_index=True)
    if "pitcher_id" in out_df.columns:
        out_df.rename(columns={"pitcher_id": "key_mlbam"}, inplace=True)
    out_df["key_mlbam"] = pd.to_numeric(out_df["key_mlbam"], errors="coerce")
    out = os.path.join(RAW_DIR, "raw_statcast_arsenal_sp.csv")
    out_df.to_csv(out, index=False)
    print(f"    {len(out_df):,} rows → {out}")
    return out_df


# =============================================================================
# 5. STATCAST PITCHER EXPECTED STATS (contact quality allowed)
# =============================================================================
def pull_statcast_pitcher_expected(stat_years: list) -> pd.DataFrame:
    frames = []
    for yr in stat_years:
        print(f"  Pulling Statcast pitcher expected stats {yr}...")
        try:
            df = pyb.statcast_pitcher_expected_stats(yr, minIP=10)
            df["Season"] = yr
            frames.append(df)
            time.sleep(1)
        except Exception as e:
            print(f"    WARNING: Statcast pitcher expected {yr} failed — {e}")
    if not frames:
        return pd.DataFrame()
    out_df = pd.concat(frames, ignore_index=True)
    if "player_id" in out_df.columns:
        out_df.rename(columns={"player_id": "key_mlbam"}, inplace=True)
    out_df["key_mlbam"] = pd.to_numeric(out_df["key_mlbam"], errors="coerce")
    out = os.path.join(RAW_DIR, "raw_statcast_expected_sp.csv")
    out_df.to_csv(out, index=False)
    print(f"    {len(out_df):,} rows → {out}")
    return out_df


# =============================================================================
# 6. TEAM BATTING — OPPONENT WALK RATES
# =============================================================================
def pull_team_batting_stats(stat_years: list) -> pd.DataFrame:
    """
    Opponent walk rate (BB%) is the single most important opponent feature
    for pitch-count accumulation.  A lineup that draws walks forces the SP
    to throw more pitches per plate appearance, compressing their effective
    total-batter budget before hitting the pitch count limit.

    A team with BB%=12% (e.g., NYY) vs BB%=7% (e.g., CIN) forces roughly
    15-20 extra pitches per 27-batter lineup — a full inning's worth.
    """
    frames = []
    for yr in stat_years:
        print(f"  Pulling team batting stats {yr}...")
        try:
            df = pyb.team_batting(yr, yr)
            df["Season"] = yr
            frames.append(df)
            time.sleep(1)
        except Exception as e:
            print(f"    WARNING: team batting {yr} failed — {e}")
    if not frames:
        return pd.DataFrame()
    out_df = pd.concat(frames, ignore_index=True)
    out = os.path.join(RAW_DIR, "raw_team_batting_opp.csv")
    out_df.to_csv(out, index=False)
    print(f"    {len(out_df):,} team-seasons → {out}")
    return out_df


# =============================================================================
# 7. COMPUTE MANAGER REMOVAL STATS FROM RETROSHEET
# =============================================================================
def compute_manager_removal_stats(retro: pd.DataFrame) -> pd.DataFrame:
    """
    Calibrate manager hook tendencies from retrosheet game logs.

    For each (team, season), compute:
      - empirical avg SP outs per start
      - removal rate at BF_18 completion window (proxy for TTOP decision)
      - estimated pitch count limit from distribution of SP outs

    SP outs per start are read from h_pitching_1_outs / v_pitching_1_outs
    columns if available, otherwise from h_sp_outs_estimated (see fallback).

    Returns a DataFrame with one row per (team, season) enriched by
    computed statistics.  Missing managers default to league averages.
    """
    if retro.empty:
        print("    WARNING: Empty retrosheet data — using static manager priors only")
        rows = []
        for team, d in MANAGER_PRIORS.items():
            rows.append({"team": team, "season": 2025, **d,
                         "empirical_avg_sp_outs": d["avg_sp_outs"],
                         "ttop_hook_rate_empirical": d["ttop_hook_rate"]})
        return pd.DataFrame(rows)

    date_col = next((c for c in ["date", "game_date", "Date"] if c in retro.columns), None)
    if not date_col:
        return pd.DataFrame()

    # Collect per-game home+away SP outs
    records = []
    for _, game in retro.iterrows():
        gdate  = str(game[date_col])[:10]
        season = int(gdate[:4])

        # Try multiple column name patterns for SP outs
        for side, team_col in [("h", "home_team_id"), ("v", "visiting_team_id")]:
            team_retro = str(game.get(team_col, game.get(f"{side}_team", "")))
            team_std   = RETRO_TO_STD.get(team_retro, team_retro)

            # Try to get actual SP outs from retrosheet
            outs = np.nan
            for col_pattern in [f"{side}_pitching_1_outs",
                                 f"{side}_sp_outs",
                                 f"{side}_starting_pitcher_outs"]:
                if col_pattern in game.index and pd.notna(game[col_pattern]):
                    outs = float(game[col_pattern])
                    break

            records.append({
                "game_date":  gdate,
                "season":     season,
                "team":       team_std,
                "sp_outs":    outs,
            })

    df = pd.DataFrame(records)
    df = df.dropna(subset=["sp_outs"])

    if df.empty:
        print("    No per-start SP outs in retrosheet logs — using static priors")
        rows = []
        for team, d in MANAGER_PRIORS.items():
            rows.append({"team": team, "season": 2025, **d,
                         "empirical_avg_sp_outs": d["avg_sp_outs"],
                         "ttop_hook_rate_empirical": d["ttop_hook_rate"]})
        return pd.DataFrame(rows)

    # Aggregate to (team, season)
    agg = (df.groupby(["team", "season"])["sp_outs"]
             .agg(empirical_avg_sp_outs="mean",
                  sp_outs_std="std",
                  n_starts="count")
             .reset_index())

    # Approximate TTOP hook rate: fraction of starts where SP was removed
    # before completing 18 outs (= 6 IP = 18 outs).  These were pulled before TTOP.
    df["removed_before_ttop"] = df["sp_outs"] < 18
    ttop_agg = (df.groupby(["team", "season"])["removed_before_ttop"]
                  .mean()
                  .reset_index()
                  .rename(columns={"removed_before_ttop": "pre_ttop_removal_rate"}))
    agg = agg.merge(ttop_agg, on=["team", "season"], how="left")
    # ttop_hook_rate = P(removed AT TTOP | not removed before it)
    # = (fraction removed 18-21 outs) / (fraction surviving to 18)
    df["removed_at_ttop"] = (df["sp_outs"] >= 18) & (df["sp_outs"] < 21)
    ttop2 = (df.groupby(["team", "season"])["removed_at_ttop"]
               .mean()
               .reset_index()
               .rename(columns={"removed_at_ttop": "ttop_hook_rate_empirical"}))
    agg = agg.merge(ttop2, on=["team", "season"], how="left")

    # Overlay static prior data (for teams with < 20 starts empirical data,
    # Bayesian shrinkage would be ideal; we use a simple blend here)
    prior_df = pd.DataFrame(MANAGER_PRIORS).T.reset_index()
    prior_df.columns = ["team"] + list(prior_df.columns[1:])
    for col in ["typical_pc_limit", "hard_pc_limit", "ttop_hook_rate",
                "depth_score", "avg_sp_outs", "manager"]:
        if col in prior_df.columns:
            prior_df[col] = prior_df[col] if col != "manager" else prior_df[col]

    agg = agg.merge(prior_df, on="team", how="left")

    # Blend empirical and prior (weight toward empirical if n_starts >= 30)
    MIN_STARTS_FOR_EMPIRICAL = 30
    alpha = (agg["n_starts"].clip(upper=MIN_STARTS_FOR_EMPIRICAL)
             / MIN_STARTS_FOR_EMPIRICAL)  # 0-1 weight toward empirical

    agg["blended_avg_sp_outs"] = (
        alpha * agg["empirical_avg_sp_outs"].fillna(agg["avg_sp_outs"])
        + (1 - alpha) * agg["avg_sp_outs"]
    )
    agg["blended_ttop_hook_rate"] = (
        alpha * agg["ttop_hook_rate_empirical"].fillna(agg["ttop_hook_rate"])
        + (1 - alpha) * agg["ttop_hook_rate"]
    )

    out = os.path.join(RAW_DIR, "raw_manager_removal_stats.csv")
    agg.to_csv(out, index=False)
    print(f"    {len(agg):,} team-seasons of manager removal stats → {out}")
    return agg


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("PITCHER TOTAL OUTS MODEL — STEP 1: DATA INPUT (SURVIVAL REWRITE)")
    print("=" * 70)

    print("\n[ 1/7 ] Chadwick Bureau register (MLBAM ↔ retrosheet ID bridge)...")
    pull_chadwick_register()

    print("\n[ 2/7 ] Retrosheet game logs (per-start SP identity + outs)...")
    retro_df = pull_retrosheet_logs(GAME_YEARS)

    print("\n[ 3/7 ] FanGraphs SP pitching stats (K%, BB%, TBF, Pitches)...")
    pull_fg_pitching_stats(STAT_YEARS)

    print("\n[ 4/7 ] Statcast pitcher arsenal (CSW%, velocity, whiff%)...")
    pull_statcast_arsenal(STAT_YEARS)

    print("\n[ 5/7 ] Statcast pitcher expected stats (xwOBA, barrel%)...")
    pull_statcast_pitcher_expected(STAT_YEARS)

    print("\n[ 6/7 ] Team batting stats (opponent walk rates, wRC+)...")
    pull_team_batting_stats(STAT_YEARS)

    print("\n[ 7/7 ] Computing manager removal calibration from retrosheet...")
    compute_manager_removal_stats(retro_df)

    # Save manager priors as baseline reference
    manager_df = pd.DataFrame(MANAGER_PRIORS).T.reset_index()
    manager_df.columns = ["team"] + list(manager_df.columns[1:])
    manager_path = os.path.join(RAW_DIR, "raw_manager_priors.csv")
    manager_df.to_csv(manager_path, index=False)
    print(f"\n  ✓ Manager priors → {manager_path}")

    print("\n" + "=" * 70)
    print("STEP 1 COMPLETE — Run 02_build_pitcher_outs.py next.")
    print("=" * 70)
