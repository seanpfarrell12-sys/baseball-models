"""
=============================================================================
PITCHER TOTAL OUTS MODEL — FILE 2 OF 4: FEATURE ENGINEERING  (SURVIVAL)
=============================================================================
This file builds TWO datasets:

  A) per_start_dataset.csv  — one row per SP start (the survival event)
     Used for walk-forward CV evaluation and final model training.
     Columns: game_date, season, team, sp_mlbam, outs_recorded, censored,
              [feature matrix], [manager hazard features]

  B) bf_level_dataset.csv   — one row per (start × batter faced)
     The discrete-time survival expansion used for hazard model training.
     Each row represents the state at a specific batter faced k, with:
       - event = 1 only at the row where removal occurred
       - Time-varying features (TTOP flag, estimated pitch count) update per k

Survival framing:
  - "Time" = batters faced before removal (integer 1..27)
  - "Event" = manager removes pitcher (event=1, censored=0)
  - "Censored" = pitcher completes the start (CG or near-CG) — right-censored
  - Hazard h(k) = P(removed at BF k | survived through BF k-1)

Third Time Through Order Penalty (TTOP):
  The "Third Time Through the Order" effect is well-documented:
    1st time through (BF 1-9):   Batters unfamiliar with SP's arsenal
    2nd time through (BF 10-18): Batters have seen all pitch types once
    3rd time through (BF 19-27): Batters most familiar → worst outcomes

  Research (Tango, Lichtman, Dolphin) shows ERA increases ~0.6-1.0 runs
  in the 3rd time through.  Modern managers use this to set their pull point.

  Hard 18th batter threshold:
  BF=18 completes the 2nd time through — a natural decision point where:
    (a) Manager evaluates: is this SP ready to face the lineup a 3rd time?
    (b) Left-handed specialists may be warming in the bullpen for the 3rd TTO
    (c) Pitch count is typically 65-85 at this point → within manager's window

  We encode this as a step-function spike in the hazard function:
    bf_18_decision_point  = 1 if bf_k == 18
    bf_19_ttop_start      = 1 if bf_k == 19
    is_ttop               = 1 if bf_k >= 19
    batters_into_ttop     = max(0, bf_k - 18)

Pitch Count Accumulation:
  Estimated pitch count at BF k:
    pc_k = sp_pitches_per_pa × k   (linear accumulation model)

  Manager pitch count limit creates a hard ceiling:
    pc_fraction_k = pc_k / manager_typical_pc_limit  (0-1)
    approaching_pc_limit = 1 if pc_fraction_k >= 0.85
    at_pc_limit          = 1 if pc_fraction_k >= 1.0

  This is mathematically correct: even a dominant Gerrit Cole type pitcher
  will be removed by any manager when his estimated pitch count crosses
  the team's limit, regardless of performance quality.

Input  : data/raw/raw_*.csv
Output : data/processed/pitcher_outs_per_start.csv
         data/processed/pitcher_outs_bf_level.csv
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR   = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

GAME_YEARS = [2023, 2024, 2025]

# Hard-coded structural constants (from survival analysis literature)
TTOP_BF_START         = 19      # First batter of 3rd time through
FIRST_THROUGH_END     = 9       # Last batter of 1st time through
SECOND_THROUGH_END    = 18      # Last batter of 2nd time through (decision point)
MAX_BF_MODELED        = 27      # Maximum BF we model (full 3rd time through)
CG_THRESHOLD_OUTS     = 24      # 24+ outs = treat as censored (manager can't pull him)

# Empirical out rate per batter faced for average SP
# Derived from league average: OBP against ≈ 0.315
# P(out on given PA) ≈ 1 - OBP = 0.685
LEAGUE_AVG_OUT_RATE = 0.685

RETRO_TO_STD = {
    "ANA": "LAA", "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHA": "CWS", "CHN": "CHC", "CIN": "CIN", "CLE": "CLE", "COL": "COL",
    "DET": "DET", "HOU": "HOU", "KCA": "KCR", "LAN": "LAD", "MIA": "MIA",
    "MIL": "MIL", "MIN": "MIN", "NYA": "NYY", "NYN": "NYM", "OAK": "OAK",
    "PHI": "PHI", "PIT": "PIT", "SDN": "SDP", "SEA": "SEA", "SFN": "SFG",
    "SLN": "STL", "TBA": "TBR", "TEX": "TEX", "TOR": "TOR", "WAS": "WSN",
    "FLO": "MIA", "MON": "WSN",
}

FG_TO_STD = {
    "CHW": "CWS", "SD": "SDP", "SF": "SFG", "TB": "TBR",
    "KC": "KCR", "WAS": "WSN", "MIA": "MIA",
}


# =============================================================================
# LOAD RAW DATA
# =============================================================================
def load_raw() -> dict:
    print("  Loading raw data files...")
    raw = {}

    def _load(name, fname):
        path = os.path.join(RAW_DIR, fname)
        if not os.path.exists(path):
            print(f"    MISSING: {fname}")
            return pd.DataFrame()
        df = pd.read_csv(path, low_memory=False)
        print(f"    {name}: {len(df):,} rows")
        return df

    raw["chad"]     = _load("Chadwick register",      "raw_chadwick.csv")
    raw["fg_pit"]   = _load("FG pitching stats",      "raw_fg_pitching_sp.csv")
    raw["ars"]      = _load("Statcast arsenal",        "raw_statcast_arsenal_sp.csv")
    raw["sc_exp"]   = _load("Statcast expected SP",   "raw_statcast_expected_sp.csv")
    raw["team_bat"] = _load("Team batting",            "raw_team_batting_opp.csv")
    raw["mgr"]      = _load("Manager removal stats",  "raw_manager_removal_stats.csv")
    raw["mgr_pri"]  = _load("Manager priors",         "raw_manager_priors.csv")

    frames = []
    for yr in GAME_YEARS:
        p = os.path.join(RAW_DIR, f"raw_retrosheet_{yr}.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p, low_memory=False))
    raw["retro"] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    print(f"    Retrosheet logs: {len(raw['retro']):,} games")
    return raw


# =============================================================================
# BUILD ID MAPS
# =============================================================================
def build_id_maps(chad: pd.DataFrame) -> tuple:
    if chad.empty:
        return {}, {}
    retro_to_mlbam = {}
    if "key_retro" in chad.columns:
        sub = chad.dropna(subset=["key_retro", "key_mlbam"])
        retro_to_mlbam = dict(zip(sub["key_retro"].astype(str),
                                   sub["key_mlbam"].astype(int)))
    mlbam_to_fg = {}
    sub2 = chad.dropna(subset=["key_mlbam", "key_fangraphs"])
    mlbam_to_fg = dict(zip(sub2["key_mlbam"].astype(int),
                            sub2["key_fangraphs"].astype(int)))
    print(f"    {len(retro_to_mlbam):,} retro→MLBAM | {len(mlbam_to_fg):,} MLBAM→FG")
    return retro_to_mlbam, mlbam_to_fg


# =============================================================================
# BUILD SP FEATURE LOOKUP  keyed by (mlbam, season)
# =============================================================================
def build_sp_feature_lookup(fg_pit: pd.DataFrame,
                              sc_exp: pd.DataFrame,
                              ars: pd.DataFrame,
                              mlbam_to_fg: dict) -> dict:
    """
    Returns sp_lookup[(mlbam, season)] with all pitcher features.
    Uses prior-year stats (season - 1) to avoid look-ahead bias.

    Key features for survival model:
      pitches_per_pa  : rate of pitch count accumulation (most important)
      k_pct / bb_pct  : efficiency — high K, low BB = deeper starts
      csw_pct         : called-strike + whiff — quick outs
      xwoba_against   : contact quality allowed — higher = more HBP/H = more BF
      obp_proxy       : used to estimate P(out per PA) in simulation
    """
    sp_lookup = {}

    # ── FanGraphs features ──────────────────────────────────────────────────
    if not fg_pit.empty:
        FG_RENAME = {
            "K%": "k_pct", "BB%": "bb_pct", "K-BB%": "k_minus_bb_pct",
            "SIERA": "siera", "xFIP": "xfip", "FIP": "fip",
            "SwStr%": "swstr_pct", "F-Strike%": "fstrike_pct", "CSW%": "csw_pct",
        }
        fg = fg_pit.rename(columns={k: v for k, v in FG_RENAME.items()
                                     if k in fg_pit.columns})

        # Normalise team abbreviation
        if "Team" in fg.columns:
            fg["team_std"] = fg["Team"].map(FG_TO_STD).fillna(fg["Team"])

        # Normalise percentage columns (FG returns 0.255 for 25.5%)
        for pct_col in ["k_pct", "bb_pct", "k_minus_bb_pct",
                         "swstr_pct", "fstrike_pct", "csw_pct"]:
            if pct_col in fg.columns:
                vals = pd.to_numeric(fg[pct_col], errors="coerce")
                if vals.dropna().max() < 1.5:
                    fg[pct_col] = vals * 100

        FG_FEAT_COLS = ["k_pct", "bb_pct", "k_minus_bb_pct", "siera", "xfip",
                        "fip", "swstr_pct", "fstrike_pct", "csw_pct",
                        "pitches_per_pa", "avg_bf_per_start", "outs_per_start",
                        "ip_per_start", "team_std"]

        # Merge with Chadwick to get MLBAM key
        # FanGraphs uses IDfg
        if "IDfg" in fg.columns or "playerid" in fg.columns:
            id_col = "IDfg" if "IDfg" in fg.columns else "playerid"
            fg.rename(columns={id_col: "IDfg"}, inplace=True)
            fg["IDfg"] = pd.to_numeric(fg["IDfg"], errors="coerce")

            # Reverse map fg→mlbam
            fg_to_mlbam = {v: k for k, v in mlbam_to_fg.items()}

            for _, row in fg.iterrows():
                fgid = int(row["IDfg"]) if pd.notna(row.get("IDfg")) else None
                yr   = int(row["Season"]) if pd.notna(row.get("Season")) else None
                if not fgid or not yr:
                    continue
                mlbam = fg_to_mlbam.get(fgid)
                if not mlbam:
                    continue
                entry = sp_lookup.setdefault((mlbam, yr), {})
                for col in FG_FEAT_COLS:
                    if col in row.index:
                        entry[col] = row[col]

    # ── Statcast expected stats ──────────────────────────────────────────
    if not sc_exp.empty:
        SC_RENAME = {
            "est_woba": "xwoba_against",
            "barrel_percent": "barrel_pct",
            "hard_hit_percent": "hard_hit_pct",
        }
        sc = sc_exp.rename(columns={k: v for k, v in SC_RENAME.items()
                                     if k in sc_exp.columns})
        for _, row in sc.iterrows():
            mlbam = int(row["key_mlbam"]) if pd.notna(row.get("key_mlbam")) else None
            yr    = int(row["Season"])    if pd.notna(row.get("Season")) else None
            if not mlbam or not yr:
                continue
            entry = sp_lookup.setdefault((mlbam, yr), {})
            for col in ["xwoba_against", "barrel_pct", "hard_hit_pct"]:
                if col in row.index:
                    entry[col] = row[col]

    # ── Arsenal CSW% ──────────────────────────────────────────────────────
    if not ars.empty:
        # Aggregate: PA-weighted CSW% and avg fastball velocity
        for (mlbam, yr), grp in ars.groupby(["key_mlbam", "Season"]):
            usage_col = next((c for c in ["pitch_percent", "pitch_usage"]
                              if c in grp.columns), None)
            csw_col   = next((c for c in ["csw_percent", "csw_pct"]
                              if c in grp.columns), None)

            entry = sp_lookup.setdefault((int(mlbam), int(yr)), {})

            if csw_col and usage_col:
                grp = grp.copy()
                grp[usage_col] = pd.to_numeric(grp[usage_col], errors="coerce")
                grp[csw_col]   = pd.to_numeric(grp[csw_col],   errors="coerce")
                denom = grp[usage_col].sum()
                if denom > 0:
                    entry["csw_pct_arsenal"] = float(
                        (grp[usage_col] * grp[csw_col]).sum() / denom
                    )

            # Primary fastball velocity
            fb = grp[grp.get("pitch_type", pd.Series()).isin(
                {"FF", "SI", "FT", "FC"}) if "pitch_type" in grp.columns
                else pd.Series([False] * len(grp))
            ]
            speed_col = next((c for c in ["avg_speed", "avg_velocity"]
                               if c in grp.columns), None)
            if not fb.empty and speed_col:
                entry.setdefault("avg_fb_velo", float(
                    pd.to_numeric(fb[speed_col], errors="coerce").mean()
                ))

    print(f"    SP feature lookup: {len(sp_lookup):,} (mlbam, season) entries")
    return sp_lookup


# =============================================================================
# BUILD MANAGER HAZARD LOOKUP  keyed by (team, season)
# =============================================================================
def build_manager_hazard_lookup(mgr: pd.DataFrame,
                                  mgr_pri: pd.DataFrame) -> dict:
    """
    Returns manager_lookup[(team, season)] = {
        typical_pc_limit, hard_pc_limit, ttop_hook_rate,
        depth_score, blended_avg_sp_outs, manager_name
    }
    """
    lookup = {}

    # Prefer calibrated stats, fall back to priors
    primary = mgr if not mgr.empty else mgr_pri

    if primary.empty:
        return lookup

    for _, row in primary.iterrows():
        team   = str(row.get("team", ""))
        season = int(row["season"]) if pd.notna(row.get("season")) else 2025
        lookup[(team, season)] = {
            "typical_pc_limit":    float(row.get("typical_pc_limit", 95)),
            "hard_pc_limit":       float(row.get("hard_pc_limit", 105)),
            "ttop_hook_rate":      float(row.get("blended_ttop_hook_rate",
                                                   row.get("ttop_hook_rate", 0.35))),
            "depth_score":         float(row.get("depth_score", 0.52)),
            "avg_sp_outs":         float(row.get("blended_avg_sp_outs",
                                                   row.get("avg_sp_outs", 15.0))),
            "manager_name":        str(row.get("manager", "Unknown")),
        }

    print(f"    Manager hazard lookup: {len(lookup):,} (team, season) entries")
    return lookup


# =============================================================================
# BUILD OPPONENT FEATURE LOOKUP  keyed by (team, season)
# =============================================================================
def build_opponent_lookup(team_bat: pd.DataFrame) -> dict:
    """
    Opponent walk rate (BB%) is the key driver of pitch-count acceleration.
    Also capture wRC+ and K% for completeness.
    """
    if team_bat.empty:
        return {}

    FG_TEAM_RENAME = {
        "CHW": "CWS", "SD": "SDP", "SF": "SFG", "TB": "TBR",
        "KC": "KCR", "WAS": "WSN",
    }

    team_col = next((c for c in ["Team", "Tm", "team"] if c in team_bat.columns), None)
    if not team_col:
        return {}

    team_bat = team_bat.copy()
    team_bat["team_std"] = team_bat[team_col].map(FG_TEAM_RENAME).fillna(
        team_bat[team_col]
    )

    # Normalise BB% / K%
    for col in ["BB%", "K%"]:
        if col in team_bat.columns:
            vals = pd.to_numeric(team_bat[col].astype(str)
                                 .str.replace("%", "", regex=False), errors="coerce")
            if vals.dropna().max() > 1.5:
                vals /= 100.0
            team_bat[col.lower().replace("%", "_pct")] = vals

    opp_lookup = {}
    for _, row in team_bat.iterrows():
        team = str(row["team_std"])
        yr   = int(row.get("Season", row.get("season", 2025)))
        opp_lookup[(team, yr)] = {
            "opp_bb_pct":   float(row.get("bb_pct", row.get("BB%", 0.085))),
            "opp_k_pct":    float(row.get("k_pct",  row.get("K%",  0.230))),
            "opp_wrc_plus": float(row.get("wRC+", row.get("wrc_plus", 100.0))),
            "opp_obp":      float(row.get("OBP", row.get("obp", 0.315))),
        }

    print(f"    Opponent lookup: {len(opp_lookup):,} (team, season) entries")
    return opp_lookup


# =============================================================================
# EXTRACT PER-START SP DATA FROM RETROSHEET
# =============================================================================
def extract_per_start_data(retro: pd.DataFrame,
                            retro_to_mlbam: dict) -> list:
    """
    Returns list of dicts, one per SP start:
      game_date, season, team, sp_retro, sp_mlbam,
      outs_recorded (if available), home_flag
    """
    if retro.empty:
        return []

    date_col = next((c for c in ["date", "game_date", "Date"]
                     if c in retro.columns), None)
    if not date_col:
        return []

    records = []
    for _, game in retro.iterrows():
        gdate  = str(game[date_col])[:10]
        season = int(gdate[:4])

        sides = [
            ("h", "home_team_id",     "h_team",  "h_starting_pitcher_id",
             "h_pitching_1_outs"),
            ("v", "visiting_team_id", "v_team",  "v_starting_pitcher_id",
             "v_pitching_1_outs"),
        ]
        for (side, team_col1, team_col2, sp_col, sp_outs_col) in sides:
            team_retro = str(game.get(team_col1, game.get(team_col2, "")))
            team_std   = RETRO_TO_STD.get(team_retro, team_retro)
            sp_retro   = str(game.get(sp_col, ""))

            outs = np.nan
            if sp_outs_col in game.index and pd.notna(game[sp_outs_col]):
                outs = float(game[sp_outs_col])
            # Fallback: check alternative column names
            for alt in [f"{side}_sp_outs", f"{side}_starting_pitcher_outs"]:
                if alt in game.index and pd.notna(game[alt]) and np.isnan(outs):
                    outs = float(game[alt])

            records.append({
                "game_date":     gdate,
                "season":        season,
                "team":          team_std,
                "sp_retro":      sp_retro,
                "sp_mlbam":      retro_to_mlbam.get(sp_retro),
                "outs_recorded": outs,
                "home_flag":     1 if side == "h" else 0,
            })

    n_labeled = sum(1 for r in records if not np.isnan(r["outs_recorded"]))
    print(f"    {len(records):,} starts extracted | {n_labeled:,} with outs data")
    return records


# =============================================================================
# BUILD PER-START FEATURE MATRIX
# =============================================================================
def build_per_start_dataset(starts: list,
                              sp_lookup: dict,
                              manager_lookup: dict,
                              opp_lookup: dict) -> pd.DataFrame:
    """
    Assembles one row per SP start with:
      - Prior-year SP features (key: (sp_mlbam, season-1))
      - Manager hazard features (key: (home_team, season))
      - Opponent features (key: (opp_team, season))
      - TTOP interaction: sp_pitches_per_pa * opp_bb_pct (compounded accumulation)
      - Target: outs_recorded (actual outs from retrosheet, or imputed from FG)
      - Censored: 1 if outs_recorded >= CG_THRESHOLD (right-censored)
    """
    rows = []
    for rec in starts:
        sp_mlbam = rec.get("sp_mlbam")
        team     = rec.get("team", "")
        opp_team = rec.get("opp_team", "")  # populated in scoring; NA in training
        season   = rec.get("season", 2025)

        if not sp_mlbam:
            continue

        # Prior-year SP features
        sp_feat = sp_lookup.get((int(sp_mlbam), int(season) - 1), {})
        if not sp_feat:
            # Try current season as fallback (use only if prior unavailable)
            sp_feat = sp_lookup.get((int(sp_mlbam), int(season)), {})
            if not sp_feat:
                continue  # skip starts with no SP data

        # Manager features (team is the SP's team)
        mgr_feat = manager_lookup.get((team, season), {})
        if not mgr_feat:
            # Try most recent season for this team
            for s in sorted([k[1] for k in manager_lookup if k[0] == team], reverse=True):
                mgr_feat = manager_lookup.get((team, s), {})
                if mgr_feat:
                    break

        # Opponent features (for training use league averages since we track home/away)
        # In scoring, the actual opponent is passed in
        opp_feat = opp_lookup.get((opp_team, season), {})
        if not opp_feat:
            # Use league-average opponent as proxy for training rows
            all_opp = [v for (t, s), v in opp_lookup.items() if s == season]
            if all_opp:
                opp_feat = {
                    "opp_bb_pct":   np.mean([x["opp_bb_pct"]  for x in all_opp]),
                    "opp_k_pct":    np.mean([x["opp_k_pct"]   for x in all_opp]),
                    "opp_wrc_plus": np.mean([x["opp_wrc_plus"] for x in all_opp]),
                    "opp_obp":      np.mean([x["opp_obp"]      for x in all_opp]),
                }

        # Pitch count accumulation rate:
        # how many pitches does this SP throw per PA? × how often does opponent walk?
        pitches_per_pa = float(sp_feat.get("pitches_per_pa", 3.75))
        opp_bb_pct     = float(opp_feat.get("opp_bb_pct",   0.085))

        # Effective pitches per PA: SP base rate + 15% extra for high-walk lineups
        # Rationale: walks add pitch-count without recording outs.  A walk-heavy
        # lineup both extends PAs AND adds baserunners that inflate the SP's workload.
        effective_ppp = pitches_per_pa * (1.0 + 0.15 * (opp_bb_pct / 0.085 - 1.0))

        # Estimated pitch count at the 18th batter (TTOP decision point)
        est_pc_at_bf18 = effective_ppp * 18.0
        typical_pc_limit = float(mgr_feat.get("typical_pc_limit", 95))

        row = {
            "game_date":           rec.get("game_date"),
            "season":              season,
            "team":                team,
            "sp_mlbam":            sp_mlbam,
            "outs_recorded":       rec.get("outs_recorded", np.nan),
            "censored":            0,  # set below

            # SP efficiency features (prior-year)
            "k_pct":               float(sp_feat.get("k_pct",          22.0)),
            "bb_pct":              float(sp_feat.get("bb_pct",           8.0)),
            "k_minus_bb_pct":      float(sp_feat.get("k_minus_bb_pct",  14.0)),
            "siera":               float(sp_feat.get("siera",            4.20)),
            "xfip":                float(sp_feat.get("xfip",             4.10)),
            "swstr_pct":           float(sp_feat.get("swstr_pct",        10.5)),
            "fstrike_pct":         float(sp_feat.get("fstrike_pct",      63.0)),
            "csw_pct":             float(sp_feat.get("csw_pct",
                                          sp_feat.get("csw_pct_arsenal", 28.0))),
            "avg_fb_velo":         float(sp_feat.get("avg_fb_velo",      93.0)),
            "xwoba_against":       float(sp_feat.get("xwoba_against",     0.330)),
            "barrel_pct":          float(sp_feat.get("barrel_pct",        7.5)),

            # Pitch count efficiency
            "pitches_per_pa":      pitches_per_pa,
            "effective_ppp":       effective_ppp,

            # Opponent features
            "opp_bb_pct":          float(opp_feat.get("opp_bb_pct",      0.085)),
            "opp_k_pct":           float(opp_feat.get("opp_k_pct",       0.230)),
            "opp_wrc_plus":        float(opp_feat.get("opp_wrc_plus",   100.0)),
            "opp_obp":             float(opp_feat.get("opp_obp",         0.315)),

            # Manager hazard features
            "typical_pc_limit":    typical_pc_limit,
            "hard_pc_limit":       float(mgr_feat.get("hard_pc_limit",  105)),
            "ttop_hook_rate":      float(mgr_feat.get("ttop_hook_rate",   0.35)),
            "depth_score":         float(mgr_feat.get("depth_score",      0.52)),
            "mgr_avg_sp_outs":     float(mgr_feat.get("avg_sp_outs",     15.0)),

            # Pre-computed TTOP-relevant states
            "est_pc_at_bf18":      est_pc_at_bf18,
            "pc_fraction_at_bf18": est_pc_at_bf18 / max(typical_pc_limit, 1),

            # Interaction: high pitcher + low manager patience = short outing
            "efficiency_x_depth":  float(sp_feat.get("k_minus_bb_pct", 14.0))
                                   * float(mgr_feat.get("depth_score", 0.52)),
            # Pitch count pressure at TTOP: does SP have pitch count room at BF18?
            "pc_headroom_at_ttop": max(0.0, typical_pc_limit - est_pc_at_bf18),
        }

        # Censoring: complete game or near-CG (manager's choice doesn't apply)
        outs = row["outs_recorded"]
        if pd.notna(outs) and outs >= CG_THRESHOLD_OUTS:
            row["censored"] = 1

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"    {len(df):,} per-start rows built")
    if "outs_recorded" in df.columns:
        n_labeled = df["outs_recorded"].notna().sum()
        print(f"    {n_labeled:,} starts with observed outs")
    return df


# =============================================================================
# EXPAND TO BATTER-FACED LEVEL  (discrete-time survival expansion)
# =============================================================================
def expand_to_bf_level(starts_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each start row, create one row per batter faced from 1 to observed_bf.

    The observed_bf (total batters faced) is estimated from outs_recorded:
      BF ≈ outs / (1 - OBP_against) = outs / out_rate_per_bf
      Using out_rate = 1 - opp_obp (or LEAGUE_AVG_OUT_RATE if opp_obp missing)

    At each BF row k:
      event = 1 only at k == observed_bf and censored == 0
      event = 0 for all k < observed_bf (survived)
      For censored starts: event = 0 for all k up to observed_bf

    Time-varying features computed at each BF k:
      bf_k                 : current batters faced count
      times_through_order  : ceil(k / 9)  [1, 2, 3, 4+]
      is_ttop              : bf_k >= TTOP_BF_START (19)
      is_18th_batter       : bf_k == 18  (2nd-through completion = decision point)
      is_19th_batter       : bf_k == 19  (first batter of TTOP)
      batters_into_ttop    : max(0, bf_k - 18)
      est_pc_k             : effective_ppp × bf_k
      pc_fraction_k        : est_pc_k / typical_pc_limit
      approaching_pc_limit : pc_fraction_k >= 0.85
      at_pc_limit          : pc_fraction_k >= 1.0

    The TTOP penalty and pitch-count ceiling are "hard mathematical penalties":
    they are baked into the feature matrix as step functions that the hazard
    model will learn to weight correctly via its training data.
    """
    if starts_df.empty:
        return pd.DataFrame()

    bf_rows = []
    start_id_col = "start_id"
    starts_df = starts_df.copy()
    starts_df[start_id_col] = np.arange(len(starts_df))

    for _, row in starts_df.iterrows():
        outs   = row.get("outs_recorded", np.nan)
        opp_obp = float(row.get("opp_obp", LEAGUE_AVG_OUT_RATE))
        out_rate = max(0.5, min(0.85, 1.0 - opp_obp))  # P(out per BF)

        # Estimate observed BF from outs
        if pd.notna(outs) and outs > 0:
            observed_bf = int(round(float(outs) / out_rate))
            observed_bf = max(3, min(MAX_BF_MODELED, observed_bf))
        else:
            continue  # skip unlabeled starts for expansion

        censored   = int(row.get("censored", 0))
        eff_ppp    = float(row.get("effective_ppp",    row.get("pitches_per_pa", 3.75)))
        pc_limit   = float(row.get("typical_pc_limit", 95.0))
        hard_limit = float(row.get("hard_pc_limit",   105.0))

        # Time-invariant features carried into every BF row
        base_feats = {col: row[col] for col in row.index
                      if col not in {start_id_col, "outs_recorded", "censored",
                                     "game_date", "sp_mlbam", "team",
                                     "est_pc_at_bf18", "pc_fraction_at_bf18"}}
        base_feats[start_id_col] = int(row[start_id_col])
        base_feats["outs_recorded"] = float(outs)
        base_feats["censored"]      = censored

        for k in range(1, observed_bf + 1):
            # Time-varying feature block
            times_through = int(np.ceil(k / 9))
            is_ttop       = int(k >= TTOP_BF_START)
            bf_into_ttop  = max(0, k - SECOND_THROUGH_END)
            est_pc_k      = eff_ppp * k
            pc_frac_k     = est_pc_k / max(pc_limit, 1.0)

            bf_feats = {
                **base_feats,
                "bf_k":                   k,
                "times_through_order":    times_through,
                "is_first_through":       int(times_through == 1),
                "is_second_through":      int(times_through == 2),
                "is_ttop":                is_ttop,
                "is_18th_batter":         int(k == SECOND_THROUGH_END),       # decision point
                "is_19th_batter":         int(k == TTOP_BF_START),            # TTOP onset
                "batters_into_ttop":      bf_into_ttop,
                "est_pc_k":               est_pc_k,
                "pc_fraction_k":          pc_frac_k,
                "approaching_pc_limit":   int(pc_frac_k >= 0.85),
                "at_pc_limit":            int(pc_frac_k >= 1.0),
                "past_hard_limit":        int(est_pc_k >= hard_limit),
                # Interaction: TTOP × low patience manager
                "ttop_x_low_patience":    is_ttop * (1.0 - float(row.get("depth_score", 0.52))),
                # Pitch count stress index at this BF
                "pc_stress_k":            max(0.0, pc_frac_k - 0.7) * 10.0,
                # Event label: 1 only at the last BF row and not censored
                "event":                  int(k == observed_bf and censored == 0),
            }
            bf_rows.append(bf_feats)

    bf_df = pd.DataFrame(bf_rows)
    print(f"    BF-level expansion: {len(bf_df):,} rows "
          f"({len(starts_df):,} starts × avg {len(bf_df)/len(starts_df):.1f} BF/start)")
    print(f"    Event rate (removal): {bf_df['event'].mean():.4f} per BF row")
    return bf_df


# =============================================================================
# IMPUTE MISSING OUTS FROM FG SEASON AVERAGES
# =============================================================================
def impute_outs_from_fg_averages(starts_df: pd.DataFrame,
                                   sp_lookup: dict) -> pd.DataFrame:
    """
    For starts where outs_recorded is missing (retrosheet lacked pitching_1_outs),
    impute using the SP's season-average outs/start from FanGraphs.

    These imputed rows get censored=1 to downweight their survival contribution.
    They still provide value for the time-invariant features in the hazard model.
    """
    df = starts_df.copy()
    missing_mask = df["outs_recorded"].isna()
    n_missing = missing_mask.sum()

    if n_missing == 0:
        return df

    print(f"    Imputing {n_missing:,} starts with missing outs from FG season averages...")
    for idx in df[missing_mask].index:
        row    = df.loc[idx]
        mlbam  = row.get("sp_mlbam")
        season = row.get("season")
        if pd.notna(mlbam) and pd.notna(season):
            feat = sp_lookup.get((int(mlbam), int(season)), {})
            if "outs_per_start" in feat and pd.notna(feat["outs_per_start"]):
                df.at[idx, "outs_recorded"] = float(feat["outs_per_start"])
                df.at[idx, "censored"]       = 1  # Imputed → treat as censored
    n_filled = df.loc[missing_mask, "outs_recorded"].notna().sum()
    print(f"    Filled {n_filled:,} / {n_missing:,} missing starts via FG averages.")
    return df


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("PITCHER TOTAL OUTS MODEL — STEP 2: FEATURE ENGINEERING (SURVIVAL)")
    print("=" * 70)

    print("\n[ 1/6 ] Loading raw data...")
    raw = load_raw()

    print("\n[ 2/6 ] Building ID maps...")
    retro_to_mlbam, mlbam_to_fg = build_id_maps(raw["chad"])

    print("\n[ 3/6 ] Building SP feature lookup (prior-year Statcast + FG)...")
    sp_lookup = build_sp_feature_lookup(raw["fg_pit"], raw["sc_exp"],
                                         raw["ars"], mlbam_to_fg)

    print("\n[ 4/6 ] Building manager hazard lookup...")
    manager_lookup = build_manager_hazard_lookup(raw["mgr"], raw["mgr_pri"])

    print("\n[ 5/6 ] Building opponent lookup (walk rates)...")
    opp_lookup = build_opponent_lookup(raw["team_bat"])

    print("\n[ 6/6 ] Assembling per-start dataset + BF-level expansion...")
    print("  Extracting per-start data from retrosheet logs...")
    starts = extract_per_start_data(raw["retro"], retro_to_mlbam)

    if not starts:
        print("  ERROR: No starts extracted from retrosheet. Check retrosheet data.")
        exit(1)

    starts_df = build_per_start_dataset(starts, sp_lookup, manager_lookup, opp_lookup)
    starts_df = impute_outs_from_fg_averages(starts_df, sp_lookup)

    # Drop starts still missing outs (can't train without target)
    starts_df = starts_df.dropna(subset=["outs_recorded"])
    starts_df["outs_recorded"] = starts_df["outs_recorded"].clip(lower=0, upper=27)
    print(f"\n  Per-start dataset: {len(starts_df):,} labeled starts")
    print(f"  Mean outs/start:   {starts_df['outs_recorded'].mean():.2f}")
    print(f"  Censored (CG):     {starts_df['censored'].sum():,} "
          f"({starts_df['censored'].mean():.1%})")

    outs_path = os.path.join(PROC_DIR, "pitcher_outs_per_start.csv")
    starts_df.to_csv(outs_path, index=False)
    print(f"  ✓ Saved → {outs_path}")

    # BF-level expansion for hazard model
    bf_df = expand_to_bf_level(starts_df)
    if not bf_df.empty:
        bf_path = os.path.join(PROC_DIR, "pitcher_outs_bf_level.csv")
        bf_df.to_csv(bf_path, index=False)
        print(f"  ✓ BF-level dataset saved → {bf_path}")
        print(f"    Columns: {list(bf_df.columns)}")

    print("\n" + "=" * 70)
    print("STEP 2 COMPLETE — Run 03_analysis_pitcher_outs.py next.")
    print("=" * 70)
