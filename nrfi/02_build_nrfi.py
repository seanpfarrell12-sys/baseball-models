"""
=============================================================================
NRFI / YRFI MODEL — FILE 2 OF 4: FEATURE ENGINEERING
=============================================================================
Purpose : Build a per-game dataset where each row is one MLB game and the
          target is YRFI (1 = at least one run scored in the 1st inning by
          either team; 0 = no runs in the 1st inning = NRFI).

One-row-per-game design:
  Each game contributes exactly ONE training row.  Both SPs' first-inning
  profiles are captured as parallel feature columns (home_sp_* / away_sp_*),
  and both batting lineups' top-3 platoon stats are captured as
  home_top3_* / away_top3_*.

YRFI label construction (from Statcast first-inning pitch data):
  - Top half of 1st (inning_topbot='Top'): away team bats vs home SP.
    If max(away_score) > 0 for that game_pk, away team scored.
  - Bottom half of 1st (inning_topbot='Bot'): home team bats vs away SP.
    If max(home_score) > 0 for that game_pk, home team scored.
  - yrfi = 1 if EITHER condition is met.

SP first-inning stats:
  Derived from Statcast pitch data (inning=1 only).  For each pitcher, the
  terminal pitch of each PA gives the event (K, BB, HR, hit, etc.).  We
  compute per-start then season-aggregate.  Features are PRIOR-YEAR values
  to avoid look-ahead bias (same (pitcher_mlbam, season-1) key approach as
  all other models).

Lineup top-3 features:
  Retrosheet game logs identify the confirmed batting order.  Slots 1-2-3
  are the only hitters guaranteed to face the opposing SP in inning=1.
  Batting platoon splits (wRC+, ISO, OBP, K%, BB%) are split by SP hand
  (vs LHP / vs RHP) via the FanGraphs splits files.
  Retrosheet player IDs → MLBAM IDs → FG IDs via Chadwick register.

Environmental features:
  - Wind component toward center field (wind_toward_cf).
    Positive = tail wind (HR-friendly); negative = head wind (HR-suppressing).
  - Temperature carry factor (based on 1% HR increase per degree F above 70°F).
  - Park HR factor from PARK_HR_FACTORS dict (in 01_input_nrfi.py).
  - Altitude: higher altitude = less air resistance = more carry.
  - Dome indicator: weather irrelevant for dome/retractable-closed parks.

Input  : data/raw/ (raw_nrfi_statcast_*.csv, raw_fg_pitching_nrfi.csv,
                    raw_batting_splits_lhp.csv, raw_batting_splits_rhp.csv,
                    raw_weather_historical.csv, raw_nrfi_park_meta.json,
                    raw_chadwick.csv, raw_retrosheet_*.csv)
Output : data/processed/nrfi_dataset.csv
=============================================================================
"""

import os
import json
import math
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

GAME_YEARS = [2023, 2024, 2025]

# Retrosheet team code → standard abbreviation (used across all models)
RETRO_TO_STD = {
    "ANA": "LAA", "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHA": "CWS", "CHN": "CHC", "CIN": "CIN", "CLE": "CLE", "COL": "COL",
    "DET": "DET", "HOU": "HOU", "KCA": "KCR", "LAN": "LAD", "MIA": "MIA",
    "MIL": "MIL", "MIN": "MIN", "NYA": "NYY", "NYN": "NYM", "OAK": "OAK",
    "PHI": "PHI", "PIT": "PIT", "SDN": "SDP", "SEA": "SEA", "SFN": "SFG",
    "SLN": "STL", "TBA": "TBR", "TEX": "TEX", "TOR": "TOR", "WAS": "WSN",
    "FLO": "MIA",
}

# ─────────────────────────────────────────────────────────────────────────────
# Temperature HR carry model: each degree F above 70°F adds ~0.5% HR distance.
# Empirical estimate: ~2% HR rate increase per 10°F (source: Alan Nathan physics)
TEMP_HR_CARRY_PER_DEGREE = 0.002   # fractional change per °F above 70°F
TEMP_BASE_F              = 70.0

# Altitude HR boost: thin air at Coors (5200 ft) is well established.
# ~1% HR boost per 1000 ft (rough estimate from park factors and physics).
ALT_HR_PER_1000FT = 0.01


# =============================================================================
# LOAD RAW DATA
# =============================================================================

def load_statcast_first_inning() -> pd.DataFrame:
    """Load and concatenate all raw_nrfi_statcast_*.csv files."""
    frames = []
    for yr in GAME_YEARS:
        path = os.path.join(RAW_DIR, f"raw_nrfi_statcast_{yr}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, low_memory=False)
            df["season"] = yr
            frames.append(df)
        else:
            print(f"  WARNING: {path} not found — run 01_input_nrfi.py first")
    if not frames:
        raise FileNotFoundError("No first-inning Statcast files found.")
    df = pd.concat(frames, ignore_index=True)
    # Standardize pitcher column → key_mlbam
    if "pitcher" in df.columns:
        df.rename(columns={"pitcher": "key_mlbam"}, inplace=True)
    df["key_mlbam"] = pd.to_numeric(df["key_mlbam"], errors="coerce")
    # game_date to datetime
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    print(f"  Statcast 1st-inn loaded: {len(df):,} pitches across "
          f"{df['game_pk'].nunique():,} games")
    return df


def load_fg_pitching() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "raw_fg_pitching_nrfi.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found — run 01_input_nrfi.py first")
    df = pd.read_csv(path, low_memory=False)
    # Normalise FG id column
    for col in ["IDfg", "playerid", "PlayerID"]:
        if col in df.columns:
            df.rename(columns={col: "IDfg"}, inplace=True)
            break
    df["IDfg"]   = pd.to_numeric(df.get("IDfg"), errors="coerce")
    df["Season"] = pd.to_numeric(df.get("Season"), errors="coerce")
    print(f"  FG pitching loaded: {len(df):,} pitcher-seasons")
    return df


def load_batting_splits() -> tuple:
    lhp_path = os.path.join(RAW_DIR, "raw_batting_splits_lhp.csv")
    rhp_path = os.path.join(RAW_DIR, "raw_batting_splits_rhp.csv")
    if not os.path.exists(lhp_path) or not os.path.exists(rhp_path):
        print("  WARNING: batting splits files not found — "
              "top-3 platoon features will be null")
        return pd.DataFrame(), pd.DataFrame()
    lhp = pd.read_csv(lhp_path, low_memory=False)
    rhp = pd.read_csv(rhp_path, low_memory=False)
    for df in [lhp, rhp]:
        for col in ["IDfg", "playerid", "PlayerID"]:
            if col in df.columns:
                df.rename(columns={col: "IDfg"}, inplace=True)
                break
        df["IDfg"]   = pd.to_numeric(df.get("IDfg"), errors="coerce")
        df["season"] = pd.to_numeric(df.get("season"), errors="coerce")
    print(f"  Batting splits loaded: LHP {len(lhp):,} | RHP {len(rhp):,} rows")
    return lhp, rhp


def load_weather() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "raw_weather_historical.csv")
    if not os.path.exists(path):
        print("  WARNING: weather file not found — environmental features will be null")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    print(f"  Weather loaded: {len(df):,} team-date records")
    return df


def load_park_meta() -> tuple:
    path = os.path.join(RAW_DIR, "raw_nrfi_park_meta.json")
    if not os.path.exists(path):
        print("  WARNING: park meta not found — using empty dicts")
        return {}, {}
    with open(path) as f:
        data = json.load(f)
    return data.get("stadium_meta", {}), data.get("park_hr_factors", {})


def load_chadwick() -> pd.DataFrame:
    """Chadwick register: key_mlbam ↔ key_fangraphs ↔ key_retro bridge."""
    path = os.path.join(RAW_DIR, "raw_chadwick.csv")
    if not os.path.exists(path):
        print("  WARNING: raw_chadwick.csv not found — "
              "pitcher MLBAM→FGid bridge unavailable")
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False,
                     usecols=lambda c: c in [
                         "key_mlbam", "key_fangraphs", "key_retro",
                         "name_first", "name_last",
                     ])
    df["key_mlbam"]      = pd.to_numeric(df.get("key_mlbam"), errors="coerce")
    df["key_fangraphs"]  = pd.to_numeric(df.get("key_fangraphs"), errors="coerce")
    print(f"  Chadwick register: {len(df):,} player mappings")
    return df


def load_retrosheet() -> pd.DataFrame:
    """Load retrosheet game logs for all GAME_YEARS (batting order + teams)."""
    frames = []
    for yr in GAME_YEARS:
        path = os.path.join(RAW_DIR, f"raw_retrosheet_{yr}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, low_memory=False)
            df["season"] = yr
            frames.append(df)
        else:
            print(f"  WARNING: retrosheet {yr} not found — lineup info unavailable")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    print(f"  Retrosheet loaded: {len(df):,} games across {GAME_YEARS}")
    return df


# =============================================================================
# STEP 1: YRFI LABELS + SP IDENTIFICATION FROM STATCAST
# =============================================================================

def build_yrfi_game_index(sc: pd.DataFrame) -> pd.DataFrame:
    """
    For each game_pk, determine:
      - YRFI label (did any run score in inning=1?)
      - home_sp_mlbam / away_sp_mlbam (first pitcher seen in each half-inning)
      - sp handedness (p_throws)
      - game_date, season, home_team, away_team

    Logic:
      Top half (inning_topbot='Top'): away team bats, home SP pitches.
        YRFI contribution: max(away_score) > 0 (away scored in 1st).
      Bottom half (inning_topbot='Bot'): home team bats, away SP pitches.
        YRFI contribution: max(home_score) > 0 (home scored in 1st).

    SP identity = first pitcher row in each half-inning (= the starter).
    """
    # Sort by game_pk, inning_topbot, at_bat_number (or pitch_number)
    sort_cols = ["game_pk"]
    if "at_bat_number" in sc.columns:
        sort_cols += ["at_bat_number"]
    if "pitch_number" in sc.columns:
        sort_cols += ["pitch_number"]
    sc = sc.sort_values(sort_cols).copy()

    # ── Top half ──────────────────────────────────────────────────────────────
    top = sc[sc["inning_topbot"] == "Top"].copy()
    top_agg = top.groupby("game_pk").agg(
        game_date    = ("game_date",   "first"),
        season       = ("season",      "first"),
        home_team    = ("home_team",   "first"),
        away_team    = ("away_team",   "first"),
        away_runs_1  = ("away_score",  "max"),   # runs scored by away team in top 1st
        home_sp_mlbam= ("key_mlbam",  "first"),  # home pitcher (faces away batters)
        home_sp_hand = ("p_throws",   "first"),
    ).reset_index()

    # ── Bottom half ───────────────────────────────────────────────────────────
    bot = sc[sc["inning_topbot"] == "Bot"].copy()
    bot_agg = bot.groupby("game_pk").agg(
        home_runs_1  = ("home_score",  "max"),   # runs scored by home team in bot 1st
        away_sp_mlbam= ("key_mlbam",  "first"),  # away pitcher (faces home batters)
        away_sp_hand = ("p_throws",   "first"),
    ).reset_index()

    # ── Merge and compute label ───────────────────────────────────────────────
    games = top_agg.merge(bot_agg, on="game_pk", how="left")
    games["away_runs_1"] = games["away_runs_1"].fillna(0)
    games["home_runs_1"] = games["home_runs_1"].fillna(0)
    games["yrfi"]        = ((games["away_runs_1"] > 0) |
                            (games["home_runs_1"] > 0)).astype(int)

    # Standardize team abbreviations (Statcast uses full/alt codes sometimes)
    # — Statcast generally uses standard codes already; map just in case
    STD_MAP = {"KC": "KCR", "SD": "SDP", "SF": "SFG", "TB": "TBR",
               "WSH": "WSN", "CWS": "CWS"}
    games["home_team"] = games["home_team"].replace(STD_MAP)
    games["away_team"] = games["away_team"].replace(STD_MAP)

    print(f"  Game index: {len(games):,} games | "
          f"YRFI rate: {games['yrfi'].mean():.3f}")
    return games


# =============================================================================
# STEP 2: SP FIRST-INNING AGGREGATE STATS (prior-year, from Statcast)
# =============================================================================

def build_sp_first_inning_stats(sc: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate first-inning stats per (pitcher, season) from Statcast PA events.

    Returns one row per (pitcher_mlbam, season) with:
      fi_gs          — first-inning starts (games appeared as 1st-inn pitcher)
      fi_pa          — total batters faced in 1st inning
      fi_k_pct       — strikeout rate (K/PA)
      fi_bb_pct      — walk rate (BB/PA, includes HBP)
      fi_hr_per_9    — HR allowed per 9 innings (= HR / fi_gs × 9)
      fi_era         — estimated ERA: (runs_allowed / fi_gs) × 9
      fi_hits_per_9  — hits per 9 innings
      fi_whiff_pct   — swinging strike pct across all pitches in 1st inning

    Note: fi_era uses runs_allowed from score tracking (not ER), which is
    appropriate for NRFI because earned vs unearned distinction does not
    matter for a run to cross the plate.
    """
    # ── Per-game, per-pitcher first-inning runs allowed ───────────────────────
    # Track runs via score deltas: compare away/home_score to game start (=0)
    # In Top half: away_score is batting team; pitcher is the home SP
    # In Bot half: home_score is batting team; pitcher is the away SP

    top = sc[sc["inning_topbot"] == "Top"].groupby(
        ["game_pk", "key_mlbam", "season"]
    ).agg(runs_allowed=("away_score", "max")).reset_index()

    bot = sc[sc["inning_topbot"] == "Bot"].groupby(
        ["game_pk", "key_mlbam", "season"]
    ).agg(runs_allowed=("home_score", "max")).reset_index()

    runs_df = pd.concat([top, bot], ignore_index=True)

    per_game_runs = (runs_df.groupby(["key_mlbam", "season"])
                    .agg(
                        fi_gs          = ("game_pk",      "count"),
                        fi_runs_total  = ("runs_allowed", "sum"),
                    ).reset_index())

    # ── PA-level stats from terminal pitch events ─────────────────────────────
    pa = sc[sc["events"].notna()].copy()
    pa["is_k"]   = pa["events"].isin(
        ["strikeout", "strikeout_double_play"])
    pa["is_bb"]  = pa["events"].isin(
        ["walk", "intent_walk", "hit_by_pitch"])
    pa["is_hr"]  = pa["events"] == "home_run"
    pa["is_hit"] = pa["events"].isin(
        ["single", "double", "triple", "home_run"])

    pa_agg = (pa.groupby(["key_mlbam", "season"])
              .agg(
                  fi_pa    = ("events",  "count"),
                  fi_k     = ("is_k",    "sum"),
                  fi_bb    = ("is_bb",   "sum"),
                  fi_hr    = ("is_hr",   "sum"),
                  fi_hits  = ("is_hit",  "sum"),
              ).reset_index())

    # ── Pitch-level whiff rate ────────────────────────────────────────────────
    if "description" in sc.columns:
        whiff_map = {"swinging_strike": 1, "swinging_strike_blocked": 1,
                     "foul_tip": 0}  # foul_tip is a borderline; exclude
        sc["is_whiff"] = sc["description"].map(whiff_map).fillna(0)
        whiff_agg = (sc.groupby(["key_mlbam", "season"])
                     .agg(
                         pitch_count = ("description", "count"),
                         whiff_count = ("is_whiff",    "sum"),
                     ).reset_index())
        whiff_agg["fi_whiff_pct"] = (whiff_agg["whiff_count"] /
                                     whiff_agg["pitch_count"].clip(lower=1))
    else:
        whiff_agg = pa_agg[["key_mlbam", "season"]].copy()
        whiff_agg["fi_whiff_pct"] = np.nan

    # ── Combine ───────────────────────────────────────────────────────────────
    sp_stats = (per_game_runs
                .merge(pa_agg,    on=["key_mlbam", "season"], how="left")
                .merge(whiff_agg, on=["key_mlbam", "season"], how="left"))

    sp_stats["fi_pa"]       = sp_stats["fi_pa"].fillna(0)
    sp_stats["fi_gs"]       = sp_stats["fi_gs"].fillna(1).clip(lower=1)
    safe_pa                 = sp_stats["fi_pa"].clip(lower=1)

    sp_stats["fi_k_pct"]    = sp_stats["fi_k"]    / safe_pa
    sp_stats["fi_bb_pct"]   = sp_stats["fi_bb"]   / safe_pa
    sp_stats["fi_hits_per_9"]= sp_stats["fi_hits"] / sp_stats["fi_gs"] * 9
    sp_stats["fi_hr_per_9"] = sp_stats["fi_hr"]   / sp_stats["fi_gs"] * 9
    sp_stats["fi_era"]      = sp_stats["fi_runs_total"] / sp_stats["fi_gs"] * 9

    keep_cols = ["key_mlbam", "season",
                 "fi_gs", "fi_pa", "fi_era",
                 "fi_k_pct", "fi_bb_pct", "fi_hr_per_9",
                 "fi_hits_per_9", "fi_whiff_pct"]
    sp_stats = sp_stats[[c for c in keep_cols if c in sp_stats.columns]]
    print(f"  SP first-inning stats: {len(sp_stats):,} pitcher-season rows")
    return sp_stats


# =============================================================================
# STEP 3: FANGRAPHS STUFF+/LOCATION+ → SP FEATURES
# =============================================================================

def build_sp_fg_features(fg: pd.DataFrame, chad: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-pitcher FanGraphs features keyed by (key_mlbam, Season).

    Features: Stuff+, Location+, Pitching+, K%, BB%, SwStr%, F-Strike%, CSW%

    Chadwick join: fg.IDfg ↔ chad.key_fangraphs → chad.key_mlbam
    """
    # ── Bridge FG id → MLBAM ──────────────────────────────────────────────────
    if not chad.empty and "key_fangraphs" in chad.columns:
        bridge = chad[["key_fangraphs", "key_mlbam"]].dropna()
        fg = fg.merge(bridge.rename(columns={"key_fangraphs": "IDfg"}),
                      on="IDfg", how="left")

    # ── Select best available Stuff+/quality columns ──────────────────────────
    CANDIDATE_COLS = {
        "stuff_plus":    ["Stuff+", "stuff_plus", "Stuff"],
        "location_plus": ["Location+", "location_plus", "Location"],
        "pitching_plus": ["Pitching+", "pitching_plus", "Pitching+1"],
        "k_pct":         ["K%", "SO%", "K_pct"],
        "bb_pct":        ["BB%", "BB_pct", "uBB%"],
        "swstr_pct":     ["SwStr%", "swstr_pct", "Swing-Strike%"],
        "f_strike_pct":  ["F-Strike%", "fstrike_pct", "First Strike%"],
        "csw_pct":       ["CSW%", "csw_pct"],
    }

    result = fg[["IDfg", "Season"]].copy()
    if "key_mlbam" in fg.columns:
        result["key_mlbam"] = fg["key_mlbam"]

    for feat, candidates in CANDIDATE_COLS.items():
        for col in candidates:
            if col in fg.columns:
                result[feat] = pd.to_numeric(fg[col], errors="coerce")
                break
        else:
            result[feat] = np.nan

    # Convert FG percentage strings ("12.5%") to floats if needed
    for col in ["k_pct", "bb_pct", "swstr_pct", "f_strike_pct", "csw_pct"]:
        if col in result.columns and result[col].dtype == object:
            result[col] = (result[col].astype(str)
                           .str.replace("%", "", regex=False)
                           .pipe(pd.to_numeric, errors="coerce")
                           .div(100))

    result.rename(columns={"Season": "season"}, inplace=True)
    print(f"  SP FG features built: {len(result):,} rows "
          f"| Stuff+ coverage: {result['stuff_plus'].notna().mean():.1%}")
    return result


# =============================================================================
# STEP 4: JOIN SP FEATURES ONTO GAME INDEX (PRIOR-YEAR)
# =============================================================================

def join_sp_features(games: pd.DataFrame,
                     sp_fi: pd.DataFrame,
                     sp_fg: pd.DataFrame) -> pd.DataFrame:
    """
    Join SP features for BOTH the home SP and the away SP.

    Uses (key_mlbam, season-1) to avoid look-ahead bias.
    For partial seasons (when season-1 stats are unavailable), we attempt
    same-season stats with a fill.

    Returns games with columns prefixed home_sp_* and away_sp_*.
    """
    # Combine first-inning Statcast stats + FG stats into a single SP lookup
    # Key: (key_mlbam, season)
    if not sp_fg.empty and "key_mlbam" in sp_fg.columns:
        sp_all = sp_fi.merge(
            sp_fg.drop(columns=["IDfg"], errors="ignore"),
            on=["key_mlbam", "season"],
            how="left",
        )
    else:
        sp_all = sp_fi.copy()
        for col in ["stuff_plus", "location_plus", "pitching_plus",
                    "k_pct", "bb_pct", "swstr_pct", "f_strike_pct"]:
            sp_all[col] = np.nan

    # Feature columns to join (all columns except the key columns)
    feat_cols = [c for c in sp_all.columns
                 if c not in ("key_mlbam", "season")]

    def _join_one_sp(side: str, games_in: pd.DataFrame) -> pd.DataFrame:
        """Join SP features for 'home' or 'away' side."""
        sp_col  = f"{side}_sp_mlbam"
        hand_col = f"{side}_sp_hand"
        if sp_col not in games_in.columns:
            return games_in

        # Prior-year key
        lookup = sp_all.copy()
        lookup["join_season"] = lookup["season"] + 1   # stats from yr-1 join to yr

        g = games_in.copy()
        g["join_season"] = g["season"]

        merged = g.merge(
            lookup[["key_mlbam", "join_season"] + feat_cols]
            .rename(columns={"key_mlbam": sp_col,
                             **{c: f"{side}_sp_{c}" for c in feat_cols}}),
            on=[sp_col, "join_season"],
            how="left",
        ).drop(columns=["join_season"])

        # Fallback: same-season stats where prior-year is missing
        # (handles rookies / first partial season)
        missing_mask = merged[f"{side}_sp_fi_gs"].isna()
        if missing_mask.any():
            same_yr = g.merge(
                sp_all[["key_mlbam", "season"] + feat_cols]
                .rename(columns={"key_mlbam": sp_col,
                                 "season": "join_season",
                                 **{c: f"{side}_sp_{c}_same" for c in feat_cols}}),
                left_on=[sp_col, "season"],
                right_on=[sp_col, "join_season"],
                how="left",
            ).drop(columns=["join_season"], errors="ignore")

            for c in feat_cols:
                dest  = f"{side}_sp_{c}"
                src   = f"{side}_sp_{c}_same"
                if src in same_yr.columns:
                    fill_vals = same_yr.loc[missing_mask, src]
                    merged.loc[missing_mask, dest] = fill_vals.values

        return merged

    games = _join_one_sp("home", games)
    games = _join_one_sp("away", games)

    home_cov = games["home_sp_fi_era"].notna().mean()
    away_cov = games["away_sp_fi_era"].notna().mean()
    print(f"  SP feature join: home {home_cov:.1%} coverage | "
          f"away {away_cov:.1%} coverage")
    return games


# =============================================================================
# STEP 5: TOP-3 LINEUP EXTRACTION FROM STATCAST
# =============================================================================

def extract_top3_lineups(retro: pd.DataFrame,
                         chad: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the actual batting slots 1-2-3 for each team from the inning=1
    Statcast data (already pulled in Step 1).

    Logic:
      - away top-3 = first 3 unique batters with inning_topbot='Top',
                     ordered by at_bat_number ascending
      - home top-3 = first 3 unique batters with inning_topbot='Bot',
                     ordered by at_bat_number ascending

    MLBAM IDs are already present in Statcast — no retrosheet bridge needed.
    The retro / chad arguments are accepted but unused (kept for call-site
    compatibility).

    Returns a DataFrame with:
      game_date, season, home_team, away_team,
      home_slot_{1,2,3}_mlbam, away_slot_{1,2,3}_mlbam
    """
    sc_frames = []
    for yr in GAME_YEARS:
        p = os.path.join(RAW_DIR, f"raw_nrfi_statcast_{yr}.csv")
        if os.path.exists(p):
            sc_frames.append(pd.read_csv(p, low_memory=False,
                                         usecols=["game_pk", "game_date",
                                                  "inning_topbot", "at_bat_number",
                                                  "batter", "home_team", "away_team",
                                                  "season"]))
    if not sc_frames:
        print("  WARNING: no Statcast files — top-3 lineup features will be null")
        return pd.DataFrame()

    sc = pd.concat(sc_frames, ignore_index=True)
    sc["game_date"] = pd.to_datetime(sc["game_date"], errors="coerce")

    # One row per at-bat (deduplicate pitches within same AB)
    ab = (sc.drop_duplicates(subset=["game_pk", "at_bat_number"])
            .sort_values(["game_pk", "at_bat_number"]))

    rows = []
    for game_pk, grp in ab.groupby("game_pk"):
        top  = grp[grp["inning_topbot"] == "Top"]
        bot  = grp[grp["inning_topbot"] == "Bot"]

        away3 = top["batter"].tolist()[:3]
        home3 = bot["batter"].tolist()[:3]

        # Pad if fewer than 3 batters appeared (e.g., walk-off first half)
        while len(away3) < 3:
            away3.append(np.nan)
        while len(home3) < 3:
            home3.append(np.nan)

        row = grp.iloc[0]
        rows.append({
            "game_date":          row["game_date"],
            "season":             row["season"],
            "home_team":          row["home_team"],
            "away_team":          row["away_team"],
            "home_slot_1_mlbam":  home3[0],
            "home_slot_2_mlbam":  home3[1],
            "home_slot_3_mlbam":  home3[2],
            "away_slot_1_mlbam":  away3[0],
            "away_slot_2_mlbam":  away3[1],
            "away_slot_3_mlbam":  away3[2],
        })

    lu = pd.DataFrame(rows)
    lu["game_date"] = pd.to_datetime(lu["game_date"], errors="coerce")
    print(f"  Statcast lineups extracted: {len(lu):,} games")
    return lu


# =============================================================================
# STEP 6: BATTING PLATOON SPLITS FOR TOP-3 HITTERS
# =============================================================================

def build_top3_features(games: pd.DataFrame,
                        lu: pd.DataFrame,
                        df_lhp: pd.DataFrame,
                        df_rhp: pd.DataFrame,
                        chad: pd.DataFrame) -> pd.DataFrame:
    """
    For each game, compute the average platoon-adjusted batting stats for the
    top-3 hitters (batting slots 1-2-3) facing the OPPOSING SP's handedness.

    Example:
      - away_sp_hand = 'R' → home top-3 hitters' vs-RHP splits
      - home_sp_hand = 'L' → away top-3 hitters' vs-LHP splits

    FG batting splits columns expected:
      wRC+, OBP, ISO, BB%, K% (or close variants)

    Returns games with added columns:
      home_top3_wrc_plus, home_top3_obp, home_top3_iso,
      home_top3_k_pct, home_top3_bb_pct  (vs away SP hand)
      away_top3_wrc_plus, ...              (vs home SP hand)
    """
    # ── Merge retrosheet lineup onto games ────────────────────────────────────
    if lu.empty:
        print("  WARNING: no lineup data — top-3 features will be null")
        for col in ["home_top3_wrc_plus", "home_top3_obp", "home_top3_iso",
                    "home_top3_k_pct", "home_top3_bb_pct",
                    "away_top3_wrc_plus", "away_top3_obp", "away_top3_iso",
                    "away_top3_k_pct", "away_top3_bb_pct"]:
            games[col] = np.nan
        return games

    games = games.merge(
        lu.drop(columns=["season"], errors="ignore"),
        on=["game_date", "home_team", "away_team"],
        how="left",
    )

    # ── FG splits column discovery ────────────────────────────────────────────
    SPLIT_COLS = {
        "wrc_plus": ["wRC+", "wRC", "wRC+1"],
        "obp":      ["OBP", "On-Base%", "OBP+"],
        "iso":      ["ISO", "Isolated Power", "ISO+"],
        "k_pct":    ["K%", "SO%", "Strikeout%"],
        "bb_pct":   ["BB%", "uBB%", "Walk%"],
    }

    def _norm_splits(df: pd.DataFrame) -> pd.DataFrame:
        """Normalise splits dataframe column names to standard names."""
        df = df.copy()
        for feat, candidates in SPLIT_COLS.items():
            for col in candidates:
                if col in df.columns and feat not in df.columns:
                    df[feat] = pd.to_numeric(df[col], errors="coerce")
                    break
        # Normalise FG % strings
        for pct_col in ["k_pct", "bb_pct"]:
            if pct_col in df.columns and df[pct_col].dtype == object:
                df[pct_col] = (df[pct_col].astype(str)
                               .str.replace("%", "", regex=False)
                               .pipe(pd.to_numeric, errors="coerce")
                               .div(100))
        return df

    df_lhp = _norm_splits(df_lhp)
    df_rhp = _norm_splits(df_rhp)

    # ── Bridge MLBAM → FGid for batter lookups ────────────────────────────────
    if not chad.empty and "key_fangraphs" in chad.columns:
        mlbam_to_fg = (chad[["key_mlbam", "key_fangraphs"]]
                       .dropna()
                       .drop_duplicates("key_mlbam")
                       .set_index("key_mlbam")["key_fangraphs"]
                       .to_dict())
    else:
        mlbam_to_fg = {}

    # ── Build splits lookup: (fgid, season) → dict of stats ──────────────────
    def _splits_lookup(splits_df: pd.DataFrame) -> dict:
        key_feats = ["wrc_plus", "obp", "iso", "k_pct", "bb_pct"]
        avail     = [c for c in key_feats if c in splits_df.columns]
        lkp       = {}
        for _, row in splits_df.iterrows():
            fgid = row.get("IDfg")
            yr   = row.get("season")
            if pd.isna(fgid) or pd.isna(yr):
                continue
            lkp[(int(fgid), int(yr))] = {c: row.get(c) for c in avail}
        return lkp

    lhp_lkp = _splits_lookup(df_lhp)
    rhp_lkp = _splits_lookup(df_rhp)

    # Also build a direct MLBAM-keyed lookup as fallback (our Statcast-derived
    # splits store key_mlbam; some players lack a FGid in Chadwick)
    def _splits_lookup_mlbam(splits_df: pd.DataFrame) -> dict:
        key_feats = ["wrc_plus", "obp", "iso", "k_pct", "bb_pct"]
        avail = [c for c in key_feats if c in splits_df.columns]
        lkp = {}
        for _, row in splits_df.iterrows():
            mid = row.get("key_mlbam")
            yr  = row.get("season")
            if pd.isna(mid) or pd.isna(yr):
                continue
            lkp[(int(mid), int(yr))] = {c: row.get(c) for c in avail}
        return lkp

    lhp_lkp_mlbam = _splits_lookup_mlbam(df_lhp) if "key_mlbam" in df_lhp.columns else {}
    rhp_lkp_mlbam = _splits_lookup_mlbam(df_rhp) if "key_mlbam" in df_rhp.columns else {}

    def _get_stats(mlbam_id, season, sp_hand, lookup_lhp, lookup_rhp):
        """Return batting split stats dict for one batter vs one SP hand."""
        if pd.isna(mlbam_id) or pd.isna(season):
            return {}
        lookup = lookup_lhp if sp_hand == "L" else lookup_rhp
        lkp_mlbam = lhp_lkp_mlbam if sp_hand == "L" else rhp_lkp_mlbam
        mid = int(mlbam_id)
        fgid = mlbam_to_fg.get(mid)
        # Try FGid prior-year, FGid same-season, MLBAM prior-year, MLBAM same-season
        for key in [
            (int(fgid), int(season) - 1) if fgid else None,
            (int(fgid), int(season))     if fgid else None,
            (mid, int(season) - 1),
            (mid, int(season)),
        ]:
            if key is None:
                continue
            stats = lookup.get(key) or lkp_mlbam.get(key)
            if stats:
                return stats
        return {}

    # ── Compute per-game top-3 averages ───────────────────────────────────────
    feat_names = ["wrc_plus", "obp", "iso", "k_pct", "bb_pct"]
    for side in ("home", "away"):
        opp_hand = "away_sp_hand" if side == "home" else "home_sp_hand"

        for feat in feat_names:
            games[f"{side}_top3_{feat}"] = np.nan

        for idx, row in games.iterrows():
            hand = row.get(opp_hand)
            seas = row.get("season")
            vals = {f: [] for f in feat_names}
            for slot in (1, 2, 3):
                col   = f"{side}_slot_{slot}_mlbam"
                mid   = row.get(col)
                stats = _get_stats(mid, seas, hand, lhp_lkp, rhp_lkp)
                for f in feat_names:
                    v = stats.get(f)
                    if v is not None and not (isinstance(v, float) and math.isnan(v)):
                        vals[f].append(v)
            for f in feat_names:
                if vals[f]:
                    games.at[idx, f"{side}_top3_{f}"] = np.mean(vals[f])

    home_cov = games["home_top3_wrc_plus"].notna().mean()
    away_cov = games["away_top3_wrc_plus"].notna().mean()
    print(f"  Top-3 lineup features: home {home_cov:.1%} | away {away_cov:.1%} coverage")
    return games


# =============================================================================
# STEP 7: ENVIRONMENTAL FEATURES (WEATHER + PARK)
# =============================================================================

def build_environmental_features(games: pd.DataFrame,
                                  wx: pd.DataFrame,
                                  stadium_meta: dict,
                                  park_hr_factors: dict) -> pd.DataFrame:
    """
    Compute per-game environmental features:

      temperature_f       — game-time temperature (°F)
      wind_speed_mph      — wind speed (mph)
      wind_toward_cf      — wind component toward CF (+ve = tailwind = HR-friendly)
      humidity_pct        — relative humidity
      temp_carry_factor   — HR carry boost from temperature
      alt_carry_factor    — HR carry boost from altitude
      hr_park_factor      — park HR factor (100 = average)
      is_dome             — 1 if dome or retractable stadium (weather irrelevant)
      hr_environment      — composite: park_factor × (1 + temp_carry) × (1 + alt_carry)
                            × (1 + wind_carry × 0.01) — overall first-inning HR index

    Weather is keyed by (home_team, date). For dome/retractable parks, wind and
    temperature carry factors are zeroed (controlled environment).
    """
    games = games.copy()
    games["game_date_str"] = games["game_date"].dt.strftime("%Y-%m-%d")

    # ── Park static features ──────────────────────────────────────────────────
    games["hr_park_factor"] = games["home_team"].map(
        {k: v["hr_factor"] for k, v in park_hr_factors.items()}
    ).fillna(100)

    games["is_dome"] = games["home_team"].map(
        {k: 1 if v["roof"] in ("dome", "retractable") else 0
         for k, v in park_hr_factors.items()}
    ).fillna(0)

    games["altitude_ft"] = games["home_team"].map(
        {k: v["alt_ft"] for k, v in stadium_meta.items()}
    ).fillna(0)

    games["cf_bearing"] = games["home_team"].map(
        {k: v["cf_bearing"] for k, v in stadium_meta.items()}
    ).fillna(0)

    # ── Weather join ──────────────────────────────────────────────────────────
    if not wx.empty:
        wx_daily = wx.copy()
        wx_daily["date_str"] = pd.to_datetime(
            wx_daily["date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")

        games = games.merge(
            wx_daily.rename(columns={"team": "home_team",
                                     "date_str": "game_date_str"}),
            on=["home_team", "game_date_str"],
            how="left",
        )
    else:
        for col in ["temperature_f", "humidity_pct",
                    "wind_speed_mph", "wind_dir_deg"]:
            games[col] = np.nan

    # ── Wind toward CF decomposition ──────────────────────────────────────────
    def _wind_toward_cf(row):
        wspd = row.get("wind_speed_mph", np.nan)
        wdir = row.get("wind_dir_deg",   np.nan)
        cfbr = row.get("cf_bearing",     0)
        dome = row.get("is_dome",        0)
        if dome or pd.isna(wspd) or pd.isna(wdir):
            return 0.0
        # Wind direction is FROM which direction (meteorological convention)
        # Wind component TOWARD CF = wspd × cos(angle between wind_toward and CF)
        # wind_toward = (wdir + 180) % 360  (direction wind is blowing toward)
        wind_toward_deg = (wdir + 180) % 360
        angle_diff      = math.radians((wind_toward_deg - cfbr + 360) % 360)
        return float(wspd) * math.cos(angle_diff)

    games["wind_toward_cf"] = games.apply(_wind_toward_cf, axis=1)

    # ── Temperature carry ─────────────────────────────────────────────────────
    def _temp_carry(row):
        temp   = row.get("temperature_f", np.nan)
        is_dom = row.get("is_dome", 0)
        if is_dom or pd.isna(temp):
            return 0.0
        return max(0.0, (float(temp) - TEMP_BASE_F) * TEMP_HR_CARRY_PER_DEGREE)

    games["temp_carry_factor"] = games.apply(_temp_carry, axis=1)

    # ── Altitude carry ────────────────────────────────────────────────────────
    games["alt_carry_factor"] = (games["altitude_ft"] / 1000.0
                                 * ALT_HR_PER_1000FT)

    # ── Composite HR environment index ────────────────────────────────────────
    # Normalize park factor to 1.0 scale (100 = 1.0)
    games["hr_environment"] = (
        games["hr_park_factor"] / 100.0
        * (1.0 + games["temp_carry_factor"])
        * (1.0 + games["alt_carry_factor"])
        * (1.0 + games["wind_toward_cf"].clip(lower=-20, upper=20) * 0.005)
    )

    # Drop intermediate column
    games.drop(columns=["game_date_str", "cf_bearing"], errors="ignore",
               inplace=True)

    env_cov = games["temperature_f"].notna().mean()
    print(f"  Environmental features: {env_cov:.1%} temperature coverage")
    return games


# =============================================================================
# STEP 8: IMPUTE MISSING VALUES AND COMPUTE FINAL DATASET
# =============================================================================

def impute_and_finalise(games: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing SP and lineup features with league medians.
    Add interaction features, encode SP hand as binary.
    Remove rows without a valid YRFI label.
    """
    games = games.copy()

    # ── Drop rows without YRFI label ─────────────────────────────────────────
    games = games[games["yrfi"].notna()].copy()

    # ── SP hand → binary ──────────────────────────────────────────────────────
    games["home_sp_is_lhp"] = (games["home_sp_hand"] == "L").astype(float)
    games["away_sp_is_lhp"] = (games["away_sp_hand"] == "L").astype(float)

    # ── Impute numeric columns with league median per season ─────────────────
    numeric_cols = games.select_dtypes(include=np.number).columns.tolist()
    exclude      = {"game_pk", "yrfi", "season",
                    "home_sp_mlbam", "away_sp_mlbam",
                    "home_slot_1_mlbam", "home_slot_2_mlbam", "home_slot_3_mlbam",
                    "away_slot_1_mlbam", "away_slot_2_mlbam", "away_slot_3_mlbam"}
    impute_cols  = [c for c in numeric_cols if c not in exclude]

    for col in impute_cols:
        if games[col].isna().any():
            med              = games.groupby("season")[col].transform("median")
            global_med       = games[col].median()
            games[col]       = games[col].fillna(med).fillna(global_med)

    # ── Interaction features ──────────────────────────────────────────────────
    # Combined first-inning walk pressure from both SPs
    if "home_sp_fi_bb_pct" in games and "away_sp_fi_bb_pct" in games:
        games["combined_fi_bb_pct"] = (games["home_sp_fi_bb_pct"] +
                                       games["away_sp_fi_bb_pct"]) / 2

    # Lineup contact quality vs SP difficulty (wRC+ / SP K% proxy)
    if "home_top3_wrc_plus" in games and "away_sp_fi_k_pct" in games:
        games["home_lineup_vs_away_sp"] = (games["home_top3_wrc_plus"] *
                                           (1 - games["away_sp_fi_k_pct"]))
    if "away_top3_wrc_plus" in games and "home_sp_fi_k_pct" in games:
        games["away_lineup_vs_home_sp"] = (games["away_top3_wrc_plus"] *
                                           (1 - games["home_sp_fi_k_pct"]))

    # HR environment × top-3 ISO (power + park/weather combined)
    if "hr_environment" in games:
        for side in ("home", "away"):
            iso_col = f"{side}_top3_iso"
            if iso_col in games:
                games[f"{side}_hr_threat"] = (games["hr_environment"] *
                                              games[iso_col].clip(lower=0))

    print(f"  Final dataset: {len(games):,} game rows | "
          f"YRFI rate: {games['yrfi'].mean():.3f} | "
          f"features: {len(games.columns)}")
    return games


# =============================================================================
# MAIN
# =============================================================================

def build_nrfi_dataset(game_years: list = GAME_YEARS) -> pd.DataFrame:
    """End-to-end pipeline: raw data → nrfi_dataset.csv"""
    print("=" * 70)
    print("NRFI / YRFI MODEL — STEP 2: FEATURE ENGINEERING")
    print("=" * 70)

    # ── Load raw data ─────────────────────────────────────────────────────────
    print("\n[ Load ] Raw data...")
    sc      = load_statcast_first_inning()
    fg      = load_fg_pitching()
    lhp, rhp= load_batting_splits()
    wx      = load_weather()
    chad    = load_chadwick()
    retro   = load_retrosheet()
    stadium_meta, park_hr_factors = load_park_meta()

    # ── Step 1: YRFI labels + SP IDs ──────────────────────────────────────────
    print("\n[ 1/7 ] Building YRFI game index from Statcast...")
    games = build_yrfi_game_index(sc)

    # ── Step 2: SP first-inning stats ─────────────────────────────────────────
    print("\n[ 2/7 ] Building SP first-inning aggregate stats...")
    sp_fi = build_sp_first_inning_stats(sc)

    # ── Step 3: FG Stuff+/Location+ ───────────────────────────────────────────
    print("\n[ 3/7 ] Building FanGraphs SP quality features...")
    sp_fg = build_sp_fg_features(fg, chad)

    # ── Step 4: Join SP features ──────────────────────────────────────────────
    print("\n[ 4/7 ] Joining SP features to game index (prior-year)...")
    games = join_sp_features(games, sp_fi, sp_fg)

    # ── Step 5: Top-3 lineup extraction ───────────────────────────────────────
    print("\n[ 5/7 ] Extracting top-3 lineup slots from retrosheet...")
    lu    = extract_top3_lineups(retro, chad)

    # ── Step 6: Platoon batting splits ────────────────────────────────────────
    print("\n[ 6/7 ] Building top-3 platoon batting features...")
    games = build_top3_features(games, lu, lhp, rhp, chad)

    # ── Step 7: Environmental features ───────────────────────────────────────
    print("\n[ 7/7 ] Building environmental (weather + park) features...")
    games = build_environmental_features(games, wx, stadium_meta, park_hr_factors)

    # ── Impute + finalise ─────────────────────────────────────────────────────
    print("\n[ Final ] Imputing missing values and computing interactions...")
    games = impute_and_finalise(games)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(PROC_DIR, "nrfi_dataset.csv")
    games.to_csv(out_path, index=False)
    print(f"\n  ✓ nrfi_dataset.csv saved → {out_path}")
    print(f"    {len(games):,} rows × {len(games.columns)} columns")

    print("\n" + "=" * 70)
    print("STEP 2 COMPLETE — Run 03_analysis_nrfi.py next.")
    print("=" * 70)
    return games


if __name__ == "__main__":
    build_nrfi_dataset()
