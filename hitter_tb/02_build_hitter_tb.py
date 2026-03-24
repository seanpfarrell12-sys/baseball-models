"""
=============================================================================
HITTER TOTAL BASES MODEL — FILE 2 OF 4: FEATURE ENGINEERING  (REFACTORED)
=============================================================================
Purpose : Build a player-game-level dataset where each row is one batter
          appearing in one game, and the target is the DISCRETE total-bases
          outcome: exactly 0, 1, 2, 3, or 4+ total bases.

Key changes from prior version:
  - Target is tb_class ∈ {0, 1, 2, 3, 4} — multinomial, not continuous.
  - All ID joins use MLBAM numeric keys (via Chadwick register).
    The old name-matching merge that produced "0 hitters matched" is gone.
  - Lineup position (batting slot 1–9) is pulled from retrosheet game logs
    and used to compute expected PA volume (pa_proj) per game.
  - Batter features: xBA, xSLG, xwOBA, exit_velocity_avg, barrel_batted_rate,
    launch_angle_avg, hard_hit_percent — taken from PRIOR year to avoid
    look-ahead bias.
  - Matchup features: SP handedness, SP pitch arsenal (fastball velo/spin/break,
    offspeed whiff%), SP xwOBA-against, barrel%-against.
  - Platoon-adjusted batter wRC+/wOBA/K%/ISO vs SP handedness.

Output row schema (one row per batter-game):
  game_date, season, team, batter_mlbam, batting_slot,
  pa_proj,            # projected PA in a 9-inning game for this slot
  -- batter features (prior-year Statcast + FG splits) --
  xba, xslg, xwoba, ev_avg, barrel_pct, la_avg, hard_hit_pct,
  wrc_plus_vs_hand, woba_vs_hand, k_pct_vs_hand, iso_vs_hand,
  -- SP features (prior-year) --
  sp_mlbam, sp_hand, sp_xwoba_against, sp_barrel_pct, sp_hard_hit_pct,
  sp_fb_velo, sp_fb_spin, sp_fb_h_break, sp_fb_v_break, sp_fb_whiff_pct,
  sp_fb_usage, sp_os_whiff_pct, sp_os_usage,
  -- matchup interactions --
  ev_vs_sp_xwoba,     # batter ev_avg * sp_xwoba_against (contact quality)
  barrel_vs_sp_barrel, # batter barrel_pct - sp_barrel_pct_against
  -- target --
  tb_actual           # int 0/1/2/3/4 (4 = 4+ TB)

Input  : data/raw/raw_*.csv files from 01_input_hitter_tb.py
Output : data/processed/hitter_tb_dataset.csv
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

# Expected PA per batting slot in a 9-inning game (league averages)
# Slots 1-3 see ~4.3 PA; slot 9 sees ~3.9 PA
SLOT_PA_PROJ = {
    1: 4.33, 2: 4.27, 3: 4.22, 4: 4.17,
    5: 4.10, 6: 4.05, 7: 4.00, 8: 3.95, 9: 3.90,
}

FASTBALL_TYPES = {"FF", "SI", "FT", "FC"}
OFFSPEED_TYPES = {"SL", "CU", "CH", "KC", "FS", "EP", "KN", "SC"}

# Retrosheet team-code → standard abbreviation (same map used in moneyline model)
RETRO_TO_STD = {
    "ANA": "LAA", "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHA": "CWS", "CHN": "CHC", "CIN": "CIN", "CLE": "CLE", "COL": "COL",
    "DET": "DET", "HOU": "HOU", "KCA": "KC",  "LAN": "LAD", "MIA": "MIA",
    "MIL": "MIL", "MIN": "MIN", "NYA": "NYY", "NYN": "NYM", "OAK": "OAK",
    "PHI": "PHI", "PIT": "PIT", "SDN": "SD",  "SEA": "SEA", "SFN": "SF",
    "SLN": "STL", "TBA": "TB",  "TEX": "TEX", "TOR": "TOR", "WAS": "WSH",
    "FLO": "MIA",
}


# =============================================================================
# LOAD RAW DATA
# =============================================================================
def load_raw() -> dict:
    print("  Loading raw data files...")
    raw = {}

    def _load(name, file_key):
        path = os.path.join(RAW_DIR, f"raw_{file_key}.csv")
        if not os.path.exists(path):
            print(f"    WARNING: {path} not found — skipping")
            return pd.DataFrame()
        df = pd.read_csv(path, low_memory=False)
        print(f"    {name}: {len(df):,} rows")
        return df

    raw["chad"]          = _load("Chadwick register",     "chadwick")
    raw["bat_exp"]       = _load("Batter expected stats", "batter_expected")
    raw["bat_ev"]        = _load("Batter EV/barrel",      "batter_ev_barrels")
    raw["splits_lhp"]    = _load("Batting splits vs LHP", "batting_splits_lhp")
    raw["splits_rhp"]    = _load("Batting splits vs RHP", "batting_splits_rhp")
    raw["pit_arsenal"]   = _load("Pitcher arsenal",       "pitcher_arsenal")
    raw["pit_exp"]       = _load("Pitcher expected",      "pitcher_expected")

    frames = []
    for yr in GAME_YEARS:
        path = os.path.join(RAW_DIR, f"raw_retrosheet_{yr}.csv")
        if os.path.exists(path):
            frames.append(pd.read_csv(path, low_memory=False))
    raw["retro"] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    print(f"    Retrosheet logs: {len(raw['retro']):,} games")

    return raw


# =============================================================================
# BUILD ID MAP  (retrosheet retro_id → MLBAM, MLBAM → FGid)
# =============================================================================
def build_id_map(chad: pd.DataFrame) -> tuple:
    """
    Returns:
      retro_to_mlbam: {retro_id_str: mlbam_int}
      mlbam_to_fg:    {mlbam_int: fgid_int}
    """
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

    print(f"    ID map: {len(retro_to_mlbam):,} retro→MLBAM, "
          f"{len(mlbam_to_fg):,} MLBAM→FG")
    return retro_to_mlbam, mlbam_to_fg


# =============================================================================
# BUILD BATTER FEATURE LOOKUP  (keyed by (mlbam, season))
# =============================================================================
def build_batter_features(bat_exp: pd.DataFrame,
                           bat_ev: pd.DataFrame) -> dict:
    """
    Merges expected stats and EV/barrel stats on MLBAM key.
    Returns: batter_lookup[(mlbam, season)] = feature dict
    """
    EXP_COLS = ["key_mlbam", "season", "est_ba", "est_slg", "est_woba"]
    EV_COLS  = ["key_mlbam", "season",
                "exit_velocity_avg", "launch_angle_avg",
                "barrel_batted_rate", "hard_hit_percent"]

    def _norm(df, cols):
        keep = [c for c in cols if c in df.columns]
        return df[keep].copy()

    exp = _norm(bat_exp, EXP_COLS)
    ev  = _norm(bat_ev,  EV_COLS)

    if exp.empty or ev.empty:
        return {}

    merged = pd.merge(exp, ev, on=["key_mlbam", "season"], how="outer")

    rename = {
        "est_ba":               "xba",
        "est_slg":              "xslg",
        "est_woba":             "xwoba",
        "exit_velocity_avg":    "ev_avg",
        "launch_angle_avg":     "la_avg",
        "barrel_batted_rate":   "barrel_pct",
        "hard_hit_percent":     "hard_hit_pct",
    }
    merged.rename(columns={k: v for k, v in rename.items()
                            if k in merged.columns}, inplace=True)

    feat_cols = ["xba", "xslg", "xwoba", "ev_avg", "la_avg",
                 "barrel_pct", "hard_hit_pct"]
    for col in feat_cols:
        if col not in merged.columns:
            merged[col] = np.nan

    batter_lookup = {}
    for _, row in merged.iterrows():
        mlbam = int(row["key_mlbam"]) if pd.notna(row.get("key_mlbam")) else None
        yr    = int(row["season"])    if pd.notna(row.get("season")) else None
        if mlbam and yr:
            batter_lookup[(mlbam, yr)] = {c: row.get(c, np.nan) for c in feat_cols}

    print(f"    Batter feature lookup: {len(batter_lookup):,} (mlbam, season) entries")
    return batter_lookup


# =============================================================================
# BUILD PLATOON FEATURE LOOKUP  (keyed by (fgid, season, hand))
# =============================================================================
def build_platoon_features(splits_lhp: pd.DataFrame,
                            splits_rhp: pd.DataFrame) -> dict:
    """
    Returns: platoon_lookup[(fgid, season, hand)] = feature dict
      hand ∈ {"L", "R"}  (handedness of the SP the batter faces)
    """
    SPLIT_COLS = ["IDfg", "season", "wRC+", "wOBA", "K%", "ISO"]

    def _norm(df, hand):
        keep = [c for c in SPLIT_COLS if c in df.columns]
        out  = df[keep].copy()
        out["hand"] = hand
        return out

    lhp = _norm(splits_lhp, "L")
    rhp = _norm(splits_rhp, "R")

    combined = pd.concat([lhp, rhp], ignore_index=True)
    if combined.empty:
        return {}

    rename = {"wRC+": "wrc_plus", "wOBA": "woba", "K%": "k_pct", "ISO": "iso"}
    combined.rename(columns={k: v for k, v in rename.items()
                              if k in combined.columns}, inplace=True)

    if "k_pct" in combined.columns:
        combined["k_pct"] = (combined["k_pct"].astype(str)
                             .str.replace("%", "", regex=False)
                             .pipe(pd.to_numeric, errors="coerce")) / 100.0

    feat_cols = ["wrc_plus", "woba", "k_pct", "iso"]

    platoon_lookup = {}
    for _, row in combined.iterrows():
        fgid = int(row["IDfg"])   if pd.notna(row.get("IDfg")) else None
        yr   = int(row["season"]) if pd.notna(row.get("season")) else None
        hand = str(row["hand"])
        if fgid and yr:
            platoon_lookup[(fgid, yr, hand)] = {
                c: row.get(c, np.nan) for c in feat_cols
            }

    print(f"    Platoon feature lookup: {len(platoon_lookup):,} entries")
    return platoon_lookup


# =============================================================================
# BUILD SP FEATURE LOOKUP  (keyed by (mlbam, season))
# =============================================================================
def build_sp_features(pit_exp: pd.DataFrame,
                       pit_arsenal: pd.DataFrame) -> dict:
    """
    Returns: sp_lookup[(mlbam, season)] = feature dict
    """
    EXP_COLS = ["key_mlbam", "season", "est_woba", "barrel_percent",
                "hard_hit_percent", "p_throws"]
    exp = pit_exp[[c for c in EXP_COLS if c in pit_exp.columns]].copy()
    exp.rename(columns={
        "est_woba":         "sp_xwoba_against",
        "barrel_percent":   "sp_barrel_pct",
        "hard_hit_percent": "sp_hard_hit_pct",
        "p_throws":         "sp_hand",
    }, inplace=True)

    sp_lookup = {}

    if not pit_arsenal.empty:
        ars = pit_arsenal.copy()
        if "pitch_type" not in ars.columns:
            ars["pitch_type"] = "FF"

        usage_col = next(
            (c for c in ["pitch_percent", "pitch_usage"] if c in ars.columns),
            None
        )
        count_col = next(
            (c for c in ["pitches", "n_pitches", "pitch_count"] if c in ars.columns),
            None
        )
        sort_col = count_col or usage_col

        for (mlbam, yr), grp in ars.groupby(["key_mlbam", "season"]):
            fb  = grp[grp["pitch_type"].isin(FASTBALL_TYPES)]
            os_ = grp[grp["pitch_type"].isin(OFFSPEED_TYPES)]

            # Primary fastball = highest-count FB type
            if not fb.empty and sort_col and sort_col in fb.columns:
                primary_fb = fb.sort_values(sort_col, ascending=False).iloc[0]
            elif not fb.empty:
                primary_fb = fb.iloc[0]
            else:
                primary_fb = pd.Series(dtype=float)

            # PA-weighted offspeed whiff %
            if (not os_.empty and usage_col and usage_col in os_.columns
                    and "whiff_percent" in os_.columns):
                os_ = os_.copy()
                os_[usage_col]       = pd.to_numeric(os_[usage_col],       errors="coerce")
                os_["whiff_percent"] = pd.to_numeric(os_["whiff_percent"],  errors="coerce")
                total_use = os_[usage_col].sum()
                os_whiff = (
                    (os_[usage_col] * os_["whiff_percent"]).sum() / total_use
                    if total_use > 0 else np.nan
                )
                os_total_use = total_use
            else:
                os_whiff, os_total_use = np.nan, np.nan

            sp_lookup[(int(mlbam), int(yr))] = {
                "sp_fb_velo":      float(primary_fb.get("avg_speed",     np.nan)),
                "sp_fb_spin":      float(primary_fb.get("avg_spin",      np.nan)),
                "sp_fb_h_break":   float(primary_fb.get("pfx_x",         np.nan)),
                "sp_fb_v_break":   float(primary_fb.get("pfx_z",         np.nan)),
                "sp_fb_whiff_pct": float(primary_fb.get("whiff_percent",  np.nan)),
                "sp_fb_usage":     float(primary_fb.get(usage_col, np.nan)
                                         if usage_col else np.nan),
                "sp_os_whiff_pct": float(os_whiff),
                "sp_os_usage":     float(os_total_use),
            }

    # Overlay expected stats
    for _, row in exp.iterrows():
        mlbam = int(row["key_mlbam"]) if pd.notna(row.get("key_mlbam")) else None
        yr    = int(row["season"])    if pd.notna(row.get("season")) else None
        if not mlbam or not yr:
            continue
        entry = sp_lookup.setdefault((mlbam, yr), {})
        entry["sp_xwoba_against"] = row.get("sp_xwoba_against", np.nan)
        entry["sp_barrel_pct"]    = row.get("sp_barrel_pct",    np.nan)
        entry["sp_hard_hit_pct"]  = row.get("sp_hard_hit_pct",  np.nan)
        entry["sp_hand"]          = row.get("sp_hand",          np.nan)

    print(f"    SP feature lookup: {len(sp_lookup):,} (mlbam, season) entries")
    return sp_lookup


# =============================================================================
# PARSE BATTING ORDER FROM RETROSHEET
# =============================================================================
def extract_batting_orders(retro: pd.DataFrame,
                            retro_to_mlbam: dict) -> list:
    """
    Retrosheet logs have columns h_bat_1_id … h_bat_9_id and
    v_bat_1_id … v_bat_9_id.

    Returns list of dicts, one per batter-game:
      game_date, season, team_retro, batter_retro_id, batting_slot,
      sp_retro (the SP they faced), home_flag
    """
    if retro.empty:
        return []

    date_col = next((c for c in ["date", "game_date", "Date", "GameDate"]
                     if c in retro.columns), None)
    if not date_col:
        print("    WARNING: no date column found in retrosheet logs")
        return []

    rows = []
    for _, game in retro.iterrows():
        gdate  = str(game[date_col])[:10]
        season = int(gdate[:4])

        home_retro = str(game.get("home_team_id", game.get("h_team", "")))
        away_retro = str(game.get("visiting_team_id", game.get("v_team", "")))
        h_sp = str(game.get("h_starting_pitcher_id", ""))
        v_sp = str(game.get("v_starting_pitcher_id", ""))

        # Home batters face away SP (v_sp)
        for slot in range(1, 10):
            bat_col = f"h_bat_{slot}_id"
            if bat_col in game.index and pd.notna(game[bat_col]):
                rows.append({
                    "game_date":    gdate,
                    "season":       season,
                    "team_retro":   home_retro,
                    "batter_retro": str(game[bat_col]),
                    "batting_slot": slot,
                    "sp_retro":     v_sp,
                    "home_flag":    1,
                })

        # Away batters face home SP (h_sp)
        for slot in range(1, 10):
            bat_col = f"v_bat_{slot}_id"
            if bat_col in game.index and pd.notna(game[bat_col]):
                rows.append({
                    "game_date":    gdate,
                    "season":       season,
                    "team_retro":   away_retro,
                    "batter_retro": str(game[bat_col]),
                    "batting_slot": slot,
                    "sp_retro":     h_sp,
                    "home_flag":    0,
                })

    print(f"    Extracted {len(rows):,} batter-game appearances from retrosheet")
    return rows


# =============================================================================
# EXTRACT ACTUAL TB OUTCOMES FROM RETROSHEET
# =============================================================================
def extract_tb_actuals(retro: pd.DataFrame) -> pd.DataFrame:
    """
    Retrosheet game logs carry per-batter box-score columns:
      h_bat_1_1b, h_bat_1_2b, h_bat_1_3b, h_bat_1_hr  (and v_ equivalents).

    Returns DataFrame keyed by (game_date, batter_retro) with tb_actual.

    NOTE: standard retrosheet game log files include only lineup IDs, not
    individual hit totals.  The detailed extended files (retrosheet event files)
    carry hit-by-hit data, but the pybaseball helper returns the summary logs.
    If tb_actual is all NaN after this function, you must supplement with an
    external source (e.g., Baseball Reference game logs API or statcast
    per-game batter data via pyb.statcast()).
    """
    if retro.empty:
        return pd.DataFrame()

    date_col = next((c for c in ["date", "game_date", "Date", "GameDate"]
                     if c in retro.columns), None)

    records = []
    for _, game in retro.iterrows():
        gdate = str(game[date_col])[:10]
        for side in ["h", "v"]:
            for slot in range(1, 10):
                bid = str(game.get(f"{side}_bat_{slot}_id", ""))
                s1b = pd.to_numeric(game.get(f"{side}_bat_{slot}_1b", np.nan),
                                    errors="coerce")
                s2b = pd.to_numeric(game.get(f"{side}_bat_{slot}_2b", np.nan),
                                    errors="coerce")
                s3b = pd.to_numeric(game.get(f"{side}_bat_{slot}_3b", np.nan),
                                    errors="coerce")
                shr = pd.to_numeric(game.get(f"{side}_bat_{slot}_hr", np.nan),
                                    errors="coerce")

                if any(pd.notna(v) for v in [s1b, s2b, s3b, shr]):
                    tb = (
                        np.nan_to_num(s1b, 0) * 1
                        + np.nan_to_num(s2b, 0) * 2
                        + np.nan_to_num(s3b, 0) * 3
                        + np.nan_to_num(shr, 0) * 4
                    )
                    records.append({
                        "game_date":    gdate,
                        "batter_retro": bid,
                        "tb_actual":    min(int(tb), 4),  # cap 4+ → class 4
                    })

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


# =============================================================================
# ASSEMBLE FINAL DATASET
# =============================================================================
def build_dataset(raw: dict) -> pd.DataFrame:
    print("\n  Building ID maps...")
    retro_to_mlbam, mlbam_to_fg = build_id_map(raw["chad"])

    print("\n  Building batter feature lookups...")
    batter_lookup = build_batter_features(raw["bat_exp"], raw["bat_ev"])

    print("\n  Building platoon feature lookups...")
    platoon_lookup = build_platoon_features(raw["splits_lhp"], raw["splits_rhp"])

    print("\n  Building SP feature lookups...")
    sp_lookup = build_sp_features(raw["pit_exp"], raw["pit_arsenal"])

    print("\n  Extracting batting orders from retrosheet logs...")
    batting_rows = extract_batting_orders(raw["retro"], retro_to_mlbam)

    print("\n  Extracting actual total-bases targets from retrosheet logs...")
    tb_actuals = extract_tb_actuals(raw["retro"])

    if not batting_rows:
        print("  ERROR: No batting order rows extracted. Check retrosheet data.")
        return pd.DataFrame()

    df = pd.DataFrame(batting_rows)

    # Map retro IDs → MLBAM
    df["batter_mlbam"] = df["batter_retro"].map(retro_to_mlbam)
    df["sp_mlbam"]     = df["sp_retro"].map(retro_to_mlbam)
    df["team"]         = df["team_retro"].map(RETRO_TO_STD).fillna(df["team_retro"])
    df["pa_proj"]      = df["batting_slot"].map(SLOT_PA_PROJ)

    # ── Batter features (prior-year Statcast) ─────────────────────────────
    bat_feat_cols = ["xba", "xslg", "xwoba", "ev_avg", "la_avg",
                     "barrel_pct", "hard_hit_pct"]
    for col in bat_feat_cols:
        df[col] = np.nan

    for idx, row in df.iterrows():
        mlbam = row["batter_mlbam"]
        yr    = row["season"]
        if pd.notna(mlbam) and pd.notna(yr):
            feat = batter_lookup.get((int(mlbam), int(yr) - 1), {})
            for col in bat_feat_cols:
                df.at[idx, col] = feat.get(col, np.nan)

    # ── SP features (prior-year) ──────────────────────────────────────────
    sp_feat_cols = ["sp_hand", "sp_xwoba_against", "sp_barrel_pct",
                    "sp_hard_hit_pct", "sp_fb_velo", "sp_fb_spin",
                    "sp_fb_h_break", "sp_fb_v_break", "sp_fb_whiff_pct",
                    "sp_fb_usage", "sp_os_whiff_pct", "sp_os_usage"]
    for col in sp_feat_cols:
        df[col] = np.nan

    for idx, row in df.iterrows():
        mlbam = row["sp_mlbam"]
        yr    = row["season"]
        if pd.notna(mlbam) and pd.notna(yr):
            feat = sp_lookup.get((int(mlbam), int(yr) - 1), {})
            for col in sp_feat_cols:
                df.at[idx, col] = feat.get(col, np.nan)

    # ── Platoon features (batter vs SP handedness) ─────────────────────────
    plat_feat_cols = ["wrc_plus_vs_hand", "woba_vs_hand",
                      "k_pct_vs_hand", "iso_vs_hand"]
    for col in plat_feat_cols:
        df[col] = np.nan

    for idx, row in df.iterrows():
        mlbam = row["batter_mlbam"]
        yr    = row["season"]
        hand  = row.get("sp_hand")
        if pd.notna(mlbam) and pd.notna(yr) and pd.notna(hand):
            fgid = mlbam_to_fg.get(int(mlbam))
            if fgid:
                feat = platoon_lookup.get((int(fgid), int(yr) - 1, str(hand)), {})
                df.at[idx, "wrc_plus_vs_hand"] = feat.get("wrc_plus", np.nan)
                df.at[idx, "woba_vs_hand"]     = feat.get("woba",     np.nan)
                df.at[idx, "k_pct_vs_hand"]    = feat.get("k_pct",    np.nan)
                df.at[idx, "iso_vs_hand"]       = feat.get("iso",      np.nan)

    # ── Interaction features ──────────────────────────────────────────────
    df["ev_vs_sp_xwoba"]      = df["ev_avg"] * df["sp_xwoba_against"]
    df["barrel_vs_sp_barrel"] = df["barrel_pct"] - df["sp_barrel_pct"]

    # ── Merge actual TB outcomes ──────────────────────────────────────────
    if not tb_actuals.empty:
        df = pd.merge(df, tb_actuals,
                      on=["game_date", "batter_retro"], how="left")
        n_labeled = df["tb_actual"].notna().sum()
        print(f"\n    TB actuals merged: {n_labeled:,} / {len(df):,} rows labeled")
    else:
        df["tb_actual"] = np.nan
        print("\n    WARNING: No TB actuals found in retrosheet logs.")
        print("    Retrosheet summary game logs carry lineup IDs only — not hit totals.")
        print("    Supplement with Baseball Reference or Statcast per-game batter data.")

    # Keep only labeled rows for training
    df_train = df.dropna(subset=["tb_actual"]).copy()
    df_train["tb_actual"] = df_train["tb_actual"].astype(int)
    print(f"\n    Training-eligible rows (labeled): {len(df_train):,}")
    if len(df_train) > 0:
        print(f"    TB distribution:\n{df_train['tb_actual'].value_counts().sort_index()}")

    return df_train


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("HITTER TB MODEL — STEP 2: FEATURE ENGINEERING (REFACTORED)")
    print("=" * 70)

    print("\n[ 1/3 ] Loading raw data...")
    raw = load_raw()

    print("\n[ 2/3 ] Building dataset...")
    df = build_dataset(raw)

    if df.empty:
        print("\nERROR: Empty dataset.  Run 01_input_hitter_tb.py first and verify "
              "that retrosheet logs contain box-score hit columns.")
        exit(1)

    print(f"\n[ 3/3 ] Saving dataset...")
    out_path = os.path.join(PROC_DIR, "hitter_tb_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"  ✓ {len(df):,} rows saved → {out_path}")
    print(f"  Columns: {list(df.columns)}")

    print("\n" + "=" * 70)
    print("STEP 2 COMPLETE — Run 03_analysis_hitter_tb.py next.")
    print("=" * 70)
