"""
=============================================================================
MONEYLINE MODEL — FILE 2 OF 4: DATASET CONSTRUCTION  (REFACTORED)
=============================================================================
Builds a game-level feature matrix using granular, matchup-specific signals.

Feature categories (no team-level aggregate stats as primary features):
  SP expected run prevention : xwOBA against, barrel%, hard_hit%, K%, BB%
  SP arsenal (pitch-level)   : fastball velocity, spin, h_break, v_break,
                               offspeed whiff%, fastball whiff%
  Bullpen                    : RP pool ERA, K%, FIP (GS < 5 pitchers)
  Platoon lineup             : opposing lineup wRC+/wOBA/K%/ISO split by
                               SP handedness (L/R), PA-weighted
  Context                    : home park factor, home field advantage

Each game row uses PRIOR-YEAR stats to avoid look-ahead bias.
  Game in 2024  →  features from 2023 season stats
  Game in 2025  →  features from 2024 season stats

Input  : data/raw/ (from 01_input_moneyline.py)
Output : data/processed/moneyline_dataset.csv
=============================================================================
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)


# ── Retrosheet team code → FanGraphs / standard abbreviation ────────────────
RETRO_TO_STD = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHN": "CHC", "CHA": "CWS", "CIN": "CIN", "CLE": "CLE",
    "COL": "COL", "DET": "DET", "HOU": "HOU", "KCA": "KCR",
    "ANA": "LAA", "LAA": "LAA", "LAN": "LAD", "MIA": "MIA",
    "MIL": "MIL", "MIN": "MIN", "NYN": "NYM", "NYA": "NYY",
    "OAK": "OAK", "PHI": "PHI", "PIT": "PIT", "SDN": "SDP",
    "SEA": "SEA", "SFN": "SFG", "SLN": "STL", "TBA": "TBR",
    "TEX": "TEX", "TOR": "TOR", "WAS": "WSN",
    # FG aliases
    "CHW": "CWS", "SD": "SDP", "SF": "SFG", "TB": "TBR",
    "KC": "KCR",
}

# Fastball pitch type codes for arsenal feature extraction
FASTBALL_TYPES  = {"FF", "SI", "FT", "FC"}
OFFSPEED_TYPES  = {"SL", "CU", "CH", "KC", "FS", "EP", "KN", "SC"}

# Park run factors (2024 estimates; update annually)
PARK_FACTORS = {
    "ARI": 97,  "ATL": 102, "BAL": 104, "BOS": 105, "CHC": 101,
    "CWS": 96,  "CIN": 103, "CLE": 98,  "COL": 116, "DET": 96,
    "HOU": 97,  "KCR": 97,  "LAA": 97,  "LAD": 95,  "MIA": 97,
    "MIL": 99,  "MIN": 101, "NYM": 97,  "NYY": 105, "OAK": 96,
    "PHI": 102, "PIT": 97,  "SDP": 96,  "SEA": 94,  "SFG": 93,
    "STL": 99,  "TBR": 97,  "TEX": 101, "TOR": 102, "WSN": 100,
}


# =============================================================================
# LOAD
# =============================================================================
def load_raw() -> dict:
    files = {
        "retro":      "raw_retrosheet.csv",
        "chadwick":   "raw_chadwick.csv",
        "sc_exp":     "raw_statcast_expected.csv",
        "sc_ars":     "raw_statcast_arsenal.csv",
        "fg_pit":     "raw_fg_pitching.csv",
        "plat_lhp":   "raw_platoon_vs_lhp.csv",
        "plat_rhp":   "raw_platoon_vs_rhp.csv",
    }
    data = {}
    for key, fname in files.items():
        path = os.path.join(RAW_DIR, fname)
        if os.path.exists(path):
            data[key] = pd.read_csv(path, low_memory=False)
            print(f"  ✓ {fname}: {len(data[key]):,} rows")
        else:
            print(f"  ✗ {fname}: NOT FOUND — run 01_input_moneyline.py first")
            data[key] = pd.DataFrame()
    return data


# =============================================================================
# BUILD ID MAP: retrosheet ID → MLBAM + FanGraphs ID
# =============================================================================
def build_id_map(chad: pd.DataFrame) -> tuple:
    """
    Returns two dicts:
      retro_to_mlbam    : {"kersc001": 477132, ...}
      retro_to_fg       : {"kersc001": 2036, ...}
    """
    if chad.empty or "key_retro" not in chad.columns:
        return {}, {}

    retro_to_mlbam = {}
    retro_to_fg    = {}

    for _, row in chad.iterrows():
        rid = row.get("key_retro")
        if pd.isna(rid):
            continue
        mid = row.get("key_mlbam")
        fid = row.get("key_fangraphs")
        if pd.notna(mid):
            retro_to_mlbam[rid] = int(mid)
        if pd.notna(fid):
            retro_to_fg[rid] = int(fid)

    return retro_to_mlbam, retro_to_fg


# =============================================================================
# BUILD SP FEATURE LOOKUP
# Returns dict keyed by (mlbam_id, stat_season)
# =============================================================================
def build_sp_features(sc_exp: pd.DataFrame, sc_ars: pd.DataFrame,
                      fg_pit: pd.DataFrame) -> dict:
    """
    Combines Statcast expected stats + arsenal stats + FG pitching stats
    into a single lookup dict: (mlbam_id, season) → feature dict.

    Statcast expected features:
      sp_xwoba          : expected wOBA against
      sp_barrel_pct     : barrel%
      sp_hard_hit_pct   : hard hit%
      sp_whiff_pct      : overall whiff%

    Arsenal features (MLBAM-keyed):
      sp_fb_velo        : primary fastball avg velocity
      sp_fb_spin        : primary fastball avg spin rate
      sp_fb_h_break     : primary fastball horizontal break
      sp_fb_v_break     : primary fastball vertical break
      sp_fb_whiff_pct   : primary fastball whiff%
      sp_os_whiff_pct   : offspeed (SL/CU/CH) weighted whiff%

    FG features (used as fallback & for handedness):
      sp_siera          : SIERA
      sp_xfip           : xFIP
      sp_k_pct          : K%
      sp_bb_pct         : BB%
      sp_throws         : 1=LHP, 0=RHP
    """
    sp_lookup = {}

    # ── Statcast expected ────────────────────────────────────────────────────
    if not sc_exp.empty:
        # Normalize column names
        col_map = {
            "player_id": "mlbam", "pitcher": "mlbam",
            "year": "season",
            "xwoba": "sp_xwoba", "est_woba_using_speedangle": "sp_xwoba",
            "barrel_batted_rate": "sp_barrel_pct",
            "hard_hit_percent":   "sp_hard_hit_pct",
            "whiff_percent":      "sp_whiff_pct",
            "k_percent":          "sp_sc_k_pct",
            "bb_percent":         "sp_sc_bb_pct",
        }
        exp = sc_exp.rename(columns={k: v for k, v in col_map.items()
                                     if k in sc_exp.columns}).copy()
        if "season" not in exp.columns and "year" in sc_exp.columns:
            exp["season"] = sc_exp["year"]

        for _, row in exp.iterrows():
            mlbam  = row.get("mlbam")
            season = row.get("season")
            if pd.isna(mlbam) or pd.isna(season):
                continue
            key = (int(mlbam), int(season))
            entry = sp_lookup.setdefault(key, {})
            for feat in ["sp_xwoba", "sp_barrel_pct", "sp_hard_hit_pct",
                         "sp_whiff_pct", "sp_sc_k_pct", "sp_sc_bb_pct"]:
                val = row.get(feat)
                if pd.notna(val):
                    entry[feat] = float(val)

    # ── Statcast arsenal ─────────────────────────────────────────────────────
    if not sc_ars.empty:
        ars = sc_ars.copy()
        # Normalize player_id column
        for pid_col in ("player_id", "pitcher_id", "pitcher", "mlbam"):
            if pid_col in ars.columns:
                ars = ars.rename(columns={pid_col: "mlbam"})
                break
        if "season" not in ars.columns and "year" in ars.columns:
            ars = ars.rename(columns={"year": "season"})

        # Normalize pitch count column
        for cnt_col in ("pitches", "pitch_count", "n_pitches"):
            if cnt_col in ars.columns:
                ars = ars.rename(columns={cnt_col: "n"})
                break
        if "n" not in ars.columns:
            ars["n"] = 1

        for (mlbam, season), grp in ars.groupby(["mlbam", "season"]):
            if pd.isna(mlbam) or pd.isna(season):
                continue
            key   = (int(mlbam), int(season))
            entry = sp_lookup.setdefault(key, {})
            grp   = grp.copy()
            grp["pitch_cat"] = grp["pitch_type"].apply(
                lambda t: "FB" if str(t).upper() in FASTBALL_TYPES else
                          "OS" if str(t).upper() in OFFSPEED_TYPES else "OTHER"
            )

            # Primary fastball: highest-n fastball pitch type
            fb_rows = grp[grp["pitch_cat"] == "FB"]
            if not fb_rows.empty:
                fb = fb_rows.sort_values("n", ascending=False).iloc[0]
                for src, dst in [("avg_speed",    "sp_fb_velo"),
                                  ("avg_spin_rate","sp_fb_spin"),
                                  ("avg_break_x", "sp_fb_h_break"),
                                  ("avg_break_z", "sp_fb_v_break"),
                                  ("avg_whiff_pct","sp_fb_whiff_pct")]:
                    val = fb.get(src)
                    if pd.notna(val):
                        entry[dst] = float(val)

            # Offspeed weighted whiff%
            os_rows = grp[grp["pitch_cat"] == "OS"]
            if not os_rows.empty and "avg_whiff_pct" in os_rows.columns:
                os_n     = os_rows["n"].fillna(1)
                os_whiff = os_rows["avg_whiff_pct"].fillna(0)
                total_n  = os_n.sum()
                if total_n > 0:
                    entry["sp_os_whiff_pct"] = float((os_whiff * os_n).sum() / total_n)

    # ── FanGraphs pitching (SIERA, xFIP, handedness) ────────────────────────
    # FG data is keyed by IDfg, not MLBAM.  We store it separately and merge
    # by IDfg during game-row construction (see join_sp_features).
    # Here we just store it indexed by (IDfg, season) for fast lookup.
    fg_sp_lookup = {}
    if not fg_pit.empty:
        fg = fg_pit.copy()
        if "Season" in fg.columns:
            fg = fg.rename(columns={"Season": "season"})
        for _, row in fg.iterrows():
            fgid   = row.get("IDfg")
            season = row.get("season")
            if pd.isna(fgid) or pd.isna(season):
                continue
            key = (int(fgid), int(season))
            entry = {}
            for src, dst in [("SIERA",  "sp_siera"),
                              ("xFIP",   "sp_xfip"),
                              ("K%",     "sp_k_pct"),
                              ("BB%",    "sp_bb_pct"),
                              ("GS",     "_gs"),
                              ("G",      "_g"),
                              ("IP",     "_ip")]:
                val = row.get(src)
                if pd.notna(val):
                    entry[dst] = float(val)
            throws = row.get("Throws", "")
            entry["sp_throws"] = 1 if str(throws).strip().upper() == "L" else 0
            fg_sp_lookup[key] = entry

    return sp_lookup, fg_sp_lookup


# =============================================================================
# BUILD BULLPEN FEATURE LOOKUP
# Keyed by (team_std, stat_season) using prior-year FG pitching data
# =============================================================================
def build_bullpen_features(fg_pit: pd.DataFrame) -> dict:
    """
    For each team-season, aggregate stats of relievers (GS < 5 and IP >= 5)
    to build a bullpen quality profile.

    Features per team-season:
      bp_era     : IP-weighted bullpen ERA
      bp_k_pct   : IP-weighted bullpen K%
      bp_bb_pct  : IP-weighted bullpen BB%
      bp_fip     : IP-weighted bullpen FIP
    """
    bp_lookup = {}
    if fg_pit.empty:
        return bp_lookup

    fg = fg_pit.copy()
    if "Season" in fg.columns:
        fg = fg.rename(columns={"Season": "season"})

    # Relievers: fewer than 5 GS and at least 5 IP
    gs_col = "GS" if "GS" in fg.columns else None
    if gs_col:
        rp = fg[(fg[gs_col].fillna(0) < 5) & (fg.get("IP", pd.Series(0)) >= 5)].copy()
    else:
        rp = fg.copy()

    rp["team_std"] = rp["Team"].map(RETRO_TO_STD).fillna(rp["Team"])
    rp["ip"]       = pd.to_numeric(rp.get("IP", 0), errors="coerce").fillna(0)

    for (team, season), grp in rp.groupby(["team_std", "season"]):
        total_ip = grp["ip"].sum()
        if total_ip < 10:
            continue
        entry = {}
        for col, feat in [("ERA", "bp_era"), ("K%", "bp_k_pct"),
                          ("BB%", "bp_bb_pct"), ("FIP", "bp_fip")]:
            if col in grp.columns:
                weights = grp["ip"].fillna(0)
                vals    = pd.to_numeric(grp[col], errors="coerce").fillna(grp[col].median())
                entry[feat] = float((vals * weights).sum() / weights.sum())
        bp_lookup[(team, int(season))] = entry

    return bp_lookup


# =============================================================================
# BUILD PLATOON LINEUP LOOKUP
# Keyed by (team_std, stat_season, hand) where hand = "L" or "R"
# =============================================================================
def build_platoon_lineup(plat_lhp: pd.DataFrame,
                         plat_rhp: pd.DataFrame) -> dict:
    """
    For each team-season-handedness, compute PA-weighted lineup platoon stats.

    Used to capture: "how does this lineup hit against a LHP vs a RHP?"

    Features:
      lineup_wrc_plus   : PA-weighted wRC+ vs that pitcher handedness
      lineup_woba       : PA-weighted wOBA
      lineup_k_pct      : PA-weighted K%
      lineup_iso        : PA-weighted ISO

    Key: (team_std, stat_season, "L") → lineup stats vs LHP
         (team_std, stat_season, "R") → lineup stats vs RHP
    """
    platoon_lookup = {}

    for hand, df in [("L", plat_lhp), ("R", plat_rhp)]:
        if df.empty:
            continue

        df = df.copy()
        if "season" not in df.columns:
            continue

        # Normalize team column
        team_col = None
        for tc in ("Team", "team", "Tm"):
            if tc in df.columns:
                team_col = tc
                break
        if team_col is None:
            continue
        df["team_std"] = df[team_col].map(RETRO_TO_STD).fillna(df[team_col])

        # Normalize PA column
        pa_col = None
        for pc in ("PA", "pa"):
            if pc in df.columns:
                pa_col = pc
                break
        if pa_col is None:
            df["PA"] = 1
            pa_col = "PA"
        df[pa_col] = pd.to_numeric(df[pa_col], errors="coerce").fillna(0)

        for (team, season), grp in df.groupby(["team_std", "season"]):
            grp = grp[grp[pa_col] >= 20].copy()  # filter tiny samples
            if grp.empty:
                continue
            total_pa = grp[pa_col].sum()
            if total_pa == 0:
                continue

            entry = {}
            for col_candidates, feat in [
                (["wRC+", "wrc_plus", "wRC"],   "lineup_wrc_plus"),
                (["wOBA", "woba"],               "lineup_woba"),
                (["K%", "k_pct", "SO%"],         "lineup_k_pct"),
                (["ISO", "iso"],                 "lineup_iso"),
                (["OBP", "obp"],                 "lineup_obp"),
            ]:
                col = next((c for c in col_candidates if c in grp.columns), None)
                if col:
                    vals = pd.to_numeric(grp[col], errors="coerce")
                    valid_mask = vals.notna()
                    if valid_mask.sum() > 0:
                        w = grp.loc[valid_mask, pa_col]
                        v = vals[valid_mask]
                        entry[feat] = float((v * w).sum() / w.sum())

            if entry:
                platoon_lookup[(team, int(season), hand)] = entry

    return platoon_lookup


# =============================================================================
# BUILD GAME DATASET
# =============================================================================
def build_games(retro: pd.DataFrame, retro_to_mlbam: dict, retro_to_fg: dict,
                sp_lookup: dict, fg_sp_lookup: dict,
                bp_lookup: dict, platoon_lookup: dict) -> pd.DataFrame:
    """
    For each game in the retrosheet game log, assemble the full feature row.

    Feature naming convention:
      home_sp_*    : home starting pitcher stats
      away_sp_*    : away starting pitcher stats
      home_bp_*    : home team bullpen stats
      away_bp_*    : away team bullpen stats
      home_lineup_*: home lineup platoon stats (vs away SP hand)
      away_lineup_*: away lineup platoon stats (vs home SP hand)
      diff_*       : home value minus away value
    """
    if retro.empty:
        print("  ERROR: No retrosheet data to build from.")
        return pd.DataFrame()

    # ── Normalize retrosheet columns ─────────────────────────────────────────
    retro = retro.copy()

    # Date
    date_col = next((c for c in ["date", "Date"] if c in retro.columns), None)
    if date_col:
        retro["game_date"] = pd.to_datetime(retro[date_col].astype(str), format="%Y%m%d",
                                            errors="coerce")
    else:
        retro["game_date"] = pd.NaT

    # Season
    if "season" not in retro.columns:
        retro["season"] = retro["game_date"].dt.year

    # Team codes
    v_team_col = next((c for c in ["v_name", "visiting_team"] if c in retro.columns), None)
    h_team_col = next((c for c in ["h_name", "home_team"]     if c in retro.columns), None)
    if not v_team_col or not h_team_col:
        print("  ERROR: Cannot identify team columns in retrosheet data.")
        return pd.DataFrame()

    retro["away_team"] = retro[v_team_col].map(RETRO_TO_STD).fillna(retro[v_team_col])
    retro["home_team"] = retro[h_team_col].map(RETRO_TO_STD).fillna(retro[h_team_col])

    # Scores and target
    v_score_col = next((c for c in ["v_score", "visiting_score", "v_final_score"]
                        if c in retro.columns), None)
    h_score_col = next((c for c in ["h_score", "home_score", "h_final_score"]
                        if c in retro.columns), None)
    if v_score_col and h_score_col:
        retro["h_score"] = pd.to_numeric(retro[h_score_col], errors="coerce")
        retro["v_score"] = pd.to_numeric(retro[v_score_col], errors="coerce")
        retro["home_win"]    = (retro["h_score"] > retro["v_score"]).astype(int)
        retro["total_runs"]  = retro["h_score"] + retro["v_score"]
    else:
        print("  ERROR: Cannot identify score columns in retrosheet data.")
        return pd.DataFrame()

    # SP IDs
    h_sp_col = next((c for c in ["h_starting_pitcher_id", "home_starting_pitcher_id"]
                     if c in retro.columns), None)
    v_sp_col = next((c for c in ["v_starting_pitcher_id", "visiting_starting_pitcher_id"]
                     if c in retro.columns), None)

    retro["home_sp_retro"] = retro[h_sp_col].fillna("") if h_sp_col else ""
    retro["away_sp_retro"] = retro[v_sp_col].fillna("") if v_sp_col else ""

    # ── Filter to completed regular-season games ─────────────────────────────
    retro = retro[retro["home_win"].notna()].copy()
    retro = retro[retro["season"].notna()].copy()
    retro = retro[retro["season"] >= 2023].copy()  # earliest game year we use

    rows = []
    n_sp_matched = 0

    for _, game in retro.iterrows():
        season      = int(game["season"])
        stat_season = season - 1           # prior-year stats as features

        home_team   = game["home_team"]
        away_team   = game["away_team"]

        row = {
            "game_date":   game["game_date"],
            "season":      season,
            "home_team":   home_team,
            "away_team":   away_team,
            "home_win":    int(game["home_win"]),
            "total_runs":  game.get("total_runs", np.nan),
        }

        # ── SP feature join ──────────────────────────────────────────────────
        for side, retro_id in [("home", game["home_sp_retro"]),
                                ("away", game["away_sp_retro"])]:
            mlbam = retro_to_mlbam.get(retro_id)
            fgid  = retro_to_fg.get(retro_id)

            sp_feats = {}
            if mlbam:
                sp_feats.update(sp_lookup.get((mlbam, stat_season), {}))
                n_sp_matched += 1

            fg_feats = fg_sp_lookup.get((fgid, stat_season), {}) if fgid else {}
            # FG fills in SIERA/xFIP/handedness; don't overwrite Statcast values
            for k, v in fg_feats.items():
                if k not in sp_feats or k.startswith("sp_throws"):
                    sp_feats[k] = v

            for feat, val in sp_feats.items():
                row[f"{side}_{feat}"] = val

        # ── Bullpen feature join ─────────────────────────────────────────────
        for side, team in [("home", home_team), ("away", away_team)]:
            bp_feats = bp_lookup.get((team, stat_season), {})
            for feat, val in bp_feats.items():
                row[f"{side}_{feat}"] = val

        # ── Platoon lineup feature join ──────────────────────────────────────
        # Home lineup vs away SP handedness
        away_sp_throws = row.get("away_sp_throws", 0)   # 1=LHP, 0=RHP
        away_sp_hand   = "L" if away_sp_throws == 1 else "R"
        home_plat = platoon_lookup.get((home_team, stat_season, away_sp_hand), {})
        for feat, val in home_plat.items():
            row[f"home_{feat}"] = val

        # Away lineup vs home SP handedness
        home_sp_throws = row.get("home_sp_throws", 0)
        home_sp_hand   = "L" if home_sp_throws == 1 else "R"
        away_plat = platoon_lookup.get((away_team, stat_season, home_sp_hand), {})
        for feat, val in away_plat.items():
            row[f"away_{feat}"] = val

        # ── Park factor ──────────────────────────────────────────────────────
        row["park_factor"] = PARK_FACTORS.get(home_team, 100)
        row["home_field"]  = 1

        rows.append(row)

    df = pd.DataFrame(rows)
    pct_matched = 100 * n_sp_matched / (2 * len(df)) if len(df) else 0
    print(f"    SP Statcast match rate: {pct_matched:.1f}% ({n_sp_matched}/{2*len(df)} sides)")
    return df


# =============================================================================
# COMPUTE DIFFERENTIALS AND SELECT FEATURES
# =============================================================================
def finalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add home-minus-away differential features and select final feature columns.
    """
    # Differential pairs: (home_col, away_col, diff_col)
    diff_pairs = [
        ("home_sp_xwoba",        "away_sp_xwoba",        "diff_sp_xwoba"),
        ("home_sp_barrel_pct",   "away_sp_barrel_pct",   "diff_sp_barrel_pct"),
        ("home_sp_siera",        "away_sp_siera",        "diff_sp_siera"),
        ("home_sp_xfip",         "away_sp_xfip",         "diff_sp_xfip"),
        ("home_sp_k_pct",        "away_sp_k_pct",        "diff_sp_k_pct"),
        ("home_sp_fb_velo",      "away_sp_fb_velo",      "diff_sp_fb_velo"),
        ("home_sp_os_whiff_pct", "away_sp_os_whiff_pct", "diff_sp_os_whiff_pct"),
        ("home_bp_era",          "away_bp_era",           "diff_bp_era"),
        ("home_lineup_wrc_plus", "away_lineup_wrc_plus",  "diff_lineup_wrc_plus"),
        ("home_lineup_woba",     "away_lineup_woba",      "diff_lineup_woba"),
    ]
    for hc, ac, dc in diff_pairs:
        if hc in df.columns and ac in df.columns:
            df[dc] = df[hc] - df[ac]

    # Feature column groups
    sp_feats = [c for c in df.columns if c.startswith(("home_sp_", "away_sp_"))
                and not c.endswith("_throws")]
    bp_feats = [c for c in df.columns if c.startswith(("home_bp_", "away_bp_"))]
    lineup_feats = [c for c in df.columns if c.startswith(("home_lineup_", "away_lineup_"))]
    diff_feats   = [c for c in df.columns if c.startswith("diff_")]
    ctx_feats    = [c for c in ["park_factor", "home_field"] if c in df.columns]

    feature_cols = sp_feats + bp_feats + lineup_feats + diff_feats + ctx_feats
    id_cols      = ["game_date", "season", "home_team", "away_team", "total_runs"]
    id_cols      = [c for c in id_cols if c in df.columns]
    target_col   = "home_win"

    df = df[id_cols + feature_cols + [target_col]].copy()

    # Impute missing values with column means
    col_means = df[feature_cols].mean()
    n_imputed = 0
    for col in feature_cols:
        n_miss = df[col].isna().sum()
        if n_miss:
            df[col] = df[col].fillna(col_means[col])
            n_imputed += n_miss

    df = df.dropna(subset=[target_col])
    print(f"    Imputed {n_imputed:,} missing values across {len(feature_cols)} features.")
    print(f"    Final dataset: {len(df):,} games, {len(feature_cols)} features")
    print(f"    Home win rate: {df[target_col].mean():.3f}")
    print(f"    Seasons: {sorted(df['season'].unique())}")
    return df


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("MONEYLINE MODEL — STEP 2: DATASET CONSTRUCTION (REFACTORED)")
    print("=" * 70)

    print("\n[ 1/5 ] Loading raw data...")
    data = load_raw()

    print("\n[ 2/5 ] Building player ID maps (retroID ↔ MLBAM ↔ FanGraphs)...")
    retro_to_mlbam, retro_to_fg = build_id_map(data["chadwick"])
    print(f"  ✓ retroID → MLBAM:     {len(retro_to_mlbam):,} mappings")
    print(f"  ✓ retroID → FanGraphs: {len(retro_to_fg):,} mappings")

    print("\n[ 3/5 ] Building feature lookups...")
    sp_lookup, fg_sp_lookup = build_sp_features(
        data["sc_exp"], data["sc_ars"], data["fg_pit"]
    )
    bp_lookup      = build_bullpen_features(data["fg_pit"])
    platoon_lookup = build_platoon_lineup(data["plat_lhp"], data["plat_rhp"])
    print(f"  ✓ SP Statcast lookup : {len(sp_lookup):,} pitcher-seasons")
    print(f"  ✓ SP FanGraphs lookup: {len(fg_sp_lookup):,} pitcher-seasons")
    print(f"  ✓ Bullpen lookup     : {len(bp_lookup):,} team-seasons")
    print(f"  ✓ Platoon lookup     : {len(platoon_lookup):,} team-season-hand entries")

    print("\n[ 4/5 ] Building game-level dataset...")
    game_df = build_games(
        data["retro"], retro_to_mlbam, retro_to_fg,
        sp_lookup, fg_sp_lookup, bp_lookup, platoon_lookup
    )

    print("\n[ 5/5 ] Finalizing dataset...")
    final_df = finalize_dataset(game_df)

    out_path = os.path.join(PROC_DIR, "moneyline_dataset.csv")
    final_df.to_csv(out_path, index=False)
    print(f"\n  ✓ Saved moneyline_dataset.csv ({len(final_df):,} rows)")

    print("\n" + "=" * 70)
    print("STEP 2 COMPLETE — Run 03_analysis_moneyline.py next.")
    print("=" * 70)
