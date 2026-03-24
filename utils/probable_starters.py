"""
=============================================================================
PROBABLE STARTERS UTILITY
=============================================================================
Purpose : Fetch today's probable starting pitchers and confirmed lineups
          from MLB Stats API (free, no authentication required), then match
          each pitcher's name to their 2025 FanGraphs individual season stats.

Used by : All 4 export files to score real games with actual SP matchups
          instead of team-average pitching stats.

For R users:
  - unicodedata.normalize() removes accents from names (Jesús → Jesus)
  - difflib.get_close_matches() is fuzzy string matching (like agrep() in R)
  - dict.get(key, default) safely accesses a key with a fallback value
=============================================================================
"""

import os
import re
import requests
import unicodedata
import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"

BASE_DIR = Path(__file__).parent.parent
RAW_DIR  = BASE_DIR / "data" / "raw"

# MLB Stats API team abbreviations that differ from our standard ones
MLB_TO_STANDARD = {
    "WSH": "WSN",   # Washington Nationals
    "ATH": "OAK",   # Oakland/Sacramento Athletics
    "AZ":  "ARI",   # Arizona Diamondbacks
    "TB":  "TBR",   # Tampa Bay Rays
    "KC":  "KCR",   # Kansas City Royals
    "SD":  "SDP",   # San Diego Padres
    "SF":  "SFG",   # San Francisco Giants
    "CWS": "CWS",   # same — just confirming
}

# All 30 valid MLB team abbreviations. Games involving any other team
# abbreviation (e.g. spring training opponents, MiLB affiliates) are excluded.
MLB_TEAMS = {
    "ARI", "ATL", "BAL", "BOS", "CHC", "CWS", "CIN", "CLE",
    "COL", "DET", "HOU", "KCR", "LAA", "LAD", "MIA", "MIL",
    "MIN", "NYM", "NYY", "OAK", "PHI", "PIT", "SDP", "SEA",
    "SFG", "STL", "TBR", "TEX", "TOR", "WSN",
}

# SP feature columns used by the moneyline and totals models
SP_FEATURE_COLS = ["SIERA", "xFIP", "FIP", "K%", "BB%", "K-BB%"]

# Rename map: FanGraphs column → our model feature names
SP_RENAME = {
    "SIERA":  "sp_siera",
    "xFIP":   "sp_xfip",
    "FIP":    "sp_fip",
    "K%":     "sp_k_pct",
    "BB%":    "sp_bb_pct",
    "K-BB%":  "sp_k_bb_pct",
}


# =============================================================================
# NAME NORMALIZATION
# =============================================================================

def normalize_name(name: str) -> str:
    """
    Normalize a pitcher name for fuzzy matching.

    Steps:
      1. Remove accents/diacritics (Jesús → Jesus, Yoán → Yoan)
      2. Lowercase
      3. Remove punctuation (Jr., II, etc.)
      4. Strip extra whitespace

    In R: iconv(name, to="ASCII//TRANSLIT") does something similar.
    """
    if not name:
        return ""
    # Remove accents: decompose Unicode, keep only ASCII characters
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Lowercase and strip punctuation
    ascii_name = re.sub(r"[^a-zA-Z\s]", "", ascii_name).lower().strip()
    # Collapse multiple spaces
    return re.sub(r"\s+", " ", ascii_name)


def _build_name_index(pitching_df: pd.DataFrame) -> dict:
    """
    Build a dict mapping normalized pitcher name → DataFrame row index.
    Used for fast O(1) lookup after normalization.
    """
    index = {}
    for idx, row in pitching_df.iterrows():
        norm = normalize_name(str(row.get("Name", "")))
        if norm:
            index[norm] = idx
    return index


# =============================================================================
# DATA LOADING
# =============================================================================

def load_pitching_stats(season: int = None) -> pd.DataFrame:
    """
    Load FanGraphs individual pitcher stats for the given season.

    If season is None (default), returns the most recent season available
    in the CSV — so 2026 data is used automatically once it is pulled.
    Falls back to 2025 if the current year has fewer than 10 qualifying
    starters (GS >= 5), indicating the season hasn't started yet.

    Returns DataFrame with Name, Team, GS, IP, SIERA, xFIP, FIP, K%, BB%, K-BB%
    Filtered to starters only (GS >= 5).
    """
    path = RAW_DIR / "raw_pitching_stats.csv"
    if not path.exists():
        print(f"  WARNING: {path} not found. Run 01_input_*.py first.")
        return pd.DataFrame()

    df = pd.read_csv(path)

    if season is None:
        latest = int(df["Season"].max())
        latest_starters = df[(df["Season"] == latest) &
                             (df.get("GS", pd.Series(0, index=df.index)) >= 5)]
        if len(latest_starters) < 10:
            season = latest - 1   # fall back to prior season if too sparse
        else:
            season = latest

    df = df[df["Season"] == season].copy()
    df = df[df.get("GS", pd.Series(0, index=df.index)) >= 5].copy()

    # Normalize team abbreviations
    from utils.action_network import AN_TEAM_MAP
    _fg_map = {"CHW": "CWS", "SD": "SDP", "SF": "SFG", "TB": "TBR",
               "KC": "KCR", "WAS": "WSN"}
    df["team_std"] = df["Team"].map(lambda t: _fg_map.get(t, t))

    return df.reset_index(drop=True)


def load_batting_stats(season: int = None) -> pd.DataFrame:
    """
    Load individual FanGraphs batting stats.

    If season is None (default), returns the most recent season available
    in the CSV — so 2026 data is used automatically once it is pulled.
    Falls back to 2025 if 2026 has fewer than 20 qualifying batters (PA ≥ 50).
    """
    path = RAW_DIR / "raw_batting_stats.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)

    if season is None:
        latest = int(df["Season"].max())
        latest_df = df[df["Season"] == latest].copy()
        # Fall back to prior season if current year data is too sparse
        if len(latest_df[latest_df.get("PA", pd.Series(999, index=latest_df.index)) >= 50]) < 20:
            prior = latest - 1
            latest_df = df[df["Season"] == prior].copy()
        return latest_df.reset_index(drop=True)

    return df[df["Season"] == season].copy().reset_index(drop=True)


# =============================================================================
# PITCHER STAT LOOKUP
# =============================================================================

def lookup_pitcher_stats(pitcher_name: str,
                         pitching_df: pd.DataFrame,
                         name_index: dict = None) -> dict:
    """
    Find a pitcher's individual 2025 stats by name.

    Uses exact normalized match first, then falls back to last-name match.
    Returns a dict with sp_siera, sp_xfip, sp_fip, sp_k_pct, sp_bb_pct, sp_k_bb_pct.
    Returns empty dict if no match found.

    Parameters
    ----------
    pitcher_name : str   Full pitcher name (e.g. "Jesús Luzardo")
    pitching_df  : pd.DataFrame   2025 pitching stats from load_pitching_stats()
    name_index   : dict  Pre-built name→index map (pass for speed in loops)
    """
    if not pitcher_name or pitching_df.empty:
        return {}

    if name_index is None:
        name_index = _build_name_index(pitching_df)

    norm = normalize_name(pitcher_name)

    # 1. Exact normalized match
    if norm in name_index:
        row = pitching_df.loc[name_index[norm]]
        return _row_to_sp_features(row)

    # 2. Last-name only match (handles "Jose" vs "Joseph" type differences)
    last_name = norm.split()[-1] if norm else ""
    candidates = [k for k in name_index if k.split()[-1] == last_name]
    if len(candidates) == 1:
        row = pitching_df.loc[name_index[candidates[0]]]
        return _row_to_sp_features(row)

    # 3. Partial first-name + last-name match
    for k in name_index:
        parts = k.split()
        norm_parts = norm.split()
        if len(parts) >= 2 and len(norm_parts) >= 2:
            if parts[-1] == norm_parts[-1] and parts[0][0] == norm_parts[0][0]:
                row = pitching_df.loc[name_index[k]]
                return _row_to_sp_features(row)

    return {}


def _row_to_sp_features(row) -> dict:
    """Convert a pitching stats row to the SP feature dict used by models."""
    result = {}
    for fg_col, model_col in SP_RENAME.items():
        val = row.get(fg_col, np.nan)
        result[model_col] = float(val) if pd.notna(val) else np.nan
    result["pitcher_name"] = str(row.get("Name", ""))
    result["pitcher_team"] = str(row.get("team_std", ""))
    result["pitcher_ip"]   = float(row.get("IP", 0))
    result["pitcher_gs"]   = int(row.get("GS", 0))
    return result


def get_team_avg_sp_stats(team_abbr: str, pitching_df: pd.DataFrame) -> dict:
    """
    Fallback: return IP-weighted average stats for all starters on a team.
    Used when no probable pitcher is announced.
    """
    team_pit = pitching_df[pitching_df["team_std"] == team_abbr]
    if team_pit.empty:
        return {}

    weights = team_pit["IP"].fillna(0)
    total_w = weights.sum()
    if total_w == 0:
        return {v: team_pit[k].mean() for k, v in SP_RENAME.items() if k in team_pit.columns}

    result = {}
    for fg_col, model_col in SP_RENAME.items():
        if fg_col in team_pit.columns:
            result[model_col] = float(
                (team_pit[fg_col].fillna(team_pit[fg_col].mean()) * weights).sum() / total_w
            )
        else:
            result[model_col] = np.nan
    result["pitcher_name"] = f"{team_abbr} rotation avg"
    result["pitcher_team"] = team_abbr
    return result


# =============================================================================
# MLB STATS API — PROBABLE STARTERS + LINEUPS
# =============================================================================

def get_probable_starters(game_date: str = None) -> pd.DataFrame:
    """
    Fetch today's probable starting pitchers from MLB Stats API.

    Returns
    -------
    pd.DataFrame with columns:
      home_team, away_team, home_sp_name, away_sp_name,
      game_time, game_pk, game_type
    """
    if game_date is None:
        game_date = date.today().strftime("%Y-%m-%d")

    try:
        response = requests.get(
            MLB_SCHEDULE_URL,
            params={"sportId": 1, "date": game_date,
                    "hydrate": "probablePitcher,team"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"  WARNING: Could not fetch probable starters: {e}")
        return pd.DataFrame()

    records = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            home_info = game["teams"]["home"]
            away_info = game["teams"]["away"]

            home_abbr = MLB_TO_STANDARD.get(
                home_info["team"].get("abbreviation", ""),
                home_info["team"].get("abbreviation", ""),
            )
            away_abbr = MLB_TO_STANDARD.get(
                away_info["team"].get("abbreviation", ""),
                away_info["team"].get("abbreviation", ""),
            )

            # Skip spring training / exhibition games involving non-MLB teams
            if home_abbr not in MLB_TEAMS or away_abbr not in MLB_TEAMS:
                continue

            home_sp = home_info.get("probablePitcher", {}).get("fullName", "")
            away_sp = away_info.get("probablePitcher", {}).get("fullName", "")

            records.append({
                "game_pk":      game.get("gamePk", ""),
                "game_time":    game.get("gameDate", ""),
                "game_type":    game.get("gameType", ""),   # R=regular, S=spring
                "home_team":    home_abbr,
                "away_team":    away_abbr,
                "home_sp_name": home_sp,
                "away_sp_name": away_sp,
            })

    df = pd.DataFrame(records)
    if not df.empty:
        n_with_sp = (df["home_sp_name"] != "").sum()
        print(f"  ✓ {len(df)} games found, {n_with_sp} with probable starters announced.")
    return df


def _parse_lineup_from_game(game: dict) -> tuple:
    """
    Extract (home_abbr, away_abbr, home_players, away_players) from a game dict.
    Shared by get_lineups() and _fetch_recent_lineups_bulk().
    """
    home_info = game["teams"]["home"]
    away_info = game["teams"]["away"]

    home_abbr = MLB_TO_STANDARD.get(
        home_info["team"].get("abbreviation", ""),
        home_info["team"].get("abbreviation", ""),
    )
    away_abbr = MLB_TO_STANDARD.get(
        away_info["team"].get("abbreviation", ""),
        away_info["team"].get("abbreviation", ""),
    )

    game_lineups = game.get("lineups", {})
    home_players = [p["fullName"] for p in game_lineups.get("homePlayers", [])
                    if p.get("primaryPosition", {}).get("type") != "Pitcher"]
    away_players = [p["fullName"] for p in game_lineups.get("awayPlayers", [])
                    if p.get("primaryPosition", {}).get("type") != "Pitcher"]

    return home_abbr, away_abbr, home_players, away_players


def _fetch_recent_lineups_bulk(lookback_days: int = 3) -> dict:
    """
    Fetch each team's most recent confirmed lineup from the past N days.

    Makes one MLB Stats API call per day (at most 3 calls), then returns
    the most recent available lineup per team. Used as a projected-lineup
    fallback when today's confirmed lineups haven't posted yet.

    Returns
    -------
    dict  {team_abbr: [player_full_name, ...]}  (most recent game's lineup)
    """
    from datetime import timedelta, date as _date_cls

    recent: dict = {}   # team_abbr → players; first hit wins (most recent day first)
    today = _date_cls.today()

    for days_back in range(1, lookback_days + 1):
        check_date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
        try:
            resp = requests.get(
                MLB_SCHEDULE_URL,
                params={"sportId": 1, "date": check_date,
                        "hydrate": "team,lineups"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            continue

        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                home_abbr, away_abbr, home_pl, away_pl = _parse_lineup_from_game(game)
                # Keep first match only (days_back=1 = most recent)
                if home_pl and home_abbr not in recent:
                    recent[home_abbr] = home_pl
                if away_pl and away_abbr not in recent:
                    recent[away_abbr] = away_pl

    return recent


def get_lineups(game_date: str = None,
                use_projected_fallback: bool = True,
                projected_lookback_days: int = 3,
                return_sources: bool = False):
    """
    Fetch today's batting lineups from MLB Stats API.

    Returns confirmed lineups where posted; falls back to each team's most
    recent lineup (from the past `projected_lookback_days` days) for any
    team that hasn't posted yet.

    Parameters
    ----------
    game_date : str
        Date string "YYYY-MM-DD". Defaults to today.
    use_projected_fallback : bool
        If True (default), fill missing lineups from recent game history.
    projected_lookback_days : int
        How many days back to look for a recent lineup (default 3).
    return_sources : bool
        If True, return a tuple (lineups, sources) where sources is a dict
        mapping team_abbr -> "confirmed" | "projected". Default False.

    Returns
    -------
    dict  {team_abbr: [player_full_name, ...]}
        or tuple (dict, dict) if return_sources=True.
    """
    if game_date is None:
        game_date = date.today().strftime("%Y-%m-%d")

    # ── Step 1: confirmed lineups from today ──────────────────────────────────
    try:
        response = requests.get(
            MLB_SCHEDULE_URL,
            params={"sportId": 1, "date": game_date,
                    "hydrate": "probablePitcher,team,lineups"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"  WARNING: Could not fetch lineups: {e}")
        return ({}, {}) if return_sources else {}

    lineups: dict = {}
    sources: dict = {}
    teams_playing: set = set()

    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            home_abbr, away_abbr, home_pl, away_pl = _parse_lineup_from_game(game)
            # Skip spring training / exhibition games
            if home_abbr not in MLB_TEAMS or away_abbr not in MLB_TEAMS:
                continue
            teams_playing.update([home_abbr, away_abbr])
            if home_pl:
                lineups[home_abbr] = home_pl
                sources[home_abbr] = "confirmed"
            if away_pl:
                lineups[away_abbr] = away_pl
                sources[away_abbr] = "confirmed"

    n_confirmed = len(lineups)
    n_missing   = len(teams_playing) - n_confirmed
    print(f"  ✓ Confirmed lineups: {n_confirmed} teams "
          f"| {n_missing} not yet posted.")

    # ── Step 2: projected fallback for teams still missing ────────────────────
    if use_projected_fallback and n_missing > 0:
        missing_teams = teams_playing - set(lineups.keys())
        print(f"  Fetching projected lineups (last {projected_lookback_days}d) "
              f"for: {', '.join(sorted(missing_teams))}")

        recent = _fetch_recent_lineups_bulk(projected_lookback_days)
        n_projected = 0
        for team in missing_teams:
            if team in recent:
                lineups[team] = recent[team]
                sources[team] = "projected"
                n_projected += 1

        n_still_missing = n_missing - n_projected
        print(f"  ✓ Projected lineups filled: {n_projected} teams"
              + (f" | {n_still_missing} still missing (team-avg fallback)"
                 if n_still_missing else "") + ".")

    if return_sources:
        return lineups, sources
    return lineups


# =============================================================================
# GAME STATUS — which games have confirmed lineups vs still pending
# =============================================================================

def get_todays_game_status(game_date: str = None) -> tuple:
    """
    Return two lists of 'AWAY @ HOME' strings for today's MLB games:
      scored  — both lineups confirmed (model has/will run)
      pending — at least one lineup not yet posted

    Parameters
    ----------
    game_date : str  "YYYY-MM-DD", defaults to today.

    Returns
    -------
    tuple (scored: list[str], pending: list[str])
    """
    if game_date is None:
        game_date = date.today().strftime("%Y-%m-%d")

    # Get all games scheduled today
    try:
        resp = requests.get(
            MLB_SCHEDULE_URL,
            params={"sportId": 1, "date": game_date,
                    "hydrate": "probablePitcher,team,lineups"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return [], []

    all_games  = []   # [(away_abbr, home_abbr)]
    confirmed  = set()

    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            home_abbr, away_abbr, home_pl, away_pl = _parse_lineup_from_game(game)
            if home_abbr not in MLB_TEAMS or away_abbr not in MLB_TEAMS:
                continue
            all_games.append((away_abbr, home_abbr))
            if home_pl:
                confirmed.add(home_abbr)
            if away_pl:
                confirmed.add(away_abbr)

    scored  = []
    pending = []
    for away, home in all_games:
        label = f"{away}@{home}"
        if away in confirmed and home in confirmed:
            scored.append(label)
        else:
            pending.append(label)

    return scored, pending


# =============================================================================
# COMBINED LOOKUP — MAIN ENTRY POINT FOR EXPORT FILES
# =============================================================================

def get_games_with_sp_stats(game_date: str = None,
                             season: int = 2025) -> pd.DataFrame:
    """
    Fetch today's games and attach individual 2025 SP stats for each starter.

    This is the main function called by moneyline and totals export files.

    Returns
    -------
    pd.DataFrame with columns:
      home_team, away_team, game_time, game_type,
      home_sp_name, away_sp_name,
      home_sp_siera, home_sp_xfip, home_sp_fip, home_sp_k_pct,
      home_sp_bb_pct, home_sp_k_bb_pct,
      away_sp_siera, away_sp_xfip, away_sp_fip, away_sp_k_pct,
      away_sp_bb_pct, away_sp_k_bb_pct,
      home_sp_source, away_sp_source   (individual vs team_avg)
    """
    print(f"  Fetching probable starters for {game_date or date.today().strftime('%Y-%m-%d')}...")
    starters_df = get_probable_starters(game_date)
    if starters_df.empty:
        print("  WARNING: No games from MLB API. Cannot score with individual SP stats.")
        return pd.DataFrame()

    pitching_df = load_pitching_stats(season)
    if pitching_df.empty:
        print("  WARNING: No pitching stats loaded.")
        return pd.DataFrame()

    name_index = _build_name_index(pitching_df)

    rows = []
    for _, game in starters_df.iterrows():
        home_team = game["home_team"]
        away_team = game["away_team"]

        # Look up home SP stats (individual, then team-average fallback)
        home_stats = lookup_pitcher_stats(game["home_sp_name"], pitching_df, name_index)
        home_source = "individual"
        if not home_stats:
            home_stats = get_team_avg_sp_stats(home_team, pitching_df)
            home_source = "team_avg"
            if game["home_sp_name"]:
                print(f"    No match for home SP '{game['home_sp_name']}' — using {home_team} avg")

        # Look up away SP stats
        away_stats = lookup_pitcher_stats(game["away_sp_name"], pitching_df, name_index)
        away_source = "individual"
        if not away_stats:
            away_stats = get_team_avg_sp_stats(away_team, pitching_df)
            away_source = "team_avg"
            if game["away_sp_name"]:
                print(f"    No match for away SP '{game['away_sp_name']}' — using {away_team} avg")

        row = {
            "home_team":     home_team,
            "away_team":     away_team,
            "game_time":     game["game_time"],
            "game_type":     game["game_type"],
            "home_sp_name":  game["home_sp_name"] or home_stats.get("pitcher_name", ""),
            "away_sp_name":  game["away_sp_name"] or away_stats.get("pitcher_name", ""),
            "home_sp_source": home_source,
            "away_sp_source": away_source,
        }
        # Prefix home/away onto stat columns
        for col, val in home_stats.items():
            if col.startswith("sp_"):
                row[f"home_{col}"] = val
        for col, val in away_stats.items():
            if col.startswith("sp_"):
                row[f"away_{col}"] = val

        rows.append(row)

    result = pd.DataFrame(rows)
    n_individual = (result["home_sp_source"] == "individual").sum()
    print(f"  ✓ SP stats: {n_individual}/{len(result)} games using individual pitcher stats.")
    return result


# =============================================================================
# LINEUP-BASED TEAM BATTING FEATURES
# =============================================================================

def get_lineup_batting_features(lineups: dict,
                                 batting_df: pd.DataFrame) -> dict:
    """
    Build per-team batting feature aggregates from today's confirmed lineups.

    Instead of using a team-level season average, this aggregates the
    individual stats of the nine players in today's actual lineup.
    This captures player availability (injuries, rest days, roster moves)
    and uses their most current season performance.

    Parameters
    ----------
    lineups : dict
        From get_lineups() — {team_abbr: [player_full_name, ...]}
    batting_df : pd.DataFrame
        From load_batting_stats() — individual batting stats, most recent season.

    Returns
    -------
    dict
        {team_abbr: {off_woba, off_iso, off_babip, off_obp, off_k_pct,
                     off_bb_pct, hr_per_game, xbh_per_game, base_runs_per_game,
                     off_pa_weighted_woba, off_top3_woba, off_top3_k_pct}}
        off_pa_weighted_woba: wOBA weighted by PA rate per lineup spot
        off_top3_woba/k_pct: avg stats for lineup spots 1-3 only
        Empty dict for teams whose lineup players cannot be matched.
    """
    if batting_df.empty or not lineups:
        return {}

    name_index = {normalize_name(str(n)): i for i, n in enumerate(batting_df["Name"])}

    # PA rates by lineup spot (1-indexed MLB averages — top of order bats more)
    _PA_RATES = {1: 4.47, 2: 4.30, 3: 4.16, 4: 3.99, 5: 3.84,
                 6: 3.69, 7: 3.54, 8: 3.38, 9: 3.19}
    _PA_MEAN  = 3.84

    team_feats = {}
    for team, players in lineups.items():
        # matched_slots: list of (0-based lineup slot, batting_df row)
        # Preserving slot index allows correct PA weighting even with partial matches
        matched_slots = []

        for slot_0, player_name in enumerate(players):
            norm = normalize_name(player_name)
            idx  = name_index.get(norm)

            # Fallback: last-name match
            if idx is None:
                last = norm.split()[-1] if norm else ""
                candidates = [k for k in name_index if k.split()[-1] == last]
                if len(candidates) == 1:
                    idx = name_index[candidates[0]]

            # Fallback: first initial + last name
            if idx is None:
                parts = norm.split()
                if len(parts) >= 2:
                    for k in name_index:
                        kp = k.split()
                        if len(kp) >= 2 and kp[-1] == parts[-1] and kp[0][0] == parts[0][0]:
                            idx = name_index[k]
                            break

            if idx is not None:
                matched_slots.append((slot_0, batting_df.iloc[idx]))

        if len(matched_slots) < 3:
            # Too few matches — fall back to full lineup average at call site
            continue

        rows = pd.DataFrame([brow for _, brow in matched_slots])

        def col_mean(col):
            return float(rows[col].fillna(0).mean()) if col in rows.columns else np.nan

        # Per-game counting stats summed across the lineup
        # (each player's per-game rate × 9 lineup slots ≈ team per-game total)
        g_vals = rows["G"].fillna(162).clip(lower=1)
        hr_sum  = (rows["HR"].fillna(0)  / g_vals).sum()
        b2_sum  = (rows["2B"].fillna(0)  / g_vals).sum()
        b3_sum  = (rows["3B"].fillna(0)  / g_vals).sum()
        h_sum   = (rows["H"].fillna(0)   / g_vals).sum()
        bb_sum  = (rows["BB"].fillna(0)  / g_vals).sum()
        ab_sum  = (rows["AB"].fillna(400) / g_vals).sum()

        singles = h_sum - b2_sum - b3_sum - hr_sum
        A = h_sum + bb_sum - hr_sum
        B = 0.8 * singles + 2.1 * b2_sum + 3.4 * b3_sum + 1.8 * hr_sum + 0.1 * bb_sum
        C = ab_sum - h_sum
        base_runs_per_game = float((A * B) / (B + C + 1e-9) + hr_sum)

        # PA-weighted wOBA: lineup spots that bat more often get higher weight.
        # Uses actual slot index so partial lineups are weighted correctly.
        pa_w_list, woba_list = [], []
        for slot_0, brow in matched_slots:
            spot  = slot_0 + 1
            woba_v = brow.get("wOBA", np.nan)
            if pd.notna(woba_v):
                pa_w_list.append(_PA_RATES.get(spot, _PA_MEAN))
                woba_list.append(float(woba_v))
        if pa_w_list:
            total_w = sum(pa_w_list)
            off_pa_weighted_woba = sum(w * v for w, v in zip(pa_w_list, woba_list)) / total_w
        else:
            off_pa_weighted_woba = col_mean("wOBA")

        # Top-3 stats: leadoff, 2-hole, 3-hole — face the pitcher most often
        top3_slots = [(s, r) for s, r in matched_slots if s < 3]
        def top3_mean(col):
            vals = [float(r.get(col, np.nan)) for _, r in top3_slots
                    if pd.notna(r.get(col, np.nan))]
            return float(np.mean(vals)) if vals else col_mean(col)

        season = int(batting_df["Season"].max()) if "Season" in batting_df.columns else "?"
        n_matched = len(matched_slots)

        # Infer lineup source: "confirmed" if ≥8 players matched (full 9-man
        # order), "projected" if fewer (recent-game proxy may have sub-ins)
        lineup_source = "confirmed" if n_matched >= 8 else "projected"

        team_feats[team] = {
            "off_woba":              col_mean("wOBA"),
            "off_iso":               col_mean("ISO"),
            "off_babip":             col_mean("BABIP"),
            "off_obp":               col_mean("OBP"),
            "off_k_pct":             col_mean("K%"),
            "off_bb_pct":            col_mean("BB%"),
            "off_slg":               col_mean("SLG"),
            "off_wrc_plus":          col_mean("wRC+"),
            "hr_per_game":           float(hr_sum),
            "xbh_per_game":          float(b2_sum + b3_sum + hr_sum),
            "base_runs_per_game":    base_runs_per_game,
            "off_pa_weighted_woba":  off_pa_weighted_woba,  # top-of-order weighted
            "off_top3_woba":         top3_mean("wOBA"),     # spots 1-3 only
            "off_top3_k_pct":        top3_mean("K%"),       # spots 1-3 K% (for pitcher outs)
            "_lineup_season":        season,
            "_n_matched":            n_matched,
            "_lineup_source":        lineup_source,
        }

    n_teams = len(team_feats)
    if n_teams:
        n_conf = sum(1 for v in team_feats.values() if v["_lineup_source"] == "confirmed")
        n_proj = n_teams - n_conf
        seasons = set(v["_lineup_season"] for v in team_feats.values())
        print(f"  ✓ Lineup batting: {n_teams} teams "
              f"({n_conf} confirmed, {n_proj} projected) "
              f"| season {'/'.join(str(s) for s in seasons)}.")
    return team_feats
