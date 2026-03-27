"""
utils/bullpen.py — Bullpen availability assessment using recent game data.

Queries the MLB Stats API for the last 3 days of game boxscores to compute:
  - bp_recent_ip : total bullpen innings pitched in last 3 days
  - bp_arms_used_yesterday : number of distinct bullpen arms used yesterday
  - bp_taxed : True if bullpen has thrown >= 9 IP in last 3 days or 5+ arms yesterday

This is used as a POST-MODEL lambda adjustment in totals scoring.
A taxed bullpen means the SP needs to go deeper, or the team gives up more runs.
The adjustment is conservative (+2% on run totals) to avoid over-correcting.

Usage:
    from utils.bullpen import get_bullpen_availability
    avail = get_bullpen_availability("2026-03-26")
    # Returns: {team_abbr: {"bp_recent_ip": 12.3, "bp_arms_used_yesterday": 4,
    #                        "bp_taxed": True, "bp_adjustment": 1.02}}
"""

import time
import requests
from datetime import date, timedelta
from collections import defaultdict

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# MLB team abbreviation → standardized abbreviation used in our models
_MLB_TO_STD = {
    "LAA": "LAA", "HOU": "HOU", "OAK": "OAK", "TOR": "TOR", "ATL": "ATL",
    "MIL": "MIL", "STL": "STL", "CHC": "CHC", "ARI": "ARI", "LAD": "LAD",
    "SF":  "SF",  "CLE": "CLE", "SEA": "SEA", "MIA": "MIA", "NYM": "NYM",
    "WSH": "WSH", "BAL": "BAL", "SD":  "SD",  "PHI": "PHI", "PIT": "PIT",
    "TEX": "TEX", "TB":  "TB",  "BOS": "BOS", "CIN": "CIN", "COL": "COL",
    "KC":  "KC",  "DET": "DET", "MIN": "MIN", "CWS": "CWS", "NYY": "NYY",
}

# Thresholds for labeling a bullpen as "taxed"
BP_IP_TAXED_THRESHOLD   = 9.0   # >= 9 total BP innings in last 3 days
BP_ARMS_TAXED_THRESHOLD = 5     # >= 5 distinct arms used yesterday


def _ip_to_float(ip_val) -> float:
    """
    Convert MLB Stats API inningsPitched value to decimal innings.

    The API returns IP as a string like "6.1" (6 full innings + 1 out),
    where the decimal digit represents thirds, not tenths.
    e.g. "2.1" = 2 + 1/3 = 2.333..., "2.2" = 2 + 2/3 = 2.667...
    """
    try:
        ip = float(ip_val)
        full = int(ip)
        thirds = round((ip - full) * 10)
        return full + thirds / 3.0
    except (TypeError, ValueError):
        return 0.0


def get_bullpen_availability(game_date: str = None) -> dict:
    """
    Return bullpen availability dict for all teams as of game_date.

    Looks back 3 days (yesterday, 2 days ago, 3 days ago) at completed MLB
    regular-season games to accumulate bullpen workload.

    Parameters
    ----------
    game_date : str, optional
        Date in "YYYY-MM-DD" format. Defaults to today.

    Returns
    -------
    dict[str, dict]
        Keyed by standardized team abbreviation. Each value contains:
          bp_recent_ip          : float  — total BP innings in last 3 days
          bp_arms_used_yesterday: int    — distinct relief arms used yesterday
          bp_taxed              : bool   — True if workload exceeds thresholds
          bp_adjustment         : float  — run-total multiplier (1.02 if taxed, else 1.0)
    """
    if game_date is None:
        game_date = date.today().isoformat()

    target = date.fromisoformat(game_date)
    # Yesterday, 2 days ago, 3 days ago
    lookback_dates = [
        (target - timedelta(days=d)).isoformat()
        for d in range(1, 4)
    ]
    yesterday = lookback_dates[0]

    recent_ip      = defaultdict(float)  # total BP IP over last 3 days
    arms_yesterday = defaultdict(set)    # distinct pitcher IDs used yesterday

    for check_date in lookback_dates:
        try:
            url = (f"{MLB_API_BASE}/schedule"
                   f"?sportId=1&gameType=R&date={check_date}&hydrate=boxscore")
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            time.sleep(0.3)
            continue

        for date_block in data.get("dates", []):
            for game in date_block.get("games", []):
                if game.get("status", {}).get("abstractGameState") != "Final":
                    continue
                bs = game.get("boxscore", {})
                for side in ("home", "away"):
                    team_info = bs.get("teams", {}).get(side, {})
                    team_abbr = team_info.get("team", {}).get("abbreviation", "")
                    team_std  = _MLB_TO_STD.get(team_abbr, team_abbr)

                    pitchers = team_info.get("pitchers", [])
                    # Bullpen = everyone after the starter (index 0)
                    bp_ids = pitchers[1:] if len(pitchers) > 1 else []

                    players = team_info.get("players", {})
                    for pid in bp_ids:
                        key    = f"ID{pid}"
                        pdata  = players.get(key, {})
                        gstats = pdata.get("stats", {}).get("pitching", {})
                        ip_val = gstats.get("inningsPitched", 0)
                        ip_float = _ip_to_float(ip_val)
                        recent_ip[team_std] += ip_float
                        if check_date == yesterday:
                            arms_yesterday[team_std].add(pid)

        time.sleep(0.3)  # courteous rate-limiting

    # Build result dict
    all_teams = set(list(recent_ip.keys()) + list(arms_yesterday.keys()))
    result = {}
    for team in all_teams:
        r_ip   = recent_ip[team]
        n_arms = len(arms_yesterday[team])
        taxed  = (r_ip >= BP_IP_TAXED_THRESHOLD) or (n_arms >= BP_ARMS_TAXED_THRESHOLD)
        # Conservative: taxed bullpen raises expected opponent run total by ~2%
        bp_adj = 1.02 if taxed else 1.0
        result[team] = {
            "bp_recent_ip":          round(r_ip, 2),
            "bp_arms_used_yesterday": n_arms,
            "bp_taxed":              taxed,
            "bp_adjustment":         bp_adj,
        }

    return result
