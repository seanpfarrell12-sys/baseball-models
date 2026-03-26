"""
=============================================================================
SEASON GATE — controls when current-season data enters model training
=============================================================================
The gate opens once MIN_GAMES completed regular-season games have been played.
At that point each model's 01_input script automatically extends its year
lists to include the current season, so training incorporates live results.

MIN_GAMES default: 20  (~first 10–12 days of the season)

Usage (from any 01_input_*.py):
    from utils.season_gate import season_gate_open
    games_played, gate_open = season_gate_open(CURRENT_YEAR)
    if gate_open:
        GAME_YEARS.append(CURRENT_YEAR)
=============================================================================
"""

import json
import requests
from datetime import date
from pathlib import Path

MIN_GAMES  = 20
CACHE_FILE = Path(__file__).parent.parent / "data" / "raw" / ".season_gate_cache.json"


def games_played_this_season(year: int = None) -> int:
    """
    Return the number of completed regular-season MLB games for `year`.
    Result is cached to disk for the calendar day to avoid repeated API
    calls across the 5 model input scripts in a single retrain run.
    """
    if year is None:
        year = date.today().year

    today = date.today().isoformat()

    # ── Check daily cache ─────────────────────────────────────────────────
    if CACHE_FILE.exists():
        try:
            cached = json.loads(CACHE_FILE.read_text())
            if cached.get("date") == today and cached.get("year") == year:
                return int(cached["games_played"])
        except Exception:
            pass

    # ── Fetch from MLB Stats API ──────────────────────────────────────────
    try:
        resp = requests.get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={
                "sportId":   1,
                "startDate": f"{year}-03-01",
                "endDate":   today,
                "gameType":  "R",       # regular season only
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()

        count = 0
        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                if game.get("status", {}).get("abstractGameState") == "Final":
                    count += 1

        # ── Write daily cache ─────────────────────────────────────────────
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(
            {"date": today, "year": year, "games_played": count},
            indent=2,
        ))
        return count

    except Exception as exc:
        print(f"  (season_gate) Could not fetch game count from MLB API: {exc}")
        return 0


def season_gate_open(year: int = None, min_games: int = MIN_GAMES) -> tuple:
    """
    Returns (games_played: int, gate_open: bool).

    gate_open is True once `min_games` completed regular-season games
    have been played for `year`.  Default threshold: 20 games.

    Example
    -------
    games_played, gate_open = season_gate_open(2026)
    # games_played=23, gate_open=True  → add 2026 to training years
    # games_played=14, gate_open=False → stay on prior-season data
    """
    n = games_played_this_season(year)
    return n, n >= min_games
