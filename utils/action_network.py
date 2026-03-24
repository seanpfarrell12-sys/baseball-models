"""
=============================================================================
ACTION NETWORK — ODDS AND PROPS DATA UTILITY
=============================================================================
Purpose : Pull MLB moneyline odds, totals lines, and player props from
          Action Network using your paid subscription session token.

SETUP (one-time, ~2 minutes):
─────────────────────────────────────────────────────────────────────────────
Token management is now fully automated via utils/an_login.py.

Run once to save your credentials:
  python utils/an_login.py --setup

After that, all export files auto-refresh the token before each run.
No manual DevTools work needed.
─────────────────────────────────────────────────────────────────────────────

Data available with paid Action Network PRO subscription:
  - Moneyline consensus odds (aggregated from 10+ books)
  - Over/Under totals lines with juice
  - Public betting percentages and money percentages
  - Sharp money indicators
  - Player props: hitter total bases, pitcher outs/IP, strikeouts, etc.
  - Line movement history

For R users:
  - `os.environ.get()` = Sys.getenv() in R
  - `requests.Session()` = persistent HTTP connection with shared cookies/headers
  - `.get(key, default)` on a dict = safely access with fallback value
=============================================================================
"""

import os
import json
import time
import requests
from datetime import datetime, date
from pathlib import Path
import pandas as pd
import numpy as np

# =============================================================================
# CONFIGURATION — paste your token here OR set AN_TOKEN env variable
# =============================================================================

# Option 1: Hard-code your token (do not commit to git if sharing code)
AUTH_TOKEN = ""   # <-- PASTE YOUR ACTION NETWORK SESSION TOKEN HERE

# Option 2: Read from environment variable (more secure)
# In terminal: export AN_TOKEN="your_token_here"
# In R: Sys.setenv(AN_TOKEN = "your_token_here")
AUTH_TOKEN = AUTH_TOKEN or os.environ.get("AN_TOKEN", "")

# Action Network internal API base URL
BASE_URL = "https://api.actionnetwork.com/web/v1"

# Book IDs for consensus odds (Action Network internal IDs)
# These are the major US sportsbooks
BOOK_IDS = {
    15:  "DraftKings",
    30:  "FanDuel",
    76:  "BetMGM",
    75:  "PointsBet",
    123: "Caesars",
    69:  "WynnBet",
    68:  "Unibet",
    972: "ESPN BET",
    71:  "Barstool",
    247: "SuperBook",
    79:  "BetRivers",
}
DEFAULT_BOOK_IDS = list(BOOK_IDS.keys())

# Team name mapping: Action Network full names → our abbreviations
AN_TEAM_MAP = {
    "Arizona Diamondbacks":     "ARI",
    "Atlanta Braves":           "ATL",
    "Baltimore Orioles":        "BAL",
    "Boston Red Sox":           "BOS",
    "Chicago Cubs":             "CHC",
    "Chicago White Sox":        "CWS",
    "Cincinnati Reds":          "CIN",
    "Cleveland Guardians":      "CLE",
    "Colorado Rockies":         "COL",
    "Detroit Tigers":           "DET",
    "Houston Astros":           "HOU",
    "Kansas City Royals":       "KCR",
    "Los Angeles Angels":       "LAA",
    "Los Angeles Dodgers":      "LAD",
    "Miami Marlins":            "MIA",
    "Milwaukee Brewers":        "MIL",
    "Minnesota Twins":          "MIN",
    "New York Mets":            "NYM",
    "New York Yankees":         "NYY",
    "Oakland Athletics":        "OAK",
    "Philadelphia Phillies":    "PHI",
    "Pittsburgh Pirates":       "PIT",
    "San Diego Padres":         "SDP",
    "Seattle Mariners":         "SEA",
    "San Francisco Giants":     "SFG",
    "St. Louis Cardinals":      "STL",
    "Tampa Bay Rays":           "TBR",
    "Texas Rangers":            "TEX",
    "Toronto Blue Jays":        "TOR",
    "Washington Nationals":     "WSN",
    # Short versions Action Network sometimes uses
    "D-backs":       "ARI",
    "Braves":        "ATL",
    "Orioles":       "BAL",
    "Red Sox":       "BOS",
    "Cubs":          "CHC",
    "White Sox":     "CWS",
    "Reds":          "CIN",
    "Guardians":     "CLE",
    "Rockies":       "COL",
    "Tigers":        "DET",
    "Astros":        "HOU",
    "Royals":        "KCR",
    "Angels":        "LAA",
    "Dodgers":       "LAD",
    "Marlins":       "MIA",
    "Brewers":       "MIL",
    "Twins":         "MIN",
    "Mets":          "NYM",
    "Yankees":       "NYY",
    "Athletics":     "OAK",
    "Phillies":      "PHI",
    "Pirates":       "PIT",
    "Padres":        "SDP",
    "Mariners":      "SEA",
    "Giants":        "SFG",
    "Cardinals":     "STL",
    "Rays":          "TBR",
    "Rangers":       "TEX",
    "Blue Jays":     "TOR",
    "Nationals":     "WSN",
}


# =============================================================================
# HTTP SESSION SETUP
# =============================================================================

def _load_token_from_file() -> str:
    """Load the current token from the credentials file saved by an_login.py."""
    try:
        creds_path = Path(__file__).parent / ".an_credentials.json"
        if creds_path.exists():
            data = json.loads(creds_path.read_text())
            return data.get("token", "")
    except Exception:
        pass
    return ""


def build_session(token: str = None) -> requests.Session:
    """
    Create a requests Session pre-loaded with Action Network auth headers.

    Using a Session object persists cookies and headers across all requests
    in the same session — like a logged-in browser tab.

    In R: httr uses add_headers() + set_cookies() for similar functionality.
    """
    session = requests.Session()

    # Headers that mimic a real browser (important for Cloudflare bypass)
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept":          "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin":          "https://www.actionnetwork.com",
        "Referer":         "https://www.actionnetwork.com/",
    })

    # Add auth token if provided — priority order:
    #   1. Explicitly passed token argument
    #   2. Module-level AUTH_TOKEN (hard-coded or env var)
    #   3. Token saved by an_login.py in .an_credentials.json
    active_token = token or AUTH_TOKEN or _load_token_from_file()
    if active_token:
        # Action Network uses AN_SESSION_TOKEN_V1 as a cookie for its web API.
        # We also send it as a Bearer header in case certain endpoints check that.
        # Sending both ensures compatibility regardless of which the endpoint expects.
        session.headers["Authorization"] = f"Bearer {active_token}"
        session.cookies.set(
            "AN_SESSION_TOKEN_V1",
            active_token,
            domain="api.actionnetwork.com",
        )
    else:
        print("  WARNING: No AUTH_TOKEN set. Paid features (props) will not be available.")
        print("  See module docstring for setup instructions.")

    return session


# =============================================================================
# FUNCTION 1: Fetch Today's MLB Games with Moneyline + Totals Odds
# =============================================================================

def fetch_mlb_odds(game_date: str = None, token: str = None) -> pd.DataFrame:
    """
    Fetch MLB game odds from Action Network for a given date.

    Returns consensus moneyline and totals odds across major US sportsbooks.

    Parameters
    ----------
    game_date : str, optional
        Date in 'YYYY-MM-DD' format. Defaults to today.
    token : str, optional
        Override the module-level AUTH_TOKEN.

    Returns
    -------
    pd.DataFrame
        One row per game with columns:
          - home_team, away_team  (abbreviated)
          - home_team_full, away_team_full
          - game_time
          - home_ml, away_ml      (consensus American moneyline odds)
          - ou_line               (consensus over/under line)
          - over_juice, under_juice
          - home_ml_book          (best home moneyline book)
          - away_ml_book
          - public_home_pct       (% of bets on home team)
          - public_away_pct

    Usage
    -----
        from utils.action_network import fetch_mlb_odds
        odds = fetch_mlb_odds("2025-04-01")
        print(odds[["home_team", "away_team", "home_ml", "ou_line"]])
    """
    if game_date is None:
        game_date = date.today().strftime("%Y-%m-%d")

    print(f"  Fetching Action Network MLB odds for {game_date}...")

    session  = build_session(token)
    book_ids = ",".join(map(str, DEFAULT_BOOK_IDS))

    url    = f"{BASE_URL}/scoreboard/mlb"
    params = {
        "period":  "game",
        "bookIds": book_ids,
        "date":    game_date.replace("-", ""),  # AN uses YYYYMMDD format
    }

    try:
        response = session.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.HTTPError as e:
        print(f"  ERROR: HTTP {e.response.status_code} — {e}")
        if e.response.status_code == 401:
            print("  Token may be expired. Re-grab from browser DevTools.")
        _dump_debug(url, params, getattr(e, 'response', None))
        return pd.DataFrame()
    except Exception as e:
        print(f"  ERROR: Request failed — {e}")
        return pd.DataFrame()

    # --- Parse response ------------------------------------------------------
    games_raw = data.get("games", data.get("data", []))
    if not games_raw:
        print(f"  WARNING: No games found in response. Raw keys: {list(data.keys())}")
        _dump_debug(url, params, response=None, raw_data=data)
        return pd.DataFrame()

    records = []
    for game in games_raw:
        try:
            record = _parse_game_odds(game)
            if record:
                records.append(record)
        except Exception as e:
            print(f"  WARNING: Could not parse game {game.get('id', '?')}: {e}")

    if not records:
        print("  WARNING: Parsed 0 games. Response structure may have changed.")
        print("  Dumping first game structure for debugging:")
        if games_raw:
            print(json.dumps(games_raw[0], indent=2)[:2000])
        return pd.DataFrame()

    df = pd.DataFrame(records)
    print(f"  ✓ Fetched {len(df)} MLB games from Action Network.")
    return df


def _parse_game_odds(game: dict) -> dict:
    """
    Parse a single game object from Action Network's scoreboard response.

    Action Network's JSON structure (approximate — may vary):
    {
      "id": 12345,
      "home_team": {"full_name": "New York Yankees", "abbr": "NYY"},
      "away_team": {"full_name": "Boston Red Sox",   "abbr": "BOS"},
      "start_time": "2025-04-01T19:05:00Z",
      "status": "scheduled",
      "odds": [
        {
          "book_id": 15,        -- DraftKings
          "ml_home": -140,
          "ml_away": 120,
          "total": 8.5,
          "over_odds": -110,
          "under_odds": -110,
          "home_spread": -1.5,  -- run line
          ...
        },
        ...
      ],
      "public_betting": {
        "home_bets_pct": 58,
        "away_bets_pct": 42,
        "home_money_pct": 65,
        "away_money_pct": 35,
      }
    }

    Note: Exact field names may differ. We try multiple key variations.
    """
    def safe_get(d, *keys):
        """Try multiple key names and return first found. Returns None if all missing."""
        for k in keys:
            if isinstance(d, dict) and k in d:
                return d[k]
        return None

    # Team info — actual structure uses a "teams" array with away_team_id/home_team_id
    home_id = game.get("home_team_id")
    away_id = game.get("away_team_id")
    teams   = {t["id"]: t for t in game.get("teams", []) if isinstance(t, dict) and "id" in t}

    home = teams.get(home_id, {})
    away = teams.get(away_id, {})

    home_full = safe_get(home, "full_name", "name", "display_name") or ""
    away_full = safe_get(away, "full_name", "name", "display_name") or ""

    # AN abbreviations (e.g. "TB") may differ from our standard ones (e.g. "TBR")
    # Use our AN_TEAM_MAP for full-name lookup first, then fall back to abbr field
    _abbr_map = {"TB": "TBR", "KC": "KCR", "SD": "SDP", "SF": "SFG",
                 "WAS": "WSN", "WS": "WSN", "CHW": "CWS"}
    home_abbr = AN_TEAM_MAP.get(home_full) or _abbr_map.get(
        safe_get(home, "abbr", "abbreviation"), safe_get(home, "abbr", "abbreviation") or ""
    )
    away_abbr = AN_TEAM_MAP.get(away_full) or _abbr_map.get(
        safe_get(away, "abbr", "abbreviation"), safe_get(away, "abbr", "abbreviation") or ""
    )

    # Odds — compute consensus (median) across all available books
    odds_list = safe_get(game, "odds", "books", "book_odds") or []
    if isinstance(odds_list, dict):
        # Some versions nest odds differently
        odds_list = list(odds_list.values())

    home_mls, away_mls, totals, over_juices, under_juices = [], [], [], [], []

    for book_odds in odds_list:
        if not isinstance(book_odds, dict):
            continue

        # Moneyline — actual AN field names are ml_home / ml_away
        home_ml = safe_get(book_odds, "ml_home", "home_ml", "moneyline_home", "home_money_line")
        away_ml = safe_get(book_odds, "ml_away", "away_ml", "moneyline_away", "away_money_line")
        total   = safe_get(book_odds, "total", "over_under", "ou", "game_total")
        # AN uses "over" / "under" for juice (not "over_odds")
        ov_odds = safe_get(book_odds, "over", "over_odds", "over_price")
        un_odds = safe_get(book_odds, "under", "under_odds", "under_price")

        if home_ml is not None:
            home_mls.append(float(home_ml))
        if away_ml is not None:
            away_mls.append(float(away_ml))
        if total is not None:
            totals.append(float(total))
        if ov_odds is not None:
            over_juices.append(float(ov_odds))
        if un_odds is not None:
            under_juices.append(float(un_odds))

    # Consensus = median across books (more robust than mean to outliers)
    # In R: median(x, na.rm=TRUE)
    def med(lst, default=None):
        return float(np.median(lst)) if lst else default

    consensus_home_ml  = med(home_mls,   default=None)
    consensus_away_ml  = med(away_mls,   default=None)
    consensus_total    = med(totals,      default=None)
    consensus_ov_juice = med(over_juices, default=-110)
    consensus_un_juice = med(under_juices,default=-110)

    # Public betting percentages — in AN these live directly in each odds entry
    # We take the value from the first odds entry that has them
    home_bet_pct   = None
    away_bet_pct   = None
    home_money_pct = None
    for book_odds in (odds_list if isinstance(odds_list, list) else []):
        if not isinstance(book_odds, dict):
            continue
        if home_bet_pct is None:
            home_bet_pct   = book_odds.get("ml_home_public")
            away_bet_pct   = book_odds.get("ml_away_public")
            home_money_pct = book_odds.get("ml_home_money")
        if home_bet_pct is not None:
            break

    return {
        "game_id":        game.get("id", ""),
        "game_date":      game.get("start_time", game.get("game_time", ""))[:10],
        "game_time":      game.get("start_time", game.get("game_time", "")),
        "home_team":      home_abbr,
        "away_team":      away_abbr,
        "home_team_full": home_full,
        "away_team_full": away_full,
        # Moneyline
        "home_ml":        consensus_home_ml,
        "away_ml":        consensus_away_ml,
        "n_ml_books":     len(home_mls),
        # Totals
        "ou_line":        consensus_total,
        "over_juice":     consensus_ov_juice,
        "under_juice":    consensus_un_juice,
        "n_ou_books":     len(totals),
        # Public betting
        "public_home_bet_pct":  home_bet_pct,
        "public_away_bet_pct":  away_bet_pct,
        "public_home_money_pct": home_money_pct,
    }


# =============================================================================
# FUNCTION 2: Fetch Player Props (Paid Subscription Required)
# =============================================================================

def fetch_player_props(game_id: str, prop_type: str = "batter_total_bases",
                        token: str = None) -> pd.DataFrame:
    """
    Fetch player prop lines from Action Network for a specific game.

    Requires Action Network PRO subscription and valid auth token.

    Parameters
    ----------
    game_id : str
        Action Network's internal game ID (from fetch_mlb_odds output).
    prop_type : str
        Type of prop to fetch. Known types:
          - "batter_total_bases"    : Hitter total bases O/U
          - "pitcher_outs_recorded" : Pitcher outs recorded O/U
          - "batter_hits"           : Hitter hits O/U
          - "pitcher_strikeouts"    : Pitcher K total O/U
          - "batter_home_runs"      : HR props (alt lines)
    token : str, optional
        Override module-level AUTH_TOKEN.

    Returns
    -------
    pd.DataFrame
        One row per player-line with:
          player_name, team, prop_line, over_juice, under_juice,
          prop_type, game_id, book_name

    Notes
    -----
    The exact endpoint path for props may be:
      /games/{game_id}/props
      /props/mlb/{game_id}
      /events/{game_id}/player-props
    We try multiple paths and return whichever works.
    """
    if not (AUTH_TOKEN or token):
        print("  ERROR: Props require AUTH_TOKEN. See module docstring for setup.")
        return pd.DataFrame()

    session = build_session(token)

    # Try multiple endpoint patterns (AN has changed these historically)
    prop_endpoints = [
        f"{BASE_URL}/games/{game_id}/props",
        f"{BASE_URL}/props/mlb/{game_id}",
        f"{BASE_URL}/events/{game_id}/player-props",
        f"{BASE_URL}/scoreboard/mlb/{game_id}/player-props",
    ]

    data = None
    for endpoint in prop_endpoints:
        try:
            params = {"type": prop_type} if prop_type else {}
            resp   = session.get(endpoint, params=params, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                print(f"  ✓ Props endpoint: {endpoint}")
                break
            elif resp.status_code == 404:
                continue  # Try next endpoint
            elif resp.status_code == 401:
                print("  ERROR: 401 Unauthorized — token expired or invalid.")
                return pd.DataFrame()
        except Exception as e:
            continue

    if data is None:
        print(f"  WARNING: No props endpoint responded for game {game_id}.")
        print("  The exact props URL may have changed — check AN DevTools network tab.")
        return pd.DataFrame()

    # --- Parse props response ------------------------------------------------
    props_raw = data.get("props", data.get("player_props", data.get("data", [])))
    if not props_raw:
        print(f"  WARNING: No props in response. Keys: {list(data.keys())}")
        return pd.DataFrame()

    records = []
    for prop in props_raw:
        try:
            record = _parse_prop(prop)
            if record:
                records.append(record)
        except Exception as e:
            pass

    if not records:
        print("  WARNING: Could not parse props. Raw structure:")
        if props_raw:
            print(json.dumps(props_raw[0], indent=2)[:1000])
        return pd.DataFrame()

    return pd.DataFrame(records)


def _parse_prop(prop: dict) -> dict:
    """Parse a single prop offer from Action Network response."""
    def safe_get(d, *keys):
        for k in keys:
            if isinstance(d, dict) and k in d:
                return d[k]
        return None

    player = safe_get(prop, "player", "athlete") or {}
    player_name = (
        safe_get(player, "full_name", "name", "display_name")
        or safe_get(prop, "player_name", "name")
        or "Unknown"
    )
    team = (
        safe_get(player, "team", "team_abbr")
        or safe_get(prop, "team")
        or ""
    )
    if isinstance(team, dict):
        team = safe_get(team, "abbr", "abbreviation", "short_name") or ""

    team = AN_TEAM_MAP.get(team, team)

    line      = safe_get(prop, "line", "total", "prop_line", "value")
    over_odds = safe_get(prop, "over_odds", "over_price", "over")
    under_odds= safe_get(prop, "under_odds", "under_price", "under")
    book_name = safe_get(prop, "book", "book_name", "sportsbook") or "Consensus"
    prop_type = safe_get(prop, "prop_type", "type", "market_name") or ""

    if line is None:
        return None

    return {
        "player_name":  str(player_name),
        "team":         str(team),
        "prop_type":    str(prop_type),
        "prop_line":    float(line),
        "over_juice":   float(over_odds) if over_odds is not None else -110.0,
        "under_juice":  float(under_odds) if under_odds is not None else -110.0,
        "book_name":    str(book_name),
    }


# =============================================================================
# FUNCTION 3: Fetch Props for All Today's Games (Batch)
# =============================================================================

def fetch_all_props_today(prop_type: str = "batter_total_bases",
                           game_date: str = None,
                           token: str = None) -> pd.DataFrame:
    """
    Fetch player props across all of today's MLB games.

    Workflow:
      1. Fetch today's games to get Action Network game IDs
      2. For each game, fetch props of the requested type
      3. Combine and return all player-prop rows

    Parameters
    ----------
    prop_type : str
        "batter_total_bases" or "pitcher_outs_recorded"
    game_date : str
        "YYYY-MM-DD" format, defaults to today.

    Returns
    -------
    pd.DataFrame
        All player props for today's slate with player_name, team,
        prop_line, over_juice, under_juice columns.
    """
    if game_date is None:
        game_date = date.today().strftime("%Y-%m-%d")

    print(f"  Fetching all {prop_type} props for {game_date}...")

    # Step 1: Get today's games and their IDs
    games_df = fetch_mlb_odds(game_date, token=token)
    if games_df.empty or "game_id" not in games_df.columns:
        print("  ERROR: Could not fetch game IDs. Cannot retrieve props.")
        return pd.DataFrame()

    all_props = []
    game_ids  = games_df["game_id"].dropna().unique()

    print(f"  Found {len(game_ids)} games. Fetching props for each...")

    for i, game_id in enumerate(game_ids, 1):
        # Find the teams for this game (for display)
        game_row  = games_df[games_df["game_id"] == game_id].iloc[0]
        home_team = game_row.get("home_team", "")
        away_team = game_row.get("away_team", "")

        print(f"    [{i}/{len(game_ids)}] {away_team} @ {home_team} (id={game_id})")

        props = fetch_player_props(str(game_id), prop_type=prop_type, token=token)
        if not props.empty:
            props["home_team"] = home_team
            props["away_team"] = away_team
            props["game_id"]   = game_id
            all_props.append(props)

        time.sleep(0.5)  # Polite delay between requests

    if not all_props:
        print("  WARNING: No props fetched for any game.")
        return pd.DataFrame()

    combined = pd.concat(all_props, ignore_index=True)
    print(f"  ✓ Total props fetched: {len(combined)} player-prop rows.")
    return combined


# =============================================================================
# FUNCTION 4: Format for Export Files
# =============================================================================

def get_moneyline_odds(game_date: str = None, token: str = None) -> pd.DataFrame:
    """
    Wrapper that returns moneyline odds in the format expected by
    04_export_moneyline.py.

    Returns columns: home_team, away_team, home_odds_american, away_odds_american
    Drops rows with missing moneyline data.
    """
    df = fetch_mlb_odds(game_date, token=token)
    if df.empty:
        return df

    result = df[["home_team", "away_team", "home_ml", "away_ml",
                 "public_home_bet_pct", "public_away_bet_pct"]].copy()
    result = result.rename(columns={
        "home_ml": "home_odds_american",
        "away_ml": "away_odds_american",
    })
    result = result.dropna(subset=["home_odds_american", "away_odds_american"])
    return result


def get_totals_odds(game_date: str = None, token: str = None) -> pd.DataFrame:
    """
    Wrapper that returns totals odds in the format expected by
    04_export_totals.py.

    Returns columns: home_team, away_team, ou_line, over_juice, under_juice
    """
    df = fetch_mlb_odds(game_date, token=token)
    if df.empty:
        return df

    result = df[["home_team", "away_team", "ou_line",
                 "over_juice", "under_juice"]].copy()
    result = result.dropna(subset=["ou_line"])
    return result


def get_hitter_tb_odds(game_date: str = None, token: str = None) -> pd.DataFrame:
    """
    Wrapper that returns hitter total bases props in the format expected by
    04_export_hitter_tb.py.

    Returns columns: player_name, team, prop_line, over_juice, under_juice
    """
    return fetch_all_props_today(
        prop_type="batter_total_bases",
        game_date=game_date,
        token=token
    )


def get_pitcher_outs_odds(game_date: str = None, token: str = None) -> pd.DataFrame:
    """
    Wrapper that returns pitcher outs props in the format expected by
    04_export_pitcher_outs.py.

    Returns columns: pitcher_name, team, prop_line, over_juice, under_juice
    """
    df = fetch_all_props_today(
        prop_type="pitcher_outs_recorded",
        game_date=game_date,
        token=token
    )
    # Rename player_name → pitcher_name to match the export file's expected column
    if not df.empty and "player_name" in df.columns:
        df = df.rename(columns={"player_name": "pitcher_name"})
    return df


# =============================================================================
# DEBUG HELPER
# =============================================================================

def _dump_debug(url: str, params: dict, response=None, raw_data: dict = None):
    """
    Print debug information to help diagnose API issues.

    When the API structure changes, this output helps identify what
    field names Action Network is now using.
    """
    print("\n  ── DEBUG INFO ──────────────────────────────────────────────")
    print(f"  URL: {url}")
    print(f"  Params: {params}")
    if response is not None:
        try:
            print(f"  Response status: {response.status_code}")
            print(f"  Response preview: {response.text[:500]}")
        except Exception:
            pass
    if raw_data is not None:
        print(f"  Response keys: {list(raw_data.keys())}")
        print(f"  First 500 chars: {json.dumps(raw_data)[:500]}")
    print("  ─────────────────────────────────────────────────────────────\n")


def test_connection(token: str = None) -> bool:
    """
    Quick test to verify your auth token and API connectivity.

    Run this first to confirm everything is working before your daily pulls.

    Usage:
        python utils/action_network.py
        # Or from another file:
        from utils.action_network import test_connection
        test_connection()
    """
    print("=" * 60)
    print("ACTION NETWORK CONNECTION TEST")
    print("=" * 60)

    active_token = token or AUTH_TOKEN or _load_token_from_file()
    if not active_token:
        print("  FAIL: No AUTH_TOKEN set.")
        print("  See module docstring for setup instructions.")
        return False

    print(f"  Token found: {active_token[:8]}...{active_token[-4:]}")

    today = date.today().strftime("%Y-%m-%d")
    df = fetch_mlb_odds(today, token=active_token)

    if df.empty:
        print(f"\n  FAIL: No games returned for {today}.")
        print("  Possible causes:")
        print("    1. Token expired — re-grab from browser DevTools")
        print("    2. Off-season / no games scheduled today")
        print("    3. API endpoint changed — check network tab in DevTools")
        return False

    print(f"\n  SUCCESS: {len(df)} games found for {today}")
    print(f"\n  Sample games:")
    for _, row in df.head(3).iterrows():
        ml_str  = f"ML: {row['away_team']} {row.get('away_ml','-'):+.0f} / {row['home_team']} {row.get('home_ml','-'):+.0f}" \
                  if row.get('home_ml') else "ML: N/A"
        ou_str  = f"O/U: {row.get('ou_line','N/A')}" if row.get('ou_line') else "O/U: N/A"
        print(f"    {row['away_team']:3s} @ {row['home_team']:3s}  |  {ml_str}  |  {ou_str}")

    print("\n  " + "=" * 58)
    return True


# =============================================================================
# MAIN — Run this file directly to test your connection
# =============================================================================
if __name__ == "__main__":
    """
    Run this file directly to test your Action Network connection:
        python utils/action_network.py

    If it works, you'll see today's MLB games with odds.
    If it fails, follow the setup instructions at the top of the file.
    """
    success = test_connection()
    if success:
        print("\nYour Action Network token is working.")
        print("The 4 export files will now use Action Network as the odds source.")
    else:
        print("\nSetup needed — see instructions at top of this file.")
