"""
=============================================================================
OVER/UNDER TOTALS MODEL — FILE 1 OF 4: DATA INPUT
=============================================================================
Purpose : Pull all raw data needed for the Poisson regression totals model.
Sources : FanGraphs (via pybaseball), Baseball Reference (via pybaseball),
          National Weather Service API (free), park factors (static lookup)
Output  : CSV files saved to ../data/raw/

Totals modeling background:
  - We predict total combined runs scored in a game (e.g., over/under 8.5).
  - Key signals: team offense (wRC+, wOBA), SP quality (SIERA), park factor,
    weather (temperature, wind speed/direction), umpire tendencies.
  - Poisson regression models the RATE of run scoring — appropriate because
    runs are count data (0, 1, 2, ... non-negative integers).

For R users:
  - Poisson regression = glm(runs ~ features, family=poisson) in R
  - We'll fit this in statsmodels (Python's equivalent of base R's glm)
  - Weather data comes from the NWS API (free, no key required)
=============================================================================
"""

import os
import time
import json
import requests        # For HTTP API calls (like httr or curl in R)
import pandas as pd
import numpy as np
import pybaseball as pyb
from datetime import date as _date

pyb.cache.enable()

# --- Configuration ----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")

TRAIN_YEARS  = [2023, 2024, 2025]
CURRENT_YEAR = _date.today().year   # 2026 — refreshed each run

# All 30 MLB team abbreviations for Baseball Reference schedule pulls
ALL_TEAMS_BREF = [
    "ARI", "ATL", "BAL", "BOS", "CHC", "CWS", "CIN", "CLE",
    "COL", "DET", "HOU", "KCR", "LAA", "LAD", "MIA", "MIL",
    "MIN", "NYM", "NYY", "OAK", "PHI", "PIT", "SDP", "SEA",
    "SFG", "STL", "TBR", "TEX", "TOR", "WSN"
]

# =============================================================================
# STATIC DATA: Park Factors and Stadium Metadata
# =============================================================================
# Park factors from FanGraphs (2024 values — update annually).
# 100 = league average scoring environment.
# >100 = hitter-friendly (Coors Field = ~118), <100 = pitcher-friendly.
#
# We also include lat/lon for each stadium for weather API lookups.
# In R you'd store this as a named list or data.frame.
PARK_FACTORS = {
    # fmt: off  (turns off code formatter for this block to keep table aligned)
    # Team     PF_runs  PF_HR   City          Lat       Lon       Roof    Surface
    "ARI": {"pf_runs": 97,  "pf_hr": 101, "city": "Phoenix",        "lat": 33.4453, "lon": -112.0667, "roof": "retractable", "surface": "artificial"},
    "ATL": {"pf_runs": 102, "pf_hr": 104, "city": "Atlanta",         "lat": 33.8908, "lon": -84.4678,  "roof": "open",        "surface": "natural"},
    "BAL": {"pf_runs": 104, "pf_hr": 110, "city": "Baltimore",       "lat": 39.2839, "lon": -76.6218,  "roof": "open",        "surface": "natural"},
    "BOS": {"pf_runs": 105, "pf_hr": 95,  "city": "Boston",          "lat": 42.3467, "lon": -71.0972,  "roof": "open",        "surface": "natural"},
    "CHC": {"pf_runs": 101, "pf_hr": 104, "city": "Chicago",         "lat": 41.9484, "lon": -87.6553,  "roof": "open",        "surface": "natural"},
    "CWS": {"pf_runs": 96,  "pf_hr": 99,  "city": "Chicago",         "lat": 41.8300, "lon": -87.6339,  "roof": "open",        "surface": "natural"},
    "CIN": {"pf_runs": 103, "pf_hr": 106, "city": "Cincinnati",      "lat": 39.0975, "lon": -84.5061,  "roof": "open",        "surface": "natural"},
    "CLE": {"pf_runs": 98,  "pf_hr": 96,  "city": "Cleveland",       "lat": 41.4962, "lon": -81.6852,  "roof": "open",        "surface": "natural"},
    "COL": {"pf_runs": 116, "pf_hr": 116, "city": "Denver",          "lat": 39.7559, "lon": -104.9942, "roof": "open",        "surface": "natural"},
    "DET": {"pf_runs": 96,  "pf_hr": 93,  "city": "Detroit",         "lat": 42.3390, "lon": -83.0485,  "roof": "open",        "surface": "natural"},
    "HOU": {"pf_runs": 97,  "pf_hr": 94,  "city": "Houston",         "lat": 29.7573, "lon": -95.3555,  "roof": "retractable", "surface": "natural"},
    "KCR": {"pf_runs": 97,  "pf_hr": 97,  "city": "Kansas City",     "lat": 39.0517, "lon": -94.4803,  "roof": "open",        "surface": "natural"},
    "LAA": {"pf_runs": 97,  "pf_hr": 99,  "city": "Anaheim",         "lat": 33.8003, "lon": -117.8827, "roof": "open",        "surface": "natural"},
    "LAD": {"pf_runs": 95,  "pf_hr": 91,  "city": "Los Angeles",     "lat": 34.0739, "lon": -118.2400, "roof": "open",        "surface": "natural"},
    "MIA": {"pf_runs": 97,  "pf_hr": 95,  "city": "Miami",           "lat": 25.7781, "lon": -80.2197,  "roof": "retractable", "surface": "artificial"},
    "MIL": {"pf_runs": 99,  "pf_hr": 100, "city": "Milwaukee",       "lat": 43.0280, "lon": -87.9712,  "roof": "retractable", "surface": "natural"},
    "MIN": {"pf_runs": 101, "pf_hr": 107, "city": "Minneapolis",     "lat": 44.9817, "lon": -93.2776,  "roof": "open",        "surface": "natural"},
    "NYM": {"pf_runs": 97,  "pf_hr": 97,  "city": "New York",        "lat": 40.7571, "lon": -73.8458,  "roof": "open",        "surface": "natural"},
    "NYY": {"pf_runs": 105, "pf_hr": 114, "city": "New York",        "lat": 40.8296, "lon": -73.9262,  "roof": "open",        "surface": "natural"},
    "OAK": {"pf_runs": 96,  "pf_hr": 96,  "city": "Oakland",         "lat": 37.7516, "lon": -122.2005, "roof": "open",        "surface": "natural"},
    "PHI": {"pf_runs": 102, "pf_hr": 106, "city": "Philadelphia",    "lat": 39.9061, "lon": -75.1665,  "roof": "open",        "surface": "natural"},
    "PIT": {"pf_runs": 97,  "pf_hr": 97,  "city": "Pittsburgh",      "lat": 40.4469, "lon": -80.0057,  "roof": "open",        "surface": "natural"},
    "SDP": {"pf_runs": 96,  "pf_hr": 91,  "city": "San Diego",       "lat": 32.7073, "lon": -117.1566, "roof": "open",        "surface": "natural"},
    "SEA": {"pf_runs": 94,  "pf_hr": 89,  "city": "Seattle",         "lat": 47.5914, "lon": -122.3325, "roof": "retractable", "surface": "natural"},
    "SFG": {"pf_runs": 93,  "pf_hr": 86,  "city": "San Francisco",   "lat": 37.7786, "lon": -122.3893, "roof": "open",        "surface": "natural"},
    "STL": {"pf_runs": 99,  "pf_hr": 101, "city": "St. Louis",       "lat": 38.6226, "lon": -90.1928,  "roof": "open",        "surface": "natural"},
    "TBR": {"pf_runs": 97,  "pf_hr": 98,  "city": "St. Petersburg",  "lat": 27.7683, "lon": -82.6534,  "roof": "fixed",       "surface": "artificial"},
    "TEX": {"pf_runs": 101, "pf_hr": 104, "city": "Arlington",       "lat": 32.7473, "lon": -97.0831,  "roof": "retractable", "surface": "natural"},
    "TOR": {"pf_runs": 102, "pf_hr": 109, "city": "Toronto",         "lat": 43.6414, "lon": -79.3894,  "roof": "retractable", "surface": "artificial"},
    "WSN": {"pf_runs": 100, "pf_hr": 104, "city": "Washington",      "lat": 38.8730, "lon": -77.0074,  "roof": "open",        "surface": "natural"},
    # fmt: on
}

# Manager hook tendencies — how often each manager allows SP to face lineup 3rd time.
# Scale: 1.0 = very aggressive hook (pulls SP early), 0.0 = lets SP go deep.
# Source: FanGraphs "Manager Hook" data, updated from 2024 season.
MANAGER_HOOK = {
    "ARI": {"manager": "Torey Lovullo",     "hook_rate": 0.45},
    "ATL": {"manager": "Brian Snitker",     "hook_rate": 0.35},
    "BAL": {"manager": "Brandon Hyde",      "hook_rate": 0.50},
    "BOS": {"manager": "Alex Cora",         "hook_rate": 0.48},
    "CHC": {"manager": "Craig Counsell",    "hook_rate": 0.55},
    "CWS": {"manager": "Pedro Grifol",      "hook_rate": 0.40},
    "CIN": {"manager": "David Bell",        "hook_rate": 0.45},
    "CLE": {"manager": "Stephen Vogt",      "hook_rate": 0.42},
    "COL": {"manager": "Bud Black",         "hook_rate": 0.38},
    "DET": {"manager": "A.J. Hinch",        "hook_rate": 0.52},
    "HOU": {"manager": "Joe Espada",        "hook_rate": 0.58},
    "KCR": {"manager": "Matt Quatraro",     "hook_rate": 0.47},
    "LAA": {"manager": "Ron Washington",    "hook_rate": 0.40},
    "LAD": {"manager": "Dave Roberts",      "hook_rate": 0.55},
    "MIA": {"manager": "Skip Schumaker",    "hook_rate": 0.50},
    "MIL": {"manager": "Pat Murphy",        "hook_rate": 0.45},
    "MIN": {"manager": "Rocco Baldelli",    "hook_rate": 0.60},
    "NYM": {"manager": "Carlos Mendoza",    "hook_rate": 0.50},
    "NYY": {"manager": "Aaron Boone",       "hook_rate": 0.52},
    "OAK": {"manager": "Mark Kotsay",       "hook_rate": 0.35},
    "PHI": {"manager": "Rob Thomson",       "hook_rate": 0.45},
    "PIT": {"manager": "Derek Shelton",     "hook_rate": 0.42},
    "SDP": {"manager": "Mike Shildt",       "hook_rate": 0.48},
    "SEA": {"manager": "Scott Servais",     "hook_rate": 0.55},
    "SFG": {"manager": "Bob Melvin",        "hook_rate": 0.48},
    "STL": {"manager": "Oliver Marmol",     "hook_rate": 0.50},
    "TBR": {"manager": "Kevin Cash",        "hook_rate": 0.72},  # Very aggressive
    "TEX": {"manager": "Bruce Bochy",       "hook_rate": 0.38},
    "TOR": {"manager": "John Schneider",    "hook_rate": 0.50},
    "WSN": {"manager": "Dave Martinez",     "hook_rate": 0.45},
}


# =============================================================================
# FUNCTION 1: Pull Game Schedules with Run Totals (Training Labels)
# =============================================================================
def pull_game_totals(years: list, teams: list) -> pd.DataFrame:
    """
    Pull game-by-game results to get historical total runs scored.

    For each game we capture: R (home runs) + RA (away runs) = total.
    This is the TARGET VARIABLE (y) in our Poisson regression model.

    Returns
    -------
    pd.DataFrame
        One row per game (home team perspective) with total runs.
    """
    all_games = []

    for year in years:
        print(f"  Pulling game results for {year}...")
        for team in teams:
            try:
                df = pyb.schedule_and_record(year, team)
                df["Season"] = year
                df["Team"]   = team
                all_games.append(df)
                time.sleep(0.5)
            except Exception as e:
                print(f"    WARNING: {team} {year} failed: {e}")

    games = pd.concat(all_games, ignore_index=True)

    # Keep only completed games (R column has actual runs, not NaN)
    games = games[games["R"].notna()].copy()

    # Convert runs columns to numeric (they can come through as strings)
    # In R: as.numeric() — same concept
    games["R"]  = pd.to_numeric(games["R"],  errors="coerce")
    games["RA"] = pd.to_numeric(games["RA"], errors="coerce")

    print(f"  ✓ Game totals: {len(games):,} team-game rows pulled.")
    return games


# =============================================================================
# FUNCTION 2: Pull Team Offensive and Pitching Stats
# =============================================================================
def pull_team_offense_defense(years: list) -> tuple:
    """
    Pull team-level offensive (wRC+, wOBA) and pitching (ERA-, FIP-) stats.

    These form the primary predictive features for total run scoring.
    Stronger offense + weaker pitching = higher expected total.
    """
    all_bat, all_pit = [], []

    for year in years:
        print(f"  Pulling team offense/defense for {year}...")
        tb = pyb.team_batting(year, year)
        tb["Season"] = year
        all_bat.append(tb)

        tp = pyb.team_pitching(year, year)
        tp["Season"] = year
        all_pit.append(tp)
        time.sleep(1)

    team_bat = pd.concat(all_bat, ignore_index=True)
    team_pit = pd.concat(all_pit, ignore_index=True)

    print(f"  ✓ Team offense:  {len(team_bat):,} team-seasons")
    print(f"  ✓ Team pitching: {len(team_pit):,} team-seasons")
    return team_bat, team_pit


# =============================================================================
# FUNCTION 3: Fetch Current Weather for an Upcoming Game (Free NWS API)
# =============================================================================
def fetch_weather_nws(lat: float, lon: float) -> dict:
    """
    Fetch current/forecast weather from the National Weather Service (NWS) API.

    NWS API is completely FREE — no API key required.
    URL: https://api.weather.gov

    This is used for SCORING upcoming games, not for training.
    For historical training data, we use static average conditions
    (handled in the build file with seasonal temperature estimates).

    Parameters
    ----------
    lat : float   Stadium latitude
    lon : float   Stadium longitude

    Returns
    -------
    dict with keys: temperature_f, wind_speed_mph, wind_direction,
                    humidity_pct, weather_condition
    """
    headers = {
        "User-Agent": "baseball-models/1.0 (contact@example.com)"
        # NWS requires a User-Agent string identifying your application
    }

    try:
        # Step 1: Get the forecast grid endpoint for the lat/lon
        # NWS API docs: https://www.weather.gov/documentation/services-web-api
        point_url = f"https://api.weather.gov/points/{lat},{lon}"
        response  = requests.get(point_url, headers=headers, timeout=10)
        response.raise_for_status()  # Raises exception if HTTP error (like stop() in R)

        point_data    = response.json()
        forecast_url  = point_data["properties"]["forecastHourly"]

        # Step 2: Get hourly forecast
        forecast_resp = requests.get(forecast_url, headers=headers, timeout=10)
        forecast_resp.raise_for_status()
        forecast_data = forecast_resp.json()

        # Get the first forecast period (current or next hour)
        periods = forecast_data["properties"]["periods"]
        current = periods[0]

        # Parse wind speed (NWS returns "12 mph" as a string)
        # In R: as.numeric(gsub(" mph", "", wind_speed))
        wind_speed_str = current.get("windSpeed", "0 mph")
        wind_speed_mph = float(wind_speed_str.replace(" mph", "").split(" to ")[0])

        # Map cardinal directions to degrees and to "in/out/cross" classification
        wind_dir = current.get("windDirection", "CALM")

        return {
            "temperature_f":    current.get("temperature", 72),
            "wind_speed_mph":   wind_speed_mph,
            "wind_direction":   wind_dir,
            "weather_condition": current.get("shortForecast", "Unknown"),
            "humidity_pct":     70,  # NWS hourly doesn't always include humidity
        }

    except Exception as e:
        print(f"    WARNING: Weather fetch failed for ({lat}, {lon}): {e}")
        # Return neutral weather defaults on failure
        return {
            "temperature_f":   72,
            "wind_speed_mph":   5,
            "wind_direction":  "CALM",
            "weather_condition": "Unknown",
            "humidity_pct":    50,
        }


def classify_wind(wind_dir: str, stadium_orientation: str = "NE") -> str:
    """
    Classify wind direction relative to the stadium as 'out', 'in', or 'cross'.

    Wind 'out' (toward outfield) increases scoring.
    Wind 'in' (toward home plate) suppresses scoring.

    Most stadiums face roughly northeast (home plate in southwest corner).
    This is an approximation — exact orientation would require stadium-specific data.

    Parameters
    ----------
    wind_dir : str   Cardinal direction (e.g., 'SW', 'NW', 'N', 'S')

    Returns
    -------
    str : 'out', 'in', or 'cross'
    """
    # Wind blowing FROM the SW goes TO the NE = blowing "out" for NE-facing stadiums
    out_winds   = {"SW", "S", "SSW", "WSW"}
    in_winds    = {"NE", "N", "NNE", "ENE"}

    if wind_dir in out_winds:
        return "out"
    elif wind_dir in in_winds:
        return "in"
    else:
        return "cross"


# =============================================================================
# FUNCTION 4: Pull FanGraphs SP Stats (for daily game scoring)
# =============================================================================
def pull_sp_stats(years: list) -> pd.DataFrame:
    """
    Pull individual starting pitcher stats (SIERA, xFIP) from FanGraphs.

    When scoring upcoming games, we need the specific starting pitcher's
    metrics — not just team-average pitching. This is a key differentiator
    from simple Elo-based models.
    """
    all_sp = []

    for year in years:
        print(f"  Pulling SP stats for {year}...")
        # qual=50 = minimum 50 IP (ensures meaningful sample for starters)
        df = pyb.pitching_stats(year, year, qual=50, ind=1)
        df["Season"] = year
        # Filter to starters (GS = Games Started > 5)
        if "GS" in df.columns:
            df = df[df["GS"] > 5].copy()
        all_sp.append(df)
        time.sleep(2)

    sp = pd.concat(all_sp, ignore_index=True)
    print(f"  ✓ SP stats: {len(sp):,} starter-seasons pulled.")
    return sp


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("TOTALS MODEL — STEP 1: DATA INPUT")
    print("=" * 70)

    # Pull training data
    print("\n[ 1/4 ] Pulling game totals (training labels)...")
    games_df = pull_game_totals(TRAIN_YEARS, ALL_TEAMS_BREF)

    print("\n[ 2/4 ] Pulling team offense/defense stats...")
    team_bat_df, team_pit_df = pull_team_offense_defense(TRAIN_YEARS)

    print("\n[ 3/4 ] Pulling starting pitcher stats...")
    sp_df = pull_sp_stats(TRAIN_YEARS)

    print("\n[ 4/4 ] Saving park factors and manager hook data...")
    # Convert dictionaries to DataFrames for CSV export
    # In R: as.data.frame(do.call(rbind, park_factors))
    park_df    = pd.DataFrame(PARK_FACTORS).T.reset_index()
    park_df.columns = ["team"] + list(park_df.columns[1:])

    manager_df = pd.DataFrame(MANAGER_HOOK).T.reset_index()
    manager_df.columns = ["team"] + list(manager_df.columns[1:])

    # --- Save all raw files --------------------------------------------------
    print("\n[ SAVING ] Writing raw data files...")

    games_df.to_csv(os.path.join(RAW_DIR, "raw_game_schedules.csv"), index=False)
    print(f"  ✓ raw_game_schedules.csv    ({len(games_df):,} rows)")

    team_bat_df.to_csv(os.path.join(RAW_DIR, "raw_team_batting.csv"), index=False)
    print(f"  ✓ raw_team_batting.csv      ({len(team_bat_df):,} rows)")

    team_pit_df.to_csv(os.path.join(RAW_DIR, "raw_team_pitching.csv"), index=False)
    print(f"  ✓ raw_team_pitching.csv     ({len(team_pit_df):,} rows)")

    sp_df.to_csv(os.path.join(RAW_DIR, "raw_sp_stats.csv"), index=False)
    print(f"  ✓ raw_sp_stats.csv          ({len(sp_df):,} rows)")

    park_df.to_csv(os.path.join(RAW_DIR, "raw_park_factors.csv"), index=False)
    print(f"  ✓ raw_park_factors.csv      ({len(park_df):,} rows)")

    manager_df.to_csv(os.path.join(RAW_DIR, "raw_manager_hook.csv"), index=False)
    print(f"  ✓ raw_manager_hook.csv      ({len(manager_df):,} rows)")

    # Save park factors as JSON too (useful for quick lookups in scoring)
    with open(os.path.join(RAW_DIR, "park_factors.json"), "w") as f:
        json.dump(PARK_FACTORS, f, indent=2)
    print(f"  ✓ park_factors.json")

    # -------------------------------------------------------------------------
    # REFRESH CURRENT SEASON DATA (re-run weekly to pick up 2026 stats)
    # -------------------------------------------------------------------------
    if CURRENT_YEAR not in TRAIN_YEARS:
        print(f"\n[ REFRESH ] Pulling {CURRENT_YEAR} current-season data...")

        try:
            cur_team_bat, cur_team_pit = pull_team_offense_defense([CURRENT_YEAR])
            if not cur_team_bat.empty:
                team_bat_df = pd.concat(
                    [team_bat_df[team_bat_df["Season"] != CURRENT_YEAR], cur_team_bat],
                    ignore_index=True,
                )
                team_bat_df.to_csv(os.path.join(RAW_DIR, "raw_team_batting.csv"), index=False)
                print(f"  ✓ Merged {CURRENT_YEAR} team batting into raw_team_batting.csv")
            if not cur_team_pit.empty:
                team_pit_df = pd.concat(
                    [team_pit_df[team_pit_df["Season"] != CURRENT_YEAR], cur_team_pit],
                    ignore_index=True,
                )
                team_pit_df.to_csv(os.path.join(RAW_DIR, "raw_team_pitching.csv"), index=False)
                print(f"  ✓ Merged {CURRENT_YEAR} team pitching into raw_team_pitching.csv")
        except Exception as e:
            print(f"  WARNING: Could not pull {CURRENT_YEAR} team offense/defense — {e}")

        try:
            cur_sp = pull_sp_stats([CURRENT_YEAR])
            if not cur_sp.empty:
                sp_df = pd.concat(
                    [sp_df[sp_df["Season"] != CURRENT_YEAR], cur_sp],
                    ignore_index=True,
                )
                sp_df.to_csv(os.path.join(RAW_DIR, "raw_sp_stats.csv"), index=False)
                print(f"  ✓ Merged {CURRENT_YEAR} SP stats into raw_sp_stats.csv")
        except Exception as e:
            print(f"  WARNING: Could not pull {CURRENT_YEAR} SP stats — {e}")

    print(f"\n  TIP: Re-run this file weekly during the {CURRENT_YEAR} season to refresh live stats.")

    print("\n" + "=" * 70)
    print("STEP 1 COMPLETE — Run 02_build_totals.py next.")
    print("=" * 70)
