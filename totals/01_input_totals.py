"""
=============================================================================
OVER/UNDER TOTALS MODEL — FILE 1 OF 4: DATA INPUT  (REFACTORED)
=============================================================================
Statistical framework: Negative Binomial regression (two-equation system)
  Runs are discrete, overdispersed count data. The Poisson assumption
  (Var = Mean) is violated for MLB runs — empirically Var ≈ 1.15 × Mean.
  Negative Binomial NB2 adds a dispersion parameter α that accommodates
  this: Var[Y] = μ + α·μ².

  We model HOME runs and AWAY runs with separate NB equations, then
  convolve the distributions via Monte Carlo to produce O/U probabilities.
  This captures asymmetric park/weather effects better than modeling total
  runs as a single outcome.

Weather physics:
  Air density governs ball carry. We pull REAL historical conditions from
  Open-Meteo (free, no key) rather than using seasonal averages.
  Dynamic park factor = f(base_pf, temp, humidity, wind×orientation).

Data sources:
  - B-Ref game schedules (run totals, training labels)
  - FanGraphs pitching/batting (SP quality, team offense)
  - Open-Meteo archive API (historical hourly weather per stadium)
  - Open-Meteo forecast API (live weather for scoring day-of)
  - Static: park factors, stadium CF bearings, altitude, manager hooks

Run once per season (weekly in-season for fresh stats):
    python3 totals/01_input_totals.py
=============================================================================
"""

import os
import time
import json
import requests
import pandas as pd
import numpy as np
import pybaseball as pyb
from datetime import date as _date

pyb.cache.enable()

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR      = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

TRAIN_YEARS  = [2023, 2024, 2025]
CURRENT_YEAR = _date.today().year

ALL_TEAMS_BREF = [
    "ARI", "ATL", "BAL", "BOS", "CHC", "CWS", "CIN", "CLE",
    "COL", "DET", "HOU", "KCR", "LAA", "LAD", "MIA", "MIL",
    "MIN", "NYM", "NYY", "OAK", "PHI", "PIT", "SDP", "SEA",
    "SFG", "STL", "TBR", "TEX", "TOR", "WSN",
]


# =============================================================================
# STADIUM METADATA
# All physical + environmental properties needed for the weather physics model.
# =============================================================================

# Park run/HR factors (FanGraphs 2024; update annually)
# pf_runs : run-based park factor (100 = league average)
# pf_hr   : HR-specific park factor
PARK_FACTORS = {
    # fmt: off
    "ARI": {"pf_runs": 97,  "pf_hr": 101, "roof": "retractable", "surface": "artificial"},
    "ATL": {"pf_runs": 102, "pf_hr": 104, "roof": "open",        "surface": "natural"},
    "BAL": {"pf_runs": 104, "pf_hr": 110, "roof": "open",        "surface": "natural"},
    "BOS": {"pf_runs": 105, "pf_hr": 95,  "roof": "open",        "surface": "natural"},
    "CHC": {"pf_runs": 101, "pf_hr": 104, "roof": "open",        "surface": "natural"},
    "CWS": {"pf_runs": 96,  "pf_hr": 99,  "roof": "open",        "surface": "natural"},
    "CIN": {"pf_runs": 103, "pf_hr": 106, "roof": "open",        "surface": "natural"},
    "CLE": {"pf_runs": 98,  "pf_hr": 96,  "roof": "open",        "surface": "natural"},
    "COL": {"pf_runs": 116, "pf_hr": 116, "roof": "open",        "surface": "natural"},
    "DET": {"pf_runs": 96,  "pf_hr": 93,  "roof": "open",        "surface": "natural"},
    "HOU": {"pf_runs": 97,  "pf_hr": 94,  "roof": "retractable", "surface": "natural"},
    "KCR": {"pf_runs": 97,  "pf_hr": 97,  "roof": "open",        "surface": "natural"},
    "LAA": {"pf_runs": 97,  "pf_hr": 99,  "roof": "open",        "surface": "natural"},
    "LAD": {"pf_runs": 95,  "pf_hr": 91,  "roof": "open",        "surface": "natural"},
    "MIA": {"pf_runs": 97,  "pf_hr": 95,  "roof": "retractable", "surface": "artificial"},
    "MIL": {"pf_runs": 99,  "pf_hr": 100, "roof": "retractable", "surface": "natural"},
    "MIN": {"pf_runs": 101, "pf_hr": 107, "roof": "open",        "surface": "natural"},
    "NYM": {"pf_runs": 97,  "pf_hr": 97,  "roof": "open",        "surface": "natural"},
    "NYY": {"pf_runs": 105, "pf_hr": 114, "roof": "open",        "surface": "natural"},
    "OAK": {"pf_runs": 96,  "pf_hr": 96,  "roof": "open",        "surface": "natural"},
    "PHI": {"pf_runs": 102, "pf_hr": 106, "roof": "open",        "surface": "natural"},
    "PIT": {"pf_runs": 97,  "pf_hr": 97,  "roof": "open",        "surface": "natural"},
    "SDP": {"pf_runs": 96,  "pf_hr": 91,  "roof": "open",        "surface": "natural"},
    "SEA": {"pf_runs": 94,  "pf_hr": 89,  "roof": "retractable", "surface": "natural"},
    "SFG": {"pf_runs": 93,  "pf_hr": 86,  "roof": "open",        "surface": "natural"},
    "STL": {"pf_runs": 99,  "pf_hr": 101, "roof": "open",        "surface": "natural"},
    "TBR": {"pf_runs": 97,  "pf_hr": 98,  "roof": "fixed",       "surface": "artificial"},
    "TEX": {"pf_runs": 101, "pf_hr": 104, "roof": "retractable", "surface": "natural"},
    "TOR": {"pf_runs": 102, "pf_hr": 109, "roof": "retractable", "surface": "artificial"},
    "WSN": {"pf_runs": 100, "pf_hr": 104, "roof": "open",        "surface": "natural"},
    # fmt: on
}

# Stadium coordinates (lat/lon) and altitude for weather API + physics model
# Altitude in feet above sea level — higher altitude = thinner air = more carry
STADIUM_GEO = {
    # fmt: off
    "ARI": {"lat": 33.4453, "lon": -112.0667, "altitude_ft": 1082},
    "ATL": {"lat": 33.8908, "lon":  -84.4678, "altitude_ft": 1050},
    "BAL": {"lat": 39.2839, "lon":  -76.6218, "altitude_ft":   25},
    "BOS": {"lat": 42.3467, "lon":  -71.0972, "altitude_ft":   21},
    "CHC": {"lat": 41.9484, "lon":  -87.6553, "altitude_ft":  595},
    "CWS": {"lat": 41.8300, "lon":  -87.6339, "altitude_ft":  595},
    "CIN": {"lat": 39.0975, "lon":  -84.5061, "altitude_ft":  490},
    "CLE": {"lat": 41.4962, "lon":  -81.6852, "altitude_ft":  660},
    "COL": {"lat": 39.7559, "lon": -104.9942, "altitude_ft": 5280},
    "DET": {"lat": 42.3390, "lon":  -83.0485, "altitude_ft":  601},
    "HOU": {"lat": 29.7573, "lon":  -95.3555, "altitude_ft":   43},
    "KCR": {"lat": 39.0517, "lon":  -94.4803, "altitude_ft":  908},
    "LAA": {"lat": 33.8003, "lon": -117.8827, "altitude_ft":  160},
    "LAD": {"lat": 34.0739, "lon": -118.2400, "altitude_ft":  512},
    "MIA": {"lat": 25.7781, "lon":  -80.2197, "altitude_ft":    6},
    "MIL": {"lat": 43.0280, "lon":  -87.9712, "altitude_ft":  635},
    "MIN": {"lat": 44.9817, "lon":  -93.2776, "altitude_ft":  830},
    "NYM": {"lat": 40.7571, "lon":  -73.8458, "altitude_ft":   16},
    "NYY": {"lat": 40.8296, "lon":  -73.9262, "altitude_ft":   55},
    "OAK": {"lat": 37.7516, "lon": -122.2005, "altitude_ft":   25},
    "PHI": {"lat": 39.9061, "lon":  -75.1665, "altitude_ft":   40},
    "PIT": {"lat": 40.4469, "lon":  -80.0057, "altitude_ft":  730},
    "SDP": {"lat": 32.7073, "lon": -117.1566, "altitude_ft":   17},
    "SEA": {"lat": 47.5914, "lon": -122.3325, "altitude_ft":   11},
    "SFG": {"lat": 37.7786, "lon": -122.3893, "altitude_ft":   10},
    "STL": {"lat": 38.6226, "lon":  -90.1928, "altitude_ft":  466},
    "TBR": {"lat": 27.7683, "lon":  -82.6534, "altitude_ft":   28},
    "TEX": {"lat": 32.7473, "lon":  -97.0831, "altitude_ft":  551},
    "TOR": {"lat": 43.6414, "lon":  -79.3894, "altitude_ft":   76},
    "WSN": {"lat": 38.8730, "lon":  -77.0074, "altitude_ft":   25},
    # fmt: on
}

# Compass bearing from home plate TOWARD center field (degrees from true North).
# This is the "outfield axis" used to decompose wind into in/out/cross components.
# Wind blowing FROM (cf_bearing + 180°) is blowing "in" from CF → suppresses HRs.
# Wind blowing FROM cf_bearing is blowing "out" toward CF → boosts HRs.
# Source: Google Maps satellite imagery of each stadium (approximate ±15°).
STADIUM_CF_BEARING = {
    "ARI":   0,   # Chase Field: CF faces roughly N
    "ATL":  25,   # Truist Park: CF faces NNE
    "BAL": 320,   # Camden Yards: CF faces NW
    "BOS":  55,   # Fenway Park: CF faces NE
    "CHC":  50,   # Wrigley Field: CF faces NE
    "CWS":  10,   # Guaranteed Rate Field: CF faces N
    "CIN": 340,   # Great American Ball Park: CF faces NNW
    "CLE": 355,   # Progressive Field: CF faces N
    "COL": 340,   # Coors Field: CF faces NNW
    "DET":  20,   # Comerica Park: CF faces NNE
    "HOU":  30,   # Minute Maid Park: CF faces NNE
    "KCR": 340,   # Kauffman Stadium: CF faces NNW
    "LAA": 340,   # Angel Stadium: CF faces NNW
    "LAD":  30,   # Dodger Stadium: CF faces NNE
    "MIA":   0,   # loanDepot park: CF faces N (retractable — limited impact)
    "MIL": 330,   # American Family Field: CF faces NNW
    "MIN":  20,   # Target Field: CF faces NNE
    "NYM":  30,   # Citi Field: CF faces NNE
    "NYY":  10,   # Yankee Stadium: CF faces N
    "OAK": 350,   # Oakland Coliseum: CF faces N
    "PHI": 320,   # Citizens Bank Park: CF faces NW
    "PIT":  10,   # PNC Park: CF faces N
    "SDP": 340,   # Petco Park: CF faces NNW
    "SEA": 330,   # T-Mobile Park: CF faces NNW
    "SFG":  60,   # Oracle Park: CF faces ENE
    "STL": 350,   # Busch Stadium: CF faces N
    "TBR":   0,   # Tropicana Field: fixed dome — CF bearing unused
    "TEX": 340,   # Globe Life Field: CF faces NNW (retractable)
    "TOR":   0,   # Rogers Centre: retractable/turf — bearing mostly unused
    "WSN": 340,   # Nationals Park: CF faces NNW
}

# Manager hook tendencies (how aggressively manager pulls SP)
MANAGER_HOOK = {
    "ARI": {"manager": "Torey Lovullo",   "hook_rate": 0.45},
    "ATL": {"manager": "Brian Snitker",   "hook_rate": 0.35},
    "BAL": {"manager": "Brandon Hyde",    "hook_rate": 0.50},
    "BOS": {"manager": "Alex Cora",       "hook_rate": 0.48},
    "CHC": {"manager": "Craig Counsell",  "hook_rate": 0.55},
    "CWS": {"manager": "Pedro Grifol",    "hook_rate": 0.40},
    "CIN": {"manager": "David Bell",      "hook_rate": 0.45},
    "CLE": {"manager": "Stephen Vogt",    "hook_rate": 0.42},
    "COL": {"manager": "Bud Black",       "hook_rate": 0.38},
    "DET": {"manager": "A.J. Hinch",      "hook_rate": 0.52},
    "HOU": {"manager": "Joe Espada",      "hook_rate": 0.58},
    "KCR": {"manager": "Matt Quatraro",   "hook_rate": 0.47},
    "LAA": {"manager": "Ron Washington",  "hook_rate": 0.40},
    "LAD": {"manager": "Dave Roberts",    "hook_rate": 0.55},
    "MIA": {"manager": "Skip Schumaker",  "hook_rate": 0.50},
    "MIL": {"manager": "Pat Murphy",      "hook_rate": 0.45},
    "MIN": {"manager": "Rocco Baldelli",  "hook_rate": 0.60},
    "NYM": {"manager": "Carlos Mendoza",  "hook_rate": 0.50},
    "NYY": {"manager": "Aaron Boone",     "hook_rate": 0.52},
    "OAK": {"manager": "Mark Kotsay",     "hook_rate": 0.35},
    "PHI": {"manager": "Rob Thomson",     "hook_rate": 0.45},
    "PIT": {"manager": "Derek Shelton",   "hook_rate": 0.42},
    "SDP": {"manager": "Mike Shildt",     "hook_rate": 0.48},
    "SEA": {"manager": "Scott Servais",   "hook_rate": 0.55},
    "SFG": {"manager": "Bob Melvin",      "hook_rate": 0.48},
    "STL": {"manager": "Oliver Marmol",   "hook_rate": 0.50},
    "TBR": {"manager": "Kevin Cash",      "hook_rate": 0.72},
    "TEX": {"manager": "Bruce Bochy",     "hook_rate": 0.38},
    "TOR": {"manager": "John Schneider",  "hook_rate": 0.50},
    "WSN": {"manager": "Dave Martinez",   "hook_rate": 0.45},
}


# =============================================================================
# WEATHER — Open-Meteo API (free, no key required)
# =============================================================================

def pull_historical_weather(game_years: list) -> pd.DataFrame:
    """
    Pull historical hourly weather for each stadium from Open-Meteo archive.

    One API call per stadium per year covers the full season (Mar–Nov).
    We then extract conditions at ~7pm local time (proxy for first pitch).

    Open-Meteo archive endpoint (free, no key):
      https://archive.open-meteo.com/v1/archive
      Returns temperature (°F), relative humidity (%), wind speed (mph),
      wind direction (°) at hourly resolution.

    Returns
    -------
    pd.DataFrame
        Columns: team, date, temperature_f, humidity_pct,
                 wind_speed_mph, wind_direction_deg
        One row per team per game date (evening conditions).
    """
    all_weather = []
    headers = {"User-Agent": "baseball-models/2.0 (totals-weather-pull)"}

    for team in ALL_TEAMS_BREF:
        geo = STADIUM_GEO.get(team)
        if not geo:
            continue
        lat, lon = geo["lat"], geo["lon"]

        for year in game_years:
            print(f"  Pulling historical weather: {team} {year}...")
            try:
                url = "https://archive.open-meteo.com/v1/archive"
                params = {
                    "latitude":          lat,
                    "longitude":         lon,
                    "start_date":        f"{year}-03-20",
                    "end_date":          f"{year}-11-10",
                    "hourly":            "temperature_2m,relative_humidity_2m,"
                                         "wind_speed_10m,wind_direction_10m",
                    "timezone":          "auto",
                    "wind_speed_unit":   "mph",
                    "temperature_unit":  "fahrenheit",
                }
                resp = requests.get(url, params=params, headers=headers, timeout=60)
                resp.raise_for_status()
                data = resp.json()

                hourly = data.get("hourly", {})
                if not hourly:
                    continue

                df_h = pd.DataFrame({
                    "datetime":          pd.to_datetime(hourly["time"]),
                    "temperature_f":     hourly.get("temperature_2m", [np.nan] * len(hourly["time"])),
                    "humidity_pct":      hourly.get("relative_humidity_2m", [np.nan] * len(hourly["time"])),
                    "wind_speed_mph":    hourly.get("wind_speed_10m", [np.nan] * len(hourly["time"])),
                    "wind_direction_deg":hourly.get("wind_direction_10m", [np.nan] * len(hourly["time"])),
                })

                df_h["hour"] = df_h["datetime"].dt.hour
                df_h["date"] = df_h["datetime"].dt.strftime("%Y-%m-%d")

                # Game-time proxy: average conditions between 6pm–8pm local
                # Captures evening first pitches; day games will use this as fallback
                game_wx = (
                    df_h[df_h["hour"].between(18, 20)]
                    .groupby("date")
                    .agg(
                        temperature_f    =("temperature_f",     "mean"),
                        humidity_pct     =("humidity_pct",      "mean"),
                        wind_speed_mph   =("wind_speed_mph",    "mean"),
                        wind_direction_deg=("wind_direction_deg","mean"),
                    )
                    .reset_index()
                )
                game_wx["team"] = team
                game_wx["year"] = year
                all_weather.append(game_wx)
                time.sleep(0.4)   # Be polite to free API

            except Exception as e:
                print(f"    WARNING: Weather archive {team} {year}: {e}")

    if not all_weather:
        return pd.DataFrame()
    result = pd.concat(all_weather, ignore_index=True)
    print(f"  ✓ Historical weather: {len(result):,} team-date rows")
    return result


def fetch_weather_open_meteo(lat: float, lon: float,
                              game_datetime_iso: str = None) -> dict:
    """
    Fetch current or forecast weather from Open-Meteo (free, no key).

    Used for SCORING upcoming games day-of.  Returns conditions for the
    specified ISO datetime (e.g., "2026-04-15T19:00") or the next
    available hour if game_datetime_iso is None.

    Parameters
    ----------
    lat, lon           : stadium coordinates
    game_datetime_iso  : ISO format datetime string for first pitch,
                         e.g. "2026-04-15T19:00".  None = use next hour.

    Returns
    -------
    dict with keys: temperature_f, humidity_pct, wind_speed_mph,
                    wind_direction_deg, weather_description
    """
    headers = {"User-Agent": "baseball-models/2.0"}
    defaults = {
        "temperature_f":     72.0,
        "humidity_pct":      55.0,
        "wind_speed_mph":     5.0,
        "wind_direction_deg": 180.0,
        "weather_description": "Unknown (API fallback)",
    }

    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude":         lat,
            "longitude":        lon,
            "hourly":           "temperature_2m,relative_humidity_2m,"
                                 "wind_speed_10m,wind_direction_10m,precipitation_probability",
            "timezone":         "auto",
            "wind_speed_unit":  "mph",
            "temperature_unit": "fahrenheit",
            "forecast_days":    3,
        }
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        hourly = data["hourly"]
        times  = pd.to_datetime(hourly["time"])

        df_h = pd.DataFrame({
            "datetime":          times,
            "temperature_f":     hourly.get("temperature_2m", []),
            "humidity_pct":      hourly.get("relative_humidity_2m", []),
            "wind_speed_mph":    hourly.get("wind_speed_10m", []),
            "wind_direction_deg":hourly.get("wind_direction_10m", []),
            "precip_prob_pct":   hourly.get("precipitation_probability", []),
        })

        if game_datetime_iso:
            target_dt = pd.to_datetime(game_datetime_iso)
            # Find closest hour to game time
            df_h["dt_diff"] = abs((df_h["datetime"] - target_dt).dt.total_seconds())
            row = df_h.loc[df_h["dt_diff"].idxmin()]
        else:
            # Use the next available hour
            row = df_h.iloc[1]

        return {
            "temperature_f":     float(row["temperature_f"]),
            "humidity_pct":      float(row["humidity_pct"]),
            "wind_speed_mph":    float(row["wind_speed_mph"]),
            "wind_direction_deg":float(row["wind_direction_deg"]),
            "precip_prob_pct":   float(row.get("precip_prob_pct", 0)),
            "weather_description": f"Open-Meteo forecast",
        }

    except Exception as e:
        print(f"    WARNING: Open-Meteo forecast failed ({lat},{lon}): {e}")
        return defaults


# =============================================================================
# BASEBALL DATA PULLS
# =============================================================================

def pull_game_totals(years: list, teams: list) -> pd.DataFrame:
    """
    Pull game-by-game results (home_runs, away_runs, total_runs).
    TARGET variables for the NB two-equation model.
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
                print(f"    WARNING: {team} {year}: {e}")

    games = pd.concat(all_games, ignore_index=True)
    games = games[games["R"].notna()].copy()
    games["R"]  = pd.to_numeric(games["R"],  errors="coerce")
    games["RA"] = pd.to_numeric(games["RA"], errors="coerce")
    print(f"  ✓ Game totals: {len(games):,} team-game rows")
    return games


def pull_team_offense_defense(years: list) -> tuple:
    """Pull team batting (wRC+, wOBA, ISO) and pitching (SIERA, xFIP, ERA)."""
    all_bat, all_pit = [], []
    for year in years:
        print(f"  Pulling team offense/defense for {year}...")
        try:
            tb = pyb.team_batting(year, year);  tb["Season"] = year; all_bat.append(tb)
            time.sleep(1)
            tp = pyb.team_pitching(year, year); tp["Season"] = year; all_pit.append(tp)
            time.sleep(1)
        except Exception as e:
            print(f"    WARNING: team stats {year}: {e}")
    team_bat = pd.concat(all_bat, ignore_index=True) if all_bat else pd.DataFrame()
    team_pit = pd.concat(all_pit, ignore_index=True) if all_pit else pd.DataFrame()
    print(f"  ✓ Team batting: {len(team_bat):,} rows  |  pitching: {len(team_pit):,} rows")
    return team_bat, team_pit


def pull_sp_stats(years: list) -> pd.DataFrame:
    """Pull individual SP stats (SIERA, xFIP, K%, GB%) for each team's rotation."""
    all_sp = []
    for year in years:
        print(f"  Pulling SP stats for {year}...")
        try:
            df = pyb.pitching_stats(year, year, qual=40, ind=1)
            df["Season"] = year
            if "GS" in df.columns:
                df = df[df["GS"] >= 5].copy()
            all_sp.append(df)
            time.sleep(2)
        except Exception as e:
            print(f"    WARNING: SP stats {year}: {e}")
    sp = pd.concat(all_sp, ignore_index=True) if all_sp else pd.DataFrame()
    print(f"  ✓ SP stats: {len(sp):,} starter-seasons")
    return sp


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("TOTALS MODEL — STEP 1: DATA INPUT  (REFACTORED — NB + WEATHER)")
    print(f"Training years: {TRAIN_YEARS}")
    print("=" * 70)

    # ── 1. Game results (training labels) ───────────────────────────────────
    print("\n[ 1/5 ] Pulling game totals (training labels)...")
    games_df = pull_game_totals(TRAIN_YEARS, ALL_TEAMS_BREF)
    games_df.to_csv(os.path.join(RAW_DIR, "raw_game_schedules.csv"), index=False)
    print(f"  ✓ Saved raw_game_schedules.csv ({len(games_df):,} rows)")

    # ── 2. Team offense/defense ──────────────────────────────────────────────
    print("\n[ 2/5 ] Pulling team offense/defense stats...")
    team_bat_df, team_pit_df = pull_team_offense_defense(TRAIN_YEARS)
    team_bat_df.to_csv(os.path.join(RAW_DIR, "raw_team_batting.csv"), index=False)
    team_pit_df.to_csv(os.path.join(RAW_DIR, "raw_team_pitching.csv"), index=False)
    print(f"  ✓ Saved raw_team_batting.csv, raw_team_pitching.csv")

    # ── 3. SP stats ──────────────────────────────────────────────────────────
    print("\n[ 3/5 ] Pulling starting pitcher stats...")
    sp_df = pull_sp_stats(TRAIN_YEARS)
    sp_df.to_csv(os.path.join(RAW_DIR, "raw_sp_stats.csv"), index=False)
    print(f"  ✓ Saved raw_sp_stats.csv ({len(sp_df):,} rows)")

    # ── 4. Historical weather (Open-Meteo archive) ───────────────────────────
    print("\n[ 4/5 ] Pulling historical weather from Open-Meteo archive...")
    print("  (One API call per stadium per year — this takes a few minutes)")
    weather_df = pull_historical_weather(TRAIN_YEARS)
    if not weather_df.empty:
        weather_df.to_csv(os.path.join(RAW_DIR, "raw_weather_historical.csv"), index=False)
        print(f"  ✓ Saved raw_weather_historical.csv ({len(weather_df):,} team-date rows)")
    else:
        print("  WARNING: No weather data pulled — will use city averages as fallback")

    # ── 5. Static metadata (park factors, geo, CF bearings, manager hooks) ──
    print("\n[ 5/5 ] Saving static stadium metadata...")

    park_df = pd.DataFrame([
        {"team": k, **v} for k, v in PARK_FACTORS.items()
    ])
    geo_df = pd.DataFrame([
        {"team": k, **v} for k, v in STADIUM_GEO.items()
    ])
    cf_df = pd.DataFrame([
        {"team": k, "cf_bearing_deg": v} for k, v in STADIUM_CF_BEARING.items()
    ])
    mgr_df = pd.DataFrame([
        {"team": k, **v} for k, v in MANAGER_HOOK.items()
    ])

    park_df.to_csv(os.path.join(RAW_DIR, "raw_park_factors.csv"), index=False)
    geo_df.to_csv(os.path.join(RAW_DIR, "raw_stadium_geo.csv"), index=False)
    cf_df.to_csv(os.path.join(RAW_DIR, "raw_stadium_cf_bearings.csv"), index=False)
    mgr_df.to_csv(os.path.join(RAW_DIR, "raw_manager_hook.csv"), index=False)

    # Combined metadata JSON (used by 02_build and scoring scripts)
    meta = {}
    for team in ALL_TEAMS_BREF:
        meta[team] = {
            **PARK_FACTORS.get(team, {}),
            **STADIUM_GEO.get(team, {}),
            "cf_bearing_deg": STADIUM_CF_BEARING.get(team, 0),
            **MANAGER_HOOK.get(team, {}),
        }
    with open(os.path.join(RAW_DIR, "stadium_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  ✓ Saved raw_park_factors.csv, raw_stadium_geo.csv,")
    print(f"      raw_stadium_cf_bearings.csv, raw_manager_hook.csv, stadium_meta.json")

    # ── In-season refresh of current-year stats ──────────────────────────────
    if CURRENT_YEAR not in TRAIN_YEARS:
        print(f"\n[ REFRESH ] Pulling {CURRENT_YEAR} in-season data...")
        for puller, csv_name, desc in [
            (lambda: pull_team_offense_defense([CURRENT_YEAR])[0],
             "raw_team_batting.csv",  "team batting"),
            (lambda: pull_sp_stats([CURRENT_YEAR]),
             "raw_sp_stats.csv",      "SP stats"),
        ]:
            try:
                new_df = puller()
                if not new_df.empty:
                    path = os.path.join(RAW_DIR, csv_name)
                    if os.path.exists(path):
                        existing = pd.read_csv(path)
                        existing = existing[existing.get("Season", 0) != CURRENT_YEAR]
                        new_df = pd.concat([existing, new_df], ignore_index=True)
                    new_df.to_csv(path, index=False)
                    print(f"  ✓ {desc} {CURRENT_YEAR} merged into {csv_name}")
            except Exception as e:
                print(f"  WARNING: {desc} {CURRENT_YEAR}: {e}")

    print("\n" + "=" * 70)
    print("STEP 1 COMPLETE — Run 02_build_totals.py next.")
    print("=" * 70)
