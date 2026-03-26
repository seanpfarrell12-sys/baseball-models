"""
=============================================================================
MODEL PERFORMANCE TRACKER
=============================================================================
Automatically saves today's value bets and grades yesterday's picks using
the MLB Stats API (free, no auth required).

Flow (called from run_daily.py):
  1. grade_picks(yesterday)  — fetch MLB results, score pending picks
  2. save_picks(today, results_dict)  — log today's value bets
  3. print_performance_summary()  — ROI table by model

Picks are stored in:      tracking/picks.xlsx
Model changelog stored in: tracking/changelog.json

To log a model change (run once after making changes):
  from utils.tracker import log_model_change
  log_model_change("Added batting order PA multiplier to hitter TB model")
=============================================================================
"""

import re
import json
import difflib
import unicodedata
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, timedelta, datetime

BASE_DIR       = Path(__file__).parent.parent
TRACKING_DIR   = BASE_DIR / "tracking"
PICKS_FILE     = TRACKING_DIR / "picks.xlsx"
CHANGELOG_FILE = TRACKING_DIR / "changelog.json"

TRACKING_DIR.mkdir(exist_ok=True)

MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"

# Same abbreviation map as probable_starters.py
MLB_TO_STANDARD = {
    "WSH": "WSN",
    "ATH": "OAK",
    "AZ":  "ARI",
    "TB":  "TBR",
    "KC":  "KCR",
    "SD":  "SDP",
    "SF":  "SFG",
}

PICKS_COLS = [
    "pick_date", "model", "game", "subject",
    "bet_type", "line", "odds", "model_prob", "edge",
    "result", "actual", "pnl", "pnl_50", "notes",
]


# =============================================================================
# HELPERS
# =============================================================================

def _normalize_name(name: str) -> str:
    """Lowercase, strip accents, remove punctuation for fuzzy matching."""
    name = unicodedata.normalize("NFD", str(name))
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z ]", "", name.lower()).strip()


def _best_name_match(target: str, candidates: list, cutoff: float = 0.75):
    t = _normalize_name(target)
    norm_map = {_normalize_name(c): c for c in candidates}
    matches = difflib.get_close_matches(t, norm_map.keys(), n=1, cutoff=cutoff)
    return norm_map[matches[0]] if matches else None


def _american_pnl(odds: float, stake: float = 100.0) -> float:
    """Profit on a winning bet at given American odds for a $100 stake."""
    try:
        odds = float(odds)
    except (TypeError, ValueError):
        return 90.91  # default -110
    if odds >= 100:
        return round(stake * odds / 100, 2)
    else:
        return round(stake * 100 / abs(odds), 2)


def _parse_outs(ip_str) -> int:
    """Convert innings-pitched string ('6.1') to total outs (19)."""
    try:
        parts = str(ip_str).split(".")
        innings = int(parts[0])
        extra   = int(parts[1]) if len(parts) > 1 else 0
        return innings * 3 + extra
    except Exception:
        return 0


def _load_picks() -> pd.DataFrame:
    if PICKS_FILE.exists():
        df = pd.read_excel(PICKS_FILE, engine="openpyxl")
        # Normalize pick_date — Excel may read it back as a datetime object.
        # Always store and compare as YYYYMMDD string.
        if "pick_date" in df.columns:
            df["pick_date"] = pd.to_datetime(df["pick_date"], errors="coerce") \
                                .dt.strftime("%Y%m%d") \
                                .fillna(df["pick_date"].astype(str))
        return df
    return pd.DataFrame(columns=PICKS_COLS)


MODEL_SHEETS = [
    ("Moneyline",    "Moneyline"),
    ("Totals",       "Totals"),
    ("Hitter TB",    "Hitter TB"),
    ("Pitcher Outs", "Pitcher Outs"),
    ("NRFI/YRFI",    "NRFI-YRFI"),
]

def _save_picks_df(df: pd.DataFrame):
    graded = (
        df[df["result"].isin(["WIN", "LOSS", "PUSH"])]
        .sort_values(["pick_date", "model"])
        .reset_index(drop=True)
    )
    with pd.ExcelWriter(PICKS_FILE, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="All Picks", index=False)
        graded.to_excel(writer, sheet_name="Season Results", index=False)
        for model_name, sheet_name in MODEL_SHEETS:
            subset = (
                df[df["model"] == model_name]
                .sort_values("pick_date")
                .reset_index(drop=True)
            )
            subset.to_excel(writer, sheet_name=sheet_name, index=False)


# =============================================================================
# CHANGELOG
# =============================================================================

def log_model_change(description: str, change_date: str = None):
    """
    Record a model change so it appears in the notes column for picks saved
    on or after this date.

    Parameters
    ----------
    description : str  — what changed, e.g. "Added PA multiplier to hitter TB"
    change_date : str  — 'YYYYMMDD', defaults to today

    Example
    -------
    from utils.tracker import log_model_change
    log_model_change("Added batting order position PA multiplier to hitter TB model")
    """
    if change_date is None:
        change_date = date.today().strftime("%Y%m%d")

    changelog = _load_changelog()
    # Append to existing entry for this date if one already exists
    if change_date in changelog and changelog[change_date]:
        changelog[change_date] = changelog[change_date] + "; " + description
    else:
        changelog[change_date] = description

    CHANGELOG_FILE.write_text(json.dumps(changelog, indent=2, sort_keys=True))
    print(f"  (tracker) Changelog updated for {change_date}: {description}")


def _load_changelog() -> dict:
    """Return {YYYYMMDD: note_str} dict."""
    if CHANGELOG_FILE.exists():
        return json.loads(CHANGELOG_FILE.read_text())
    return {}


def _get_note_for_date(pick_date: str) -> str:
    """
    Return the most recent changelog entry on or before pick_date.
    Empty string if none exists.
    """
    changelog = _load_changelog()
    if not changelog:
        return ""
    # All dates on or before pick_date, sorted descending — take the most recent
    candidates = sorted(
        (d for d in changelog if d <= pick_date),
        reverse=True,
    )
    if not candidates:
        return ""
    return changelog[candidates[0]]


# =============================================================================
# SAVE PICKS
# =============================================================================

def save_picks(pick_date: str, results: dict):
    """
    Extract value bets from each model's edge report and append to picks.csv.
    Existing rows for pick_date are replaced (safe to re-run).

    Parameters
    ----------
    pick_date : str   — 'YYYYMMDD'
    results   : dict  — {model_name: edge_report_df | None}
    """
    MODEL_EXTRACTORS = {
        "Moneyline":    _extract_moneyline,
        "Totals O/U":   _extract_totals,
        "Hitter TB":    _extract_hitter_tb,
        "Pitcher Outs": _extract_pitcher_outs,
        "NRFI/YRFI":    _extract_nrfi,
    }

    new_rows = []
    for model_name, report in results.items():
        if report is None or report.empty:
            continue
        extractor = MODEL_EXTRACTORS.get(model_name)
        if extractor is None:
            continue
        value_bets = report[report["is_value_bet"] == 1] if "is_value_bet" in report.columns else report
        for _, row in value_bets.iterrows():
            pick = extractor(row)
            pick["pick_date"] = pick_date
            pick["result"]    = "PENDING"
            pick["actual"]    = None
            pick["pnl"]       = None
            pick["pnl_50"]    = None
            pick["notes"]     = _get_note_for_date(pick_date)
            new_rows.append(pick)

    if not new_rows:
        print("  (tracker) No value bets to save.")
        return

    new_df = pd.DataFrame(new_rows)[PICKS_COLS]

    existing = _load_picks()
    combined = pd.concat([existing, new_df], ignore_index=True)

    # Deduplicate: keep the first occurrence of each unique pick so earlier
    # runs are preserved; only truly new picks from later runs are appended.
    dedup_cols = ["pick_date", "model", "game", "subject", "bet_type", "line"]
    combined = combined.drop_duplicates(subset=dedup_cols, keep="first")

    _save_picks_df(combined)
    print(f"  (tracker) Saved {len(new_rows)} value bets for {pick_date}")


def _extract_moneyline(r) -> dict:
    side  = str(r.get("bet_side", ""))
    subj  = r.get("home_team", "") if side == "HOME" else r.get("away_team", "")
    return {
        "model":      "Moneyline",
        "game":       f"{r.get('away_team','')} @ {r.get('home_team','')}",
        "subject":    subj,
        "bet_type":   side,
        "line":       None,
        "odds":       r.get("american_odds", -110),
        "model_prob": r.get("model_prob"),
        "edge":       r.get("edge"),
    }


def _extract_totals(r) -> dict:
    return {
        "model":      "Totals",
        "game":       f"{r.get('away_team','')} @ {r.get('home_team','')}",
        "subject":    f"{r.get('away_team','')} @ {r.get('home_team','')}",
        "bet_type":   r.get("bet_type", ""),
        "line":       r.get("ou_line"),
        "odds":       r.get("juice", -110),
        "model_prob": r.get("model_prob"),
        "edge":       r.get("edge"),
    }


def _extract_hitter_tb(r) -> dict:
    return {
        "model":      "Hitter TB",
        "game":       f"{r.get('opp_team','')} vs {r.get('team','')}",
        "subject":    r.get("player_name", ""),
        "bet_type":   r.get("bet_side", ""),
        "line":       r.get("prop_line"),
        "odds":       r.get("juice", -110),
        "model_prob": r.get("model_prob"),
        "edge":       r.get("edge"),
    }


def _extract_pitcher_outs(r) -> dict:
    return {
        "model":      "Pitcher Outs",
        "game":       f"{r.get('opp_team','')} vs {r.get('team','')}",
        "subject":    r.get("pitcher_name", ""),
        "bet_type":   r.get("bet_side", ""),
        "line":       r.get("prop_line_outs"),
        "odds":       r.get("juice", -110),
        "model_prob": r.get("model_prob"),
        "edge":       r.get("edge"),
    }


def _extract_nrfi(r) -> dict:
    game     = f"{r.get('away_team', '')} @ {r.get('home_team', '')}"
    bet_side = str(r.get("bet_side", ""))
    # model_prob is P(YRFI) for YRFI bets; P(NRFI) = 1 - P(YRFI) for NRFI bets
    p_yrfi   = r.get("p_yrfi")
    model_prob = p_yrfi if bet_side == "YRFI" else (1 - p_yrfi if p_yrfi is not None else None)
    return {
        "model":      "NRFI/YRFI",
        "game":       game,
        "subject":    game,   # game string doubles as the grading lookup key
        "bet_type":   bet_side,
        "line":       None,   # binary market — no numerical line
        "odds":       r.get("bet_odds", 100),
        "model_prob": model_prob,
        "edge":       r.get("edge"),
    }


# =============================================================================
# GRADE PICKS
# =============================================================================

def grade_picks(grade_date: str) -> int:
    """
    Fetch MLB results for grade_date and score any PENDING picks.
    Returns number of picks graded.

    Parameters
    ----------
    grade_date : str — 'YYYYMMDD'
    """
    picks = _load_picks()
    if picks.empty:
        return 0

    pending = picks[(picks["pick_date"] == grade_date) & (picks["result"] == "PENDING")]
    if pending.empty:
        return 0

    # Convert YYYYMMDD → YYYY-MM-DD for the API
    api_date = f"{grade_date[:4]}-{grade_date[4:6]}-{grade_date[6:]}"

    print(f"  (tracker) Grading {len(pending)} picks from {api_date}...")

    try:
        game_scores, batter_tb, pitcher_outs_map, nrfi_results = _fetch_mlb_results(api_date)
    except Exception as e:
        print(f"  (tracker) Could not fetch MLB results: {e}")
        return 0

    if not game_scores:
        print(f"  (tracker) No final scores found for {api_date} (games may still be in progress).")
        return 0

    graded = 0
    for idx in pending.index:
        row   = picks.loc[idx]
        model = row["model"]
        try:
            result, actual = _grade_row(row, model, game_scores, batter_tb, pitcher_outs_map, nrfi_results)
        except Exception as e:
            print(f"  (tracker) Error grading {model} pick '{row['subject']}': {e}")
            result, actual = "ERROR", None

        if result in ("WIN", "LOSS", "PUSH"):
            pnl    = _american_pnl(row["odds"])        if result == "WIN" else (0.0 if result == "PUSH" else -100.0)
            pnl_50 = _american_pnl(row["odds"], 50.0)  if result == "WIN" else (0.0 if result == "PUSH" else  -50.0)
            picks.at[idx, "result"] = result
            picks.at[idx, "actual"] = actual
            picks.at[idx, "pnl"]    = pnl
            picks.at[idx, "pnl_50"] = pnl_50
            graded += 1

    _save_picks_df(picks)
    print(f"  (tracker) Graded {graded} / {len(pending)} picks.")
    return graded


def _grade_row(row, model, game_scores, batter_tb, pitcher_outs_map, nrfi_results=None):
    """Return (result, actual_value) for a single pick row."""
    if model == "Moneyline":
        return _grade_moneyline(row, game_scores)
    elif model == "Totals":
        return _grade_totals(row, game_scores)
    elif model == "Hitter TB":
        return _grade_prop(row, batter_tb)
    elif model == "Pitcher Outs":
        return _grade_prop(row, pitcher_outs_map)
    elif model == "NRFI/YRFI":
        return _grade_nrfi(row, nrfi_results or {})
    return "PENDING", None


def _grade_moneyline(row, game_scores):
    """Grade a moneyline pick: subject is the team we backed."""
    backed_team = str(row["subject"])
    key = _find_game_key(backed_team, game_scores)
    if key is None:
        return "PENDING", None

    scores = game_scores[key]
    away, home = key
    winner = home if scores["home_runs"] > scores["away_runs"] else away
    actual = f"{scores['away_runs']}-{scores['home_runs']}"

    if backed_team == winner:
        return "WIN", actual
    return "LOSS", actual


def _grade_totals(row, game_scores):
    """Grade an over/under pick."""
    game_str = str(row["game"])   # "ATL @ NYY"
    parts = [p.strip() for p in game_str.replace("@", " ").split()]
    away, home = (parts[0], parts[-1]) if len(parts) >= 2 else ("", "")

    key = _find_game_key_pair(away, home, game_scores)
    if key is None:
        return "PENDING", None

    scores   = game_scores[key]
    total    = scores["home_runs"] + scores["away_runs"]
    line     = float(row["line"]) if row["line"] is not None else None
    bet_type = str(row["bet_type"]).upper()

    if line is None:
        return "PENDING", None

    if total == line:
        return "PUSH", total
    went_over = total > line
    result = "WIN" if (bet_type == "OVER" and went_over) or (bet_type == "UNDER" and not went_over) else "LOSS"
    return result, total


def _grade_prop(row, stat_map):
    """Grade an OVER/UNDER prop (TB or outs)."""
    subject  = str(row["subject"])
    line     = float(row["line"]) if row["line"] is not None else None
    bet_type = str(row["bet_type"]).upper()

    if line is None:
        return "PENDING", None

    match = _best_name_match(subject, list(stat_map.keys()))
    if match is None:
        return "PENDING", None

    actual = stat_map[match]
    if actual == line:
        return "PUSH", actual
    went_over = actual > line
    result = "WIN" if (bet_type == "OVER" and went_over) or (bet_type == "UNDER" and not went_over) else "LOSS"
    return result, actual


def _grade_nrfi(row, nrfi_results: dict):
    """
    Grade a NRFI/YRFI pick.

    nrfi_results keyed by (away_abbr, home_abbr) →
      {"yrfi": 0|1, "away_1st": int, "home_1st": int}

    bet_type = "YRFI" → WIN if yrfi==1, LOSS if yrfi==0
    bet_type = "NRFI" → WIN if yrfi==0, LOSS if yrfi==1
    No push possible (binary market).
    """
    game_str = str(row.get("subject", row.get("game", "")))
    # Parse "ATL @ NYY" → away="ATL", home="NYY"
    parts = [p.strip() for p in game_str.replace("@", " ").split()]
    away, home = (parts[0], parts[-1]) if len(parts) >= 2 else ("", "")

    key = _find_game_key_pair(away, home, nrfi_results)
    if key is None:
        return "PENDING", None

    info     = nrfi_results[key]
    yrfi     = info.get("yrfi", None)
    if yrfi is None:
        return "PENDING", None

    actual_str = f"away_1st={info['away_1st']} home_1st={info['home_1st']}"
    bet_type   = str(row.get("bet_type", "")).upper()

    if bet_type == "YRFI":
        result = "WIN" if yrfi == 1 else "LOSS"
    elif bet_type == "NRFI":
        result = "WIN" if yrfi == 0 else "LOSS"
    else:
        return "PENDING", None

    return result, actual_str


# =============================================================================
# MLB STATS API
# =============================================================================

def _fetch_mlb_results(api_date: str):
    """
    Fetch final scores + player box stats for api_date ('YYYY-MM-DD').
    Returns:
      game_scores   : {(away_abbr, home_abbr): {"home_runs", "away_runs"}}
      batter_tb     : {player_full_name: total_bases}
      pitcher_outs  : {pitcher_full_name: outs_recorded}
      nrfi_results  : {(away_abbr, home_abbr): {"yrfi": 0|1, "away_1st": int, "home_1st": int}}
    """
    # Step 1: get gamePks, final scores, and linescore (first-inning runs)
    resp = requests.get(
        MLB_SCHEDULE_URL,
        params={"sportId": 1, "date": api_date, "hydrate": "linescore"},
        timeout=20,
    )
    resp.raise_for_status()
    schedule = resp.json()

    game_scores  = {}
    nrfi_results = {}
    final_pks    = []

    for date_entry in schedule.get("dates", []):
        for game in date_entry.get("games", []):
            state = game.get("status", {}).get("abstractGameState", "")
            if state != "Final":
                continue

            home_abbr = MLB_TO_STANDARD.get(
                game["teams"]["home"]["team"].get("abbreviation", ""),
                game["teams"]["home"]["team"].get("abbreviation", ""),
            )
            away_abbr = MLB_TO_STANDARD.get(
                game["teams"]["away"]["team"].get("abbreviation", ""),
                game["teams"]["away"]["team"].get("abbreviation", ""),
            )
            home_runs = game["teams"]["home"].get("score", 0) or 0
            away_runs = game["teams"]["away"].get("score", 0) or 0

            game_scores[(away_abbr, home_abbr)] = {
                "home_runs": int(home_runs),
                "away_runs": int(away_runs),
            }
            final_pks.append(game["gamePk"])

            # Extract first-inning runs from linescore
            linescore = game.get("linescore", {})
            innings   = linescore.get("innings", [])
            away_1st  = 0
            home_1st  = 0
            for inn in innings:
                if inn.get("num") == 1:
                    away_1st = int(inn.get("away", {}).get("runs", 0) or 0)
                    home_1st = int(inn.get("home", {}).get("runs", 0) or 0)
                    break
            nrfi_results[(away_abbr, home_abbr)] = {
                "yrfi":     int(away_1st > 0 or home_1st > 0),
                "away_1st": away_1st,
                "home_1st": home_1st,
            }

    if not final_pks:
        return game_scores, {}, {}, nrfi_results

    # Step 2: fetch box scores for player stats
    batter_tb    = {}
    pitcher_outs = {}

    for pk in final_pks:
        try:
            box = requests.get(
                f"https://statsapi.mlb.com/api/v1/game/{pk}/boxscore",
                timeout=20,
            ).json()
        except Exception:
            continue

        for side in ("home", "away"):
            players = box.get("teams", {}).get(side, {}).get("players", {})
            for _, player in players.items():
                name   = player.get("person", {}).get("fullName", "")
                bstats = player.get("stats", {}).get("batting", {})
                pstats = player.get("stats", {}).get("pitching", {})

                # Total bases
                if bstats.get("atBats", 0) or bstats.get("hits", 0):
                    h  = int(bstats.get("hits", 0))
                    d  = int(bstats.get("doubles", 0))
                    t  = int(bstats.get("triples", 0))
                    hr = int(bstats.get("homeRuns", 0))
                    tb = h + d + 2 * t + 3 * hr
                    batter_tb[name] = tb

                # Pitcher outs
                ip_str = pstats.get("inningsPitched")
                if ip_str is not None:
                    pitcher_outs[name] = _parse_outs(ip_str)

    return game_scores, batter_tb, pitcher_outs, nrfi_results


def _find_game_key(team_abbr: str, game_scores: dict):
    """Find the game key where team_abbr is home or away."""
    for (away, home) in game_scores:
        if team_abbr in (away, home):
            return (away, home)
    return None


def _find_game_key_pair(away: str, home: str, game_scores: dict):
    """Find game key matching both teams (with abbreviation tolerance)."""
    direct = (away, home)
    if direct in game_scores:
        return direct
    # Try fuzzy: any key that contains both
    for key in game_scores:
        if away in key and home in key:
            return key
    return None


# =============================================================================
# PERFORMANCE SUMMARY
# =============================================================================

def print_performance_summary(n_days: int = 30):
    """Print a P&L summary table by model for the last n_days."""
    picks = _load_picks()
    if picks.empty:
        print("  (tracker) No picks logged yet.")
        return

    # Filter to graded picks within window
    picks["pick_date"] = picks["pick_date"].astype(str)
    cutoff = (date.today() - timedelta(days=n_days)).strftime("%Y%m%d")
    recent = picks[
        (picks["pick_date"] >= cutoff) &
        (picks["result"].isin(["WIN", "LOSS", "PUSH"]))
    ].copy()

    if recent.empty:
        print(f"  (tracker) No graded picks in the last {n_days} days.")
        return

    recent["pnl"] = pd.to_numeric(recent["pnl"], errors="coerce").fillna(0)

    print(f"\n  {'─' * 58}")
    print(f"  PERFORMANCE SUMMARY — last {n_days} days (${100} flat stake)")
    print(f"  {'─' * 58}")
    print(f"  {'Model':<16} {'Picks':>6} {'W':>5} {'L':>5} {'P':>5} {'Win%':>7} {'P&L':>9}")
    print(f"  {'─' * 58}")

    totals_row = {"picks": 0, "w": 0, "l": 0, "p": 0, "pnl": 0.0}

    for model in ["Moneyline", "Totals", "Hitter TB", "Pitcher Outs", "NRFI/YRFI"]:
        sub = recent[recent["model"] == model]
        if sub.empty:
            continue
        w   = int((sub["result"] == "WIN").sum())
        l   = int((sub["result"] == "LOSS").sum())
        p   = int((sub["result"] == "PUSH").sum())
        tot = w + l
        pct = f"{100 * w / tot:.1f}%" if tot > 0 else "—"
        pnl = sub["pnl"].sum() - (l * 100)
        print(f"  {model:<16} {len(sub):>6} {w:>5} {l:>5} {p:>5} {pct:>7} {pnl:>+9.2f}")
        totals_row["picks"] += len(sub)
        totals_row["w"]     += w
        totals_row["l"]     += l
        totals_row["p"]     += p
        totals_row["pnl"]   += pnl

    print(f"  {'─' * 58}")
    t = totals_row
    tot = t["w"] + t["l"]
    pct = f"{100 * t['w'] / tot:.1f}%" if tot > 0 else "—"
    print(f"  {'ALL MODELS':<16} {t['picks']:>6} {t['w']:>5} {t['l']:>5} {t['p']:>5} {pct:>7} {t['pnl']:>+9.2f}")
    print(f"  {'─' * 58}\n")
