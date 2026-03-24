"""
=============================================================================
DAILY SCHEDULER — runs run_daily.py 90 minutes before each MLB game window
=============================================================================
Launched at 8am every day by launchd (installed via setup_launchd.py).

Flow:
  1. Fetch today's MLB schedule
  2. Group games into time windows (games within 20 min of each other share
     a window — one run covers all of them)
  3. For each window, sleep until T-90 min then run run_daily.py
  4. Repeat for every window throughout the day

Example — 3:05 PM + 3:10 PM + 10:05 PM games:
  → Run 1 at 1:35 PM  (covers the early games)
  → Run 2 at 8:35 PM  (covers the night game with confirmed lineups)

No games today → exits silently.
Already past T-90 for a window → skips that window (or runs immediately
if we're between T-90 and first pitch).
=============================================================================
"""

import sys
import time
import subprocess
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path

BASE_DIR    = Path(__file__).parent
LEAD_TIME   = timedelta(minutes=90)
WINDOW_GAP  = timedelta(minutes=20)   # games within 20 min share a run
LOG_DIR     = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


def get_game_times_utc(date_str: str) -> list:
    """
    Return sorted list of UTC-aware game start times for date_str.
    """
    resp = requests.get(
        "https://statsapi.mlb.com/api/v1/schedule",
        params={"sportId": 1, "date": date_str},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()

    times = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            gt = game.get("gameDate")   # e.g. "2026-03-24T17:10:00Z"
            if gt:
                times.append(datetime.fromisoformat(gt.replace("Z", "+00:00")))

    return sorted(set(times))


def group_into_windows(game_times: list) -> list:
    """
    Cluster game times into windows — if two games start within WINDOW_GAP
    of each other they share a single pre-game run.
    Returns a list of the earliest time in each window.
    """
    if not game_times:
        return []

    windows    = []
    window_start = game_times[0]

    for t in game_times[1:]:
        if t - window_start > WINDOW_GAP:
            windows.append(window_start)
            window_start = t

    windows.append(window_start)
    return windows


def log(msg: str):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    log_file = LOG_DIR / f"scheduler_{datetime.now().strftime('%Y%m%d')}.log"
    with open(log_file, "a") as f:
        f.write(line + "\n")


def run_models():
    log("Launching run_daily.py...")
    result = subprocess.run(
        [sys.executable, str(BASE_DIR / "run_daily.py")],
        capture_output=False,
    )
    if result.returncode != 0:
        log(f"run_daily.py exited with code {result.returncode}")
    else:
        log("run_daily.py completed successfully.")


SEASON_START = datetime(2026, 3, 25).date()


def main():
    today = datetime.now().date()
    if today < SEASON_START:
        log(f"Before Opening Day ({SEASON_START}) — skipping.")
        return
    today = today.strftime("%Y-%m-%d")
    log(f"Scheduler started — checking MLB schedule for {today}")

    try:
        game_times = get_game_times_utc(today)
    except Exception as e:
        log(f"ERROR fetching schedule: {e}")
        sys.exit(1)

    if not game_times:
        log("No MLB games today — exiting.")
        return

    windows = group_into_windows(game_times)
    log(f"Found {len(game_times)} game(s) across {len(windows)} time window(s).")

    for i, window_time in enumerate(windows, 1):
        run_at    = window_time - LEAD_TIME
        now_utc   = datetime.now(timezone.utc)
        wait_secs = (run_at - now_utc).total_seconds()

        local_game = window_time.astimezone().strftime("%I:%M %p %Z")
        local_run  = run_at.astimezone().strftime("%I:%M %p %Z")
        log(f"Window {i}/{len(windows)}: first pitch {local_game} → run at {local_run}")

        if wait_secs < -LEAD_TIME.total_seconds():
            # More than 90 min past first pitch — game is underway, skip
            log(f"  Window {i} already in progress — skipping.")
            continue

        if wait_secs > 0:
            log(f"  Sleeping {wait_secs / 60:.1f} minutes until window {i}...")
            time.sleep(wait_secs)

        else:
            log(f"  T-90 already passed for window {i} — running immediately.")

        run_models()


if __name__ == "__main__":
    main()
