"""
=============================================================================
DAILY SCHEDULER — registers one-shot launchd jobs to run run_daily.py
                  90 minutes before each MLB game window
=============================================================================
Launched at 5am every day by launchd (installed via setup_launchd.py).

Flow:
  1. Clean up any stale run plists from previous days
  2. Fetch today's MLB schedule
  3. Group games into time windows (games within 20 min of each other share
     a window — one run covers all of them)
  4. For each window, write a one-shot launchd plist for T-90 and load it
     (launchd owns the timer — survives machine sleep/wake)
  5. Exit immediately — no long sleep in this process

Example — 3:05 PM + 3:10 PM + 10:05 PM games:
  → Registers plist to run at 1:35 PM  (covers the early games)
  → Registers plist to run at 8:35 PM  (covers the night game)

No games today → exits silently.
Already past T-90 for a window → runs immediately instead of scheduling.
=============================================================================
"""

import sys
import subprocess
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path

BASE_DIR    = Path(__file__).parent
LEAD_TIME   = timedelta(minutes=90)
WINDOW_GAP  = timedelta(minutes=20)   # games within 20 min share a run
LOG_DIR     = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

AGENTS_DIR  = Path.home() / "Library" / "LaunchAgents"
RUN_LABEL_PREFIX = "com.baseballmodels.run."


def get_game_times_utc(date_str: str) -> list:
    """Return sorted list of UTC-aware game start times for date_str."""
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
            gt = game.get("gameDate")   # e.g. "2026-03-25T17:10:00Z"
            if gt:
                times.append(datetime.fromisoformat(gt.replace("Z", "+00:00")))

    return sorted(set(times))


def group_into_windows(game_times: list) -> list:
    """
    Cluster game times into windows — if two games start within WINDOW_GAP
    of each other they share a single pre-game run.
    Returns a list of (window_start, [game_times_in_window]) tuples.
    """
    if not game_times:
        return []

    windows         = []
    window_start    = game_times[0]
    window_games    = [game_times[0]]

    for t in game_times[1:]:
        if t - window_start > WINDOW_GAP:
            windows.append((window_start, window_games))
            window_start = t
            window_games = [t]
        else:
            window_games.append(t)

    windows.append((window_start, window_games))
    return windows


def log(msg: str):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    log_file = LOG_DIR / f"scheduler_{datetime.now().strftime('%Y%m%d')}.log"
    with open(log_file, "a") as f:
        f.write(line + "\n")


def cleanup_stale_run_plists():
    """Unload and delete run plists from previous days."""
    today_str = datetime.now().strftime("%Y%m%d")
    for plist in AGENTS_DIR.glob(f"{RUN_LABEL_PREFIX}*.plist"):
        # Stem: com.baseballmodels.run.YYYYMMDDHHMM
        date_part = plist.stem.replace(RUN_LABEL_PREFIX, "")[:8]
        if date_part < today_str:
            subprocess.run(["launchctl", "unload", str(plist)], capture_output=True)
            plist.unlink()
            log(f"Cleaned up stale plist: {plist.name}")


def schedule_via_launchd(window_time: datetime, index: int, total: int,
                         window_game_times: list = None):
    """Write and load a one-shot launchd plist to run run_daily.py at T-90."""
    run_at       = window_time - LEAD_TIME
    run_at_local = run_at.astimezone()
    label        = f"{RUN_LABEL_PREFIX}{run_at_local.strftime('%Y%m%d%H%M')}"
    plist_path   = AGENTS_DIR / f"{label}.plist"
    stamp        = run_at_local.strftime("%Y%m%d_%H%M")

    # Build --window-games arg: comma-separated UTC ISO timestamps
    window_games_arg = ""
    if window_game_times:
        times_str = ",".join(t.strftime("%Y-%m-%dT%H:%M:%S+00:00") for t in window_game_times)
        window_games_arg = f"""
        <string>--window-games</string>
        <string>{times_str}</string>"""

    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>

    <key>ProgramArguments</key>
    <array>
        <string>{sys.executable}</string>
        <string>{BASE_DIR / "run_daily.py"}</string>{window_games_arg}
    </array>

    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>{run_at_local.hour}</integer>
        <key>Minute</key>
        <integer>{run_at_local.minute}</integer>
    </dict>

    <key>StandardOutPath</key>
    <string>{LOG_DIR / f"run_{stamp}_stdout.log"}</string>

    <key>StandardErrorPath</key>
    <string>{LOG_DIR / f"run_{stamp}_stderr.log"}</string>

    <key>KeepAlive</key>
    <false/>

    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
"""
    plist_path.write_text(plist)
    subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True)
    result = subprocess.run(["launchctl", "load", str(plist_path)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        log(f"  ERROR registering window {index}: {result.stderr.strip()}")
    else:
        n_games = len(window_game_times) if window_game_times else "?"
        log(f"  Window {index}/{total} registered → run_daily.py at "
            f"{run_at_local.strftime('%-I:%M %p %Z')} "
            f"(first pitch {window_time.astimezone().strftime('%-I:%M %p %Z')}, "
            f"{n_games} game(s))")


def run_models(window_game_times: list = None):
    """Run run_daily.py immediately, optionally scoped to a specific game window."""
    log("Launching run_daily.py immediately (T-90 already passed)...")
    cmd = [sys.executable, str(BASE_DIR / "run_daily.py")]
    if window_game_times:
        times_str = ",".join(t.strftime("%Y-%m-%dT%H:%M:%S+00:00") for t in window_game_times)
        cmd += ["--window-games", times_str]
    result = subprocess.run(cmd, capture_output=False)
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

    cleanup_stale_run_plists()

    today_str = today.strftime("%Y-%m-%d")
    log(f"Scheduler started — checking MLB schedule for {today_str}")

    try:
        game_times = get_game_times_utc(today_str)
    except Exception as e:
        log(f"ERROR fetching schedule: {e}")
        sys.exit(1)

    if not game_times:
        log("No MLB games today — exiting.")
        return

    window_groups = group_into_windows(game_times)
    total_windows = len(window_groups)
    log(f"Found {len(game_times)} game(s) across {total_windows} time window(s).")

    try:
        from utils.notifier import notify_run_status
        lines = [f"{len(game_times)} game(s) · {total_windows} model run(s) scheduled"]
        for window_time, wgames in window_groups:
            run_at   = (window_time - LEAD_TIME).astimezone().strftime("%-I:%M %p %Z")
            fp_time  = window_time.astimezone().strftime("%-I:%M %p %Z")
            lines.append(f"  First pitch {fp_time} → models run at {run_at} ({len(wgames)} game(s))")
        notify_run_status("✅ 5 AM Schedule Analyzed", lines)
    except Exception as e:
        log(f"Status notification failed: {e}")

    now_utc = datetime.now(timezone.utc)
    for i, (window_time, window_game_times) in enumerate(window_groups, 1):
        run_at    = window_time - LEAD_TIME
        wait_secs = (run_at - now_utc).total_seconds()

        if wait_secs < -LEAD_TIME.total_seconds():
            # More than 90 min past first pitch — game is underway, skip
            log(f"Window {i}/{total_windows}: game already in progress — skipping.")
            continue

        if wait_secs <= 0:
            # T-90 is now or just passed — run immediately
            log(f"Window {i}/{total_windows}: T-90 already passed — running immediately.")
            run_models(window_game_times)
        else:
            # Register a launchd job; this process can now exit cleanly
            schedule_via_launchd(window_time, i, total_windows, window_game_times)


if __name__ == "__main__":
    main()
