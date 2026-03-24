"""
=============================================================================
LAUNCHD SETUP — installs two macOS background services:
  1. com.baseballmodels.scheduler  — runs models T-90 min before each game
  2. com.baseballmodels.grader     — grades yesterday's picks at 9am daily
=============================================================================
Run once:
    python3 setup_launchd.py           # install both
    python3 setup_launchd.py --remove  # uninstall both
    python3 setup_launchd.py --status  # check status of both
=============================================================================
"""

import sys
import argparse
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent
LOG_DIR  = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

JOBS = [
    {
        "label":   "com.baseballmodels.scheduler",
        "script":  BASE_DIR / "schedule_daily.py",
        "hour":    8,
        "minute":  0,
        "stdout":  LOG_DIR / "scheduler_stdout.log",
        "stderr":  LOG_DIR / "scheduler_stderr.log",
        "desc":    "Model scheduler (wakes at 8am, runs models T-90 min before each game)",
    },
    {
        "label":   "com.baseballmodels.grader",
        "script":  BASE_DIR / "grade_daily.py",
        "hour":    9,
        "minute":  0,
        "stdout":  LOG_DIR / "grader_stdout.log",
        "stderr":  LOG_DIR / "grader_stderr.log",
        "desc":    "Results grader (runs at 9am, grades yesterday's picks)",
    },
]


def _plist_path(label: str) -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"


def _make_plist(job: dict) -> str:
    python = sys.executable
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{job['label']}</string>

    <key>ProgramArguments</key>
    <array>
        <string>{python}</string>
        <string>{job['script']}</string>
    </array>

    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>{job['hour']}</integer>
        <key>Minute</key>
        <integer>{job['minute']}</integer>
    </dict>

    <key>StandardOutPath</key>
    <string>{job['stdout']}</string>

    <key>StandardErrorPath</key>
    <string>{job['stderr']}</string>

    <key>KeepAlive</key>
    <false/>

    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
"""


def install():
    for job in JOBS:
        path = _plist_path(job["label"])
        path.write_text(_make_plist(job))
        subprocess.run(["launchctl", "unload", str(path)], capture_output=True)
        result = subprocess.run(["launchctl", "load", str(path)],
                                capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR loading {job['label']}: {result.stderr}")
        else:
            print(f"  ✓ {job['desc']}")

    print()
    print("  Daily schedule:")
    print("    9:00 AM — grade yesterday's picks → post results to Discord")
    print("    8:00 AM — scheduler wakes, waits until T-90 before each game window")
    print("    T-90    — models run → SMS + Discord picks sent")
    print(f"\n  Logs: {LOG_DIR}/")


def remove():
    for job in JOBS:
        path = _plist_path(job["label"])
        if path.exists():
            subprocess.run(["launchctl", "unload", str(path)], capture_output=True)
            path.unlink()
            print(f"  ✓ Removed {job['label']}")
        else:
            print(f"  {job['label']} was not installed.")


def status():
    for job in JOBS:
        result = subprocess.run(
            ["launchctl", "list", job["label"]],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"  ✓ ACTIVE   {job['label']}")
        else:
            print(f"  ✗ INACTIVE {job['label']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--remove", action="store_true", help="Uninstall both jobs")
    ap.add_argument("--status", action="store_true", help="Check status of both jobs")
    args = ap.parse_args()

    if args.remove:
        remove()
    elif args.status:
        status()
    else:
        install()
