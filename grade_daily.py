"""
=============================================================================
DAILY GRADER — grades yesterday's picks and posts results to Discord
=============================================================================
Runs automatically at 4am every day via launchd (installed via setup_launchd.py).
Can also be run manually: python3 grade_daily.py
=============================================================================
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.tracker  import grade_picks
from utils.notifier import send_graded_results

SEASON_START  = datetime(2026, 3, 25)
yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

if datetime.now() < SEASON_START:
    print(f"  Before Opening Day ({SEASON_START.strftime('%B %d')}) — skipping.")
    sys.exit(0)

print(f"\n{'=' * 60}")
print(f"  GRADING — {datetime.now().strftime('%A, %B %d, %Y')}")
print(f"  Grading picks from {yesterday_str[:4]}-{yesterday_str[4:6]}-{yesterday_str[6:]}")
print(f"{'=' * 60}\n")

n_graded = grade_picks(yesterday_str)

if n_graded > 0:
    send_graded_results(yesterday_str)
else:
    print("  No picks to grade or results not yet final.")
    from utils.notifier import notify_run_status
    notify_run_status("📋 4 AM Grader — No Results",
                      ["No picks from yesterday to grade, or results not yet final."])

print(f"\n{'=' * 60}\n")

# ── Push graded results to GitHub for monitoring agents ───────────────────
import subprocess
from pathlib import Path as _Path
_BASE_DIR = str(_Path(__file__).parent)
try:
    subprocess.run(
        ["git", "add", "tracking/picks.xlsx"],
        cwd=_BASE_DIR, check=True, capture_output=True
    )
    result = subprocess.run(
        ["git", "commit", "-m", f"Graded results: {yesterday_str}"],
        cwd=_BASE_DIR, capture_output=True
    )
    if result.returncode == 0:
        subprocess.run(["git", "push"], cwd=_BASE_DIR, check=True, capture_output=True)
        print("  ✓ Graded results pushed to GitHub.")
    elif b"nothing to commit" in result.stdout + result.stderr:
        print("  (git) Nothing new to commit.")
    else:
        result.check_returncode()
except Exception as e:
    print(f"  WARNING: git push failed — {e}")
