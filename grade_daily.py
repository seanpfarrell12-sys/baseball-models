"""
=============================================================================
DAILY GRADER — grades yesterday's picks and posts results to Discord
=============================================================================
Runs automatically at 9am every day via launchd (installed via setup_launchd.py).
Can also be run manually: python3 grade_daily.py
=============================================================================
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.tracker  import grade_picks
from utils.notifier import send_graded_results

SEASON_START  = datetime(2026, 3, 26)
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

print(f"\n{'=' * 60}\n")
