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

# ── Local model evaluation agent ──────────────────────────────────────────
import shutil
if shutil.which("claude"):
    print("\n  Launching local model evaluation agent...")
    prompt = f"""You are a baseball model performance analyst. Analyze the graded results from yesterday ({yesterday_str}) and provide a concise evaluation.

Working directory: {_BASE_DIR}

Steps:
1. Read tracking/picks.xlsx — focus on the 'Season Results' sheet and each per-model sheet (Moneyline, Totals, Hitter TB, Pitcher Outs, NRFI-YRFI)
2. Filter to picks from {yesterday_str} that have result WIN, LOSS, or PUSH
3. For each model, report: picks made, W/L/P record, win rate, P&L at $100 flat stake
4. Flag any model with win rate below 40% yesterday as a concern
5. Read tracking/picks.xlsx 'Season Results' sheet and compute the same stats for the last 14 days across all models
6. Compare yesterday's performance to the 14-day trend — is yesterday an outlier or consistent with recent form?
7. Check if any model produced 0 picks yesterday — note whether that is expected (no edge found) or suspicious
8. Write your findings to logs/grade_evaluation_{yesterday_str}.txt
9. Print a brief summary to the terminal

Be direct and specific. Focus on actionable observations, not general commentary."""

    subprocess.run(
        ["claude", "-p", prompt],
        cwd=_BASE_DIR,
    )
else:
    print("  (claude CLI not found — skipping local evaluation agent)")
