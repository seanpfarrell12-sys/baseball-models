"""
=============================================================================
DAILY GRADER — grades yesterday's picks and posts results to Discord
=============================================================================
Runs automatically at 4am every day via launchd (installed via setup_launchd.py).
Can also be run manually: python3 grade_daily.py
=============================================================================
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.tracker  import grade_picks
from utils.notifier import send_graded_results, notify_run_status

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
    notify_run_status(
        "📋 4 AM Grader — Complete",
        [f"Graded {n_graded} picks from {yesterday_str[:4]}-{yesterday_str[4:6]}-{yesterday_str[6:]}",
         "Results posted to Discord results channel."],
    )
else:
    print("  No picks to grade or results not yet final.")
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

# ── Local orchestrator + 5 specialist subagents ───────────────────────────
import shutil
if shutil.which("claude"):
    print("\n  Launching model evaluation orchestrator...")
    prompt = f"""You are the master baseball model evaluation orchestrator. Grading has just completed for {yesterday_str}.

graded_date={yesterday_str}
base_dir={_BASE_DIR}
exports_dir={_BASE_DIR}/exports/{yesterday_str}/
picks_file={_BASE_DIR}/tracking/picks.xlsx

Use the Agent tool to spawn all 5 specialist subagents simultaneously. Pass each the graded_date, base_dir, exports_dir, and picks_file above:
- specialist-moneyline
- specialist-totals
- specialist-pitcher-outs
- specialist-hitter-tb
- specialist-nrfi-yrfi

After all 5 return, synthesize their findings into this unified report:

# Model Evaluation Report — {yesterday_str}

## Overall Health
[1 paragraph]

## Yesterday's Results
| Model | W-L-P | Win% | P&L |
|-------|-------|------|-----|
| Moneyline | | | |
| Totals | | | |
| Pitcher Outs | | | |
| Hitter TB | | | |
| NRFI/YRFI | | | |
| **ALL** | | | |

## Per-Model Status
| Model | Status | Top Issue | Priority Action |
|-------|--------|-----------|-----------------|
| Moneyline | 🟢/🟡/🔴 | ... | ... |
| Totals | 🟢/🟡/🔴 | ... | ... |
| Pitcher Outs | 🟢/🟡/🔴 | ... | ... |
| Hitter TB | 🟢/🟡/🔴 | ... | ... |
| NRFI/YRFI | 🟢/🟡/🔴 | ... | ... |

## Cross-Model Insights
[Patterns and shared issues across models.]

## Top 3 Actions
1.
2.
3.

## Recommended Claude Prompts
For each model with a 🟡 or 🔴 status, provide a ready-to-paste Claude prompt the user can run to fix the top issue. Format each as:

**[Model] — [Issue]**
```
[Exact prompt to paste into Claude Code, referencing specific files and the precise fix needed]
```

Prompts should be surgical and specific — reference the actual file paths, function names, and values involved. Do not generate prompts for 🟢 models.

## Full Specialist Reports
[Paste all 5 specialist outputs here]

Rules:
- 0 picks from any model is HEALTHY — do not flag as a problem
- Never recommend lowering any MIN_EDGE threshold
- If exports folder for {yesterday_str} is missing, use most recent available
- If picks.xlsx has fewer than 10 graded results, note win rates are not yet statistically meaningful
- Status: 🟢 Good, 🟡 Warning, 🔴 Critical

Write the unified report to: {_BASE_DIR}/logs/grade_evaluation_{yesterday_str}.txt
Print a brief terminal summary (overall W-L-P, any 🔴 critical issues only)."""

    subprocess.run(
        ["claude", "-p", prompt],
        cwd=_BASE_DIR,
    )

    # Email the report once the agent has written it
    from utils.notifier import send_report_email
    report_path = os.path.join(_BASE_DIR, "logs", f"grade_evaluation_{yesterday_str}.txt")
    send_report_email(report_path, yesterday_str)
else:
    print("  (claude CLI not found — skipping local evaluation agent)")
