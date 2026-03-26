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

# ── Local orchestrator + 5 specialist subagents ───────────────────────────
import shutil
if shutil.which("claude"):
    print("\n  Launching model evaluation orchestrator...")
    prompt = f"""You are the master baseball model evaluation orchestrator. Grading has just completed for {yesterday_str}. Your job is to spawn 5 specialist subagents in parallel — one per model — then synthesize their findings into a unified report.

Base directory: {_BASE_DIR}
Graded date: {yesterday_str}
Exports folder: {_BASE_DIR}/exports/{yesterday_str}/
Picks file: {_BASE_DIR}/tracking/picks.xlsx

---

## YOUR TASK

Use the Agent tool to spawn all 5 specialists simultaneously. Pass each the exact prompt below. After all 5 return, synthesize their findings and write the unified report.

---

### SPECIALIST 1 — MONEYLINE
Domain: XGBoost binary classifier predicting home team win probability.
Key assumptions: 6% MIN_EDGE, quarter-Kelly 25% with 5% hard cap, SP stats matched individually via fuzzy name matching, odds from Action Network PRO only, vig-adjusted market probabilities.
Good signs: edge values 6–15%, EV% positive, Kelly ≤ 5%, 0–2 plays per day.
Red flags: any play with edge < 6%, Kelly > 5%, 5+ plays (overconfidence), SP fallback rate > 5%.
Benchmarks: target pick rate > 60% over 30 games, Brier score < 0.25.
Files to read: {_BASE_DIR}/exports/{yesterday_str}/moneyline_edges_*.xlsx (most recent), {_BASE_DIR}/tracking/picks.xlsx sheet=Moneyline.
Analyze: edge threshold compliance, EV% quality, pick volume, Kelly sizing, SP match rate, rolling win rate from picks.xlsx.
Graded results: filter picks.xlsx Moneyline sheet to pick_date={yesterday_str}, report W/L/P record, win rate, P&L at $100 flat stake. Compare to 14-day rolling win rate.
Return your findings in this exact format:
=== SPECIALIST 1: MONEYLINE ===
Yesterday ({yesterday_str}): [W-L-P record, win rate, P&L]
14-day trend: [win rate, P&L]
Key Findings: [bullet list]
Recommended Improvements: [ranked list]
What to watch next: [1-2 sentences]

### SPECIALIST 2 — TOTALS O/U
Domain: Poisson GLM predicting total runs scored (λ̂), evaluated against market O/U line.
Key assumptions: 6% MIN_EDGE, consensus O/U line with outliers >0.5 runs from median excluded, games with no valid line skipped (no fallback), umpire accuracy is a key feature, park altitude/roof included.
Good signs: λ̂ in 7.5–10.5 range, edge 6–15%, umpire data present, 0–2 plays per day.
Red flags: λ̂ outside 6–13, any play with edge < 6%, games skipped when valid line exists, umpire data missing for >10% of games, over+under edges both positive for same game (vig error).
Benchmarks: target pick rate > 60%, MAE < 1.8 runs, O/U accuracy ≥ 52%.
Files to read: {_BASE_DIR}/exports/{yesterday_str}/totals_edges_*.xlsx (most recent), {_BASE_DIR}/tracking/picks.xlsx sheet=Totals.
Analyze: lambda distribution, edge compliance, line sourcing, vig integrity, umpire join rate, rolling win rate.
Graded results: filter picks.xlsx Totals sheet to pick_date={yesterday_str}, report W/L/P record, win rate, P&L. Compare to 14-day rolling win rate.
Return your findings in this exact format:
=== SPECIALIST 2: TOTALS ===
Yesterday ({yesterday_str}): [W-L-P record, win rate, P&L]
14-day trend: [win rate, P&L]
Key Findings: [bullet list]
Recommended Improvements: [ranked list]
What to watch next: [1-2 sentences]

### SPECIALIST 3 — PITCHER OUTS
Domain: XGBoost regressor predicting E[Outs] per start, evaluated against single consensus prop line.
Key assumptions: 7% MIN_EDGE, single consensus line per pitcher (≥2 books required), normal approximation σ=3.5 outs, manager depth score included, 15% fractional Kelly with 4% hard cap, no per-game dedup (both starters can appear — if BOTH show 7%+ edge on same game, flag as potential model bias).
Good signs: E[Outs] 12–21, 0–1 picks per game, edge 7–15%, CSW% as proportion ~0.25–0.35.
Red flags: edge < 7%, E[Outs] outside 12–21, CSW% appearing as 25–35 (scaling bug — most common silent error), both starters same game both flagged, Kelly > 4%, lines from <2 books.
Benchmarks: target pick rate > 60%, RMSE < 3.2, MAE < 2.5.
Files to read: {_BASE_DIR}/exports/{yesterday_str}/pitcher_outs_edges_*.xlsx (most recent), {_BASE_DIR}/tracking/picks.xlsx sheet=Pitcher Outs.
IMPORTANT — CSW% check: CSW% should be ~0.25–0.35 (proportion). If values appear as 25–35, flag as CRITICAL scaling bug.
Graded results: filter picks.xlsx Pitcher Outs sheet to pick_date={yesterday_str}, report W/L/P record, win rate, P&L. Compare to 14-day rolling win rate.
Return your findings in this exact format:
=== SPECIALIST 3: PITCHER OUTS ===
Yesterday ({yesterday_str}): [W-L-P record, win rate, P&L]
14-day trend: [win rate, P&L]
Key Findings: [bullet list]
Recommended Improvements: [ranked list]
What to watch next: [1-2 sentences]

### SPECIALIST 4 — HITTER TOTAL BASES
Domain: XGBoost regressor predicting E[TB] per player per game, using geometric (negative binomial n=1) distribution for probabilities.
Key assumptions: 7% MIN_EDGE, geometric distribution NOT Poisson (P(TB=0) = 1/(1+λ) ≈ 35–45% for average hitter), single consensus line per player (≥2 books), max 1 pick per player per day, lineups must be confirmed before scoring, SP SwStr% from raw_pitcher_efficiency.csv as proportions (~0.085–0.15).
Good signs: E[TB] 0.8–2.2, 0–3 picks per game, edge 7–15%, max 1 pick per player.
Red flags: edge < 7%, more than 3 picks per game (model bias), more than 1 pick per player, SwStr% as percentages (8.5–15) instead of proportions, E[TB] outside 0.5–2.5.
Benchmarks: target pick rate > 60%, RMSE < 0.85, MAE < 0.65, prop line accuracy ≥ 53% at 1.5 TB line.
Files to read: {_BASE_DIR}/exports/{yesterday_str}/hitter_tb_edges_*.xlsx (most recent), {_BASE_DIR}/tracking/picks.xlsx sheet=Hitter TB.
Analyze: edge compliance, E[TB] distribution, picks per game, picks per player, SwStr% scale, lineup confirmation, rolling win rate.
Graded results: filter picks.xlsx Hitter TB sheet to pick_date={yesterday_str}, report W/L/P record, win rate, P&L. Compare to 14-day rolling win rate.
Return your findings in this exact format:
=== SPECIALIST 4: HITTER TB ===
Yesterday ({yesterday_str}): [W-L-P record, win rate, P&L]
14-day trend: [win rate, P&L]
Key Findings: [bullet list]
Recommended Improvements: [ranked list]
What to watch next: [1-2 sentences]

### SPECIALIST 5 — NRFI/YRFI
Domain: XGBoost classifier with isotonic regression calibration predicting P(NRFI) per game.
Key assumptions: 6% MIN_EDGE, raw probabilities calibrated via nrfi_calibrator.pkl, P(NRFI) ≥ 0.95 or ≤ 0.05 are expected calibrator tail artifacts NOT errors, odds fetched using period=firstinning filtered on total==0.5, scoring gated on confirmed lineups, umpire accuracy is a key feature, 15% fractional Kelly with 4% hard cap.
Good signs: P(NRFI) base rate 58–65%, calibrated probabilities applied, fi_fstrike_pct varying across pitchers, umpire data present, 0–2 picks per day.
Red flags: edge < 6%, uncalibrated probabilities, P(NRFI)+P(YRFI) ≠ 1.0, fi_fstrike_pct missing/uniform, unconfirmed lineup scored, umpire join rate < 90%, AUC < 0.520, 4+ picks.
DO NOT flag P(NRFI) ≥ 0.95 or ≤ 0.05 as errors — these are expected isotonic calibrator tail artifacts.
Benchmarks: target pick rate > 60%, AUC ≥ 0.520 (baseline 0.5374), log-loss < 0.680.
Files to read: {_BASE_DIR}/exports/{yesterday_str}/nrfi_edges_*.csv (most recent), {_BASE_DIR}/tracking/picks.xlsx sheet=NRFI-YRFI.
Analyze: edge compliance, probability calibration, fi_fstrike_pct coverage, umpire join rate, P(NRFI)+P(YRFI) integrity, lineup confirmation, rolling win rate.
Graded results: filter picks.xlsx NRFI-YRFI sheet to pick_date={yesterday_str}, report W/L/P record, win rate, P&L. Compare to 14-day rolling win rate.
Return your findings in this exact format:
=== SPECIALIST 5: NRFI/YRFI ===
Yesterday ({yesterday_str}): [W-L-P record, win rate, P&L]
14-day trend: [win rate, P&L]
Key Findings: [bullet list]
Recommended Improvements: [ranked list]
What to watch next: [1-2 sentences]

---

## AFTER ALL 5 SPECIALISTS RETURN

Synthesize their findings into this unified report:

# Model Evaluation Report — {yesterday_str}

## Overall Health
[1 paragraph — is the system healthy overall?]

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
[Patterns and shared issues across models. Most valuable section.]

## Top 3 Actions
1. [Highest priority]
2.
3.

## Full Specialist Reports
[Paste all 5 specialist outputs here]

---

## RULES
- Spawn all 5 specialists using the Agent tool before writing the synthesis
- 0 picks from any model is HEALTHY — do not flag as a problem
- Never recommend lowering any MIN_EDGE threshold
- If exports folder for {yesterday_str} is missing, note it and use most recent available
- If picks.xlsx has fewer than 10 graded results, note that win rates are not yet statistically meaningful
- Status key: 🟢 Good, 🟡 Warning (minor issues), 🔴 Critical (immediate action needed)

## FINAL STEP
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
