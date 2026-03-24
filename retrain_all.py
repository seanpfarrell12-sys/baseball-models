"""
=============================================================================
WEEKLY RETRAIN — pulls fresh data and retrains all 4 models in sequence
=============================================================================
Usage:
    python3 retrain_all.py

Runs in order:
    1. Pull fresh data      (01_input_*.py  x4)
    2. Rebuild datasets     (02_build_*.py  x4)
    3. Retrain models       (03_analysis_*.py x4)

Each step runs independently — if one model fails the others still run.
=============================================================================
"""

import os
import sys
import importlib.util
import traceback
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


def run_script(path: str):
    spec = importlib.util.spec_from_file_location("script", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


def section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


PIPELINE = [
    ("01 — Pull data",       [
        ("Moneyline",    "moneyline/01_input_moneyline.py"),
        ("Totals",       "totals/01_input_totals.py"),
        ("Hitter TB",    "hitter_tb/01_input_hitter_tb.py"),
        ("Pitcher Outs", "pitcher_outs/01_input_pitcher_outs.py"),
    ]),
    ("02 — Rebuild datasets", [
        ("Moneyline",    "moneyline/02_build_moneyline.py"),
        ("Totals",       "totals/02_build_totals.py"),
        ("Hitter TB",    "hitter_tb/02_build_hitter_tb.py"),
        ("Pitcher Outs", "pitcher_outs/02_build_pitcher_outs.py"),
    ]),
    ("03 — Retrain models", [
        ("Moneyline",    "moneyline/03_analysis_moneyline.py"),
        ("Totals",       "totals/03_analysis_totals.py"),
        ("Hitter TB",    "hitter_tb/03_analysis_hitter_tb.py"),
        ("Pitcher Outs", "pitcher_outs/03_analysis_pitcher_outs.py"),
    ]),
]

if __name__ == "__main__":
    start = datetime.now()
    print(f"\n{'=' * 70}")
    print(f"  WEEKLY RETRAIN — {start.strftime('%A, %B %d, %Y')}")
    print(f"{'=' * 70}")

    errors = []

    for stage_name, scripts in PIPELINE:
        section(stage_name.upper())
        for model_name, rel_path in scripts:
            path = os.path.join(BASE_DIR, rel_path)
            print(f"\n  >>> {model_name}")
            try:
                run_script(path)
            except Exception:
                msg = f"{stage_name} — {model_name}"
                print(f"\n  ERROR in {msg}:")
                traceback.print_exc()
                errors.append(msg)

    elapsed = (datetime.now() - start).seconds
    print(f"\n{'=' * 70}")
    print(f"  RETRAIN COMPLETE — {elapsed}s elapsed")
    if errors:
        print(f"  {len(errors)} error(s):")
        for e in errors:
            print(f"    ✗ {e}")
    else:
        print("  All 12 steps completed successfully.")
    print(f"{'=' * 70}\n")
