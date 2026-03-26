"""
=============================================================================
DAILY MASTER RUNNER — Runs all four model export files in sequence
=============================================================================
Usage:
    python run_daily.py

Runs in order:
    1. Moneyline edge report
    2. Totals (Over/Under) edge report
    3. Hitter total bases props
    4. Pitcher outs props

Each model runs independently — if one fails the others still run.
All output files are saved to the exports/ directory as usual.
=============================================================================
"""

import os
import sys
import argparse
import importlib.util
import traceback
import pandas as pd
from datetime import datetime, date, timedelta

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
today_str  = datetime.now().strftime("%Y%m%d")
run_str    = datetime.now().strftime("%H%M")
EXPORT_DIR = os.path.join(BASE_DIR, "exports", today_str)
os.makedirs(EXPORT_DIR, exist_ok=True)

sys.path.insert(0, BASE_DIR)

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--note", type=str, default=None,
                    help="Log a model change note for today's picks")
args, _ = parser.parse_known_args()


def load_export_module(subdir: str, filename: str):
    """
    Load a 04_export_*.py module by file path.
    Required because Python cannot import files whose names start with a digit.
    """
    path = os.path.join(BASE_DIR, subdir, filename)
    spec = importlib.util.spec_from_file_location(f"{subdir}_export", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# =============================================================================
# ERROR CHECKS
# =============================================================================

MIN_EDGES = {
    "Moneyline":    0.06,
    "Totals O/U":   0.06,
    "Hitter TB":    0.07,
    "Pitcher Outs": 0.07,
    "NRFI/YRFI":    0.06,
}

def run_error_checks(results: dict) -> list:
    """
    Sanity-check all model edge reports after a run.
    Returns a list of warning strings — empty means no issues.
    """
    warnings = []

    for name, report in results.items():
        if report is None:
            continue  # model skipped / no data — already logged during run

        min_edge  = MIN_EDGES.get(name, 0.06)
        value_bets = (
            report[report["is_value_bet"] == 1]
            if "is_value_bet" in report.columns
            else pd.DataFrame()
        )

        # ── Edge threshold compliance ──────────────────────────────────────
        if "edge" in value_bets.columns and not value_bets.empty:
            below = value_bets[
                pd.to_numeric(value_bets["edge"], errors="coerce") < min_edge
            ]
            if not below.empty:
                warnings.append(
                    f"⚠ {name}: {len(below)} value bet(s) below "
                    f"{min_edge*100:.0f}% edge threshold"
                )

        # ── EV% outlier — live line contamination signal ───────────────────
        for ev_col in ("ev_pct", "EV%", "ev"):
            if ev_col in value_bets.columns:
                outliers = value_bets[
                    pd.to_numeric(value_bets[ev_col], errors="coerce").abs() > 500
                ]
                if not outliers.empty:
                    warnings.append(
                        f"🚨 {name}: EV% > 500% on {len(outliers)} pick(s) "
                        f"— possible live line contamination"
                    )
                break

        # ── Model-specific checks ──────────────────────────────────────────
        if name == "Totals O/U":
            if "ou_line" in report.columns:
                fallback = report[
                    pd.to_numeric(report["ou_line"], errors="coerce") == 8.5
                ]
                if not fallback.empty:
                    warnings.append(
                        f"⚠ Totals: {len(fallback)} game(s) with ou_line=8.5 "
                        f"— odds join may have failed"
                    )

        elif name == "Hitter TB":
            # Duplicate picks for same player
            if "player_name" in value_bets.columns and not value_bets.empty:
                dupes = value_bets[value_bets["player_name"].duplicated(keep=False)]
                if not dupes.empty:
                    warnings.append(
                        f"⚠ Hitter TB: duplicate picks for "
                        f"{dupes['player_name'].unique().tolist()}"
                    )
            # E[TB] out of realistic range
            for col in ("expected_tb", "E_TB", "lambda_hat"):
                if col in report.columns:
                    vals = pd.to_numeric(report[col], errors="coerce")
                    bad  = report[(vals < 0.5) | (vals > 2.5)]
                    if not bad.empty:
                        warnings.append(
                            f"⚠ Hitter TB: {len(bad)} row(s) with {col} "
                            f"outside 0.5–2.5 range"
                        )
                    break

        elif name == "Pitcher Outs":
            # Both starters from the same game both flagged — model bias signal
            if "game" in value_bets.columns and not value_bets.empty:
                dual = value_bets["game"].value_counts()
                dual = dual[dual > 1]
                if not dual.empty:
                    warnings.append(
                        f"⚠ Pitcher Outs: both starters flagged in "
                        f"{len(dual)} game(s) — possible model bias: "
                        f"{dual.index.tolist()}"
                    )
            # CSW% scaling bug — should be ~0.25–0.35, not 25–35
            for col in ("csw_pct", "CSW%", "csw"):
                if col in report.columns:
                    scaled = report[pd.to_numeric(report[col], errors="coerce") > 10]
                    if not scaled.empty:
                        warnings.append(
                            f"🚨 Pitcher Outs: CSW% looks like a percentage "
                            f"(>10) in {len(scaled)} row(s) — scaling bug"
                        )
                    break

    return warnings


# =============================================================================
# MODEL RUNNERS
# =============================================================================

def run_moneyline():
    from utils.action_network import get_moneyline_odds
    m = load_export_module("moneyline", "04_export_moneyline.py")

    odds_df = get_moneyline_odds(date.today().strftime("%Y-%m-%d"))
    if odds_df.empty:
        odds_df = m.fetch_odds_api(m.ODDS_API_KEY) if m.ODDS_API_KEY \
                  else m.load_manual_odds()
    if odds_df.empty:
        print("  No games found — skipping moneyline.")
        return None

    predictions_df = m.score_live_games(odds_df)
    if predictions_df.empty:
        return None

    edge_report = m.build_edge_report(predictions_df, odds_df)
    edge_report.to_excel(
        os.path.join(EXPORT_DIR, f"moneyline_edges_{today_str}_{run_str}.xlsx"),
        index=False, engine="openpyxl"
    )
    value_bets = edge_report[edge_report["is_value_bet"] == 1]
    if not value_bets.empty:
        value_bets.to_excel(
            os.path.join(EXPORT_DIR, f"moneyline_plays_{today_str}_{run_str}.xlsx"),
            index=False, engine="openpyxl"
        )
    print(f"  ✓ {len(edge_report)} matchups | {len(value_bets)} value bets")
    return edge_report


def run_totals():
    from utils.action_network import get_totals_odds
    m = load_export_module("totals", "04_export_totals.py")

    odds_df = get_totals_odds(date.today().strftime("%Y-%m-%d"))
    if odds_df.empty:
        odds_df = m.fetch_totals_odds_api(m.ODDS_API_KEY) if m.ODDS_API_KEY \
                  else m.load_manual_totals_odds()
    if odds_df.empty:
        print("  No games found — skipping totals.")
        return None

    predictions_df = m.score_live_games_totals(odds_df)
    if predictions_df.empty:
        return None

    edge_report = m.build_totals_edge_report(predictions_df, odds_df)
    edge_report.to_excel(
        os.path.join(EXPORT_DIR, f"totals_edges_{today_str}_{run_str}.xlsx"),
        index=False, engine="openpyxl"
    )
    value_bets = edge_report[edge_report["is_value_bet"] == 1]
    if not value_bets.empty:
        value_bets.to_excel(
            os.path.join(EXPORT_DIR, f"totals_plays_{today_str}_{run_str}.xlsx"),
            index=False, engine="openpyxl"
        )
    print(f"  ✓ {len(edge_report)} O/U bets | {len(value_bets)} value bets")
    return edge_report


def run_hitter_tb():
    from utils.action_network import get_hitter_tb_odds
    m = load_export_module("hitter_tb", "04_export_hitter_tb.py")

    predictions_df = m.score_live_hitters()
    if predictions_df is None or predictions_df.empty:
        print("  No hitters scored — lineups may not be posted yet.")
        return None

    odds_df = get_hitter_tb_odds(date.today().strftime("%Y-%m-%d"))
    if odds_df.empty:
        manual_path = os.path.join(BASE_DIR, "data", "processed",
                                   "hitter_tb_odds_today.csv")
        if os.path.exists(manual_path):
            odds_df = m.load_prop_odds_manual(manual_path)
        else:
            print("  No prop odds available — skipping hitter TB edge calculation.")
            print("  (Check AN token or manually supply hitter_tb_odds_today.csv)")
            return None

    edge_report = m.build_tb_edge_report(predictions_df, odds_df)
    print(m.generate_daily_summary(edge_report))

    edge_report.to_excel(
        os.path.join(EXPORT_DIR, f"hitter_tb_edges_{today_str}_{run_str}.xlsx"),
        index=False, engine="openpyxl"
    )
    value_bets = edge_report[edge_report["is_value_bet"] == 1]
    if not value_bets.empty:
        value_bets.to_excel(
            os.path.join(EXPORT_DIR, f"hitter_tb_plays_{today_str}_{run_str}.xlsx"),
            index=False, engine="openpyxl"
        )
    print(f"  ✓ {len(edge_report)} props | {len(value_bets)} value bets")
    return edge_report


def run_pitcher_outs():
    from utils.action_network import get_pitcher_outs_odds
    m = load_export_module("pitcher_outs", "04_export_pitcher_outs.py")

    predictions_df = m.score_live_pitchers()
    if predictions_df is None or predictions_df.empty:
        print("  No probable starters found — skipping pitcher outs.")
        return None

    odds_df = get_pitcher_outs_odds(date.today().strftime("%Y-%m-%d"))
    if odds_df.empty:
        manual_path = os.path.join(BASE_DIR, "data", "processed",
                                   "pitcher_outs_odds_today.csv")
        if os.path.exists(manual_path):
            odds_df = m.load_prop_odds_manual(manual_path)
        else:
            print("  No prop odds available — skipping pitcher outs edge calculation.")
            print("  (Check AN token or manually supply pitcher_outs_odds_today.csv)")
            return None

    edge_report = m.build_pitcher_edge_report(predictions_df, odds_df)
    print(m.generate_daily_summary(edge_report))

    edge_report.to_excel(
        os.path.join(EXPORT_DIR, f"pitcher_outs_edges_{today_str}_{run_str}.xlsx"),
        index=False, engine="openpyxl"
    )
    value_bets = edge_report[edge_report["is_value_bet"] == 1]
    if not value_bets.empty:
        value_bets.to_excel(
            os.path.join(EXPORT_DIR, f"pitcher_outs_plays_{today_str}_{run_str}.xlsx"),
            index=False, engine="openpyxl"
        )
    print(f"  ✓ {len(edge_report)} props | {len(value_bets)} value bets")
    return edge_report


def run_nrfi():
    m = load_export_module("nrfi", "04_export_nrfi.py")
    nrfi_date = datetime.strptime(today_str, "%Y%m%d").strftime("%Y-%m-%d")
    df = m.run_nrfi_export(target_date=nrfi_date, verbose=True)
    if df is None or df.empty:
        print("  No NRFI/YRFI games scored — skipping.")
        return None
    picks = df[df.get("bet_side", pd.Series()) != ""] if "bet_side" in df.columns else df
    print(f"  ✓ {len(df)} games scored | {len(picks)} picks")
    return df


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    start = datetime.now()
    print(f"\n{'=' * 70}")
    print(f"  DAILY MODEL RUN — {start.strftime('%A, %B %d, %Y')}")
    print(f"{'=' * 70}")

    from utils.tracker import save_picks, print_performance_summary, log_model_change

    MODELS = [
        ("Moneyline",    run_moneyline),
        ("Totals O/U",   run_totals),
        ("Hitter TB",    run_hitter_tb),
        ("Pitcher Outs", run_pitcher_outs),
        ("NRFI/YRFI",    run_nrfi),
    ]

    results = {}
    for name, fn in MODELS:
        section(name.upper())
        try:
            results[name] = fn()
        except Exception:
            print(f"\n  ERROR running {name}:")
            traceback.print_exc()
            results[name] = None

    # ── Error checks ──────────────────────────────────────────────────────
    section("ERROR CHECKS")
    error_warnings = run_error_checks(results)
    if error_warnings:
        print(f"\n  {len(error_warnings)} issue(s) detected:")
        for w in error_warnings:
            print(f"    {w}")
        try:
            from utils.notifier import notify_run_status
            notify_run_status("⚠️ Model Run — Issues Detected", error_warnings)
        except Exception as e:
            print(f"  WARNING: error notification failed — {e}")
    else:
        print("  ✓ All checks passed — no issues detected.")

    # ── Log model change note if provided ─────────────────────────────────
    if args.note:
        log_model_change(args.note)

    # ── Save today's picks for future grading ─────────────────────────────
    save_picks(today_str, results)

    # ── Send Discord picks ─────────────────────────────────────────────────
    from utils.notifier import send_daily_picks
    from utils.probable_starters import get_todays_game_status
    scored_games, pending_games = get_todays_game_status()
    try:
        send_daily_picks(results, today_str, scored_games, pending_games)
    except Exception as e:
        print(f"  WARNING: notification failed — {e}")

    # ── Combined Excel workbook ────────────────────────────────────────────
    import pandas as pd
    combined_path = os.path.join(EXPORT_DIR, f"daily_report_{today_str}_{run_str}.xlsx")
    TAB_NAMES = {
        "Moneyline":    "Moneyline",
        "Totals O/U":   "Totals",
        "Hitter TB":    "Hitter TB",
        "Pitcher Outs": "Pitcher Outs",
        "NRFI/YRFI":    "NRFI-YRFI",
    }
    with pd.ExcelWriter(combined_path, engine="openpyxl") as writer:
        for name, report in results.items():
            sheet = TAB_NAMES[name]
            if report is not None and not report.empty:
                report.to_excel(writer, sheet_name=sheet, index=False)
            else:
                pd.DataFrame({"status": ["No data"]}).to_excel(
                    writer, sheet_name=sheet, index=False
                )

    # ── Run-status notification to results channel ────────────────────────
    try:
        from utils.notifier import notify_run_status
        elapsed_for_notif = (datetime.now() - start).seconds
        mins, secs = divmod(elapsed_for_notif, 60)
        elapsed_str = f"{mins}m {secs}s" if mins else f"{secs}s"
        status_lines = []
        for name, report in results.items():
            if report is None:
                status_lines.append(f"⊘ {name}: skipped / no data")
            else:
                n = int((report["is_value_bet"] == 1).sum()) \
                    if "is_value_bet" in report.columns else "?"
                status_lines.append(f"✓ {name}: {n} value bet(s)")
        status_lines.append(f"\n⏱ {elapsed_str}")
        notify_run_status("✅ T-90 Models Complete", status_lines)
    except Exception as e:
        print(f"  WARNING: run-status notification failed — {e}")

    # ── Final summary ─────────────────────────────────────────────────────
    elapsed = (datetime.now() - start).seconds
    print(f"\n{'=' * 70}")
    print(f"  COMPLETE — {elapsed}s elapsed")
    print(f"  Combined report: {combined_path}")
    print(f"{'=' * 70}")
    print(f"  {'Model':<16} Result")
    print(f"  {'-' * 36}")
    for name, report in results.items():
        if report is None:
            status = "skipped / no data"
        else:
            n = int((report["is_value_bet"] == 1).sum()) \
                if "is_value_bet" in report.columns else "?"
            status = f"✓  {n} value bet(s)"
        print(f"  {name:<16} {status}")
    print(f"{'=' * 70}\n")

    print_performance_summary(n_days=30)

    # ── Push exports and picks to GitHub for monitoring agents ────────────────
    import subprocess
    try:
        subprocess.run(
            ["git", "add", f"exports/{today_str}/", "tracking/picks.xlsx"],
            cwd=BASE_DIR, check=True, capture_output=True
        )
        result = subprocess.run(
            ["git", "commit", "-m", f"Daily picks: {today_str}"],
            cwd=BASE_DIR, capture_output=True
        )
        if result.returncode == 0:
            subprocess.run(["git", "push"], cwd=BASE_DIR, check=True, capture_output=True)
            print("  ✓ Results pushed to GitHub.")
        elif b"nothing to commit" in result.stdout + result.stderr:
            print("  (git) Nothing new to commit.")
        else:
            result.check_returncode()
    except Exception as e:
        print(f"  WARNING: git push failed — {e}")
