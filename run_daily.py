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
EXPORT_DIR = os.path.join(BASE_DIR, "exports")
today_str  = datetime.now().strftime("%Y%m%d")

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
        os.path.join(EXPORT_DIR, f"moneyline_edges_{today_str}.xlsx"),
        index=False, engine="openpyxl"
    )
    value_bets = edge_report[edge_report["is_value_bet"] == 1]
    if not value_bets.empty:
        value_bets.to_excel(
            os.path.join(EXPORT_DIR, f"moneyline_plays_{today_str}.xlsx"),
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

    import pandas as pd
    predictions_df = predictions_df.merge(
        odds_df[["home_team", "away_team", "ou_line"]].drop_duplicates(
            ["home_team", "away_team"]
        ),
        on=["home_team", "away_team"], how="left"
    )
    edge_report = m.build_totals_edge_report(predictions_df, odds_df)
    edge_report.to_excel(
        os.path.join(EXPORT_DIR, f"totals_edges_{today_str}.xlsx"),
        index=False, engine="openpyxl"
    )
    value_bets = edge_report[edge_report["is_value_bet"] == 1]
    if not value_bets.empty:
        value_bets.to_excel(
            os.path.join(EXPORT_DIR, f"totals_plays_{today_str}.xlsx"),
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
        odds_df = m.load_prop_odds_manual(manual_path)

    edge_report = m.build_tb_edge_report(predictions_df, odds_df)
    print(m.generate_daily_summary(edge_report))

    edge_report.to_excel(
        os.path.join(EXPORT_DIR, f"hitter_tb_edges_{today_str}.xlsx"),
        index=False, engine="openpyxl"
    )
    value_bets = edge_report[edge_report["is_value_bet"] == 1]
    if not value_bets.empty:
        value_bets.to_excel(
            os.path.join(EXPORT_DIR, f"hitter_tb_plays_{today_str}.xlsx"),
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
        odds_df = m.load_prop_odds_manual(manual_path)

    edge_report = m.build_pitcher_edge_report(predictions_df, odds_df)
    print(m.generate_daily_summary(edge_report))

    edge_report.to_excel(
        os.path.join(EXPORT_DIR, f"pitcher_outs_edges_{today_str}.xlsx"),
        index=False, engine="openpyxl"
    )
    value_bets = edge_report[edge_report["is_value_bet"] == 1]
    if not value_bets.empty:
        value_bets.to_excel(
            os.path.join(EXPORT_DIR, f"pitcher_outs_plays_{today_str}.xlsx"),
            index=False, engine="openpyxl"
        )
    print(f"  ✓ {len(edge_report)} props | {len(value_bets)} value bets")
    return edge_report


def run_nrfi():
    m = load_export_module("nrfi", "04_export_nrfi.py")
    df = m.run_nrfi_export(target_date=today_str, verbose=True)
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

    # ── Log model change note if provided ─────────────────────────────────
    if args.note:
        log_model_change(args.note)

    # ── Save today's picks for future grading ─────────────────────────────
    save_picks(today_str, results)

    # ── Send SMS summary ───────────────────────────────────────────────────
    from utils.notifier import send_daily_picks
    from utils.probable_starters import get_todays_game_status
    scored_games, pending_games = get_todays_game_status()
    send_daily_picks(results, today_str, scored_games, pending_games)

    # ── Combined Excel workbook ────────────────────────────────────────────
    import pandas as pd
    combined_path = os.path.join(EXPORT_DIR, f"daily_report_{today_str}.xlsx")
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
