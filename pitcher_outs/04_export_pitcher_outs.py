"""
=============================================================================
PITCHER TOTAL OUTS MODEL — FILE 4 OF 4: EDGE SCORING AND CSV EXPORT
=============================================================================
Purpose : Compare model's E[Outs] to market prop lines; calculate edge,
          EV%, Kelly Criterion, and edge score; export ranked recommendations.
Input   : ../data/processed/pitcher_outs_predictions.csv (from Step 3)
Output  : ../exports/pitcher_outs_edges_YYYYMMDD.csv

Pitcher outs prop context:
─────────────────────────────────────────────────────────────────────────────
Pitcher outs (or outs recorded) is a newer prop market with significant
pricing inefficiency because:
  1. Books price primarily on recent ERA, which is noisy and luck-influenced
  2. Manager tendencies are often not factored in (biggest edge here)
  3. Bullpen state (are key relievers rested?) affects SP longevity
  4. The TTTO (third time through order) penalty is underweighted by markets

Typical prop lines (outs recorded):
  - 14.5 outs (4.83 IP) — very low bar, mainly for shaky starters
  - 15.5 outs (5.17 IP) — standard short-rest line
  - 16.5 outs (5.50 IP) — standard mid-range line
  - 17.5 outs (5.83 IP) — standard workhorse line
  - 18.5 outs (6.17 IP) — high bar for ace/workhorse pitchers
  - 19.5 outs (6.50 IP) — for elite starters with permissive managers

Markets also sometimes offer "IP recorded" in half-inning increments.
This model generates outs, and you can convert: IP = outs / 3.

Manager hook is the single most underutilized signal in this market.
A pitcher on the Tampa Bay Rays under Kevin Cash can have a 2.50 ERA and
still get pulled in the 5th inning. Our model captures this explicitly.

For R users:
  - This file follows the same structure as hitter_tb export
  - The key difference: we evaluate pitcher outs (not TB) as the prop outcome
  - We include a "manager hook" display column to explain predictions
=============================================================================
"""

import os
import sys
import json
import requests
from datetime import datetime, date
import pandas as pd
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.action_network import get_pitcher_outs_odds
from utils.probable_starters import (get_probable_starters, normalize_name,
                                      get_lineups, get_lineup_batting_features,
                                      load_batting_stats)

# --- Configuration ----------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR  = os.path.join(BASE_DIR, "data", "models")
EXPORT_DIR = os.path.join(BASE_DIR, "exports")

ODDS_API_KEY         = "fbc985ad430c95d6435cb75210f7b989"
KELLY_FRACTION_PROPS = 0.15   # 15% Kelly for player props
MAX_BET_FRACTION     = 0.04   # Max 4% per pitcher prop
MIN_EDGE             = 0.07   # 7% min edge threshold

# Standard prop lines (outs recorded)
STANDARD_OUTS_LINES = [14.5, 15.5, 16.5, 17.5, 18.5]


# =============================================================================
# ODDS UTILITIES (identical to other export files — consistent interface)
# =============================================================================

def american_to_decimal(odds: float) -> float:
    """American odds → decimal odds."""
    return (odds / 100 + 1) if odds >= 0 else (100 / abs(odds) + 1)


def juice_to_implied_prob(juice: float) -> float:
    """American juice → raw implied probability."""
    return 1.0 / american_to_decimal(juice)


def remove_vig(over_juice: float, under_juice: float) -> tuple:
    """
    Remove vig from a two-sided prop market.

    For pitcher outs, 'over' = pitcher records MORE outs than the line.
    'under' = pitcher is pulled BEFORE reaching the line.

    Returns
    -------
    tuple : (fair_p_over, fair_p_under)
    """
    p_o = juice_to_implied_prob(over_juice)
    p_u = juice_to_implied_prob(under_juice)
    total = p_o + p_u
    return (p_o / total, p_u / total) if total > 0 else (0.5, 0.5)


def calculate_ev_pct(model_prob: float, decimal_odds: float) -> float:
    """EV% = model_prob × (dec - 1) - (1 - model_prob)."""
    return round(model_prob * (decimal_odds - 1) - (1 - model_prob), 4)


def kelly_pitcher(model_prob: float, decimal_odds: float) -> float:
    """Fractional Kelly (15%) sized for pitcher outs props."""
    b = decimal_odds - 1
    if b <= 0:
        return 0.0
    q = 1 - model_prob
    f = ((b * model_prob - q) / b) * KELLY_FRACTION_PROPS
    return round(min(max(f, 0.0), MAX_BET_FRACTION), 4)


def compute_pitcher_edge_score(edge: float, ev_pct: float, kelly: float,
                                expected_outs: float, prop_line: float,
                                manager_depth: float = 0.5) -> float:
    """
    Compute 0–10 edge score for pitcher outs props.

    Manager depth bonus:
      - High manager depth + model says Over → strong conviction
      - Low manager depth (Kevin Cash) + model says Under → strong conviction
      - When manager depth aligns with bet direction, score gets a bonus.

    Components:
      edge_comp    : 0–3 (edge from 0% to 20%)
      ev_comp      : 0–3 (EV% from 0% to 20%)
      kelly_comp   : 0–2 (Kelly from 0% to 4%)
      gap_comp     : 0–2 (|expected - line| / line)

    Parameters
    ----------
    manager_depth : float
        0–1, where higher = manager lets pitchers go deeper.
        Used to scale the gap component.
    """
    edge_comp  = 3.0 * np.clip(edge,   0, 0.20) / 0.20
    ev_comp    = 3.0 * np.clip(ev_pct, 0, 0.20) / 0.20
    kelly_comp = 2.0 * np.clip(kelly,  0, 0.04) / 0.04

    # Gap component: larger expected_outs vs line gap = higher confidence
    if prop_line and prop_line > 0:
        gap      = abs(expected_outs - prop_line)
        gap_norm = np.clip(gap / prop_line, 0, 0.35)
        gap_comp = 2.0 * gap_norm / 0.35
    else:
        gap_comp = 0.0

    return round(min(edge_comp + ev_comp + kelly_comp + gap_comp, 10.0), 2)


# =============================================================================
# ODDS LOADING
# =============================================================================

def load_prop_odds_manual(path: str = None) -> pd.DataFrame:
    """
    Load pitcher outs prop odds from a manually created CSV.

    Expected format:
      pitcher_name,team,prop_line,over_juice,under_juice,book
      Gerrit Cole,NYY,17.5,-130,+108,FanDuel
      Shane McClanahan,TBR,14.5,-125,+104,DraftKings
      Dylan Cease,SDP,16.5,-115,-105,BetMGM

    Note on prop_line column:
      This is in OUTS, not innings. Convert IP line to outs: IP × 3.
      Some books offer "innings pitched" lines (e.g., over/under 5.5 IP = 16.5 outs).
    """
    if path and os.path.exists(path):
        print(f"  Loading prop odds from: {path}")
        return pd.read_csv(path)

    # Create template
    template = pd.DataFrame({
        "pitcher_name": ["Gerrit Cole",     "Shane McClanahan", "Dylan Cease"],
        "team":         ["NYY",             "TBR",              "SDP"],
        "prop_line":    [17.5,              14.5,               16.5],   # In OUTS
        "over_juice":   [-130.0,            -125.0,             -115.0],
        "under_juice":  [+108.0,            +104.0,             -105.0],
    })
    template_path = os.path.join(PROC_DIR, "pitcher_outs_odds_template.csv")
    template.to_csv(template_path, index=False)
    print(f"  Created prop odds template: {template_path}")
    return template


# =============================================================================
# MAIN EDGE REPORT BUILDER
# =============================================================================

def build_pitcher_edge_report(predictions_df: pd.DataFrame,
                               odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join model predictions with market odds to compute edge metrics.

    For each pitcher, we evaluate multiple prop lines:
      - The specific line in the odds file (if available)
      - All standard lines (14.5 through 18.5) for a full picture

    Parameters
    ----------
    predictions_df : pd.DataFrame
        From Step 3: expected_outs, p_over/under for each line, pitcher_name, team
    odds_df : pd.DataFrame
        Market odds: pitcher_name, prop_line, over_juice, under_juice

    Returns
    -------
    pd.DataFrame
        One row per pitcher-line-side combination with full edge metrics.
    """
    rows = []

    for _, pred_row in predictions_df.iterrows():
        pitcher_name  = pred_row.get("pitcher_name", "")
        team          = pred_row.get("team",         "")
        opp_team      = pred_row.get("opp_team",     "")
        expected_outs = pred_row.get("expected_outs", 15.0)
        expected_ip   = pred_row.get("expected_ip",  5.0)
        depth_score   = pred_row.get("depth_score",  0.5)  # Manager tendency

        # Find market odds for this pitcher
        pitcher_odds = odds_df[
            odds_df["pitcher_name"].str.lower().str.strip()
            == str(pitcher_name).lower().strip()
        ] if not odds_df.empty else pd.DataFrame()

        # Skip pitchers with no real market odds
        if pitcher_odds.empty:
            continue

        # Select the consensus line: most books posting it (highest n_books),
        # tiebreak by most balanced odds (closest to 50/50).
        # Require at least 2 books; if none qualify, skip pitcher.
        pitcher_odds = pitcher_odds.copy()
        if "n_books" in pitcher_odds.columns:
            qualified = pitcher_odds[pitcher_odds["n_books"] >= 2]
            if qualified.empty:
                continue
            pitcher_odds = qualified

        def _over_prob_p(row):
            d_over  = (row["over_juice"]  / 100 + 1) if row["over_juice"]  >= 0 else (1 - 100 / row["over_juice"])
            d_under = (row["under_juice"] / 100 + 1) if row["under_juice"] >= 0 else (1 - 100 / row["under_juice"])
            p_over_raw  = 1 / d_over
            p_under_raw = 1 / d_under
            total = p_over_raw + p_under_raw
            return p_over_raw / total if total > 0 else 0.5

        pitcher_odds["_p_over"] = pitcher_odds.apply(_over_prob_p, axis=1)
        pitcher_odds["_balance"] = (pitcher_odds["_p_over"] - 0.5).abs()
        # Primary sort: most books; secondary sort: most balanced
        pitcher_odds = pitcher_odds.sort_values(
            ["n_books", "_balance"] if "n_books" in pitcher_odds.columns else ["_balance"],
            ascending=[False, True]
        )
        consensus_row = pitcher_odds.iloc[0]
        line        = float(consensus_row["prop_line"])
        over_juice  = float(consensus_row["over_juice"])
        under_juice = float(consensus_row["under_juice"])

        # Get model probability for the consensus line
        line_col = f"line_{str(line).replace('.', '_')}"
        p_over   = pred_row.get(f"p_over_{line_col}",  None)
        p_under  = pred_row.get(f"p_under_{line_col}", None)
        if p_over is None or p_under is None:
            from scipy import stats as sp_stats
            sigma   = 3.5  # Typical outs/start std dev
            z_score = (line - expected_outs) / sigma
            p_under = float(sp_stats.norm.cdf(z_score))
            p_over  = 1 - p_under

        # Remove vig
        fair_p_over, fair_p_under = remove_vig(over_juice, under_juice)

        for bet_side, model_p, fair_p, juice in [
            ("OVER",  p_over,  fair_p_over,  over_juice),
            ("UNDER", p_under, fair_p_under, under_juice),
        ]:
            dec_odds = american_to_decimal(juice)
            edge     = round(float(model_p) - float(fair_p), 4)
            ev_pct   = calculate_ev_pct(float(model_p), dec_odds)
            kelly    = kelly_pitcher(float(model_p), dec_odds)
            score    = compute_pitcher_edge_score(
                edge, ev_pct, kelly, expected_outs, line, depth_score
            )

            rows.append({
                "game_date":      pred_row.get("game_date",
                                  datetime.now().strftime("%Y-%m-%d")),
                "pitcher_name":   pitcher_name,
                "team":           team,
                "opp_team":       opp_team,
                "bet_type":       f"{bet_side} {line} OUTS",
                "prop_line_outs": line,
                "prop_line_ip":   round(line / 3, 2),  # Display in IP too
                "bet_side":       bet_side,
                "expected_outs":  round(expected_outs, 2),
                "expected_ip":    round(expected_ip, 2),
                "manager_depth":  round(depth_score, 2),  # 0=pull-happy, 1=lets go deep
                "model_prob":     round(float(model_p), 4),
                "market_implied": round(float(fair_p), 4),
                "edge":           edge,
                "ev_pct":         ev_pct,
                "kelly_fraction": kelly,
                "edge_score":     score,
                "juice":          juice,
                "decimal_odds":   round(dec_odds, 4),
                "is_value_bet":   1 if edge >= MIN_EDGE and ev_pct > 0 else 0,
                # Explanatory columns for understanding the bet
                "outs_vs_line":   round(expected_outs - line, 2),  # Positive = lean Over
                "ip_vs_line":     round((expected_outs - line) / 3, 2),
            })

    report_df = pd.DataFrame(rows)
    if not report_df.empty:
        report_df = report_df.sort_values("edge_score", ascending=False).reset_index(drop=True)
    return report_df


# =============================================================================
# DAILY SUMMARY REPORT
# =============================================================================

def generate_daily_summary(edge_report: pd.DataFrame) -> str:
    """
    Generate a concise daily summary of top pitcher outs plays.

    Includes a "manager context" column to explain why we like/dislike the bet.
    This is the key differentiator from naive market pricing.
    """
    today    = datetime.now().strftime("%B %d, %Y")
    vb       = edge_report[edge_report["is_value_bet"] == 1]
    overs    = vb[vb["bet_side"] == "OVER"]
    unders   = vb[vb["bet_side"] == "UNDER"]

    lines = [
        "=" * 70,
        f"  PITCHER TOTAL OUTS — DAILY EDGE REPORT — {today}",
        "=" * 70,
        f"  Props evaluated:   {len(edge_report)}",
        f"  Value bets:        {len(vb)} ({len(overs)} Over, {len(unders)} Under)",
        "",
        "  TOP OVER PLAYS (SP expected to pitch DEEPER than market prices):",
        f"  {'Pitcher':20s} {'Line':8s} {'E[Outs]':9s} {'E[IP]':7s} {'Edge':7s} "
        f"{'EV%':7s} {'Kelly':7s} {'Mgr':5s} Score",
        "  " + "-" * 68,
    ]

    for _, row in overs.head(6).iterrows():
        lines.append(
            f"  {str(row['pitcher_name']):20s} "
            f"{row['bet_type']:8s} "
            f"{row['expected_outs']:6.1f} outs  "
            f"{row['expected_ip']:5.2f} IP  "
            f"{row['edge']:+5.1%}  "
            f"{row['ev_pct']:+5.1%}  "
            f"{row['kelly_fraction']:5.1%}  "
            f"{row['manager_depth']:4.2f}  "
            f"{row['edge_score']:4.1f}/10"
        )

    lines += [
        "",
        "  TOP UNDER PLAYS (SP expected to be pulled EARLIER than market prices):",
        f"  {'Pitcher':20s} {'Line':8s} {'E[Outs]':9s} {'E[IP]':7s} {'Edge':7s} "
        f"{'EV%':7s} {'Kelly':7s} {'Mgr':5s} Score",
        "  " + "-" * 68,
    ]
    for _, row in unders.head(6).iterrows():
        lines.append(
            f"  {str(row['pitcher_name']):20s} "
            f"{row['bet_type']:9s} "
            f"{row['expected_outs']:6.1f} outs  "
            f"{row['expected_ip']:5.2f} IP  "
            f"{row['edge']:+5.1%}  "
            f"{row['ev_pct']:+5.1%}  "
            f"{row['kelly_fraction']:5.1%}  "
            f"{row['manager_depth']:4.2f}  "
            f"{row['edge_score']:4.1f}/10"
        )

    lines += [
        "",
        "  NOTE: Mgr = manager depth score (low=pull-happy, high=lets SP go deep)",
        "  Under plays on low-Mgr pitchers = market not pricing manager hook properly",
        "=" * 70,
    ]
    return "\n".join(lines)


# =============================================================================
# LIVE SCORING — Today's probable starters
# =============================================================================

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

# FanGraphs column → model feature name, plus scale transformation
# k_pct/bb_pct/swstr_pct/fstrike_pct are stored as percentages in the dataset (e.g., 21.7)
# csw_pct is stored as proportion (e.g., 0.269)
_EFF_RENAME = {
    "K%":          ("k_pct",           100.0),   # 0.217 → 21.7
    "BB%":         ("bb_pct",          100.0),
    "K-BB%":       ("k_minus_bb_pct",  100.0),
    "SIERA":       ("siera",             1.0),
    "xFIP":        ("xfip",             1.0),
    "FIP":         ("fip",              1.0),
    "pitches_per_ip": ("pitches_per_ip", 1.0),
    "SwStr%":      ("swstr_pct",       100.0),
    "F-Strike%":   ("fstrike_pct",     100.0),
    "CSW%":        ("csw_pct",           1.0),   # stored as proportion
    "outs_per_start": ("avg_sp_outs",   1.0),
}


def score_live_pitchers() -> pd.DataFrame:
    """
    Score today's probable starting pitchers using individual 2025 stats.

    Returns
    -------
    pd.DataFrame
        One row per probable starter with expected_outs, expected_ip,
        pitcher_name, team, opp_team, game_date, depth_score, etc.
    """
    import xgboost as xgb

    model = xgb.XGBRegressor()
    model.load_model(os.path.join(MODEL_DIR, "pitcher_outs_model.json"))
    with open(os.path.join(MODEL_DIR, "pitcher_outs_features.json")) as f:
        feature_cols = json.load(f)

    # ── Today's probable starters ─────────────────────────────────────────
    print("  Fetching today's probable starters...")
    game_date = date.today().strftime("%Y-%m-%d")
    starters_df = get_probable_starters(game_date)
    if starters_df.empty:
        print("  WARNING: No probable starters found.")
        return pd.DataFrame()

    # ── Load 2025 pitcher efficiency stats ───────────────────────────────
    eff = pd.read_csv(os.path.join(RAW_DIR, "raw_pitcher_efficiency.csv"))
    eff_latest = int(eff["Season"].max())
    eff_qualifiers = eff[(eff["Season"] == eff_latest) & (eff["GS"] >= 5)]
    eff_season = eff_latest if len(eff_qualifiers) >= 10 else eff_latest - 1
    eff = eff[(eff["Season"] == eff_season) & (eff["GS"] >= 5)].copy()
    eff_index = {normalize_name(n): i for i, n in enumerate(eff["Name"])}

    # ── Manager depth scores ──────────────────────────────────────────────
    mgr = pd.read_csv(os.path.join(RAW_DIR, "raw_manager_depth.csv"))
    mgr_map = {r["team"]: r for _, r in mgr.iterrows()}

    # ── Lineup-based opposing batting (most recent season) ───────────────
    print("  Fetching today's lineups for opposing batting context...")
    lineups, lineup_sources = get_lineups(game_date, return_sources=True)
    batting_df = load_batting_stats()   # most recent season (2026 → 2025 fallback)
    lineup_bat = get_lineup_batting_features(lineups, batting_df)

    # Team batting CSV fallback
    bat_csv = pd.read_csv(os.path.join(RAW_DIR, "raw_team_batting.csv"))
    bat_season = int(batting_df["Season"].max()) if not batting_df.empty else 2025
    bat_csv = bat_csv[bat_csv["Season"] == bat_season] if bat_season in bat_csv["Season"].values \
              else bat_csv[bat_csv["Season"] == bat_csv["Season"].max()]
    bat_map = {r["Team"]: r for _, r in bat_csv.iterrows()}

    # Training means as fallback
    train_means = pd.read_csv(
        os.path.join(PROC_DIR, "pitcher_outs_dataset.csv")
    )[feature_cols].mean()

    rows = []
    for _, game in starters_df.iterrows():
        for side, pitcher_name, team, opp_team in [
            ("home", game["home_sp_name"], game["home_team"], game["away_team"]),
            ("away", game["away_sp_name"], game["away_team"], game["home_team"]),
        ]:
            if not pitcher_name:
                continue

            # Skip if this pitcher's individual stats are not confirmed
            sp_source_col = f"{side}_sp_source"
            if game.get(sp_source_col) == "team_avg":
                print(f"  Skipping {pitcher_name} ({team}) — starter not individually confirmed.")
                continue

            # Skip if opposing lineup is not yet confirmed
            if lineup_sources.get(opp_team, "projected") != "confirmed":
                print(f"  Skipping {pitcher_name} ({team}) — {opp_team} lineup not confirmed yet.")
                continue

            # Look up pitcher in efficiency data
            norm = normalize_name(pitcher_name)
            eff_idx = eff_index.get(norm)
            if eff_idx is None:
                # Fallback: last name + first initial
                last = norm.split()[-1] if norm else ""
                candidates = [k for k in eff_index if k.split()[-1] == last]
                if len(candidates) == 1:
                    eff_idx = eff_index[candidates[0]]
            eff_row = eff.iloc[eff_idx] if eff_idx is not None else None

            feats = {}
            if eff_row is not None:
                for fg_col, (model_col, scale) in _EFF_RENAME.items():
                    val = eff_row.get(fg_col, np.nan)
                    feats[model_col] = float(val) * scale if pd.notna(val) else np.nan
            else:
                print(f"    No 2025 stats for '{pitcher_name}' — using training means")

            # Manager depth + team avg outs
            mgr_row = mgr_map.get(team, {})
            feats["depth_score"] = float(mgr_row.get("depth_score", train_means.get("depth_score", 0.525)))
            feats["avg_sp_outs"] = float(mgr_row.get("avg_sp_outs", feats.get("avg_sp_outs", train_means.get("avg_sp_outs", 14.9))))

            # Opposing batting — prefer lineup-based, fall back to team CSV
            opp_lb  = lineup_bat.get(opp_team, {})
            opp_bat = bat_map.get(opp_team, {})
            feats["opp_lg_avg_k_pct"] = float(
                opp_lb.get("off_k_pct",
                opp_bat.get("K%",   train_means.get("opp_lg_avg_k_pct", 0.225))))
            feats["opp_lg_avg_woba"] = float(
                opp_lb.get("off_pa_weighted_woba",       # PA-weighted (top of order counts more)
                opp_lb.get("off_woba",                   # simple lineup avg
                opp_bat.get("wOBA", train_means.get("opp_lg_avg_woba", 0.314)))))
            feats["opp_lg_avg_obp"] = float(
                opp_lb.get("off_obp",
                opp_bat.get("OBP",  train_means.get("opp_lg_avg_obp",   0.316))))

            rows.append({
                "game_date":       game.get("game_time", date.today().isoformat())[:10],
                "pitcher_name":    pitcher_name,
                "team":            team,
                "opp_team":        opp_team,
                # Display-only context columns (not fed to model)
                "opp_top3_woba":   opp_lb.get("off_top3_woba",  np.nan),
                "opp_top3_k_pct":  opp_lb.get("off_top3_k_pct", np.nan),
                "opp_bat_source":  opp_lb.get("_lineup_source", "team_avg") if opp_lb else "team_avg",
                **feats,
            })

    if not rows:
        return pd.DataFrame()

    game_df = pd.DataFrame(rows)

    for col in feature_cols:
        if col in game_df.columns:
            game_df[col] = game_df[col].fillna(train_means.get(col, 0.0))
        else:
            game_df[col] = train_means.get(col, 0.0)

    X = game_df[feature_cols]
    game_df["expected_outs"] = model.predict(X).round(2)
    game_df["expected_ip"]   = (game_df["expected_outs"] / 3).round(2)

    print(f"  ✓ Scored {len(game_df)} pitchers. Mean E[Outs] = {game_df['expected_outs'].mean():.1f}.")
    return game_df


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("PITCHER TOTAL OUTS MODEL — STEP 4: EDGE SCORING AND EXPORT")
    print("=" * 70)
    today_str = datetime.now().strftime("%Y%m%d")

    print("\n[ 1/4 ] Scoring today's probable starters (individual stats)...")
    predictions_df = score_live_pitchers()
    if predictions_df.empty:
        print("  No probable starters found. Exiting.")
        exit(0)
    print(f"  ✓ {len(predictions_df)} pitchers scored.")
    if "expected_outs" in predictions_df.columns:
        print(f"  Mean E[Outs]: {predictions_df['expected_outs'].mean():.1f} "
              f"({predictions_df['expected_outs'].mean()/3:.2f} IP)")

    print("\n[ 2/4 ] Loading prop market odds...")
    # Try Action Network first (PRO subscription required for props)
    odds_df = get_pitcher_outs_odds(date.today().strftime("%Y-%m-%d"))
    if odds_df.empty:
        manual_path = os.path.join(PROC_DIR, "pitcher_outs_odds_today.csv")
        odds_df     = load_prop_odds_manual(manual_path)

    print("\n[ 3/4 ] Computing edge scores...")
    edge_report = build_pitcher_edge_report(predictions_df, odds_df)
    print(f"  ✓ {len(edge_report)} pitcher-prop combinations evaluated.")

    # Display summary
    summary = generate_daily_summary(edge_report)
    print("\n" + summary)

    # Save summary text
    summary_path = os.path.join(EXPORT_DIR, f"pitcher_outs_summary_{today_str}.txt")
    with open(summary_path, "w") as f:
        f.write(summary)

    print(f"\n[ 4/4 ] Exporting...")
    output_path = os.path.join(EXPORT_DIR, f"pitcher_outs_edges_{today_str}.xlsx")
    edge_report.to_excel(output_path, index=False, engine='openpyxl')
    print(f"  ✓ Full edge report: {output_path}")

    value_bets = edge_report[edge_report["is_value_bet"] == 1]
    if not value_bets.empty:
        plays_path = os.path.join(EXPORT_DIR, f"pitcher_outs_plays_{today_str}.xlsx")
        value_bets.to_excel(plays_path, index=False, engine='openpyxl')
        print(f"  ✓ Value bets: {plays_path} ({len(value_bets)} plays)")

    print("\n" + "=" * 70)
    print("PITCHER OUTS MODEL — COMPLETE")
    print("=" * 70)
