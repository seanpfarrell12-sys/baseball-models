"""
=============================================================================
HITTER TOTAL BASES MODEL — FILE 4 OF 4: EDGE SCORING AND CSV EXPORT
=============================================================================
Purpose : Compare model's E[TB] to market prop lines; calculate edge/Kelly;
          export ranked bet recommendations for today's player slate.
Input   : ../data/processed/hitter_tb_predictions.csv (from Step 3)
Output  : ../exports/hitter_tb_edges_YYYYMMDD.csv

Player prop context:
─────────────────────────────────────────────────────────────────────────────
Hitter total bases props are one of the most exploitable markets because:
  1. Books use simple historical averages; we use Statcast quality of contact
  2. Matchup-specific adjustments are often underweighted by the market
  3. Platoon splits create significant pricing errors on some lineups
  4. High-Barrel% / low-AVG hitters are systematically underpriced

Common prop lines: 0.5, 1.5, 2.5 total bases
  - 0.5 TB: Did the batter get any hit? (Usually priced at -200 to +150)
  - 1.5 TB: Did the batter get an XBH or two singles? (Most common prop)
  - 2.5 TB: Did the batter get a double + single, or a HR? (Higher variance)

Bet sizing for player props:
  - Player props have lower limits than game lines (typically $500–$2,000)
  - Use smaller Kelly fractions (15%) due to lower liquidity
  - Multiple props from the same game are correlated — monitor total exposure

For R users:
  - The export format creates one row per player-prop combination
  - "Pivot longer" approach: each player gets rows for each prop line tested
  - In R: tidyr::pivot_longer(cols=starts_with("p_over"), names_to="line", values_to="prob")
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
from utils.action_network import get_hitter_tb_odds
from utils.probable_starters import (get_games_with_sp_stats, get_lineups,
                                      normalize_name, load_batting_stats)

# --- Configuration ----------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR  = os.path.join(BASE_DIR, "data", "models")
EXPORT_DIR = os.path.join(BASE_DIR, "exports")

ODDS_API_KEY       = "fbc985ad430c95d6435cb75210f7b989"
ODDS_API_URL_PROPS = "https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{event_id}/odds"

# For player props, use smaller Kelly fraction (lower liquidity)
KELLY_FRACTION_PROPS = 0.15  # 15% Kelly for player props
MAX_BET_FRACTION     = 0.03  # Max 3% of bankroll per player prop (lower than game lines)
MIN_EDGE             = 0.04  # 4% minimum edge (higher bar for props — more variance)


# =============================================================================
# ODDS CONVERSION (same utilities as other export files)
# =============================================================================

def american_to_decimal(american_odds: float) -> float:
    """Convert American odds to decimal odds."""
    if american_odds >= 0:
        return (american_odds / 100.0) + 1.0
    else:
        return (100.0 / abs(american_odds)) + 1.0


def juice_to_implied_prob(juice: float) -> float:
    """Convert American juice to raw implied probability."""
    return 1.0 / american_to_decimal(juice)


def remove_vig_props(over_juice: float, under_juice: float) -> tuple:
    """Remove vig from player prop lines to get fair probabilities."""
    p_over_raw  = juice_to_implied_prob(over_juice)
    p_under_raw = juice_to_implied_prob(under_juice)
    total       = p_over_raw + p_under_raw
    if total <= 0:
        return 0.5, 0.5
    return p_over_raw / total, p_under_raw / total


def calculate_ev_pct(model_prob: float, decimal_odds: float) -> float:
    """EV% = model_prob × (dec_odds - 1) - (1 - model_prob)."""
    return round(model_prob * (decimal_odds - 1) - (1 - model_prob), 4)


def kelly_criterion_props(model_prob: float, decimal_odds: float) -> float:
    """
    Kelly criterion sized for player props.

    Uses KELLY_FRACTION_PROPS (15%) instead of the standard 25% because:
      1. Player props have higher variance than game lines
      2. Liquidity is lower (books limit winning players sooner)
      3. Correlated exposure across multiple props on the same game

    The math is identical to moneyline Kelly — only the fraction differs.
    """
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - model_prob
    f_full = (b * model_prob - q) / b
    return round(min(max(f_full * KELLY_FRACTION_PROPS, 0.0), MAX_BET_FRACTION), 4)


def compute_tb_edge_score(edge: float, ev_pct: float, kelly: float,
                           expected_tb: float, prop_line: float) -> float:
    """
    Compute 0–10 edge score for hitter total bases props.

    Additional TB-specific signal:
      - Gap between expected_tb and prop_line (bigger gap = more confidence)
      - High barrel% players get a small bonus for consistency
        (not directly in this function but reflected in model_prob)

    Components:
      edge_comp : 0–3 based on edge (3% = low, 15% = high)
      ev_comp   : 0–3 based on EV% (3% = low, 20% = high — props have more EV upside)
      kelly_comp: 0–2 based on kelly sizing (captures confidence)
      gap_comp  : 0–2 based on expected_tb vs line gap

    R equivalent: see moneyline edge_score function for similar structure
    """
    edge_comp  = 3.0 * np.clip(edge,   0, 0.20) / 0.20
    ev_comp    = 3.0 * np.clip(ev_pct, 0, 0.20) / 0.20
    kelly_comp = 2.0 * np.clip(kelly,  0, 0.03) / 0.03

    # Gap component: how different is E[TB] from the prop line?
    if prop_line and prop_line > 0:
        gap      = abs(expected_tb - prop_line)
        gap_norm = np.clip(gap / prop_line, 0, 0.50)  # Cap at 50% gap
        gap_comp = 2.0 * gap_norm / 0.50
    else:
        gap_comp = 0.0

    return round(min(edge_comp + ev_comp + kelly_comp + gap_comp, 10.0), 2)


# =============================================================================
# PROPS ODDS LOADING
# =============================================================================

def load_prop_odds_manual(path: str = None) -> pd.DataFrame:
    """
    Load player prop odds from a manually created CSV file.

    Expected format:
      player_name,team,prop_line,over_juice,under_juice,book
      Juan Soto,NYM,1.5,-120,+100,FanDuel
      Aaron Judge,NYY,1.5,-150,+125,DraftKings
      Freddie Freeman,LAD,1.5,-130,+108,BetMGM

    The model's prediction is in the predictions CSV; this file provides
    the MARKET PRICE to compare against.

    Column 'prop_line' should match the line in the predictions file
    (0.5, 1.5, or 2.5).
    """
    if path and os.path.exists(path):
        print(f"  Loading prop odds from: {path}")
        return pd.read_csv(path)

    # Default template
    template = pd.DataFrame({
        "player_name":  ["Juan Soto",     "Aaron Judge",  "Freddie Freeman"],
        "team":         ["NYM",           "NYY",          "LAD"],
        "prop_line":    [1.5,             1.5,            1.5],
        "over_juice":   [-120.0,          -150.0,         -130.0],
        "under_juice":  [+100.0,          +125.0,         +108.0],
    })
    template_path = os.path.join(PROC_DIR, "hitter_tb_odds_template.csv")
    template.to_csv(template_path, index=False)
    print(f"  Created prop odds template: {template_path}")
    return template


# =============================================================================
# MAIN EDGE REPORT BUILDER
# =============================================================================

def build_tb_edge_report(predictions_df: pd.DataFrame,
                          odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join model predictions with market prop odds to compute edge metrics.

    For each player-prop combination:
      - Find market's over/under juice for that player's prop line
      - Compute model probability vs market implied probability
      - Calculate edge, EV%, Kelly, and edge score
      - Flag as value bet if edge ≥ MIN_EDGE

    Merge strategy:
      - Primary: player_name + prop_line (exact match)
      - Fallback: player_name only (use closest available line)

    Returns
    -------
    pd.DataFrame
        One row per player-prop bet with all edge metrics.
    """
    rows = []

    for _, pred_row in predictions_df.iterrows():
        player_name = pred_row.get("player_name", "")
        team        = pred_row.get("team",  "")
        opp_team    = pred_row.get("opp_team", "")
        # Prefer PA-adjusted E[TB] if available; fall back to raw model output
        expected_tb = pred_row.get("adjusted_expected_tb",
                                   pred_row.get("expected_tb", 1.2))

        # Look up this player's odds in the market odds DataFrame
        player_odds = odds_df[
            odds_df["player_name"].str.lower().str.strip()
            == str(player_name).lower().strip()
        ] if not odds_df.empty else pd.DataFrame()

        # Evaluate each standard prop line
        prop_lines_to_eval = [0.5, 1.5, 2.5]

        for line in prop_lines_to_eval:
            # Get model probabilities for this line
            line_col_over  = f"p_over_line_{str(line).replace('.', '_')}"
            line_col_under = f"p_under_line_{str(line).replace('.', '_')}"

            model_p_over  = pred_row.get(line_col_over,  pred_row.get("p_over_main",  0.5))
            model_p_under = pred_row.get(line_col_under, pred_row.get("p_under_main", 0.5))

            # Find market odds for this line
            if not player_odds.empty:
                line_odds = player_odds[
                    abs(player_odds["prop_line"] - line) < 0.01
                ]
                if not line_odds.empty:
                    over_juice  = line_odds["over_juice"].iloc[0]
                    under_juice = line_odds["under_juice"].iloc[0]
                else:
                    # Use typical market juice as default
                    over_juice  = -115.0  # Slight over juice (common for power hitters)
                    under_juice = -105.0
            else:
                # No odds data — use standard juice
                over_juice  = -110.0
                under_juice = -110.0

            # Fair market probabilities (remove vig)
            fair_p_over, fair_p_under = remove_vig_props(over_juice, under_juice)

            # Edges
            over_edge  = round(model_p_over  - fair_p_over,  4)
            under_edge = round(model_p_under - fair_p_under, 4)

            # Evaluate both Over and Under for this line
            for bet_side, model_p, fair_p, juice, edge in [
                ("OVER",  model_p_over,  fair_p_over,  over_juice,  over_edge),
                ("UNDER", model_p_under, fair_p_under, under_juice, under_edge),
            ]:
                dec_odds = american_to_decimal(juice)
                ev_pct   = calculate_ev_pct(model_p, dec_odds)
                kelly    = kelly_criterion_props(model_p, dec_odds)
                score    = compute_tb_edge_score(edge, ev_pct, kelly, expected_tb, line)

                # Skip clearly wrong directions (under on 0.5 is usually bad)
                if bet_side == "UNDER" and line == 0.5 and model_p_over > 0.85:
                    continue  # Don't recommend Under 0.5 if we project likely getting a hit

                rows.append({
                    "game_date":     pred_row.get("game_date", datetime.now().strftime("%Y-%m-%d")),
                    "player_name":   player_name,
                    "team":          team,
                    "opp_team":      opp_team,
                    "bet_type":          f"{bet_side} {line} TB",
                    "prop_line":         line,
                    "bet_side":          bet_side,
                    "batting_order_pos": pred_row.get("batting_order_pos", ""),
                    "pa_adjustment":     round(pred_row.get("pa_adjustment", 1.0), 3),
                    "expected_tb":       round(expected_tb, 3),  # PA-adjusted E[TB]
                    "model_prob":        round(model_p, 4),
                    "market_implied": round(fair_p, 4),
                    "edge":          edge,
                    "ev_pct":        ev_pct,
                    "kelly_fraction": kelly,
                    "edge_score":    score,
                    "juice":         juice,
                    "decimal_odds":  round(dec_odds, 4),
                    "is_value_bet":  1 if edge >= MIN_EDGE and ev_pct > 0 else 0,
                })

    report_df = pd.DataFrame(rows)
    if not report_df.empty:
        report_df = report_df.sort_values("edge_score", ascending=False).reset_index(drop=True)
        # Keep only the single best-scoring value bet per player (max 1 pick per batter)
        best_idx = (
            report_df[report_df["is_value_bet"] == 1]
            .groupby("player_name")["edge_score"]
            .idxmax()
        )
        report_df["is_value_bet"] = 0
        if not best_idx.empty:
            report_df.loc[best_idx.values, "is_value_bet"] = 1
    return report_df


# =============================================================================
# DAILY SUMMARY REPORT
# =============================================================================

def generate_daily_summary(edge_report: pd.DataFrame) -> str:
    """
    Generate a human-readable daily summary of top plays.

    Returns
    -------
    str
        Formatted summary string ready for display or saving.
    """
    today = datetime.now().strftime("%B %d, %Y")
    value_bets = edge_report[edge_report["is_value_bet"] == 1]
    overs  = value_bets[value_bets["bet_side"] == "OVER"]
    unders = value_bets[value_bets["bet_side"] == "UNDER"]

    lines = [
        "=" * 65,
        f"  HITTER TOTAL BASES — DAILY EDGE REPORT — {today}",
        "=" * 65,
        f"  Total props evaluated: {len(edge_report)}",
        f"  Value bets found:      {len(value_bets)} "
        f"({len(overs)} Over, {len(unders)} Under)",
        "",
        "  TOP OVER PLAYS:",
    ]

    for _, row in overs.head(5).iterrows():
        lines.append(
            f"  {row['player_name']:20s} OVER {row['prop_line']} TB  |  "
            f"E[TB]={row['expected_tb']:.2f}  |  "
            f"Edge={row['edge']:+.1%}  |  "
            f"EV%={row['ev_pct']:+.1%}  |  "
            f"Kelly={row['kelly_fraction']:.1%}  |  "
            f"Score={row['edge_score']}/10"
        )

    lines += ["", "  TOP UNDER PLAYS:"]
    for _, row in unders.head(5).iterrows():
        lines.append(
            f"  {row['player_name']:20s} UNDER {row['prop_line']} TB |  "
            f"E[TB]={row['expected_tb']:.2f}  |  "
            f"Edge={row['edge']:+.1%}  |  "
            f"EV%={row['ev_pct']:+.1%}  |  "
            f"Kelly={row['kelly_fraction']:.1%}  |  "
            f"Score={row['edge_score']}/10"
        )

    lines += ["=" * 65]
    return "\n".join(lines)


# =============================================================================
# LIVE SCORING — Today's confirmed lineups vs probable starters
# =============================================================================

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

# Park HR factor abbreviation map (FanGraphs → our standard)
_FG_TO_STD = {
    "CHW": "CWS", "SD": "SDP", "SF": "SFG", "TB": "TBR",
    "KC": "KCR", "WAS": "WSN",
}

# Average MLB plate appearances per game by lineup position (1-indexed).
# Top of order bats ~1.3 more times per game than bottom of order —
# directly proportional to total bases opportunity.
_PA_RATES = {1: 4.47, 2: 4.30, 3: 4.16, 4: 3.99, 5: 3.84,
             6: 3.69, 7: 3.54, 8: 3.38, 9: 3.19}
_PA_MEAN  = 3.84   # average across all 9 spots


def score_live_hitters() -> pd.DataFrame:
    """
    Score today's confirmed lineup hitters vs probable starters.

    Uses individual 2025 hitter stats + today's opposing SP's 2025 stats.

    Returns
    -------
    pd.DataFrame
        One row per lineup hitter with expected_tb, player_name, team, opp_team.
    """
    import xgboost as xgb

    model = xgb.XGBRegressor()
    model.load_model(os.path.join(MODEL_DIR, "hitter_tb_model.json"))
    with open(os.path.join(MODEL_DIR, "hitter_tb_features.json")) as f:
        feature_cols = json.load(f)

    # ── Today's lineups and games ─────────────────────────────────────────
    game_date = date.today().strftime("%Y-%m-%d")
    print("  Fetching today's lineups and probable starters...")
    lineups, lineup_sources = get_lineups(game_date, return_sources=True)
    sp_df = get_games_with_sp_stats(game_date)   # auto-selects most recent season

    if not lineups:
        print("  WARNING: No confirmed lineups yet — lineups typically post 2-3h before game.")
        return pd.DataFrame()

    # Only score teams with confirmed lineups — skip projected to avoid stale picks
    confirmed_teams = {t for t, src in lineup_sources.items() if src == "confirmed"}
    projected_teams = set(lineups.keys()) - confirmed_teams
    if projected_teams:
        print(f"  Skipping {len(projected_teams)} team(s) with projected lineups: "
              f"{', '.join(sorted(projected_teams))}")
    lineups = {t: v for t, v in lineups.items() if t in confirmed_teams}
    if not lineups:
        print("  WARNING: No confirmed lineups available yet.")
        return pd.DataFrame()

    # Build: team → opposing SP stats (opp_avg_sp_*)
    # Only include games where BOTH SPs are individually confirmed
    opp_sp_map = {}   # team → {opp_avg_sp_k_pct, ...}
    if not sp_df.empty:
        for _, sp_row in sp_df.iterrows():
            home = sp_row["home_team"]
            away = sp_row["away_team"]
            if sp_row.get("home_sp_source") == "team_avg" or \
               sp_row.get("away_sp_source") == "team_avg":
                print(f"  Skipping {away} @ {home} hitters — starting pitcher(s) not confirmed yet.")
                # Remove any confirmed lineups for these teams so they aren't scored
                lineups.pop(home, None)
                lineups.pop(away, None)
                continue
            # Home batters face the away SP
            opp_sp_map[home] = {
                "opp_avg_sp_k_pct":    sp_row.get("away_sp_k_pct",    np.nan),
                "opp_avg_sp_siera":    sp_row.get("away_sp_siera",    np.nan),
                "opp_avg_sp_xfip":     sp_row.get("away_sp_xfip",     np.nan),
                "opp_avg_sp_bb_pct":   sp_row.get("away_sp_bb_pct",   np.nan),
            }
            # Away batters face the home SP
            opp_sp_map[away] = {
                "opp_avg_sp_k_pct":    sp_row.get("home_sp_k_pct",    np.nan),
                "opp_avg_sp_siera":    sp_row.get("home_sp_siera",    np.nan),
                "opp_avg_sp_xfip":     sp_row.get("home_sp_xfip",     np.nan),
                "opp_avg_sp_bb_pct":   sp_row.get("home_sp_bb_pct",   np.nan),
            }

    # Add swstr_pct from pitcher efficiency file (not in probable_starters output)
    eff = pd.read_csv(os.path.join(RAW_DIR, "raw_pitcher_efficiency.csv"))
    eff_latest = int(eff["Season"].max())
    eff_qualifiers = eff[(eff["Season"] == eff_latest) & (eff["GS"] >= 5)]
    eff_season = eff_latest if len(eff_qualifiers) >= 10 else eff_latest - 1
    eff = eff[(eff["Season"] == eff_season) & (eff["GS"] >= 5)].copy()
    eff_index = {normalize_name(n): i for i, n in enumerate(eff["Name"])}

    if not sp_df.empty:
        for _, sp_row in sp_df.iterrows():
            home = sp_row["home_team"]
            away = sp_row["away_team"]
            for batter_team, sp_name in [(home, sp_row.get("away_sp_name", "")),
                                          (away, sp_row.get("home_sp_name", ""))]:
                if not sp_name:
                    continue
                norm = normalize_name(sp_name)
                idx  = eff_index.get(norm)
                if idx is None:
                    last = norm.split()[-1] if norm else ""
                    cands = [k for k in eff_index if k.split()[-1] == last]
                    if len(cands) == 1:
                        idx = eff_index[cands[0]]
                if idx is not None and batter_team in opp_sp_map:
                    opp_sp_map[batter_team]["opp_avg_sp_swstr_pct"] = float(
                        eff.iloc[idx].get("SwStr%", np.nan)
                    )

    # ── Individual batter stats (most recent season: 2026 → 2025 fallback) ─
    bat = load_batting_stats()   # auto-selects most recent season with ≥20 qualifiers
    bat = bat[bat["PA"] >= 50].copy()
    bat_season = int(bat["Season"].max()) if not bat.empty else "?"
    print(f"  Using {bat_season} batting stats for hitter lookup ({len(bat)} qualified batters).")
    bat_index = {normalize_name(n): i for i, n in enumerate(bat["Name"])}

    # ── Park HR factors ───────────────────────────────────────────────────
    park = pd.read_csv(os.path.join(RAW_DIR, "raw_park_factors.csv"))
    park_map = {r["team"]: r for _, r in park.iterrows()}

    # Build: team → home park (teams in lineup are always home or away vs a park)
    # We need to know which park they're playing in — use sp_df game info
    team_to_park = {}
    if not sp_df.empty:
        for _, sp_row in sp_df.iterrows():
            team_to_park[sp_row["home_team"]] = sp_row["home_team"]
            team_to_park[sp_row["away_team"]] = sp_row["home_team"]

    # Training means as fallback
    train_means = pd.read_csv(
        os.path.join(PROC_DIR, "hitter_tb_dataset.csv")
    )[feature_cols].mean()

    rows = []
    for team, players in lineups.items():
        # Find opponent team from sp_df
        if not sp_df.empty:
            home_mask = sp_df["home_team"] == team
            away_mask = sp_df["away_team"] == team
            if home_mask.any():
                opp_team = sp_df[home_mask].iloc[0]["away_team"]
            elif away_mask.any():
                opp_team = sp_df[away_mask].iloc[0]["home_team"]
            else:
                opp_team = ""
        else:
            opp_team = ""

        park_team = team_to_park.get(team, team)
        pk = park_map.get(park_team, {})
        home_park_hr_factor = pk.get("pf_hr", 100) if isinstance(pk, dict) else getattr(pk, "pf_hr", 100)

        opp_sp = opp_sp_map.get(team, {})

        for lineup_pos, player_name in enumerate(players, start=1):
            norm = normalize_name(player_name)
            bat_idx = bat_index.get(norm)
            if bat_idx is None:
                last = norm.split()[-1] if norm else ""
                cands = [k for k in bat_index if k.split()[-1] == last]
                if len(cands) == 1:
                    bat_idx = bat_index[cands[0]]

            if bat_idx is None:
                continue  # Skip players not found in batting stats

            b = bat.iloc[bat_idx]
            g  = max(int(b.get("G", 1)), 1)
            hr = float(b.get("HR", 0))
            b2 = float(b.get("2B", 0))
            b3 = float(b.get("3B", 0))

            rows.append({
                "game_date":       date.today().strftime("%Y-%m-%d"),
                "player_name":     player_name,
                "team":            team,
                "opp_team":        opp_team,
                "batting_order_pos": lineup_pos,   # 1-9; used for PA adjustment
                # Batter features
                "ISO":          b.get("ISO",   np.nan),
                "SLG":          b.get("SLG",   np.nan),
                "wOBA":         b.get("wOBA",  np.nan),
                "wRC+":         b.get("wRC+",  np.nan),
                "hr_per_game":  hr / g,
                "xbh_per_game": (b2 + b3 + hr) / g,
                "BB%":          b.get("BB%",   np.nan),
                "K%":           b.get("K%",    np.nan),
                "BABIP":        b.get("BABIP", np.nan),
                "home_park_hr_factor": home_park_hr_factor,
                # Opposing SP features (proportions to match training data)
                "opp_avg_sp_k_pct":    opp_sp.get("opp_avg_sp_k_pct",    np.nan),
                "opp_avg_sp_siera":    opp_sp.get("opp_avg_sp_siera",    np.nan),
                "opp_avg_sp_xfip":     opp_sp.get("opp_avg_sp_xfip",     np.nan),
                "opp_avg_sp_swstr_pct":opp_sp.get("opp_avg_sp_swstr_pct", np.nan),
                "opp_avg_sp_bb_pct":   opp_sp.get("opp_avg_sp_bb_pct",   np.nan),
            })

    if not rows:
        print("  WARNING: No hitters matched in batting stats.")
        return pd.DataFrame()

    game_df = pd.DataFrame(rows)

    for col in feature_cols:
        if col in game_df.columns:
            game_df[col] = game_df[col].fillna(train_means.get(col, 0.0))
        else:
            game_df[col] = train_means.get(col, 0.0)

    X = game_df[feature_cols]
    game_df["expected_tb"] = model.predict(X).round(3)

    # Post-model PA adjustment: the model predicts TB from a player's skills
    # without knowing their lineup position. A leadoff hitter bats ~1.3× more
    # often per game than a 9-hole hitter, so we scale E[TB] by their relative
    # PA rate. This doesn't require retraining — it's a calibration correction.
    game_df["pa_adjustment"]        = game_df["batting_order_pos"].map(_PA_RATES) / _PA_MEAN
    game_df["adjusted_expected_tb"] = (game_df["expected_tb"] * game_df["pa_adjustment"]).round(3)

    # Compute per-line probabilities using Poisson distribution.
    # TB is a non-negative integer (0,1,2,...) — Poisson(lambda=adjusted_expected_tb)
    # is a reasonable approximation. This converts E[TB] into the probabilities
    # the edge report needs to compare against market implied odds.
    from scipy.stats import poisson as _poisson
    for line in [0.5, 1.5, 2.5]:
        col_over  = f"p_over_line_{str(line).replace('.', '_')}"
        col_under = f"p_under_line_{str(line).replace('.', '_')}"
        # P(TB > line) for half-point lines = P(TB >= ceil(line))
        k = int(np.floor(line))   # e.g. line=1.5 → k=1; P(over) = P(TB >= 2)
        game_df[col_over]  = game_df["adjusted_expected_tb"].apply(
            lambda lam: float(1 - _poisson.cdf(k, max(lam, 1e-9)))
        )
        game_df[col_under] = 1.0 - game_df[col_over]

    print(f"  ✓ Scored {len(game_df)} hitters. "
          f"Mean E[TB] = {game_df['expected_tb'].mean():.3f} (raw) "
          f"| {game_df['adjusted_expected_tb'].mean():.3f} (PA-adjusted).")
    return game_df


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("HITTER TOTAL BASES MODEL — STEP 4: EDGE SCORING AND EXPORT")
    print("=" * 70)
    today_str = datetime.now().strftime("%Y%m%d")

    print("\n[ 1/4 ] Scoring today's lineup hitters (individual stats)...")
    predictions_df = score_live_hitters()
    if predictions_df.empty:
        print("  No hitters scored. Lineups may not be posted yet.")
        print("  (Lineups typically post 2-3 hours before first pitch.)")
        exit(0)
    print(f"  ✓ {len(predictions_df)} hitters scored.")
    if "expected_tb" in predictions_df.columns:
        print(f"  Mean E[TB]: {predictions_df['expected_tb'].mean():.3f}")

    print("\n[ 2/4 ] Loading prop market odds...")
    # Try Action Network first (PRO subscription gives access to player props)
    odds_df = get_hitter_tb_odds(date.today().strftime("%Y-%m-%d"))
    if odds_df.empty:
        # Fallback to manual CSV
        manual_path = os.path.join(PROC_DIR, "hitter_tb_odds_today.csv")
        odds_df     = load_prop_odds_manual(manual_path)

    print("\n[ 3/4 ] Computing edge scores...")
    edge_report = build_tb_edge_report(predictions_df, odds_df)
    print(f"  ✓ {len(edge_report)} player-prop combinations evaluated.")

    # Generate human-readable summary
    summary = generate_daily_summary(edge_report)
    print("\n" + summary)

    # Save summary to text file
    summary_path = os.path.join(EXPORT_DIR, f"hitter_tb_summary_{today_str}.txt")
    with open(summary_path, "w") as f:
        f.write(summary)

    print(f"\n[ 4/4 ] Exporting...")
    output_path = os.path.join(EXPORT_DIR, f"hitter_tb_edges_{today_str}.xlsx")
    edge_report.to_excel(output_path, index=False, engine='openpyxl')
    print(f"  ✓ Full edge report: {output_path}")

    value_bets = edge_report[edge_report["is_value_bet"] == 1]
    if not value_bets.empty:
        plays_path = os.path.join(EXPORT_DIR, f"hitter_tb_plays_{today_str}.xlsx")
        value_bets.to_excel(plays_path, index=False, engine='openpyxl')
        print(f"  ✓ Value bets only: {plays_path}")

    print("\n" + "=" * 70)
    print("HITTER TB MODEL — COMPLETE")
    print("=" * 70)
