"""
=============================================================================
NOTIFIER — Discord daily picks delivery
=============================================================================
Setup (one-time):
  python3 utils/notifier.py --setup-discord          # picks channel webhook
  python3 utils/notifier.py --setup-discord-results  # results channel webhook
  python3 utils/notifier.py --setup-discord-status   # model-status channel webhook

Discord uses a free incoming webhook — no bot account needed.

To get a Discord webhook URL:
  1. Open your Discord server → channel settings (gear icon)
  2. Integrations → Webhooks → New Webhook
  3. Copy Webhook URL and paste when prompted
=============================================================================
"""

import json
import argparse
import requests
from pathlib import Path
from datetime import datetime

import pandas as pd

BASE_DIR      = Path(__file__).parent.parent
CREDS_FILE    = Path(__file__).parent / ".notifier_credentials.json"

# Discord embed colour (green)
DISCORD_COLOR = 0x2ecc71


# =============================================================================
# CREDENTIALS
# =============================================================================

def _load_creds() -> dict:
    if CREDS_FILE.exists():
        return json.loads(CREDS_FILE.read_text())
    return {}


def _save_creds(creds: dict):
    CREDS_FILE.write_text(json.dumps(creds, indent=2))



def _setup_discord():
    print("\n" + "=" * 60)
    print("  DISCORD NOTIFIER — SETUP (picks channel)")
    print("=" * 60)
    print("\n  In your Discord server:")
    print("  Channel Settings → Integrations → Webhooks → New Webhook → Copy URL\n")
    url = input("  Paste your webhook URL: ").strip()
    creds = _load_creds()
    creds["discord_webhook"] = url
    _save_creds(creds)
    print(f"\n  ✓ Discord picks webhook saved.")
    print("  Sending test message...")
    _send_discord_raw("⚾ Baseball Models connected — picks channel setup complete!")
    print("  ✓ Test sent — check your Discord channel.\n")


def _setup_discord_results():
    print("\n" + "=" * 60)
    print("  DISCORD NOTIFIER — SETUP (results channel)")
    print("=" * 60)
    print("\n  Create a second channel (e.g. #results) in your Discord server,")
    print("  then: Channel Settings → Integrations → Webhooks → New Webhook → Copy URL\n")
    url = input("  Paste your results channel webhook URL: ").strip()
    creds = _load_creds()
    creds["discord_results_webhook"] = url
    _save_creds(creds)
    print(f"\n  ✓ Discord results webhook saved.")
    print("  Sending test message...")
    _send_discord_results_raw("📊 Baseball Models connected — results channel setup complete!")
    print("  ✓ Test sent — check your results channel.\n")


def _setup_discord_status():
    print("\n" + "=" * 60)
    print("  DISCORD NOTIFIER — SETUP (model-status channel)")
    print("=" * 60)
    print("\n  Channel Settings → Integrations → Webhooks → New Webhook → Copy URL\n")
    url = input("  Paste your model-status channel webhook URL: ").strip()
    creds = _load_creds()
    creds["discord_status_webhook"] = url
    _save_creds(creds)
    print(f"\n  ✓ Discord status webhook saved.")
    print("  Sending test message...")
    _notify_status_raw("🟢 Baseball Models connected — model-status channel setup complete!")
    print("  ✓ Test sent — check your #model-status channel.\n")


# =============================================================================
# DISCORD HELPERS
# =============================================================================

def _send_discord_raw(content: str):
    creds = _load_creds()
    url   = creds.get("discord_webhook")
    if not url:
        print("  (notifier) Discord not configured — run: python3 utils/notifier.py --setup-discord")
        return
    requests.post(url, json={"content": content}, timeout=10)


def _send_discord_results_raw(content: str):
    creds = _load_creds()
    url   = creds.get("discord_results_webhook")
    if not url:
        return
    requests.post(url, json={"content": content}, timeout=10)


def _notify_status_raw(content: str):
    creds = _load_creds()
    url   = creds.get("discord_status_webhook")
    if not url:
        return
    requests.post(url, json={"content": content}, timeout=10)


def _send_discord_results_embed(title: str, fields: list, footer: str = "", color: int = None):
    creds = _load_creds()
    url   = creds.get("discord_results_webhook")
    if not url:
        print("  (notifier) Results channel not configured — run: python3 utils/notifier.py --setup-discord-results")
        return
    embed = {"title": title, "color": color or DISCORD_COLOR, "fields": fields}
    if footer:
        embed["footer"] = {"text": footer}
    requests.post(url, json={"embeds": [embed]}, timeout=10)


def _send_discord_embed(title: str, fields: list, footer: str = ""):
    """
    Send a rich embed to Discord.
    fields: [{"name": "Model", "value": "pick1\npick2", "inline": False}]
    """
    creds = _load_creds()
    url   = creds.get("discord_webhook")
    if not url:
        print("  (notifier) Discord not configured — run: python3 utils/notifier.py --setup-discord")
        return

    embed = {
        "title":  title,
        "color":  DISCORD_COLOR,
        "fields": fields,
    }
    if footer:
        embed["footer"] = {"text": footer}

    resp = requests.post(url, json={"embeds": [embed]}, timeout=10)
    if resp.status_code not in (200, 204):
        print(f"  (notifier) Discord error {resp.status_code}: {resp.text[:200]}")
        return


# =============================================================================
# FORMAT PICKS — shared logic
# =============================================================================

MODEL_CONFIGS = {
    "Moneyline": {
        "subject_col": None,
        "line_col":    None,
        "odds_col":    "american_odds",
        "edge_col":    "edge",
        "sms_label":   "ML",
        "emoji":       "💰",
    },
    "Totals O/U": {
        "subject_col": None,
        "line_col":    "ou_line",
        "odds_col":    "juice",
        "edge_col":    "edge",
        "sms_label":   "TOT",
        "emoji":       "🔢",
    },
    "Hitter TB": {
        "subject_col": "player_name",
        "line_col":    "prop_line",
        "odds_col":    "juice",
        "edge_col":    "edge",
        "sms_label":   "TB",
        "emoji":       "🏏",
    },
    "Pitcher Outs": {
        "subject_col": "pitcher_name",
        "line_col":    "prop_line_outs",
        "odds_col":    "juice",
        "edge_col":    "edge",
        "sms_label":   "PO",
        "emoji":       "⚾",
    },
    "NRFI/YRFI": {
        "subject_col": None,          # game-level market; subject built from team cols
        "line_col":    None,
        "odds_col":    "bet_odds",
        "edge_col":    "edge",
        "sms_label":   "NR",
        "emoji":       "1️⃣",
    },
}


def _pick_rows(report, model_name: str, cfg: dict) -> list:
    """Return list of formatted pick strings for one model."""
    if report is None or report.empty:
        return []
    vb = report[report["is_value_bet"] == 1] if "is_value_bet" in report.columns else pd.DataFrame()
    if vb.empty:
        return []

    rows = []
    for _, r in vb.iterrows():
        if cfg["subject_col"]:
            subj = str(r.get(cfg["subject_col"], ""))
        elif model_name == "Moneyline":
            side = str(r.get("bet_side", ""))
            subj = str(r.get("home_team", "") if side == "HOME" else r.get("away_team", ""))
        else:
            subj = f"{r.get('away_team','')} @ {r.get('home_team','')}"

        bet  = str(r.get("bet_type", r.get("bet_side", "")))
        line = r.get(cfg["line_col"]) if cfg["line_col"] else None
        odds = r.get(cfg["odds_col"], "")
        edge = r.get(cfg["edge_col"])

        line_str = f" {line}" if line is not None else ""
        odds_str = f" ({int(odds):+d})" if odds != "" else ""
        edge_str = f" +{edge*100:.1f}% edge" if edge is not None else ""

        rows.append(f"{subj} {bet}{line_str}{odds_str}{edge_str}")

    return rows


# =============================================================================
# DISCORD FORMAT
# =============================================================================

def _format_discord(results: dict, pick_date: str) -> tuple:
    """Return (title, fields, footer) for a Discord embed."""
    date_fmt = datetime.strptime(pick_date, "%Y%m%d").strftime("%A, %B %-d")
    title    = f"⚾ Daily Picks — {date_fmt}"
    fields   = []
    total    = 0

    for model_name, cfg in MODEL_CONFIGS.items():
        picks = _pick_rows(results.get(model_name), model_name, cfg)
        if not picks:
            continue
        total += len(picks)
        # Discord field value limit is 1024 characters
        lines, remaining = [], len(picks)
        for p in picks:
            line = f"`{p}`"
            candidate = "\n".join(lines + [line])
            if len(candidate) > 980:
                lines.append(f"*… +{remaining} more*")
                break
            lines.append(line)
            remaining -= 1
        fields.append({
            "name":   f"{cfg['emoji']} {model_name} ({len(picks)})",
            "value":  "\n".join(lines),
            "inline": False,
        })

    if not fields:
        fields = [{"name": "No value bets today", "value": "Models found no edges.", "inline": False}]

    footer = f"{total} value bet(s) — flat $100 stake tracking"
    return title, fields, footer


# =============================================================================
# RESULTS FORMAT
# =============================================================================

RESULT_EMOJI = {"WIN": "✅", "LOSS": "❌", "PUSH": "➖"}
RESULT_COLOR = {"WIN": 0x2ecc71, "LOSS": 0xe74c3c, "PUSH": 0x95a5a6}


def _format_discord_results(grade_date: str) -> tuple:
    """
    Load graded picks for grade_date from picks.xlsx and format as Discord embed.
    Returns (title, fields, footer, color).
    """
    from pathlib import Path
    picks_file = Path(__file__).parent.parent / "tracking" / "picks.xlsx"

    if not picks_file.exists():
        return None, None, None, None

    df = pd.read_excel(picks_file, engine="openpyxl")
    day = df[(df["pick_date"].astype(str) == grade_date) &
             (df["result"].isin(["WIN", "LOSS", "PUSH"]))]

    if day.empty:
        return None, None, None, None

    date_fmt = datetime.strptime(grade_date, "%Y%m%d").strftime("%A, %B %-d")
    title    = f"📊 Results — {date_fmt}"
    fields   = []

    wins   = int((day["result"] == "WIN").sum())
    losses = int((day["result"] == "LOSS").sum())
    pushes = int((day["result"] == "PUSH").sum())
    pnl    = pd.to_numeric(day["pnl"], errors="coerce").fillna(0).sum()
    pnl   -= losses * 100

    for model in ["Moneyline", "Totals", "Hitter TB", "Pitcher Outs", "NRFI/YRFI"]:
        sub = day[day["model"] == model]
        if sub.empty:
            continue
        lines = []
        for _, r in sub.iterrows():
            emoji  = RESULT_EMOJI.get(r["result"], "❓")
            actual = f" (actual: {r['actual']})" if pd.notna(r.get("actual")) else ""
            pnl_r  = pd.to_numeric(r.get("pnl"), errors="coerce")
            pnl_r  = 0 if pd.isna(pnl_r) else pnl_r
            if r["result"] == "LOSS":
                pnl_r = -100
            pnl_str = f" ${pnl_r:+.0f}" if r["result"] != "PUSH" else " $0"
            lines.append(f"{emoji} {r['subject']} {r['bet_type']}{actual}{pnl_str}")
        cfg = next((v for k, v in MODEL_CONFIGS.items() if v.get("sms_label") and model in k), {})
        emoji = cfg.get("emoji", "📌")
        fields.append({"name": f"{emoji} {model}", "value": "\n".join(f"`{l}`" for l in lines), "inline": False})

    record = f"{wins}W-{losses}L" + (f"-{pushes}P" if pushes else "")
    footer = f"{record} | P&L: ${pnl:+.2f} (flat $100)"
    color  = RESULT_COLOR["WIN"] if pnl >= 0 else RESULT_COLOR["LOSS"]

    return title, fields, footer, color


# =============================================================================
# PUBLIC API
# =============================================================================

def notify_run_status(title: str, lines: list, success: bool = True):
    """
    Send a brief run-status embed to the Discord #model-status channel.
    Called by retrain_all.py, schedule_daily.py, grade_daily.py, and
    run_daily.py to confirm each scheduled job completed (or failed).
    """
    creds = _load_creds()
    url   = creds.get("discord_status_webhook")
    if not url:
        return
    color = 0x2ecc71 if success else 0xe74c3c   # green / red
    embed = {
        "title":       title,
        "description": "\n".join(lines) if lines else "",
        "color":       color,
        "footer":      {"text": datetime.now().strftime("%-I:%M %p · %b %-d, %Y")},
    }
    try:
        requests.post(url, json={"embeds": [embed]}, timeout=10)
    except Exception as e:
        print(f"  (notifier) Status notification failed: {e}")


def send_graded_results(grade_date: str):
    """Send yesterday's graded results to the Discord results channel."""
    creds = _load_creds()
    if not creds.get("discord_results_webhook"):
        print("  (notifier) Results channel not configured — run: python3 utils/notifier.py --setup-discord-results")
        return

    title, fields, footer, color = _format_discord_results(grade_date)
    if title is None:
        print(f"  (notifier) No graded results for {grade_date} to send.")
        return

    _send_discord_results_embed(title, fields, footer, color)
    print("  (notifier) Results posted to Discord.")


def send_daily_picks(results: dict, pick_date: str,
                     scored_games: list = None, pending_games: list = None):
    """Send today's value bets via Discord."""
    creds = _load_creds()

    if creds.get("discord_webhook"):
        title, fields, footer = _format_discord(results, pick_date)
        _send_discord_embed(title, fields, footer)
        print("  (notifier) Discord message sent.")
    else:
        print("  (notifier) Discord not configured — run: python3 utils/notifier.py --setup-discord")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup-discord",         action="store_true", help="Configure Discord picks channel webhook")
    ap.add_argument("--setup-discord-results", action="store_true", help="Configure Discord results channel webhook")
    ap.add_argument("--setup-discord-status",  action="store_true", help="Configure Discord model-status channel webhook")
    ap.add_argument("--test-discord",          action="store_true", help="Send a test Discord picks message")
    ap.add_argument("--test-discord-results",  action="store_true", help="Send a test Discord results message")
    ap.add_argument("--test-discord-status",   action="store_true", help="Send a test Discord model-status message")
    parsed = ap.parse_args()

    if parsed.setup_discord:
        _setup_discord()
    elif parsed.setup_discord_results:
        _setup_discord_results()
    elif parsed.setup_discord_status:
        _setup_discord_status()
    elif parsed.test_discord:
        _send_discord_raw("⚾ Test from baseball models — Discord picks channel is working!")
        print("Test Discord message sent.")
    elif parsed.test_discord_results:
        _send_discord_results_raw("📊 Test from baseball models — Discord results channel is working!")
        print("Test Discord results message sent.")
    elif parsed.test_discord_status:
        notify_run_status("🟢 Model Status Test", ["This is a test — #model-status channel is working!"])
        print("Test status message sent.")
    else:
        ap.print_help()
