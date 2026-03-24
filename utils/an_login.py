"""
=============================================================================
ACTION NETWORK — AUTO-LOGIN TOKEN MANAGER
=============================================================================
Purpose : Log into Action Network using Playwright, capture the auth token
          from live network requests, and save it for use by all export files.

FIRST-TIME SETUP (run once after installing):
─────────────────────────────────────────────────────────────────────────────
  pip install playwright
  playwright install chromium

Then run this file once to save your credentials:
  python utils/an_login.py --setup

After setup, token refresh is fully automatic. The export files call
refresh_token_if_needed() which re-logs in whenever the token expires.
─────────────────────────────────────────────────────────────────────────────

How it works:
  1. Playwright launches a real Chromium browser (can be headless/invisible)
  2. Navigates to actionnetwork.com and completes the login form
  3. Intercepts outgoing network requests to api.actionnetwork.com
  4. Extracts the Authorization: Bearer <token> header from any request
  5. Saves token + timestamp to utils/.an_credentials.json
  6. action_network.py reads the token from that file automatically

Credentials are stored in utils/.an_credentials.json (gitignored).
Never hard-code your password in any source file.

For R users:
  - `argparse` = R's optparse or argparser package for CLI arguments
  - `getpass.getpass()` = readline(prompt="Password: ") in R (hides input)
  - `asyncio.run()` = runs async code; Playwright uses async/await pattern
  - `async def` / `await` = similar to promises in JavaScript; no R equivalent
=============================================================================
"""

import os
import sys
import json
import asyncio
import getpass       # Hides password input (like readline with echo=FALSE in R)
import argparse      # Command-line argument parsing
from datetime import datetime, timedelta
from pathlib import Path

# --- Paths ------------------------------------------------------------------
UTILS_DIR    = Path(__file__).parent
CREDS_FILE   = UTILS_DIR / ".an_credentials.json"   # Saved token + credentials
GITIGNORE    = UTILS_DIR / ".gitignore"

# Action Network URLs
LOGIN_URL    = "https://www.actionnetwork.com/login"
TRIGGER_URL  = "https://www.actionnetwork.com/mlb"   # Page that triggers API calls
API_HOST     = "api.actionnetwork.com"

# Token is considered fresh for this many hours before forcing a re-login
TOKEN_TTL_HOURS = 20


# =============================================================================
# CREDENTIAL STORAGE
# =============================================================================

def save_credentials(email: str, password: str, token: str = ""):
    """
    Save login credentials and token to a local JSON file.

    The file is stored at utils/.an_credentials.json which is gitignored.
    Password is stored in plain text locally — this is acceptable for a
    personal script on your own machine (same as storing it in a .env file).

    JSON structure:
      {
        "email":      "you@example.com",
        "password":   "yourpassword",
        "token":      "eyJhbGci...",
        "token_saved_at": "2025-04-01T10:30:00"
      }
    """
    data = {}
    if CREDS_FILE.exists():
        try:
            data = json.loads(CREDS_FILE.read_text())
        except Exception:
            pass

    data["email"]          = email
    data["password"]       = password
    if token:
        data["token"]          = token
        data["token_saved_at"] = datetime.now().isoformat()

    CREDS_FILE.write_text(json.dumps(data, indent=2))

    # Make sure .gitignore exists so credentials aren't accidentally committed
    _ensure_gitignored()
    print(f"  ✓ Credentials saved to {CREDS_FILE}")


def load_credentials() -> dict:
    """
    Load saved credentials from the JSON file.

    Returns
    -------
    dict with keys: email, password, token, token_saved_at
    Returns empty dict if file doesn't exist.
    """
    if not CREDS_FILE.exists():
        return {}
    try:
        return json.loads(CREDS_FILE.read_text())
    except Exception:
        return {}


def save_token(token: str):
    """Update only the token in the credentials file (preserves email/password)."""
    data = load_credentials()
    data["token"]          = token
    data["token_saved_at"] = datetime.now().isoformat()
    CREDS_FILE.write_text(json.dumps(data, indent=2))


def is_token_fresh(creds: dict, ttl_hours: int = TOKEN_TTL_HOURS) -> bool:
    """
    Check whether the saved token is still within its TTL window.

    Returns True if token was saved less than `ttl_hours` ago.
    Returns False if token is missing, old, or timestamp is unparseable.

    In R: difftime(Sys.time(), as.POSIXct(saved_at), units="hours") < ttl_hours
    """
    token     = creds.get("token", "")
    saved_at  = creds.get("token_saved_at", "")

    if not token or not saved_at:
        return False

    try:
        saved_dt  = datetime.fromisoformat(saved_at)
        age_hours = (datetime.now() - saved_dt).total_seconds() / 3600
        return age_hours < ttl_hours
    except Exception:
        return False


def _ensure_gitignored():
    """Create a .gitignore in utils/ that excludes the credentials file."""
    if not GITIGNORE.exists():
        GITIGNORE.write_text(".an_credentials.json\n")
    else:
        content = GITIGNORE.read_text()
        if ".an_credentials.json" not in content:
            with open(GITIGNORE, "a") as f:
                f.write("\n.an_credentials.json\n")


# =============================================================================
# PLAYWRIGHT LOGIN AND TOKEN CAPTURE
# =============================================================================

async def _login_and_capture_token(email: str, password: str,
                                    headless: bool = True) -> str:
    """
    Use Playwright to log into Action Network and capture the auth token.

    Playwright launches a real Chromium browser and intercepts network
    requests to capture the Bearer token. This bypasses Cloudflare/DataDome
    because it uses a genuine browser, not a simple HTTP request.

    `async def` means this is an asynchronous function — Python's way of
    running I/O tasks without blocking. `await` pauses execution until
    the awaited operation completes (like Sys.sleep() but non-blocking).

    Parameters
    ----------
    email : str      Your Action Network login email
    password : str   Your Action Network password
    headless : bool  True = invisible browser (faster), False = visible window

    Returns
    -------
    str : The captured Bearer token, or "" if capture failed.
    """
    # Import playwright inside the function — only needed here, not globally.
    # This prevents import errors if playwright isn't installed yet.
    try:
        from playwright.async_api import async_playwright, TimeoutError as PWTimeout
    except ImportError:
        print("  ERROR: playwright not installed.")
        print("  Run: pip install playwright && playwright install chromium")
        return ""

    captured_token = ""

    async with async_playwright() as p:
        print(f"  Launching {'headless' if headless else 'visible'} Chromium browser...")

        browser = await p.chromium.launch(
            headless=headless,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )

        # Create a new browser context with realistic viewport + user agent
        context = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )

        page = await context.new_page()

        # ── Apply stealth patches to evade Cloudflare bot detection ───────
        # Playwright sets navigator.webdriver=true by default, which Cloudflare
        # detects and blocks. playwright-stealth patches ~20 fingerprinting APIs
        # to make the browser indistinguishable from a real user session.
        try:
            from playwright_stealth import stealth_async
            await stealth_async(page)
            print("  Stealth mode active (Cloudflare evasion).")
        except ImportError:
            print("  NOTE: playwright-stealth not installed. Run: pip3 install playwright-stealth")
            print("  Proceeding without stealth (login may be blocked by Cloudflare).")

        # ── Intercept outgoing requests for Authorization headers ─────────
        async def on_request(request):
            """Called for every outgoing request — look for Bearer token."""
            nonlocal captured_token
            if captured_token:
                return
            auth_header = request.headers.get("authorization", "")
            if auth_header.startswith("Bearer ") and len(auth_header) > 20:
                captured_token = auth_header.replace("Bearer ", "").strip()
                print(f"  ✓ Token captured from request header: {request.url[:70]}...")

        # ── Intercept responses — login API response body may contain token ─
        # Many SPAs return { token: "..." } or { access_token: "..." } in the
        # JSON body of the login POST response.
        async def on_response(response):
            """Called for every response — check login API response for token."""
            nonlocal captured_token
            if captured_token:
                return
            # Only check likely auth/login endpoints
            url_lower = response.url.lower()
            if not any(k in url_lower for k in ["login", "auth", "session", "token", "signin"]):
                return
            try:
                body = await response.json()
                # Flatten nested dicts one level to find token fields
                def _find_token(obj, depth=0):
                    if depth > 3 or not isinstance(obj, dict):
                        return ""
                    for key, val in obj.items():
                        if isinstance(val, str) and len(val) > 30 and any(
                            k in key.lower() for k in ["token", "jwt", "auth", "bearer"]
                        ):
                            return val.replace("Bearer ", "").strip()
                        nested = _find_token(val, depth + 1)
                        if nested:
                            return nested
                    return ""
                found = _find_token(body)
                if found:
                    captured_token = found
                    print(f"  ✓ Token captured from response body: {response.url[:70]}...")
            except Exception:
                pass  # Non-JSON response, ignore

        page.on("request", on_request)
        page.on("response", on_response)

        # ── Navigate to the login page ────────────────────────────────────
        print(f"  Navigating to {LOGIN_URL}...")
        try:
            await page.goto(LOGIN_URL, wait_until="domcontentloaded", timeout=20000)
        except PWTimeout:
            print("  WARNING: Login page load timed out. Proceeding anyway...")

        await asyncio.sleep(2)  # Brief pause for JS to render the form

        # ── Fill in login form ────────────────────────────────────────────
        print("  Filling login form...")
        try:
            # Try multiple possible selectors for the email field
            # Action Network may use different input names/IDs
            email_selectors = [
                'input[name="email"]',
                'input[type="email"]',
                'input[placeholder*="email" i]',
                '#email',
            ]
            password_selectors = [
                'input[name="password"]',
                'input[type="password"]',
                '#password',
            ]
            submit_selectors = [
                'button[type="submit"]',
                'button:has-text("Sign In")',
                'button:has-text("Log In")',
                'input[type="submit"]',
            ]

            # Fill email field — try each selector until one works
            for sel in email_selectors:
                try:
                    await page.fill(sel, email, timeout=3000)
                    print(f"    Email field found: {sel}")
                    break
                except Exception:
                    continue

            # Fill password field
            for sel in password_selectors:
                try:
                    await page.fill(sel, password, timeout=3000)
                    print(f"    Password field found: {sel}")
                    break
                except Exception:
                    continue

            # Click submit
            for sel in submit_selectors:
                try:
                    await page.click(sel, timeout=3000)
                    print(f"    Submit button clicked: {sel}")
                    break
                except Exception:
                    continue

        except Exception as e:
            print(f"  WARNING: Form fill error: {e}")
            if not headless:
                print("  Browser is visible — you can complete login manually.")
                await asyncio.sleep(15)  # Give time for manual completion

        # ── Wait for login to complete ────────────────────────────────────
        print("  Waiting for login to complete...")
        await asyncio.sleep(4)

        # ── Navigate to MLB odds page to trigger API requests ─────────────
        if not captured_token:
            print(f"  Navigating to {TRIGGER_URL} to trigger API calls...")
            try:
                await page.goto(TRIGGER_URL, wait_until="domcontentloaded", timeout=20000)
            except PWTimeout:
                pass
            await asyncio.sleep(5)  # Wait for API calls to fire

        # ── Check localStorage and sessionStorage ────────────────────────
        if not captured_token:
            print("  Checking localStorage and sessionStorage...")
            try:
                storage_data = await page.evaluate("""
                    () => {
                        const result = {};
                        const check = (store, prefix) => {
                            for (let i = 0; i < store.length; i++) {
                                const key = store.key(i);
                                result[prefix + key] = store.getItem(key);
                            }
                        };
                        try { check(localStorage, 'local:'); } catch(e) {}
                        try { check(sessionStorage, 'session:'); } catch(e) {}
                        return result;
                    }
                """)
                for key, value in storage_data.items():
                    if not value or len(value) < 20:
                        continue
                    key_lower = key.lower()
                    if any(k in key_lower for k in ["token", "auth", "jwt", "bearer"]):
                        captured_token = value.strip().replace("Bearer ", "")
                        print(f"  ✓ Token found in storage['{key}']")
                        break
                    # Also try parsing JSON values (e.g. {"token": "..."} stored as string)
                    try:
                        parsed = json.loads(value)
                        if isinstance(parsed, dict):
                            for k, v in parsed.items():
                                if isinstance(v, str) and len(v) > 30 and any(
                                    x in k.lower() for x in ["token", "auth", "jwt"]
                                ):
                                    captured_token = v.strip().replace("Bearer ", "")
                                    print(f"  ✓ Token found in storage['{key}']['{k}']")
                                    break
                    except Exception:
                        pass
                    if captured_token:
                        break
            except Exception as e:
                print(f"  Storage check failed: {e}")

        # ── Check cookies — AN may use httpOnly session cookies ──────────
        # Playwright CAN read httpOnly cookies (unlike JS running in the page).
        # Many sites store the auth token in a cookie named 'token', 'jwt',
        # 'session', 'an_token', etc.
        if not captured_token:
            print("  Checking cookies for auth token...")
            try:
                cookies = await context.cookies()
                # Sort by value length descending — tokens are long strings
                cookies_sorted = sorted(cookies, key=lambda c: len(c.get("value", "")), reverse=True)
                # AN_SESSION_TOKEN_V1 is Action Network's auth cookie (JWT)
                AN_COOKIE_NAME = "AN_SESSION_TOKEN_V1"
                for cookie in cookies_sorted:
                    name  = cookie.get("name", "")
                    value = cookie.get("value", "")
                    if len(value) < 20:
                        continue
                    # Check known AN cookie name first
                    if name == AN_COOKIE_NAME:
                        captured_token = value.strip()
                        print(f"  ✓ Token found in cookie '{name}'")
                        break
                    # Fallback: any cookie with auth-related name
                    if any(k in name.lower() for k in ["token", "auth", "jwt", "bearer", "an_"]):
                        captured_token = value.strip()
                        print(f"  ✓ Token found in cookie '{name}'")
                        break
                if not captured_token and cookies_sorted:
                    # Print the top 5 longest cookies for debugging
                    print("  Top cookies by length (for debugging):")
                    for c in cookies_sorted[:5]:
                        print(f"    {c['name']}: {len(c.get('value',''))} chars — {c.get('value','')[:40]}...")
            except Exception as e:
                print(f"  Cookie check failed: {e}")

        # ── Scroll the page to trigger lazy-loaded API calls ─────────────
        if not captured_token:
            print("  Scrolling to trigger additional API calls...")
            try:
                await page.evaluate("window.scrollTo(0, 500)")
                await asyncio.sleep(3)
                await page.evaluate("window.scrollTo(0, 1000)")
                await asyncio.sleep(3)
            except Exception:
                pass

        await browser.close()

    if not captured_token:
        print("  WARNING: Could not capture token automatically.")
        print("  Try running with headless=False to debug visually:")
        print("    python utils/an_login.py --visible")

    return captured_token


# =============================================================================
# MANUAL LOGIN — for when automated login is blocked by Cloudflare
# =============================================================================

async def _manual_login_and_capture(email: str = "") -> str:
    """
    Open a visible browser, let the user log in manually, then capture the token.

    This bypasses all bot detection because a real human is performing the login.
    Once logged in, we navigate to the AN API endpoint — the authenticated browser
    automatically includes the Bearer token in that request, and we capture it.

    Parameters
    ----------
    email : str   Pre-fill the email field (optional, user can type their own)

    Returns
    -------
    str : The captured Bearer token, or "" if capture failed.
    """
    try:
        from playwright.async_api import async_playwright, TimeoutError as PWTimeout
    except ImportError:
        print("  ERROR: playwright not installed.")
        print("  Run: pip3 install playwright && python3 -m playwright install chromium")
        return ""

    captured_token = ""

    async with async_playwright() as p:
        print()
        print("  Opening browser for manual login...")
        print("  ─────────────────────────────────────────────────────")
        print("  1. Log into Action Network in the browser window")
        print("  2. Wait until you see the MLB odds page fully loaded")
        print("  3. Come back here and press ENTER")
        print("  ─────────────────────────────────────────────────────")
        print()

        browser = await p.chromium.launch(
            headless=False,
            args=["--no-sandbox"],
        )
        context = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()

        # Navigate to login page — user will complete it themselves
        await page.goto(LOGIN_URL, wait_until="domcontentloaded", timeout=20000)

        # Pre-fill email if provided (user just has to enter password)
        if email:
            try:
                await page.fill('input[name="email"]', email, timeout=3000)
                print(f"  Email pre-filled ({email}). Please enter your password and log in.")
            except Exception:
                print("  Please log in to Action Network in the browser window.")

        # Wait for user to complete login and press Enter
        # asyncio.get_event_loop().run_in_executor allows blocking input() inside async code
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: input("  Press ENTER after you are logged in and can see the MLB page... "))

        # ── Now capture the token from the authenticated session ───────────
        # Set up network interception BEFORE navigating to the API endpoint
        async def on_request(request):
            nonlocal captured_token
            if captured_token:
                return
            auth_header = request.headers.get("authorization", "")
            if auth_header.startswith("Bearer ") and len(auth_header) > 20:
                captured_token = auth_header.replace("Bearer ", "").strip()
                print(f"\n  ✓ Token captured from: {request.url[:80]}...")

        async def on_response(response):
            nonlocal captured_token
            if captured_token:
                return
            url_lower = response.url.lower()
            if not any(k in url_lower for k in ["login", "auth", "session", "token", "signin"]):
                return
            try:
                body = await response.json()
                def _find_token(obj, depth=0):
                    if depth > 3 or not isinstance(obj, dict):
                        return ""
                    for key, val in obj.items():
                        if isinstance(val, str) and len(val) > 30 and any(
                            k in key.lower() for k in ["token", "jwt", "auth", "bearer"]
                        ):
                            return val.replace("Bearer ", "").strip()
                        nested = _find_token(val, depth + 1)
                        if nested:
                            return nested
                    return ""
                found = _find_token(body)
                if found:
                    captured_token = found
                    print(f"\n  ✓ Token in response body from: {response.url[:80]}...")
            except Exception:
                pass

        page.on("request", on_request)
        page.on("response", on_response)

        # Navigate to the AN scoreboard API — this will trigger an authenticated request
        print("\n  Navigating to Action Network API to capture token...")
        api_urls = [
            f"https://{API_HOST}/web/v1/scoreboard/mlb",
            TRIGGER_URL,
        ]
        for url in api_urls:
            if captured_token:
                break
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=15000)
                await asyncio.sleep(4)
            except Exception:
                pass

        # Check localStorage / sessionStorage
        if not captured_token:
            try:
                storage_data = await page.evaluate("""
                    () => {
                        const result = {};
                        const check = (store, prefix) => {
                            for (let i = 0; i < store.length; i++) {
                                const key = store.key(i);
                                result[prefix + key] = store.getItem(key);
                            }
                        };
                        try { check(localStorage, 'local:'); } catch(e) {}
                        try { check(sessionStorage, 'session:'); } catch(e) {}
                        return result;
                    }
                """)
                for key, value in storage_data.items():
                    if not value or len(value) < 20:
                        continue
                    key_lower = key.lower()
                    if any(k in key_lower for k in ["token", "auth", "jwt", "bearer"]):
                        captured_token = value.strip().replace("Bearer ", "")
                        print(f"  ✓ Token found in storage['{key}']")
                        break
                    try:
                        parsed = json.loads(value)
                        if isinstance(parsed, dict):
                            for k, v in parsed.items():
                                if isinstance(v, str) and len(v) > 30 and any(
                                    x in k.lower() for x in ["token", "auth", "jwt"]
                                ):
                                    captured_token = v.strip().replace("Bearer ", "")
                                    print(f"  ✓ Token in storage['{key}']['{k}']")
                                    break
                    except Exception:
                        pass
                    if captured_token:
                        break
            except Exception:
                pass

        # Check cookies
        if not captured_token:
            try:
                cookies = await context.cookies()
                AN_COOKIE_NAME = "AN_SESSION_TOKEN_V1"
                cookies_sorted = sorted(cookies, key=lambda c: len(c.get("value", "")), reverse=True)
                for cookie in cookies_sorted:
                    name  = cookie.get("name", "")
                    value = cookie.get("value", "")
                    if len(value) < 20:
                        continue
                    if name == AN_COOKIE_NAME:
                        captured_token = value.strip()
                        print(f"  ✓ Token found in cookie '{name}'")
                        break
                    if any(k in name.lower() for k in ["token", "auth", "jwt", "bearer", "an_"]):
                        captured_token = value.strip()
                        print(f"  ✓ Token found in cookie '{name}'")
                        break
                if not captured_token and cookies_sorted:
                    print("  Cookies found (for debugging):")
                    for c in cookies_sorted[:8]:
                        print(f"    {c['name']}: {len(c.get('value',''))} chars")
            except Exception:
                pass

        await browser.close()

    return captured_token


def manual_login() -> str:
    """
    Prompt user to log in manually in a visible browser, then capture the token.

    Use this when automated login is blocked by Cloudflare/bot protection.
    Run: python utils/an_login.py --manual
    """
    creds = load_credentials()
    email = creds.get("email", "")

    print("\n  Manual login mode — bypasses all bot detection.")
    token = asyncio.run(_manual_login_and_capture(email=email))

    if token:
        save_token(token)
        _patch_auth_token_in_module(token)
        print(f"\n  ✓ Token saved. Valid for ~{TOKEN_TTL_HOURS} hours.")
        print(f"  Token: {token[:12]}...{token[-6:]}")
    else:
        print("\n  WARNING: Could not capture token from authenticated session.")
        print("  Try the manual paste option: python utils/an_login.py --paste-token")

    return token


# =============================================================================
# PUBLIC INTERFACE
# =============================================================================

def refresh_token(headless: bool = True) -> str:
    """
    Log into Action Network and save a fresh token.

    This is the main function called by export files to keep the token current.
    It loads saved credentials, runs the async login, and saves the new token.

    Returns
    -------
    str : The new token, or "" if login failed.
    """
    creds = load_credentials()
    email    = creds.get("email", "")
    password = creds.get("password", "")

    if not email or not password:
        print("  ERROR: No credentials saved. Run setup first:")
        print("    python utils/an_login.py --setup")
        return ""

    print(f"  Logging into Action Network as {email}...")
    # asyncio.run() executes the async function synchronously
    # In R: there's no equivalent — R doesn't have native async/await
    token = asyncio.run(_login_and_capture_token(email, password, headless=headless))

    if token:
        save_token(token)
        # Also update the AUTH_TOKEN in action_network.py so it's live immediately
        _patch_auth_token_in_module(token)
        print(f"  ✓ Token refreshed and saved. Valid for ~{TOKEN_TTL_HOURS} hours.")
    else:
        print("  WARNING: Token refresh failed. Using last saved token if available.")
        token = creds.get("token", "")

    return token


def refresh_token_if_needed(force: bool = False) -> str:
    """
    Refresh the token only if it's stale (older than TOKEN_TTL_HOURS).

    Call this at the top of each export file's main block to ensure
    the token is always fresh before making API calls.

    Usage in export files:
        from utils.an_login import refresh_token_if_needed
        refresh_token_if_needed()

    Parameters
    ----------
    force : bool   If True, refresh even if token appears fresh.

    Returns
    -------
    str : The current valid token.
    """
    creds = load_credentials()
    token = creds.get("token", "")

    if not force and is_token_fresh(creds):
        print(f"  ✓ Action Network token is fresh (saved "
              f"{creds.get('token_saved_at','?')[:16]}). Skipping refresh.")
        # Patch the module's AUTH_TOKEN with the current saved token
        _patch_auth_token_in_module(token)
        return token

    print("  Action Network token is stale or missing. Refreshing...")
    return refresh_token()


def get_current_token() -> str:
    """Return the currently saved token without triggering a refresh."""
    return load_credentials().get("token", "")


def _patch_auth_token_in_module(token: str):
    """
    Inject the current token into the action_network module at runtime.

    This avoids needing to restart Python or reload the module — it directly
    sets the AUTH_TOKEN variable in the already-imported module.

    In R: there's no exact equivalent; this modifies a variable in another
    module's namespace dynamically, like environment() manipulation.
    """
    try:
        # Import the module if it's accessible
        module_path = Path(__file__).parent / "action_network.py"
        if not module_path.exists():
            return

        # Try to update the already-imported module's AUTH_TOKEN variable
        import importlib
        import utils.action_network as an_module
        an_module.AUTH_TOKEN = token
        # Also update session headers if a session exists
    except Exception:
        pass  # Silent fail — module may not be imported yet


# =============================================================================
# SETUP AND CLI
# =============================================================================

def interactive_setup():
    """
    Interactive first-time setup: prompt for email/password and do a test login.

    Run this once:  python utils/an_login.py --setup
    """
    print("=" * 60)
    print("  ACTION NETWORK — FIRST-TIME SETUP")
    print("=" * 60)
    print()
    print("  Your credentials will be saved locally to:")
    print(f"  {CREDS_FILE}")
    print("  (This file is gitignored and stays on your machine only)")
    print()

    email    = input("  Action Network email: ").strip()
    # getpass hides the password as you type (no echo) — like a proper password prompt
    password = getpass.getpass("  Action Network password: ").strip()

    if not email or not password:
        print("  ERROR: Email and password are required.")
        return

    # Save credentials first (so refresh_token() can read them)
    save_credentials(email, password)

    print()
    print("  Running test login (this opens a browser — may take 15–30 seconds)...")
    token = refresh_token(headless=True)

    if token:
        print()
        print("  ✓ SUCCESS! Setup complete.")
        print(f"  Token: {token[:12]}...{token[-6:]}")
        print()
        print("  You can now run the export files — they'll auto-refresh the token.")
        print("  To test odds pull: python utils/action_network.py")
    else:
        print()
        print("  Automated login blocked by Cloudflare bot detection.")
        print()
        print("  ── Next steps ─────────────────────────────────────────────────")
        print("  OPTION 1 (recommended): Manual login in browser window")
        print("    python3 utils/an_login.py --manual")
        print("    → A browser opens, YOU log in, we capture the token automatically.")
        print()
        print("  OPTION 2: Paste token manually from your browser's DevTools")
        print("    python3 utils/an_login.py --paste-token")
        print("    → In your browser: DevTools → Network → any api.actionnetwork.com")
        print("      request → Headers → Authorization: Bearer <token>")
        print("  ───────────────────────────────────────────────────────────────")


# =============================================================================
# MAIN — CLI entry point
# =============================================================================

if __name__ == "__main__":
    """
    Usage:
      python utils/an_login.py --setup      First-time credential setup
      python utils/an_login.py --refresh    Force a token refresh now
      python utils/an_login.py --visible    Refresh with visible browser (debug)
      python utils/an_login.py --status     Show current token status
    """
    parser = argparse.ArgumentParser(
        description="Action Network token manager"
    )
    parser.add_argument("--setup",       action="store_true", help="First-time setup")
    parser.add_argument("--refresh",     action="store_true", help="Force token refresh")
    parser.add_argument("--visible",     action="store_true", help="Refresh with visible browser")
    parser.add_argument("--manual",      action="store_true", help="Manual login in browser (bypasses bot detection)")
    parser.add_argument("--paste-token", action="store_true", help="Manually paste a token from DevTools")
    parser.add_argument("--status",      action="store_true", help="Show token status")

    args = parser.parse_args()

    if args.setup:
        interactive_setup()

    elif args.manual:
        manual_login()

    elif getattr(args, "paste_token", False):
        print("\n  Open your browser, log into actionnetwork.com, then:")
        print("  DevTools (F12) → Network tab → click any request to api.actionnetwork.com")
        print("  → Headers → find 'Authorization: Bearer <token>'")
        print("  Copy everything AFTER 'Bearer ' and paste it below.\n")
        token = input("  Paste token here: ").strip().replace("Bearer ", "")
        if token and len(token) > 20:
            save_token(token)
            _patch_auth_token_in_module(token)
            print(f"\n  ✓ Token saved. Valid for ~{TOKEN_TTL_HOURS} hours.")
            print(f"  Token: {token[:12]}...{token[-6:]}")
        else:
            print("  Token too short — paste the full token string.")

    elif args.refresh or args.visible:
        headless = not args.visible
        token    = refresh_token(headless=headless)
        if token:
            print(f"\n  Token: {token[:12]}...{token[-6:]}")

    elif args.status:
        creds = load_credentials()
        token = creds.get("token", "")
        saved = creds.get("token_saved_at", "never")
        fresh = is_token_fresh(creds)
        print(f"\n  Email:   {creds.get('email', 'not set')}")
        print(f"  Token:   {'SET' if token else 'NOT SET'} "
              f"({'fresh' if fresh else 'STALE'}) — saved {saved[:16]}")
        print(f"  TTL:     {TOKEN_TTL_HOURS} hours")

    else:
        # Default: refresh if needed
        token = refresh_token_if_needed()
        if token:
            print(f"\n  Ready. Token: {token[:12]}...{token[-6:]}")
