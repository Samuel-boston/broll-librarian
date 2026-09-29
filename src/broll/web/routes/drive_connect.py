"""Connect Google Drive from the browser, for an install on a server.

`broll drive login` opens a browser window on the machine it runs on, which a server doesn't have.
Here the person clicks "Connect Google Drive" in Settings, signs in at Google, and Google sends them
back to /drive/callback. The login is then stored exactly as `broll drive login` would store it.

Needs a Google OAuth client of type "Web application" with this app's /drive/callback address
registered as an authorised redirect URI, and BROLL_PUBLIC_URL set to the app's public address.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import time
from urllib.parse import quote

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse

from ...drive.auth import SCOPES, DriveAuthError, save_credentials, web_client_config
from ..app import client_state

router = APIRouter()

PUBLIC_URL_ENV = "BROLL_PUBLIC_URL"
PENDING_TTL_S = 600


def redirect_uri(request: Request) -> str:
    """This app's /drive/callback address, as Google must have it registered."""
    base = os.environ.get(PUBLIC_URL_ENV, "").strip().rstrip("/")
    if not base:
        forwarded_proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip()
        host = request.headers.get("x-forwarded-host") or request.headers.get("host", "")
        base = f"{forwarded_proto or request.url.scheme}://{host}"
    return f"{base}/drive/callback"


def new_flow(uri: str):
    from google_auth_oauthlib.flow import Flow

    return Flow.from_client_config(web_client_config(uri), scopes=SCOPES, redirect_uri=uri)


def _back(message: str) -> RedirectResponse:
    return RedirectResponse(f"/settings?message={quote(message)}", status_code=303)


@router.get("/drive/connect")
async def drive_connect(request: Request):
    state = client_state(request)
    uri = redirect_uri(request)
    try:
        flow = new_flow(uri)
    except DriveAuthError as exc:
        return _back(str(exc))
    auth_url, oauth_state = flow.authorization_url(access_type="offline", prompt="consent", include_granted_scopes="true")
    pending = state.__dict__.setdefault("oauth_pending", {})
    now = time.time()
    for key in [k for k, v in pending.items() if now - v["at"] > PENDING_TTL_S]:
        pending.pop(key, None)
    # PKCE: Google needs the same verifier again when the person comes back.
    pending[oauth_state] = {"verifier": getattr(flow, "code_verifier", None), "uri": uri, "at": now}
    return RedirectResponse(auth_url, status_code=303)


@router.get("/drive/callback")
async def drive_callback(request: Request, code: str = "", state: str = "", error: str = ""):
    app_state = client_state(request)
    if error:
        return _back("Google Drive was not connected: " + ("you declined access." if error == "access_denied" else error))
    pending = app_state.__dict__.setdefault("oauth_pending", {})
    entry = pending.pop(state, None)
    if not code or entry is None or time.time() - entry["at"] > PENDING_TTL_S:
        return _back("That sign-in link has expired or wasn't started here. Press Connect Google Drive again.")
    try:
        flow = new_flow(entry["uri"])
        flow.code_verifier = entry["verifier"]
        await asyncio.to_thread(flow.fetch_token, code=code)
        if not flow.credentials.refresh_token:
            return _back("Google didn't give a long-lived login. Remove the app under myaccount.google.com/permissions and try again.")
        save_credentials(app_state.config, flow.credentials)
    except DriveAuthError as exc:
        return _back(str(exc))
    except Exception as exc:  # noqa: BLE001 - shown to the person, who can't read a traceback
        return _back(f"Google Drive was not connected: {type(exc).__name__}: {exc}"[:300])
    return _back("Google Drive is connected.")
