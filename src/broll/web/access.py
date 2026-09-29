"""An optional shared-password gate for a hosted install.

The app has no accounts. Behind Cloudflare Access (or any other login in front of it) it doesn't need
any. On a plain server with nothing in front, set BROLL_ACCESS_PASSWORD and everyone has to enter it
once; the browser then keeps a signed cookie for 30 days.

The cookie holds only an expiry time and a signature made with a key derived from the password, so
changing the password signs everyone out. Nothing is stored on the server. Failed logins are slowed
down, so the password can't be guessed by hammering the form.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from urllib.parse import quote

from fastapi import Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

PASSWORD_ENV = "BROLL_ACCESS_PASSWORD"
COOKIE = "broll_access"
TTL_S = 30 * 24 * 3600

# Failed attempts in the last five minutes: at most this many per address, and per server in total.
WINDOW_S = 300
MAX_PER_CLIENT = 5
MAX_TOTAL = 30

# Reachable without logging in: the login page itself and the health check the container uses.
PUBLIC_PATHS = ("/login", "/logout", "/healthz")

_failures: dict[str, list[float]] = {}


def password() -> str | None:
    return os.environ.get(PASSWORD_ENV) or None


def _key(pw: str) -> bytes:
    return hashlib.sha256(b"broll-access-v1:" + pw.encode()).digest()


def make_token(pw: str, now: float | None = None) -> str:
    expires = int((now if now is not None else time.time()) + TTL_S)
    signature = hmac.new(_key(pw), str(expires).encode(), hashlib.sha256).hexdigest()
    return f"{expires}.{signature}"


def valid_token(pw: str, token: str | None, now: float | None = None) -> bool:
    if not token or "." not in token:
        return False
    expires, _, signature = token.partition(".")
    if not expires.isdigit() or int(expires) < (now if now is not None else time.time()):
        return False
    expected = hmac.new(_key(pw), expires.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, expected)


def safe_next(target: str | None) -> str:
    """Only ever send people back to a page on this site."""
    if target and target.startswith("/") and not target.startswith("//") and "\\" not in target:
        return target
    return "/library"


def _client(request: Request) -> str:
    forwarded = request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for", "")
    return (forwarded.split(",")[0].strip() or (request.client.host if request.client else "?"))[:64]


def _recent(key: str, now: float) -> list[float]:
    kept = [t for t in _failures.get(key, []) if now - t < WINDOW_S]
    _failures[key] = kept
    return kept


def too_many_failures(request: Request, now: float | None = None) -> bool:
    now = now if now is not None else time.time()
    total = sum(len(_recent(k, now)) for k in list(_failures))
    return len(_recent(_client(request), now)) >= MAX_PER_CLIENT or total >= MAX_TOTAL


def record_failure(request: Request, now: float | None = None) -> None:
    now = now if now is not None else time.time()
    _recent(_client(request), now).append(now)


def reset_failures() -> None:
    _failures.clear()


def _secure(request: Request) -> bool:
    return request.url.scheme == "https" or request.headers.get("x-forwarded-proto", "").lower() == "https"


LOGIN_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sign in · B-Roll Librarian</title>
<style>
  body{margin:0;min-height:100vh;display:grid;place-items:center;background:#0b0d12;color:#e7eaf0;font:15px system-ui,sans-serif}
  form{width:min(360px,90vw);background:#141821;border:1px solid #232a38;border-radius:14px;padding:28px}
  h1{margin:0 0 4px;font-size:18px} p{margin:0 0 18px;color:#8b93a4;font-size:13px}
  input{width:100%;box-sizing:border-box;background:#0b0d12;color:inherit;border:1px solid #232a38;border-radius:10px;padding:11px 12px;font-size:15px}
  input:focus{outline:none;border-color:#8b8cff}
  button{margin-top:14px;width:100%;padding:11px;border:0;border-radius:10px;background:#6e6fff;color:#fff;font-size:15px;font-weight:600;cursor:pointer}
  .err{margin:0 0 14px;color:#ffb0be;font-size:13px}
</style></head><body>
<form method="post" action="/login">
  <h1>B-Roll Librarian</h1><p>Enter the team password.</p>
  __ERROR__
  <input type="hidden" name="next" value="__NEXT__">
  <input type="password" name="password" autocomplete="current-password" autofocus required aria-label="Password">
  <button type="submit">Sign in</button>
</form></body></html>"""


def login_page(next_url: str, error: str = "", status: int = 200) -> HTMLResponse:
    from html import escape

    body = LOGIN_PAGE.replace("__NEXT__", escape(next_url, quote=True)).replace(
        "__ERROR__", f'<p class="err">{escape(error)}</p>' if error else ""
    )
    return HTMLResponse(body, status_code=status)


def install(app) -> None:
    """Add the login routes and the gate. Does nothing until a password is set."""
    from fastapi import Form

    @app.get("/healthz", include_in_schema=False)
    async def healthz():
        return Response("ok", media_type="text/plain")

    @app.get("/login", include_in_schema=False)
    async def login_form(request: Request, next: str = ""):
        if password() is None:
            return RedirectResponse("/library", status_code=303)
        return login_page(safe_next(next))

    @app.post("/login", include_in_schema=False)
    async def login_submit(request: Request, password_field: str = Form("", alias="password"), next: str = Form("")):
        pw = password()
        if pw is None:
            return RedirectResponse("/library", status_code=303)
        if too_many_failures(request):
            return login_page(safe_next(next), "Too many wrong attempts. Wait a few minutes and try again.", 429)
        if not hmac.compare_digest(password_field.encode(), pw.encode()):
            record_failure(request)
            return login_page(safe_next(next), "That isn't the password.", 401)
        response = RedirectResponse(safe_next(next), status_code=303)
        response.set_cookie(COOKIE, make_token(pw), max_age=TTL_S, httponly=True, samesite="lax", secure=_secure(request))
        return response

    @app.post("/logout", include_in_schema=False)
    async def logout():
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie(COOKIE)
        return response

    @app.middleware("http")
    async def gate(request: Request, call_next):
        pw = password()
        if pw is None or request.url.path in PUBLIC_PATHS:
            return await call_next(request)
        # The JSON API carries its own auth (BROLL_API_TOKEN, or loopback only),
        # and an agent cannot follow a redirect to a login form.
        if request.url.path.startswith("/api/"):
            return await call_next(request)
        if valid_token(pw, request.cookies.get(COOKIE)):
            return await call_next(request)
        if request.headers.get("hx-request"):
            return Response(status_code=401, headers={"HX-Redirect": "/login"})
        target = request.url.path + (f"?{request.url.query}" if request.url.query else "")
        return RedirectResponse(f"/login?next={quote(target, safe='')}", status_code=303)
