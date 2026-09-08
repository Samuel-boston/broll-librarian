"""Google Drive OAuth.

The full ``drive`` scope is required, not ``drive.file``: the narrower scope only
grants access to files the app itself created or the user picked, so it cannot
list an existing folder's contents, adopt a pre-existing root folder, or create
shortcuts inside folders the app did not create. See the README for why that
means each client uses their own Google Cloud project in testing mode.

Tokens are stored per workspace and never in the database.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..config import WorkspaceConfig

SCOPES = ["https://www.googleapis.com/auth/drive"]

CLIENT_ID_ENV = "GOOGLE_OAUTH_CLIENT_ID"
CLIENT_SECRET_ENV = "GOOGLE_OAUTH_CLIENT_SECRET"
CLIENT_SECRETS_FILE_ENV = "GOOGLE_OAUTH_CLIENT_SECRETS_FILE"


class DriveAuthError(RuntimeError):
    pass


def _require_libraries():
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as exc:
        raise DriveAuthError(
            "Drive support needs the extra: pip install 'broll-librarian[drive]'"
        ) from exc
    return Request, Credentials, InstalledAppFlow


def client_config() -> dict:
    """The OAuth client, from a downloaded JSON file or from two env vars."""
    secrets_file = os.environ.get(CLIENT_SECRETS_FILE_ENV)
    if secrets_file:
        path = Path(secrets_file).expanduser()
        if not path.exists():
            raise DriveAuthError(f"{CLIENT_SECRETS_FILE_ENV} points at {path}, which does not exist.")
        return json.loads(path.read_text())

    client_id = os.environ.get(CLIENT_ID_ENV)
    client_secret = os.environ.get(CLIENT_SECRET_ENV)
    if not (client_id and client_secret):
        raise DriveAuthError(
            f"No Google OAuth client. Set {CLIENT_ID_ENV} and {CLIENT_SECRET_ENV} "
            f"(or {CLIENT_SECRETS_FILE_ENV}) - see the Google Drive setup "
            "walkthrough in the README."
        )
    return {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }


def load_credentials(config: WorkspaceConfig):
    """Return stored credentials, refreshing them if needed, else None."""
    Request, Credentials, _ = _require_libraries()
    token_path = config.drive_token_path
    if not token_path.exists():
        return None
    credentials = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if credentials and credentials.expired and credentials.refresh_token:
        try:
            credentials.refresh(Request())
        except Exception as exc:
            raise DriveAuthError(
                f"Stored Drive token could not be refreshed ({exc}). "
                "Run `broll drive login` again. Note that Google expires refresh "
                "tokens for apps in testing mode after 7 days."
            ) from exc
        _save(config, credentials)
    return credentials


def login(config: WorkspaceConfig, port: int = 0) -> object:
    """Run the installed-app OAuth flow and store the token for this workspace."""
    _, _, InstalledAppFlow = _require_libraries()
    flow = InstalledAppFlow.from_client_config(client_config(), SCOPES)
    credentials = flow.run_local_server(port=port, prompt="consent")
    _save(config, credentials)
    return credentials


def logout(config: WorkspaceConfig) -> bool:
    if config.drive_token_path.exists():
        config.drive_token_path.unlink()
        return True
    return False


def _save(config: WorkspaceConfig, credentials) -> None:
    config.ensure_dirs()
    config.drive_token_path.write_text(credentials.to_json())
    config.drive_token_path.chmod(0o600)
