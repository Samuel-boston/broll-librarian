"""Mirror the library index into the Content Ops dashboard's Supabase.

The librarian keeps its working index in a local SQLite file (fast search,
embeddings). The dashboard's Library -> Footage index reads a copy of that
index from Supabase, so it works wherever the librarian runs: on a Mac, or on
a server. This module keeps the copy current.

It talks to Supabase over plain HTTPS (PostgREST and Storage), so there is no
extra dependency. Only shots whose content changed are sent, and thumbnails
are uploaded once. What was last sent is remembered in ``dashboard_sync.json``
inside the workspace, so a restart does not re-send everything.

Video files are never uploaded. The dashboard stores metadata, a thumbnail and
the Drive link.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from ..config import WorkspaceConfig
from ..db.store import Store

log = logging.getLogger(__name__)

URL_ENV = "DASHBOARD_SUPABASE_URL"
KEY_ENV = "DASHBOARD_SUPABASE_KEY"
BUCKET = "library-thumbs"
TABLE = "library_shots"
CHUNK = 200

# The columns the dashboard's library_shots table (migration 030) is fed from.
SHOT_SQL = """
    SELECT s.id, s.source_id, s.start_s, s.end_s, s.duration_s, s.caption,
           s.action, s.setting, s.shot_type, s.emotions_json, s.subjects_json,
           s.category, s.top_pick, s.featured_person, s.search_text,
           s.thumbnail_path,
           src.media_kind, src.original_filename, src.drive_file_id,
           src.drive_web_link, src.drive_path
    FROM shots s JOIN sources src ON src.id = s.source_id
    WHERE s.workspace_id = ? AND s.status = 'indexed'
"""


class DashboardSyncError(RuntimeError):
    """Something a person can act on: wrong key, missing table, no network."""


@dataclass
class SyncResult:
    total: int = 0
    upserted: int = 0
    thumbnails: int = 0
    removed: int = 0
    skipped_reason: str | None = None
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.skipped_reason:
            return f"Dashboard sync skipped: {self.skipped_reason}"
        return (
            f"Dashboard: {self.total} shots in the library, {self.upserted} updated, "
            f"{self.thumbnails} thumbnails sent, {self.removed} removed."
        )


def credentials(config: WorkspaceConfig) -> tuple[str, str] | None:
    """(project URL, service key) when the dashboard is connected, else None."""
    url = (os.environ.get(URL_ENV) or config.dashboard.supabase_url or "").strip().rstrip("/")
    key = (os.environ.get(KEY_ENV) or "").strip()
    if not config.dashboard.enabled or not url or not key:
        return None
    return url, key


def is_connected(config: WorkspaceConfig) -> bool:
    return credentials(config) is not None


def _headers(key: str, **extra: str) -> dict[str, str]:
    return {"apikey": key, "Authorization": f"Bearer {key}", **extra}


def _parse_list(text: str | None) -> list[str]:
    try:
        value = json.loads(text or "[]")
    except (TypeError, ValueError):
        return []
    return [str(v) for v in value] if isinstance(value, list) else []


def _record(row: Any, thumb_path: str | None) -> dict[str, Any]:
    return {
        "id": row["id"],
        "source_id": row["source_id"],
        "media_kind": "image" if row["media_kind"] == "image" else "video",
        "filename": row["original_filename"],
        "caption": row["caption"],
        "action": row["action"],
        "setting": row["setting"],
        "shot_type": row["shot_type"],
        "emotions": _parse_list(row["emotions_json"]),
        "subjects": _parse_list(row["subjects_json"]),
        "category": row["category"],
        "featured_person": bool(row["featured_person"]),
        "top_pick": bool(row["top_pick"]),
        "start_s": row["start_s"],
        "end_s": row["end_s"],
        "duration_s": row["duration_s"],
        "drive_file_id": row["drive_file_id"],
        "drive_web_link": row["drive_web_link"],
        "drive_path": row["drive_path"],
        "thumb_path": thumb_path,
        "search_text": row["search_text"],
    }


def _fingerprint(record: dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()


def _friendly(response: httpx.Response) -> DashboardSyncError:
    body = response.text[:300]
    if response.status_code in (401, 403):
        return DashboardSyncError(
            "The dashboard rejected the key. Use the service_role key from Supabase "
            "> Project Settings > API, not the anon key."
        )
    if response.status_code == 404 or "library_shots" in body and "does not exist" in body:
        return DashboardSyncError(
            "The dashboard database has no library_shots table yet. Run the "
            "dashboard's supabase/setup_all.sql (or migration_030) in Supabase first."
        )
    return DashboardSyncError(f"Dashboard answered {response.status_code}: {body}")


class DashboardSync:
    def __init__(self, config: WorkspaceConfig, client: httpx.Client | None = None):
        self.config = config
        self._client = client
        self.state_path: Path = config.dir / "dashboard_sync.json"

    # -- state --------------------------------------------------------------

    def _load_state(self, url: str) -> dict[str, Any]:
        try:
            state = json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return {"url": url, "shots": {}, "thumbs": []}
        if state.get("url") != url:  # pointed at a different dashboard: start over
            return {"url": url, "shots": {}, "thumbs": []}
        return state

    def _save_state(self, state: dict[str, Any]) -> None:
        self.state_path.write_text(json.dumps(state))

    # -- connection ---------------------------------------------------------

    def check(self) -> None:
        """Raise DashboardSyncError with a plain-English reason if it cannot work."""
        creds = credentials(self.config)
        if creds is None:
            raise DashboardSyncError("The dashboard connection is not set up.")
        url, key = creds
        with self._http() as http:
            try:
                r = http.get(f"{url}/rest/v1/{TABLE}", params={"select": "id", "limit": "1"},
                             headers=_headers(key))
            except httpx.HTTPError as exc:
                raise DashboardSyncError(f"Could not reach the dashboard database: {exc}") from exc
            if r.status_code >= 400:
                raise _friendly(r)

    def _http(self):
        if self._client is not None:
            return _Borrowed(self._client)
        return httpx.Client(timeout=60.0)

    # -- sync ---------------------------------------------------------------

    def run(self, prune: bool = True, force: bool = False) -> SyncResult:
        result = SyncResult()
        creds = credentials(self.config)
        if creds is None:
            result.skipped_reason = "the dashboard is not connected (run `broll connect-dashboard`)"
            return result
        url, key = creds

        store = Store.for_config(self.config)
        try:
            rows = store.conn.execute(SHOT_SQL, (self.config.id,)).fetchall()
        finally:
            store.close()
        result.total = len(rows)

        state = {"url": url, "shots": {}, "thumbs": []} if force else self._load_state(url)
        sent: dict[str, str] = state["shots"]
        thumbs: set[str] = set(state["thumbs"])

        changed: list[dict[str, Any]] = []
        current: dict[str, str] = {}
        with self._http() as http:
            for row in rows:
                thumb_file = self._thumbnail_file(row)
                thumb_path = f"{row['id']}.jpg" if thumb_file else None
                record = _record(row, thumb_path)
                digest = _fingerprint(record)
                current[row["id"]] = digest

                if thumb_file and row["id"] not in thumbs:
                    try:
                        self._upload_thumb(http, url, key, thumb_path, thumb_file)
                        thumbs.add(row["id"])
                        result.thumbnails += 1
                    except DashboardSyncError as exc:
                        result.errors.append(str(exc))
                        record["thumb_path"] = None
                        digest = _fingerprint(record)
                        current[row["id"]] = digest
                if sent.get(row["id"]) != digest:
                    record["synced_at"] = datetime.now(UTC).isoformat()
                    changed.append(record)

            for i in range(0, len(changed), CHUNK):
                chunk = changed[i:i + CHUNK]
                r = http.post(
                    f"{url}/rest/v1/{TABLE}", params={"on_conflict": "id"}, json=chunk,
                    headers=_headers(key, **{"Prefer": "resolution=merge-duplicates,return=minimal",
                                             "Content-Type": "application/json"}),
                )
                if r.status_code >= 400:
                    self._save_state({"url": url, "shots": {k: v for k, v in sent.items()},
                                      "thumbs": sorted(thumbs)})
                    raise _friendly(r)
                for rec in chunk:
                    sent[rec["id"]] = current[rec["id"]]
                result.upserted += len(chunk)

            stale = [sid for sid in sent if sid not in current]
            # Never wipe the dashboard because the local index came back empty.
            if prune and stale and rows:
                self._remove(http, url, key, stale)
                for sid in stale:
                    sent.pop(sid, None)
                    thumbs.discard(sid)
                result.removed = len(stale)

        self._save_state({"url": url, "shots": sent, "thumbs": sorted(thumbs)})
        return result

    # -- helpers ------------------------------------------------------------

    def _thumbnail_file(self, row: Any) -> Path | None:
        for candidate in (row["thumbnail_path"], self.config.thumbnails_dir / f"{row['id']}.jpg"):
            if candidate and Path(candidate).is_file():
                return Path(candidate)
        return None

    @staticmethod
    def _upload_thumb(http: httpx.Client, url: str, key: str, name: str, path: Path) -> None:
        r = http.post(
            f"{url}/storage/v1/object/{BUCKET}/{name}", content=path.read_bytes(),
            headers=_headers(key, **{"Content-Type": "image/jpeg", "x-upsert": "true"}),
        )
        if r.status_code >= 400:
            raise DashboardSyncError(f"Thumbnail {name} not uploaded ({r.status_code}): {r.text[:200]}")

    @staticmethod
    def _remove(http: httpx.Client, url: str, key: str, ids: list[str]) -> None:
        for i in range(0, len(ids), CHUNK):
            chunk = ids[i:i + CHUNK]
            quoted = ",".join(f'"{sid}"' for sid in chunk)
            r = http.delete(f"{url}/rest/v1/{TABLE}", params={"id": f"in.({quoted})"},
                            headers=_headers(key))
            if r.status_code >= 400:
                raise _friendly(r)
            http.request("DELETE", f"{url}/storage/v1/object/{BUCKET}",
                         json={"prefixes": [f"{sid}.jpg" for sid in chunk]},
                         headers=_headers(key, **{"Content-Type": "application/json"}))


class _Borrowed:
    """Lets a caller-supplied client be used in a ``with`` without being closed."""

    def __init__(self, client: httpx.Client):
        self.client = client

    def __enter__(self) -> httpx.Client:
        return self.client

    def __exit__(self, *exc: Any) -> None:
        return None
