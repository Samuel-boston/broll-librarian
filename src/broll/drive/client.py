"""A thin, retrying, rate-limit-aware Drive wrapper.

Drive is far stricter than the AI APIs, so: modest concurrency, exponential
backoff on 429 and 5xx, and an aggressively cached folder tree. Resolving a path
naively costs one API call per level per file and will burn the quota in
minutes.

Every method here is also implemented by the in-memory fake used in the tests,
so the organiser can be exercised without touching Google.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger(__name__)

FOLDER_MIME = "application/vnd.google-apps.folder"
SHORTCUT_MIME = "application/vnd.google-apps.shortcut"

MAX_RETRIES = 6
BASE_DELAY_S = 1.0
MAX_DELAY_S = 64.0
RETRY_STATUSES = {403, 429, 500, 502, 503, 504}


@dataclass
class DriveFile:
    id: str
    name: str
    mime_type: str
    parents: list[str] = field(default_factory=list)
    web_view_link: str | None = None
    shortcut_target_id: str | None = None

    @property
    def is_folder(self) -> bool:
        return self.mime_type == FOLDER_MIME

    @property
    def is_shortcut(self) -> bool:
        return self.mime_type == SHORTCUT_MIME


class DriveError(RuntimeError):
    pass


# Drive reports rate limiting as 403 as well as 429, so 403 alone is not
# enough to decide: only these reasons are worth retrying.
RETRYABLE_403_REASONS = {"rateLimitExceeded", "userRateLimitExceeded", "backendError"}


class DriveStorageFullError(RuntimeError):
    """The Drive account has no room left. Retrying cannot help."""


def _reason(exc: Exception) -> str:
    for detail in getattr(exc, "error_details", None) or []:
        if isinstance(detail, dict) and detail.get("reason"):
            return detail["reason"]
    return "storageQuotaExceeded" if "storage quota" in str(exc).lower() else ""


def _is_retryable(exc: Exception) -> bool:
    status = getattr(getattr(exc, "resp", None), "status", None)
    if status == 403:
        return _reason(exc) in RETRYABLE_403_REASONS
    return status in RETRY_STATUSES


def with_backoff(operation, *, description: str = "drive call"):
    """Retry a Drive call with exponential backoff and jitter."""
    delay = BASE_DELAY_S
    last: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return operation()
        except Exception as exc:  # googleapiclient raises HttpError
            if _reason(exc) == "storageQuotaExceeded":
                raise DriveStorageFullError(
                    "This Google Drive is full, so nothing can be uploaded. Free up "
                    "space (emptying Drive's trash often does it) or connect a Drive "
                    "with room to spare."
                ) from exc
            if not _is_retryable(exc) or attempt == MAX_RETRIES:
                raise
            last = exc
            sleep_for = min(delay, MAX_DELAY_S) + random.uniform(0, 0.5)
            log.warning("%s failed (%s), retrying in %.1fs", description, exc, sleep_for)
            time.sleep(sleep_for)
            delay *= 2
    raise DriveError(f"{description} failed after {MAX_RETRIES} attempts: {last}")


class DriveClient:
    """The real client. Construct it with credentials from drive.auth."""

    def __init__(self, credentials, page_size: int = 200):
        try:
            from googleapiclient.discovery import build
        except ImportError as exc:
            raise DriveError(
                "Drive support needs the extra: pip install 'broll-librarian[drive]'"
            ) from exc
        self.service = build("drive", "v3", credentials=credentials, cache_discovery=False)
        self.page_size = page_size
        self._children: dict[str, dict[str, DriveFile]] = {}

    # -- reads --------------------------------------------------------------

    def get(self, file_id: str) -> DriveFile | None:
        def call():
            return self.service.files().get(
                fileId=file_id,
                fields="id,name,mimeType,parents,webViewLink,shortcutDetails",
                supportsAllDrives=True,
            ).execute()

        try:
            return _to_file(with_backoff(call, description=f"get {file_id}"))
        except Exception as exc:
            if getattr(getattr(exc, "resp", None), "status", None) == 404:
                return None
            raise

    def list_children(self, parent_id: str, refresh: bool = False) -> dict[str, DriveFile]:
        """Children by name, cached. One listing per folder per run."""
        if not refresh and parent_id in self._children:
            return self._children[parent_id]

        children: dict[str, DriveFile] = {}
        page_token = None
        while True:
            def call():
                return self.service.files().list(
                    q=f"'{parent_id}' in parents and trashed = false",
                    fields=("nextPageToken, files(id,name,mimeType,parents,"
                            "webViewLink,shortcutDetails)"),
                    pageSize=self.page_size,
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                ).execute()

            response = with_backoff(call, description=f"list {parent_id}")
            for item in response.get("files", []):
                entry = _to_file(item)
                children[entry.name] = entry
            page_token = response.get("nextPageToken")
            if not page_token:
                break

        self._children[parent_id] = children
        return children

    # -- writes -------------------------------------------------------------

    def create_folder(self, name: str, parent_id: str) -> DriveFile:
        def call():
            return self.service.files().create(
                body={"name": name, "mimeType": FOLDER_MIME, "parents": [parent_id]},
                fields="id,name,mimeType,parents,webViewLink",
                supportsAllDrives=True,
            ).execute()

        created = _to_file(with_backoff(call, description=f"create folder {name}"))
        self._children.setdefault(parent_id, {})[name] = created
        return created

    def ensure_folder(self, name: str, parent_id: str) -> DriveFile:
        existing = self.list_children(parent_id).get(name)
        if existing and existing.is_folder:
            return existing
        return self.create_folder(name, parent_id)

    def resolve_path(self, parts: Iterable[str], root_id: str) -> str:
        """Folder id for a path under root, creating the missing levels."""
        current = root_id
        for part in parts:
            current = self.ensure_folder(part, current).id
        return current

    def upload(self, path: Path, name: str, parent_id: str) -> DriveFile:
        from googleapiclient.http import MediaFileUpload

        media = MediaFileUpload(str(path), resumable=True)

        def call():
            return self.service.files().create(
                body={"name": name, "parents": [parent_id]},
                media_body=media,
                fields="id,name,mimeType,parents,webViewLink",
                supportsAllDrives=True,
            ).execute()

        created = _to_file(with_backoff(call, description=f"upload {name}"))
        self._children.setdefault(parent_id, {})[name] = created
        return created

    def download(self, file_id: str, destination: Path) -> Path:
        from googleapiclient.http import MediaIoBaseDownload

        destination.parent.mkdir(parents=True, exist_ok=True)
        request = self.service.files().get_media(fileId=file_id, supportsAllDrives=True)
        with destination.open("wb") as handle:
            downloader = MediaIoBaseDownload(handle, request)
            done = False
            while not done:
                _, done = with_backoff(downloader.next_chunk, description="download chunk")
        return destination

    def create_shortcut(self, target_id: str, name: str, parent_id: str) -> DriveFile:
        def call():
            return self.service.files().create(
                body={
                    "name": name,
                    "mimeType": SHORTCUT_MIME,
                    "parents": [parent_id],
                    "shortcutDetails": {"targetId": target_id},
                },
                fields="id,name,mimeType,parents,shortcutDetails",
                supportsAllDrives=True,
            ).execute()

        created = _to_file(with_backoff(call, description=f"shortcut {name}"))
        self._children.setdefault(parent_id, {})[name] = created
        return created

    def rename(self, file_id: str, name: str) -> DriveFile:
        def call():
            return self.service.files().update(
                fileId=file_id, body={"name": name},
                fields="id,name,mimeType,parents,webViewLink",
                supportsAllDrives=True,
            ).execute()

        self._children.clear()
        return _to_file(with_backoff(call, description=f"rename {file_id}"))

    def move(self, file_id: str, add_parent: str, remove_parents: Iterable[str]) -> DriveFile:
        def call():
            return self.service.files().update(
                fileId=file_id,
                addParents=add_parent,
                removeParents=",".join(remove_parents),
                fields="id,name,mimeType,parents,webViewLink",
                supportsAllDrives=True,
            ).execute()

        self._children.clear()
        return _to_file(with_backoff(call, description=f"move {file_id}"))

    def delete_shortcut(self, file_id: str) -> None:
        """Only ever called for shortcuts - never for a user's actual footage."""
        entry = self.get(file_id)
        if entry is None:
            return
        if not entry.is_shortcut:
            raise DriveError(
                f"refusing to delete {entry.name!r}: it is not a shortcut. "
                "This tool never deletes footage."
            )

        def call():
            return self.service.files().delete(fileId=file_id, supportsAllDrives=True).execute()

        with_backoff(call, description=f"delete shortcut {file_id}")
        self._children.clear()

    def invalidate(self) -> None:
        self._children.clear()


def _to_file(payload: dict[str, Any]) -> DriveFile:
    details = payload.get("shortcutDetails") or {}
    return DriveFile(
        id=payload["id"],
        name=payload.get("name", ""),
        mime_type=payload.get("mimeType", ""),
        parents=list(payload.get("parents") or []),
        web_view_link=payload.get("webViewLink"),
        shortcut_target_id=details.get("targetId"),
    )
