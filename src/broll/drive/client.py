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
    #: Bytes, when Drive reports one (folders, shortcuts and Google Docs have none).
    size: int | None = None
    md5: str | None = None
    #: Custom key/values stored on the file (the copier stamps each copy with where it came from).
    app_properties: dict[str, str] | None = None
    #: Length in seconds, for video Drive has finished processing (a file just uploaded may not have one yet).
    duration_s: float | None = None

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
        self.credentials = credentials
        self.page_size = page_size
        self._children: dict[str, dict[str, DriveFile]] = {}

    def access_token(self) -> str:
        """A current OAuth token, for a request made outside the Drive library (reading a video over HTTPS)."""
        credentials = self.credentials
        if not getattr(credentials, "valid", True):
            from google.auth.transport.requests import Request

            credentials.refresh(Request())
        return credentials.token

    # -- reads --------------------------------------------------------------

    def get(self, file_id: str) -> DriveFile | None:
        def call():
            return self.service.files().get(
                fileId=file_id,
                fields=("id,name,mimeType,parents,webViewLink,shortcutDetails,size,md5Checksum,appProperties,trashed,"
                        "videoMediaMetadata(durationMillis)"),
                supportsAllDrives=True,
            ).execute()

        try:
            payload = with_backoff(call, description=f"get {file_id}")
            if payload.get("trashed"):
                return None  # a file Adam trashed is gone as far as filing is concerned
            return _to_file(payload)
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
                            "webViewLink,shortcutDetails,size,md5Checksum,appProperties,"
                            "videoMediaMetadata(durationMillis))"),
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

    def list_all(self, parent_id: str) -> list[DriveFile]:
        """Every child of a folder, in a list, uncached. Unlike `list_children`, two files with
        the same name both come back: a folder of footage often has them."""
        found: list[DriveFile] = []
        page_token = None
        while True:
            def call():
                return self.service.files().list(
                    q=f"'{parent_id}' in parents and trashed = false",
                    fields=("nextPageToken, files(id,name,mimeType,parents,"
                            "webViewLink,shortcutDetails,size,md5Checksum,appProperties,"
                            "videoMediaMetadata(durationMillis))"),
                    pageSize=self.page_size,
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                ).execute()

            response = with_backoff(call, description=f"list {parent_id}")
            found.extend(_to_file(item) for item in response.get("files", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                return found

    def storage_quota(self) -> tuple[int | None, int]:
        """(limit, used) in bytes for the signed-in account. The limit is None when unlimited
        or pooled in a way Drive does not report."""
        def call():
            return self.service.about().get(fields="storageQuota").execute()

        quota = with_backoff(call, description="storage quota").get("storageQuota", {})
        limit = quota.get("limit")
        return (int(limit) if limit else None), int(quota.get("usage", 0))

    # -- writes -------------------------------------------------------------

    def copy_file(self, file_id: str, name: str, parent_id: str,
                  app_properties: dict[str, str] | None = None) -> DriveFile:
        """Copy a file inside Drive, on Google's side: nothing is downloaded or uploaded. The copy
        belongs to whoever is signed in, and counts against their storage."""
        body: dict[str, Any] = {"name": name, "parents": [parent_id]}
        if app_properties:
            body["appProperties"] = app_properties

        def call():
            return self.service.files().copy(
                fileId=file_id, body=body,
                fields="id,name,mimeType,parents,webViewLink,size,md5Checksum,appProperties",
                supportsAllDrives=True,
            ).execute()

        created = _to_file(with_backoff(call, description=f"copy {name}"))
        self._children.setdefault(parent_id, {})[name] = created
        return created

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

    def upload(self, path: Path, name: str, parent_id: str,
               app_properties: dict[str, str] | None = None) -> DriveFile:
        from googleapiclient.http import MediaFileUpload

        media = MediaFileUpload(str(path), resumable=True)
        body: dict[str, Any] = {"name": name, "parents": [parent_id]}
        if app_properties:
            body["appProperties"] = app_properties

        def call():
            return self.service.files().create(
                body=body,
                media_body=media,
                fields="id,name,mimeType,parents,webViewLink,appProperties",
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

    def create_doc(self, name: str, text: str, parent_id: str) -> DriveFile:
        """A Google Doc from plain text - Drive converts it on upload."""
        import io

        from googleapiclient.http import MediaIoBaseUpload

        media = MediaIoBaseUpload(io.BytesIO(text.encode("utf-8")), mimetype="text/plain")

        def call():
            return self.service.files().create(
                body={
                    "name": name,
                    "mimeType": "application/vnd.google-apps.document",
                    "parents": [parent_id],
                },
                media_body=media,
                fields="id,name,mimeType,parents,webViewLink",
                supportsAllDrives=True,
            ).execute()

        created = _to_file(with_backoff(call, description=f"create doc {name}"))
        self._children.setdefault(parent_id, {})[name] = created
        return created

    def rename(self, file_id: str, name: str) -> DriveFile:
        def call():
            return self.service.files().update(
                fileId=file_id, body={"name": name},
                fields="id,name,mimeType,parents,webViewLink",
                supportsAllDrives=True,
            ).execute()

        done = _to_file(with_backoff(call, description=f"rename {file_id}"))
        # Only the folder the file sits in changed. Emptying the whole cache made every file re-list every
        # shortcut folder it touched: tens of thousands of calls over a 5,000-file library.
        for parent in done.parents:
            self._children.pop(parent, None)
        return done

    def move(self, file_id: str, add_parent: str, remove_parents: Iterable[str]) -> DriveFile:
        removed = list(remove_parents)

        def call():
            return self.service.files().update(
                fileId=file_id,
                addParents=add_parent,
                removeParents=",".join(removed),
                fields="id,name,mimeType,parents,webViewLink",
                supportsAllDrives=True,
            ).execute()

        done = _to_file(with_backoff(call, description=f"move {file_id}"))
        for parent in (add_parent, *removed, *done.parents):
            self._children.pop(parent, None)
        return done

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
        for parent in entry.parents:
            self._children.pop(parent, None)

    def invalidate(self) -> None:
        self._children.clear()


def _to_file(payload: dict[str, Any]) -> DriveFile:
    details = payload.get("shortcutDetails") or {}
    millis = (payload.get("videoMediaMetadata") or {}).get("durationMillis")
    return DriveFile(
        id=payload["id"],
        name=payload.get("name", ""),
        mime_type=payload.get("mimeType", ""),
        parents=list(payload.get("parents") or []),
        web_view_link=payload.get("webViewLink"),
        shortcut_target_id=details.get("targetId"),
        size=int(payload["size"]) if payload.get("size") else None,
        md5=payload.get("md5Checksum"),
        app_properties=payload.get("appProperties") or None,
        duration_s=int(millis) / 1000.0 if millis else None,
    )
