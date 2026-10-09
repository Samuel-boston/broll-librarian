"""Downloading Drive-only sources into the workspace temp directory.

Probing, shot detection and frame extraction all need local bytes, so a source
that lives only in Drive is fetched first. Uploaded and local-path sources skip
this entirely. Fetched files land in the workspace temp directory, which is the
only place cleanup is ever allowed to delete from.

A download is written to a ".part" file and only renamed to its real name once it
is the size Drive says it should be. A run killed half way therefore leaves a
".part" file, never something that looks like the clip but is cut short - which
is what used to happen, and what made the next run analyse half a video as if it
were the whole thing.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from ..config import WorkspaceConfig
from ..ingest.errors import DiskSpaceError, IncompleteDownloadError
from .auth import load_credentials
from .client import DriveClient

log = logging.getLogger(__name__)

GB = 1_000_000_000
# A ".part" file older than this belongs to a download nobody is doing any more.
STALE_PART_S = 30 * 60


def sweep_partial_downloads(config: WorkspaceConfig, now: float | None = None) -> int:
    """Delete ".part" files left behind by a run that was killed. Returns how many."""
    now = now if now is not None else time.time()
    removed = 0
    if not config.temp_dir.is_dir():
        return 0
    for part in config.temp_dir.glob("*.part"):
        try:
            if now - part.stat().st_mtime > STALE_PART_S:
                part.unlink(missing_ok=True)
                removed += 1
        except OSError:
            continue
    return removed


# Bytes promised to downloads that are running. Two files starting together would each see the same free
# space and together overshoot it, so a download reserves its room before it starts.
_reserved = 0
_reserved_lock = threading.Lock()


@contextmanager
def reserve_room(directory: Path, needed_bytes: int | None, headroom_gb: float):
    """Hold `needed_bytes` of the disk for a download, or raise DiskSpaceError if it does not fit.

    "Fit" counts the downloads already running, not just what is free this instant.
    """
    global _reserved
    if not needed_bytes:
        yield
        return
    directory.mkdir(parents=True, exist_ok=True)
    with _reserved_lock:
        free = shutil.disk_usage(directory).free
        available = free - _reserved
        if available - needed_bytes < headroom_gb * GB:
            raise DiskSpaceError(
                f"Not enough free disk to download this file: it needs {needed_bytes / GB:.1f} GB and "
                f"only {max(0.0, available) / GB:.1f} GB is free (keeping {headroom_gb:.0f} GB spare). "
                "It will be tried again when space frees up."
            )
        _reserved += needed_bytes
    try:
        yield
    finally:
        with _reserved_lock:
            _reserved -= needed_bytes


def ensure_room(directory: Path, needed_bytes: int | None, headroom_gb: float) -> None:
    """Raise DiskSpaceError unless `needed_bytes` fits with `headroom_gb` to spare (nothing is held)."""
    with reserve_room(directory, needed_bytes, headroom_gb):
        pass


def fetch_drive_file(
    config: WorkspaceConfig,
    file_id: str,
    filename: str,
    client: DriveClient | None = None,
    expected_size: int | None = None,
) -> Path:
    if client is None:
        credentials = load_credentials(config)
        if credentials is None:
            raise RuntimeError(
                "Drive is not connected for this workspace. Run `broll drive login`."
            )
        client = DriveClient(credentials)

    config.ensure_dirs()
    if expected_size is None:
        entry = client.get(file_id)
        expected_size = entry.size if entry is not None else None

    destination = config.temp_dir / f"drive-{file_id}-{Path(filename).name}"
    if destination.exists() and destination.stat().st_size > 0:
        # Only a file of the size Drive reports is the file. Anything else is a leftover.
        if not expected_size or destination.stat().st_size == expected_size:
            return destination
        log.warning("discarding %s: %d bytes on disk, Drive says %d",
                    destination.name, destination.stat().st_size, expected_size)
        destination.unlink(missing_ok=True)

    partial = destination.with_name(destination.name + ".part")
    with reserve_room(config.temp_dir, expected_size, config.ingest.disk_headroom_gb):
        partial.unlink(missing_ok=True)
        log.info("downloading %s from Drive", filename)
        try:
            client.download(file_id, partial)
            got = partial.stat().st_size if partial.exists() else 0
            if expected_size and got != expected_size:
                raise IncompleteDownloadError(
                    f"{filename}: the download stopped at {got} of {expected_size} bytes"
                )
            if got == 0:
                raise IncompleteDownloadError(f"{filename}: the download was empty")
            os.replace(partial, destination)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
    return destination
