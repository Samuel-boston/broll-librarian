"""Watched folders: footage that files itself.

An editor working across several clients should not have to open an app to add
footage. Drop a card's worth of clips into the client's folder and they are
queued, analysed, filed into that client's Drive tree, and searchable.

Two ways to say where to look:

* ``ingest.watch_dirs`` in a client's config - any folders you like.
* ``BROLL_INBOX`` - one root with a subfolder per client, created on startup.
  ``~/Broll Inbox/adam-kunder/`` is then Adam's drop folder.

Nothing in a watched folder is moved or deleted. The file is read where it
lies, and the library remembers it by content hash, so dropping the same clip
in twice costs one analysis, not two.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from ..config import WorkspaceConfig
from ..db.store import Store
from ..jobs.queue import enqueue_files
from .scanner import scan_local

log = logging.getLogger(__name__)

INBOX_ENV = "BROLL_INBOX"

# A file still being copied in grows between scans. Waiting for its size to
# settle is cruder than watching the filesystem and far more portable.
SETTLE_S = 2.0


def inbox_dir(config: WorkspaceConfig) -> Path | None:
    """This client's folder under BROLL_INBOX, if one is configured."""
    root = os.environ.get(INBOX_ENV, "").strip()
    return Path(root).expanduser() / config.id if root else None


def watched_dirs(config: WorkspaceConfig, create: bool = False) -> list[Path]:
    dirs = [Path(d).expanduser() for d in config.ingest.watch_dirs]
    inbox = inbox_dir(config)
    if inbox is not None:
        dirs.append(inbox)
    if create:
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)
    return [d for d in dirs if d.is_dir()]


def _settled(path: Path) -> bool:
    """True once the file has stopped growing - i.e. the copy has finished."""
    try:
        first = path.stat().st_size
    except OSError:
        return False
    import time

    time.sleep(SETTLE_S)
    try:
        return path.stat().st_size == first and first > 0
    except OSError:
        return False


def sweep(config: WorkspaceConfig, store: Store) -> int:
    """Queue anything new in this client's watched folders. Returns how many."""
    found = []
    for directory in watched_dirs(config):
        for discovered in scan_local(directory):
            if discovered.path and _settled(discovered.path):
                found.append(discovered)
    if not found:
        return 0
    queued = enqueue_files(store, found)
    if queued:
        log.info("watch: queued %d new file(s) for %s", len(queued), config.id)
    return len(queued)


def sweep_once(config: WorkspaceConfig, store_factory) -> int:
    """Open a database connection, sweep, close it.

    The connection is opened here rather than passed in because this runs in a
    worker thread, and a SQLite connection belongs to the thread that made it.
    """
    store = store_factory()
    try:
        return sweep(config, store)
    finally:
        store.close()


async def watch_loop(config: WorkspaceConfig, store_factory) -> None:
    """Sweep this client's watched folders for as long as the app runs."""
    interval = max(5, config.ingest.watch_interval_s)
    while True:
        try:
            if watched_dirs(config, create=True):
                await asyncio.to_thread(sweep_once, config, store_factory)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a watch problem must not stop the app
            log.warning("watch sweep failed for %s: %s", config.id, exc)
        await asyncio.sleep(interval)
