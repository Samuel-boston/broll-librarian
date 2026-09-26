"""Copies of the library database.

The index (shots, tags, vectors, corrections you made by hand) lives in one SQLite file. Losing it
loses the work of analysing the footage again, which costs money, so a hosted install keeps a few
recent copies next to it. The copy is made with SQLite's own backup call, so it is consistent even
while the app is writing.

These are copies on the same disk. They protect against a bad edit or a corrupted file, not against
losing the server: for that, switch on the host's own backups or snapshots.
"""

from __future__ import annotations

import os
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from .config import WorkspaceConfig

NAME = re.compile(r"^library-\d{8}-\d{6}\.db$")


def backups_dir(config: WorkspaceConfig) -> Path:
    return config.dir / "backups"


def backups_enabled(config: WorkspaceConfig) -> bool:
    return config.backup.enabled or os.environ.get("BROLL_BACKUPS", "").lower() in ("1", "true", "yes", "on")


def list_backups(config: WorkspaceConfig) -> list[Path]:
    directory = backups_dir(config)
    if not directory.is_dir():
        return []
    return sorted((p for p in directory.iterdir() if NAME.match(p.name)), reverse=True)


def backup_database(config: WorkspaceConfig, keep: int | None = None) -> Path:
    """Copy the database now, then delete all but the newest `keep` copies. Returns the new file."""
    keep = max(1, keep if keep is not None else config.backup.keep)
    directory = backups_dir(config)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    target = directory / f"library-{stamp}.db"
    partial = target.with_suffix(".part")

    source = sqlite3.connect(f"file:{config.db_path}?mode=ro", uri=True, timeout=30.0)
    try:
        destination = sqlite3.connect(str(partial))
        try:
            source.backup(destination)
        finally:
            destination.close()
    finally:
        source.close()
    partial.chmod(0o600)
    os.replace(partial, target)  # a half-written copy is never named like a real one

    for old in list_backups(config)[keep:]:
        old.unlink(missing_ok=True)
    return target
