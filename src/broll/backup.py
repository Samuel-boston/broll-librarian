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


class RestoreError(RuntimeError):
    pass


def _row_counts(path: Path) -> dict[str, int]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10.0)
    try:
        return {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("sources", "shots")
        }
    finally:
        connection.close()


def restore_database(config: WorkspaceConfig, backup: Path) -> dict[str, object]:
    """Put a backup in place of the library database.

    Stop the app first. SQLite keeps recent writes in library.db-wal and replays them on top of whatever
    library.db holds, so copying a backup over library.db alone would quietly bring the newer rows back. The
    -wal and -shm files are removed here, and the database being replaced is kept beside it, not deleted.
    Returns the row counts of the restored database and where the replaced one went.
    """
    backup = Path(backup)
    if not backup.is_file():
        raise RestoreError(f"There is no backup file at {backup}.")
    check = sqlite3.connect(f"file:{backup}?mode=ro", uri=True, timeout=10.0)
    try:
        if check.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RestoreError(f"{backup.name} failed SQLite's integrity check; it is not safe to restore.")
        if not check.execute("SELECT 1 FROM sqlite_master WHERE name = 'shots'").fetchone():
            raise RestoreError(f"{backup.name} is not a library database.")
    finally:
        check.close()

    live = config.db_path
    if live.exists():
        # Refuse while something is still using it: an exclusive lock cannot be taken under a running app.
        probe = sqlite3.connect(str(live), timeout=2.0)
        try:
            probe.execute("BEGIN EXCLUSIVE")
            probe.execute("ROLLBACK")
        except sqlite3.OperationalError as exc:
            raise RestoreError("The library is in use. Stop the app first (docker compose stop librarian).") from exc
        finally:
            probe.close()

    kept = None
    if live.exists():
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        kept = live.with_name(f"library.db.before-restore-{stamp}")
        connection = sqlite3.connect(str(live))
        replacement = sqlite3.connect(str(kept))
        try:
            connection.backup(replacement)  # includes whatever the -wal holds
        finally:
            replacement.close()
            connection.close()
        kept.chmod(0o600)
    for suffix in ("-wal", "-shm"):
        Path(str(live) + suffix).unlink(missing_ok=True)
    partial = live.with_suffix(".restoring")
    source = sqlite3.connect(f"file:{backup}?mode=ro", uri=True, timeout=10.0)
    try:
        destination = sqlite3.connect(str(partial))
        try:
            source.backup(destination)
        finally:
            destination.close()
    finally:
        source.close()
    partial.chmod(0o600)
    os.replace(partial, live)
    from .db.store import Store

    Store.for_config(config).close()  # brings an older backup up to the current schema
    return {"counts": _row_counts(live), "kept": kept}
