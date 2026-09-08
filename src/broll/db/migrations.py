"""Schema creation and migration.

Versioning uses SQLite's ``user_version``. Each migration is an idempotent
callable; new versions are appended to MIGRATIONS and never edited in place.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SQL_DIR = Path(__file__).parent

SCHEMA_VERSION = 2


def _apply_sql_file(conn: sqlite3.Connection, name: str) -> None:
    conn.executescript((SQL_DIR / name).read_text())


def _v1_base_schema(conn: sqlite3.Connection) -> None:
    _apply_sql_file(conn, "schema.sql")


def _v2_drive_shortcuts(conn: sqlite3.Connection) -> None:
    """Remember every shortcut we create.

    Without this the organiser can only reconcile folders a source is *still*
    planned into, so a shortcut in a folder the taxonomy no longer justifies
    would survive forever.
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS drive_shortcuts (
            workspace_id  TEXT NOT NULL,
            source_id     TEXT NOT NULL,
            shot_id       TEXT NOT NULL,
            folder_path   TEXT NOT NULL,
            folder_id     TEXT,
            name          TEXT NOT NULL,
            shortcut_id   TEXT NOT NULL,
            target_id     TEXT NOT NULL,
            created_at    TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (workspace_id, folder_path, name)
        );
        CREATE INDEX IF NOT EXISTS idx_drive_shortcuts_source
            ON drive_shortcuts (workspace_id, source_id);
        """
    )


MIGRATIONS = {1: _v1_base_schema, 2: _v2_drive_shortcuts}


def migrate(conn: sqlite3.Connection) -> int:
    """Bring a workspace database up to SCHEMA_VERSION. Returns the new version."""
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    for version in sorted(MIGRATIONS):
        if version > current:
            MIGRATIONS[version](conn)
            conn.execute(f"PRAGMA user_version = {version}")
            current = version
    conn.commit()
    return current


def migrate_registry(conn: sqlite3.Connection) -> None:
    _apply_sql_file(conn, "registry.sql")
    conn.commit()
