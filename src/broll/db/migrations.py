"""Schema creation and migration.

Versioning uses SQLite's ``user_version``. Each migration is an idempotent
callable; new versions are appended to MIGRATIONS and never edited in place.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SQL_DIR = Path(__file__).parent

SCHEMA_VERSION = 1


def _apply_sql_file(conn: sqlite3.Connection, name: str) -> None:
    conn.executescript((SQL_DIR / name).read_text())


def _v1_base_schema(conn: sqlite3.Connection) -> None:
    _apply_sql_file(conn, "schema.sql")


MIGRATIONS = {1: _v1_base_schema}


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
