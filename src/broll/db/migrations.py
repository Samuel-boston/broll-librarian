"""Schema creation and migration.

Versioning uses SQLite's ``user_version``. Each migration is an idempotent
callable; new versions are appended to MIGRATIONS and never edited in place.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SQL_DIR = Path(__file__).parent

SCHEMA_VERSION = 9


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


def _v3_client_aware_fields(conn: sqlite3.Connection) -> None:
    """Emotions, the client's category, the featured person, and Top Picks."""
    for ddl in (
        "ALTER TABLE shots ADD COLUMN emotions_json TEXT NOT NULL DEFAULT '[]'",
        "ALTER TABLE shots ADD COLUMN category TEXT",
        "ALTER TABLE shots ADD COLUMN secondary_categories_json TEXT NOT NULL DEFAULT '[]'",
        "ALTER TABLE shots ADD COLUMN featured_person INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE shots ADD COLUMN top_pick INTEGER NOT NULL DEFAULT 0",
    ):
        try:
            conn.execute(ddl)
        except sqlite3.OperationalError as exc:
            if "duplicate column" not in str(exc):
                raise
    conn.execute("CREATE INDEX IF NOT EXISTS idx_shots_category ON shots (workspace_id, category)")


def _v4_job_owner(conn: sqlite3.Connection) -> None:
    """Which process holds a running job, so a starting worker can tell an
    orphaned job from one a live sibling is working on."""
    try:
        conn.execute("ALTER TABLE jobs ADD COLUMN claimed_by TEXT")
    except sqlite3.OperationalError as exc:
        if "duplicate column" not in str(exc):
            raise


def _v5_stemmed_keyword_index(conn: sqlite3.Connection) -> None:
    """Stem the keyword index, so "meditate" finds "meditating" and "meditation".

    The triggers on `shots` refer to shots_fts by name, so they keep working
    once it is recreated; 'rebuild' repopulates it from shots.search_text.
    """
    conn.executescript(
        """
        DROP TABLE IF EXISTS shots_fts;
        CREATE VIRTUAL TABLE shots_fts USING fts5(
            search_text,
            content='shots',
            content_rowid='rowid',
            tokenize='porter unicode61 remove_diacritics 2'
        );
        INSERT INTO shots_fts(shots_fts) VALUES ('rebuild');
        """
    )


def _v6_media_kind(conn: sqlite3.Connection) -> None:
    """Stills live alongside clips, so a source has to say which it is.

    Everything indexed before this migration was a video by definition.
    """
    conn.execute(
        "ALTER TABLE sources ADD COLUMN media_kind TEXT NOT NULL DEFAULT 'video'"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sources_kind ON sources (workspace_id, media_kind)")


def _v7_shot_usage(conn: sqlite3.Connection) -> None:
    """Which videos each shot has been cut into.

    An editing agent choosing B-roll needs this to keep a shot from repeating
    inside one video, and from turning up in every video a client makes that
    month. One row per shot per video; the beat is optional and lets a video's
    own shots stay on offer to the beat that already has them.
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS shot_usage (
            workspace_id  TEXT NOT NULL,
            shot_id       TEXT NOT NULL REFERENCES shots(id) ON DELETE CASCADE,
            project       TEXT NOT NULL,
            beat          TEXT,
            used_at       TEXT NOT NULL,
            PRIMARY KEY (workspace_id, shot_id, project)
        );
        CREATE INDEX IF NOT EXISTS idx_shot_usage_project ON shot_usage (workspace_id, project);
        CREATE INDEX IF NOT EXISTS idx_shot_usage_when ON shot_usage (workspace_id, used_at);
        """
    )


def _add_column(conn: sqlite3.Connection, ddl: str) -> None:
    try:
        conn.execute(ddl)
    except sqlite3.OperationalError as exc:
        if "duplicate column" not in str(exc):
            raise


def _v8_precision_intake_and_segments(conn: sqlite3.Connection) -> None:
    """Precise tags, segments inside a clip, and a list of files that need a person.

    * Search text is split in two (what is in the picture / what it stands for) so a match on the
      first counts for more. The old single column mixed camera-setting words into everything.
    * A shot can be one usable stretch of a longer file, with a "best part" inside it.
    * `attention` lists files the library did not index (too long, too big, a download that did not
      finish...) so none of them disappears quietly. `folder_proposals` holds the new folders the
      model suggested, for a person to approve.
    """
    for ddl in (
        "ALTER TABLE shots ADD COLUMN concept_text TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE shots ADD COLUMN observations_json TEXT NOT NULL DEFAULT '[]'",
        "ALTER TABLE shots ADD COLUMN themes_json TEXT NOT NULL DEFAULT '[]'",
        "ALTER TABLE shots ADD COLUMN category_confidence REAL",
        "ALTER TABLE shots ADD COLUMN review_reasons_json TEXT NOT NULL DEFAULT '[]'",
        "ALTER TABLE shots ADD COLUMN best_start_s REAL",
        "ALTER TABLE shots ADD COLUMN best_end_s REAL",
        "ALTER TABLE sources ADD COLUMN segments_json TEXT",
    ):
        _add_column(conn, ddl)

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS attention (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id  TEXT NOT NULL,
            kind          TEXT NOT NULL,
            key           TEXT NOT NULL,
            filename      TEXT NOT NULL,
            origin        TEXT NOT NULL DEFAULT 'drive',
            drive_file_id TEXT,
            origin_path   TEXT,
            link          TEXT,
            size_bytes    INTEGER,
            duration_s    REAL,
            detail        TEXT,
            status        TEXT NOT NULL DEFAULT 'open',
            shortcut_id   TEXT,
            first_seen    TEXT NOT NULL DEFAULT (datetime('now')),
            last_seen     TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE (workspace_id, key)
        );
        CREATE INDEX IF NOT EXISTS idx_attention_status ON attention (workspace_id, status, kind);

        CREATE TABLE IF NOT EXISTS folder_proposals (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id  TEXT NOT NULL,
            path          TEXT NOT NULL,
            note          TEXT,
            status        TEXT NOT NULL DEFAULT 'open',
            created_at    TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE (workspace_id, path)
        );
        CREATE TABLE IF NOT EXISTS folder_proposal_shots (
            proposal_id   INTEGER NOT NULL REFERENCES folder_proposals(id) ON DELETE CASCADE,
            shot_id       TEXT NOT NULL,
            PRIMARY KEY (proposal_id, shot_id)
        );

        DROP TRIGGER IF EXISTS shots_ai;
        DROP TRIGGER IF EXISTS shots_ad;
        DROP TRIGGER IF EXISTS shots_au;
        DROP TABLE IF EXISTS shots_fts;
        """
    )
    # Shots indexed under the old rules have the old, noisier text. Rewrite it now, while nothing is
    # watching the table: an update with the old triggers in place would try to remove rows from an
    # index that no longer holds them. Embeddings need `broll reembed` afterwards.
    from .searchrows import rewrite_all_search_text

    rewrite_all_search_text(conn)

    conn.executescript(
        """
        CREATE VIRTUAL TABLE shots_fts USING fts5(
            search_text,
            concept_text,
            content='shots',
            content_rowid='rowid',
            tokenize='porter unicode61 remove_diacritics 2'
        );
        CREATE TRIGGER shots_ai AFTER INSERT ON shots BEGIN
            INSERT INTO shots_fts(rowid, search_text, concept_text)
            VALUES (new.rowid, new.search_text, new.concept_text);
        END;
        CREATE TRIGGER shots_ad AFTER DELETE ON shots BEGIN
            INSERT INTO shots_fts(shots_fts, rowid, search_text, concept_text)
            VALUES ('delete', old.rowid, old.search_text, old.concept_text);
        END;
        CREATE TRIGGER shots_au AFTER UPDATE ON shots BEGIN
            INSERT INTO shots_fts(shots_fts, rowid, search_text, concept_text)
            VALUES ('delete', old.rowid, old.search_text, old.concept_text);
            INSERT INTO shots_fts(rowid, search_text, concept_text)
            VALUES (new.rowid, new.search_text, new.concept_text);
        END;
        """
    )
    conn.execute("INSERT INTO shots_fts(shots_fts) VALUES ('rebuild')")


def _v9_search_phrases(conn: sqlite3.Connection) -> None:
    """How an editor would put it when looking for a clip, written by the model as it describes it."""
    _add_column(conn, "ALTER TABLE shots ADD COLUMN phrases_json TEXT NOT NULL DEFAULT '[]'")


MIGRATIONS = {
    1: _v1_base_schema,
    2: _v2_drive_shortcuts,
    3: _v3_client_aware_fields,
    4: _v4_job_owner,
    5: _v5_stemmed_keyword_index,
    6: _v6_media_kind,
    7: _v7_shot_usage,
    8: _v8_precision_intake_and_segments,
    9: _v9_search_phrases,
}


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
