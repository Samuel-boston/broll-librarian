"""All SQL lives here. Nothing else in the codebase touches SQLite, with one
exception: db/vectors.py owns the vector table, because its DDL depends on which
backend the interpreter can support.

Two databases: the registry (workspace list) and one library.db per workspace.
Every workspace query filters on workspace_id even though each workspace has
its own file today - see the note at the top of schema.sql.
"""

from __future__ import annotations

import json
import os
import re
import socket
import sqlite3
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import WorkspaceConfig, broll_home, registry_path, workspace_dir
from .migrations import migrate, migrate_registry
from .models import Job, Shot, ShotFacets, Source, Workspace
from .searchrows import embedding_for, search_columns
from .vectors import VectorIndex, get_vector_index

LIST_FACETS = ("subjects", "mood", "emotions", "usable_for", "quality_flags")
SCALAR_FACETS = (
    "action",
    "setting",
    "shot_type",
    "camera_movement",
    "time_of_day",
    "colour_profile",
    "people_count",
    "pace",
    "category",
)


def new_id() -> str:
    return uuid.uuid4().hex


def worker_identity() -> str:
    """host:pid - enough to tell whether a job's owner is still alive."""
    return f"{socket.gethostname()}:{os.getpid()}"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # it exists; it just isn't ours
    return True


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) if part.isdigit() else 0 for part in version.split("."))


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def canonical_time(value: str | datetime | None = None) -> str:
    """A timestamp as stored: UTC, to the second, ISO 8601 with its offset.

    Usage dates are compared as strings in SQL, so every one is written in the
    same form; a caller's "2026-09-29T10:00:00+01:00" and a bare date both
    become UTC before they reach the table. A value with no zone is read as UTC.
    """
    if value is None:
        return _now()
    when = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).strip())
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return when.astimezone(UTC).isoformat(timespec="seconds")


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


class Registry:
    """The workspace list at ~/.broll/registry.db."""

    def __init__(self, path: Path | None = None):
        self.path = path or registry_path()
        broll_home().mkdir(parents=True, exist_ok=True)
        self.conn = connect(self.path)
        migrate_registry(self.conn)

    def close(self) -> None:
        self.conn.close()

    def create(self, workspace: Workspace) -> Workspace:
        self.conn.execute(
            """INSERT INTO workspaces (id, name, drive_root_folder_id, provider, db_path)
               VALUES (?, ?, ?, ?, ?)""",
            (
                workspace.id,
                workspace.name,
                workspace.drive_root_folder_id,
                workspace.provider,
                workspace.db_path,
            ),
        )
        return self.get(workspace.id)  # type: ignore[return-value]

    def get(self, workspace_id: str) -> Workspace | None:
        row = self.conn.execute(
            "SELECT * FROM workspaces WHERE id = ?", (workspace_id,)
        ).fetchone()
        return Workspace.model_validate(dict(row)) if row else None

    def list(self) -> list[Workspace]:
        rows = self.conn.execute("SELECT * FROM workspaces ORDER BY created_at").fetchall()
        return [Workspace.model_validate(dict(r)) for r in rows]

    def update(self, workspace_id: str, **fields: Any) -> None:
        if not fields:
            return
        assignments = ", ".join(f"{k} = ?" for k in fields)
        self.conn.execute(
            f"UPDATE workspaces SET {assignments} WHERE id = ?",
            (*fields.values(), workspace_id),
        )

    def delete(self, workspace_id: str) -> None:
        self.conn.execute("DELETE FROM workspaces WHERE id = ?", (workspace_id,))

    def default_workspace(self) -> Workspace | None:
        workspaces = self.list()
        return workspaces[0] if len(workspaces) == 1 else None


# --------------------------------------------------------------------------
# Workspace store
# --------------------------------------------------------------------------


class Store:
    """Everything that reads or writes a workspace's library.db."""

    def __init__(self, workspace_id: str, path: Path | None = None, dimensions: int = 384):
        self.workspace_id = workspace_id
        self.path = path or (workspace_dir(workspace_id) / "library.db")
        self.conn = connect(self.path)
        migrate(self.conn)
        self._dimensions = dimensions
        self._vectors: VectorIndex | None = None
        #: What each folder is for (path -> the client's note), read into the embedding text of the clips filed
        #: there. Empty unless the store was opened for a workspace with a folder tree.
        self.folder_notes: dict[str, str] = {}

    @property
    def vectors(self) -> VectorIndex:
        """Lazily opened so a workspace that never searches never builds one."""
        if self._vectors is None:
            self._vectors = get_vector_index(self.conn, self.workspace_id, self._dimensions)
        return self._vectors

    @classmethod
    def for_config(cls, config: WorkspaceConfig) -> "Store":
        store = cls(config.id, config.db_path, config.embedder.dimensions)
        store.folder_notes = config.taxonomy.folder_descriptions() if config.taxonomy.mode == "tree" else {}
        return store

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @contextmanager
    def transaction(self):
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self.conn
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")

    # -- sources ------------------------------------------------------------

    def source_by_hash(self, content_hash: str) -> Source | None:
        row = self.conn.execute(
            "SELECT * FROM sources WHERE workspace_id = ? AND content_hash = ?",
            (self.workspace_id, content_hash),
        ).fetchone()
        return Source.model_validate(dict(row)) if row else None

    def get_source(self, source_id: str) -> Source | None:
        row = self.conn.execute(
            "SELECT * FROM sources WHERE workspace_id = ? AND id = ?",
            (self.workspace_id, source_id),
        ).fetchone()
        return Source.model_validate(dict(row)) if row else None

    def insert_source(self, source: Source) -> Source:
        data = source.model_dump()
        data["workspace_id"] = self.workspace_id
        data.pop("created_at", None)
        columns = ", ".join(data)
        placeholders = ", ".join("?" for _ in data)
        self.conn.execute(
            f"INSERT INTO sources ({columns}) VALUES ({placeholders})",
            tuple(data.values()),
        )
        return self.get_source(source.id)  # type: ignore[return-value]

    def update_source(self, source_id: str, **fields: Any) -> None:
        if not fields:
            return
        assignments = ", ".join(f"{k} = ?" for k in fields)
        self.conn.execute(
            f"UPDATE sources SET {assignments} WHERE workspace_id = ? AND id = ?",
            (*fields.values(), self.workspace_id, source_id),
        )

    def list_sources(self, status: str | None = None, limit: int = 500) -> list[Source]:
        sql = "SELECT * FROM sources WHERE workspace_id = ?"
        params: list[Any] = [self.workspace_id]
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY created_at DESC, id LIMIT ?"
        params.append(limit)
        return [
            Source.model_validate(dict(r)) for r in self.conn.execute(sql, params).fetchall()
        ]

    def recompute_source_status(self, source_id: str) -> str:
        """sources.status is derived from its shots. Call after any shot change."""
        rows = self.conn.execute(
            "SELECT status, COUNT(*) AS n FROM shots WHERE workspace_id = ? AND source_id = ?"
            " GROUP BY status",
            (self.workspace_id, source_id),
        ).fetchall()
        counts = {r["status"]: r["n"] for r in rows}
        total = sum(counts.values())
        if total and not self._planned_indexes(source_id) <= {
            r["shot_index"] for r in self.conn.execute(
                "SELECT shot_index FROM shots WHERE workspace_id = ? AND source_id = ?",
                (self.workspace_id, source_id),
            ).fetchall()
        }:
            # Some of the shots the plan calls for have not been written yet: still being indexed.
            counts["pending"] = counts.get("pending", 0) + 1
        if total == 0:
            status = "pending"
        elif counts.get("needs_review"):
            status = "needs_review"
        elif counts.get("failed", 0) == total:
            status = "failed"
        elif counts.get("pending"):
            status = "analysing"
        else:
            status = "indexed"
        fields: dict[str, Any] = {"status": status}
        if status == "indexed":
            fields["indexed_at"] = _now()
        # A source is only as current as its oldest shot. Deriving this, rather
        # than stamping it when a run starts, means a run that dies halfway can
        # never mark stale shots as current (and hide them from `reanalyse`).
        versions = [
            r[0] for r in self.conn.execute(
                "SELECT analysis_version FROM shots WHERE workspace_id = ? AND source_id = ?"
                " AND analysis_version IS NOT NULL",
                (self.workspace_id, source_id),
            ).fetchall()
        ]
        if versions:
            fields["analysis_version"] = min(versions, key=_version_key)
        self.update_source(source_id, **fields)
        return status

    def _planned_indexes(self, source_id: str) -> set[int]:
        """The shot numbers the stored plan for this file calls for (empty when there is no plan)."""
        row = self.conn.execute(
            "SELECT segments_json FROM sources WHERE workspace_id = ? AND id = ?",
            (self.workspace_id, source_id),
        ).fetchone()
        if not row or not row["segments_json"]:
            return set()
        try:
            return {int(s.get("index", 0)) for s in json.loads(row["segments_json"]) if s.get("kind") == "usable"}
        except (ValueError, AttributeError, TypeError):
            return set()

    # -- shots --------------------------------------------------------------

    def insert_shot(self, shot: Shot) -> Shot:
        payload = self._shot_row(shot)
        columns = ", ".join(payload)
        placeholders = ", ".join("?" for _ in payload)
        self.conn.execute(
            f"INSERT INTO shots ({columns}) VALUES ({placeholders})", tuple(payload.values())
        )
        self.set_shot_tags(shot.id, shot.tags)
        self.recompute_search_text(shot.id)
        return self.get_shot(shot.id)  # type: ignore[return-value]

    def update_shot(self, shot: Shot) -> Shot:
        payload = self._shot_row(shot)
        payload.pop("id")
        assignments = ", ".join(f"{k} = ?" for k in payload)
        self.conn.execute(
            f"UPDATE shots SET {assignments} WHERE workspace_id = ? AND id = ?",
            (*payload.values(), self.workspace_id, shot.id),
        )
        self.set_shot_tags(shot.id, shot.tags)
        self.recompute_search_text(shot.id)
        return self.get_shot(shot.id)  # type: ignore[return-value]

    def set_shot_fields(self, shot_id: str, **fields: Any) -> None:
        if not fields:
            return
        assignments = ", ".join(f"{k} = ?" for k in fields)
        self.conn.execute(
            f"UPDATE shots SET {assignments} WHERE workspace_id = ? AND id = ?",
            (*fields.values(), self.workspace_id, shot_id),
        )

    def _shot_row(self, shot: Shot) -> dict[str, Any]:
        return {
            "id": shot.id,
            "workspace_id": self.workspace_id,
            "source_id": shot.source_id,
            "shot_index": shot.shot_index,
            "is_primary": int(shot.is_primary),
            "start_s": shot.start_s,
            "end_s": shot.end_s,
            "duration_s": shot.duration_s,
            "thumbnail_path": shot.thumbnail_path,
            "caption": shot.caption,
            "confidence": shot.confidence,
            "action": shot.action,
            "setting": shot.setting,
            "setting_detail": shot.setting_detail,
            "shot_type": shot.shot_type,
            "camera_movement": shot.camera_movement,
            "time_of_day": shot.time_of_day,
            "colour_profile": shot.colour_profile,
            "people_count": shot.people_count,
            "has_recognisable_faces": _bool_or_none(shot.has_recognisable_faces),
            "has_text_on_screen": _bool_or_none(shot.has_text_on_screen),
            "pace": shot.pace,
            "subjects_json": json.dumps(shot.subjects),
            "mood_json": json.dumps(shot.mood),
            "usable_for_json": json.dumps(shot.usable_for),
            "quality_flags_json": json.dumps(shot.quality_flags),
            "emotions_json": json.dumps(shot.emotions),
            "category": shot.category,
            "secondary_categories_json": json.dumps(shot.secondary_categories),
            "featured_person": int(shot.featured_person),
            "top_pick": int(shot.top_pick),
            "status": shot.status,
            "error_message": shot.error_message,
            "analysis_version": shot.analysis_version,
            "analysed_at": shot.analysed_at,
            "search_text": shot.search_text,
            "raw_analysis_json": json.dumps(shot.raw_analysis) if shot.raw_analysis else None,
            "observations_json": json.dumps(shot.observations),
            "themes_json": json.dumps(shot.themes),
            "phrases_json": json.dumps(shot.search_phrases),
            "body_json": json.dumps(shot.body_language),
            "category_confidence": shot.category_confidence,
            "review_reasons_json": json.dumps(shot.review_reasons),
            "best_start_s": shot.best_start_s,
            "best_end_s": shot.best_end_s,
        }

    def get_shot(self, shot_id: str) -> Shot | None:
        row = self.conn.execute(
            "SELECT * FROM shots WHERE workspace_id = ? AND id = ?",
            (self.workspace_id, shot_id),
        ).fetchone()
        return Shot.from_row(row, self.shot_tags(shot_id)) if row else None

    def shots_for_source(self, source_id: str) -> list[Shot]:
        rows = self.conn.execute(
            "SELECT * FROM shots WHERE workspace_id = ? AND source_id = ? ORDER BY shot_index",
            (self.workspace_id, source_id),
        ).fetchall()
        return [Shot.from_row(r, self.shot_tags(r["id"])) for r in rows]

    def list_shots(self, status: str | None = None, limit: int = 1000) -> list[Shot]:
        sql = "SELECT * FROM shots WHERE workspace_id = ?"
        params: list[Any] = [self.workspace_id]
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [Shot.from_row(r, self.shot_tags(r["id"])) for r in rows]

    def sample_shot_ids(self, n: int) -> list[str]:
        """About `n` shot ids spread evenly through the library, the same ones every time."""
        total = self.count_shots()
        if total <= 0:
            return []
        step = max(1, -(-total // max(1, n)))  # round up, so n rows reach the whole library
        rows = self.conn.execute(
            "SELECT id FROM shots WHERE workspace_id = ? AND rowid % ? = 0 LIMIT ?",
            (self.workspace_id, step, n),
        ).fetchall()
        return [r["id"] for r in rows]

    def count_shots(self, status: str | None = None, media_kind: str | None = None) -> int:
        sql = "SELECT COUNT(*) FROM shots s"
        params: list[Any] = []
        if media_kind:
            sql += " JOIN sources src ON src.id = s.source_id"
        sql += " WHERE s.workspace_id = ?"
        params.append(self.workspace_id)
        if status:
            sql += " AND s.status = ?"
            params.append(status)
        if media_kind:
            sql += " AND src.media_kind = ?"
            params.append(media_kind)
        return int(self.conn.execute(sql, params).fetchone()[0])

    def shot_kinds(self, shot_ids: Iterable[str]) -> dict[str, str]:
        """shot id -> "video" or "image", for the shots asked about."""
        ids = list(dict.fromkeys(shot_ids))
        out: dict[str, str] = {}
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            marks = ", ".join("?" for _ in chunk)
            for row in self.conn.execute(
                f"""SELECT s.id, src.media_kind FROM shots s JOIN sources src ON src.id = s.source_id
                    WHERE s.workspace_id = ? AND s.id IN ({marks})""",
                (self.workspace_id, *chunk),
            ).fetchall():
                out[row["id"]] = row["media_kind"]
        return out

    def prune_shots(self, source_id: str, keep: int) -> int:
        """Delete this source's shots numbered `keep` and above: the file was planned into fewer shots.

        Returns how many were removed. Their vectors go with them. A shot a person has touched is
        never removed: one they corrected or starred, or that a video already uses.
        """
        rows = self.conn.execute(
            "SELECT id FROM shots WHERE workspace_id = ? AND source_id = ? AND shot_index >= ?"
            " AND top_pick = 0"
            " AND (raw_analysis_json IS NULL OR raw_analysis_json NOT LIKE '%\"corrected_by_operator\": true%')"
            " AND id NOT IN (SELECT shot_id FROM shot_usage WHERE workspace_id = ?)",
            (self.workspace_id, source_id, keep, self.workspace_id),
        ).fetchall()
        for row in rows:
            self.vectors.delete(row["id"])
            self.conn.execute("DELETE FROM shots WHERE workspace_id = ? AND id = ?",
                              (self.workspace_id, row["id"]))
        return len(rows)

    # -- tags ---------------------------------------------------------------

    def set_shot_tags(self, shot_id: str, tags: Iterable[str]) -> None:
        self.conn.execute(
            "DELETE FROM shot_tags WHERE workspace_id = ? AND shot_id = ?",
            (self.workspace_id, shot_id),
        )
        for tag in tags:
            name = tag.strip().lower()
            if not name:
                continue
            self.conn.execute(
                "INSERT OR IGNORE INTO tags (workspace_id, name) VALUES (?, ?)",
                (self.workspace_id, name),
            )
            row = self.conn.execute(
                "SELECT id FROM tags WHERE workspace_id = ? AND name = ?",
                (self.workspace_id, name),
            ).fetchone()
            self.conn.execute(
                "INSERT OR IGNORE INTO shot_tags (shot_id, tag_id, workspace_id) VALUES (?, ?, ?)",
                (shot_id, row["id"], self.workspace_id),
            )

    def shot_tags(self, shot_id: str) -> list[str]:
        rows = self.conn.execute(
            """SELECT t.name FROM shot_tags st
               JOIN tags t ON t.id = st.tag_id
               WHERE st.workspace_id = ? AND st.shot_id = ? ORDER BY t.name""",
            (self.workspace_id, shot_id),
        ).fetchall()
        return [r["name"] for r in rows]

    # -- search_text --------------------------------------------------------

    def recompute_search_text(self, shot_id: str) -> str:
        """The one place the search columns are built. FTS5 indexes both of them.

        Returns the embedding text, so a caller that is about to embed the shot has it.
        """
        row = self.conn.execute(
            "SELECT * FROM shots WHERE workspace_id = ? AND id = ?",
            (self.workspace_id, shot_id),
        ).fetchone()
        if row is None:
            return ""
        tags = self.shot_tags(shot_id)
        primary, concept = search_columns(row, tags)
        self.conn.execute(
            "UPDATE shots SET search_text = ?, concept_text = ? WHERE workspace_id = ? AND id = ?",
            (primary, concept, self.workspace_id, shot_id),
        )
        return embedding_for(row, tags, self.folder_notes)

    def embedding_text(self, shot_id: str) -> str:
        """What the embedder reads for this shot, from what is stored now."""
        row = self.conn.execute(
            "SELECT * FROM shots WHERE workspace_id = ? AND id = ?",
            (self.workspace_id, shot_id),
        ).fetchone()
        return embedding_for(row, self.shot_tags(shot_id), self.folder_notes) if row else ""

    # -- browsing -----------------------------------------------------------

    def vocabulary(self) -> Counter:
        """Every word in the library's search text, with how many shots use it."""
        counts: Counter = Counter()
        for (text,) in self.conn.execute(
            "SELECT search_text || ' ' || concept_text FROM shots WHERE workspace_id = ?",
            (self.workspace_id,),
        ):
            counts.update(set(re.findall(r"[\w']+", (text or "").lower())))
        return counts

    def browse_rows(self, media_kind: str | None = None) -> list[dict[str, Any]]:
        """Every organised shot, newest first - the raw material for browsing.

        `media_kind` keeps one side only: "video" or "image".
        """
        kind_clause = " AND src.media_kind = ?" if media_kind else ""
        rows = self.conn.execute(
            f"""SELECT s.id, s.category, s.secondary_categories_json, s.thumbnail_path,
                      s.top_pick, s.featured_person, src.media_kind
               FROM shots s JOIN sources src ON src.id = s.source_id
               WHERE s.workspace_id = ? AND s.status IN ('indexed', 'needs_review'){kind_clause}
               ORDER BY src.created_at DESC, s.shot_index""",
            (self.workspace_id, *([media_kind] if media_kind else [])),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "category": r["category"],
                "secondary": json.loads(r["secondary_categories_json"] or "[]"),
                "has_thumbnail": bool(r["thumbnail_path"]),
                "top_pick": bool(r["top_pick"]),
                "featured": bool(r["featured_person"]),
                "media_kind": r["media_kind"],
            }
            for r in rows
        ]

    # -- vocabulary candidates ---------------------------------------------

    def record_vocabulary_candidates(self, terms: Iterable[tuple[str, str]]) -> None:
        for field, term in terms:
            self.conn.execute(
                """INSERT INTO vocabulary_candidates (workspace_id, field, term, count)
                   VALUES (?, ?, ?, 1)
                   ON CONFLICT (workspace_id, field, term)
                   DO UPDATE SET count = count + 1, last_seen = datetime('now')""",
                (self.workspace_id, field, term),
            )

    def vocabulary_candidates(self, min_count: int = 1) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT field, term, count, first_seen, last_seen, promoted
               FROM vocabulary_candidates
               WHERE workspace_id = ? AND count >= ? AND promoted = 0
               ORDER BY count DESC, term""",
            (self.workspace_id, min_count),
        ).fetchall()
        return [dict(r) for r in rows]

    def promote_vocabulary_candidate(self, field: str, term: str) -> None:
        self.conn.execute(
            "UPDATE vocabulary_candidates SET promoted = 1"
            " WHERE workspace_id = ? AND field = ? AND term = ?",
            (self.workspace_id, field, term),
        )

    # -- facet counts (input to the taxonomy threshold rules) ---------------

    def facet_counts(self, media_kind: str | None = None) -> dict[tuple[str, str], int]:
        """How many indexed shots carry each facet value, keyed (facet, value).

        `media_kind` counts one side only ("video" or "image"), so the filters offered next to a list of
        videos never promise images that are not in it.
        """
        counts: dict[tuple[str, str], int] = {}
        join = " JOIN sources src ON src.id = s.source_id" if media_kind else ""
        only = " AND src.media_kind = ?" if media_kind else ""
        extra = [media_kind] if media_kind else []
        for facet in SCALAR_FACETS:
            rows = self.conn.execute(
                f"SELECT s.{facet} AS value, COUNT(*) AS n FROM shots s{join}"
                " WHERE s.workspace_id = ? AND s.status IN ('indexed','needs_review')"
                f" AND s.{facet} IS NOT NULL{only} GROUP BY s.{facet}",
                (self.workspace_id, *extra),
            ).fetchall()
            for row in rows:
                counts[(facet, row["value"])] = row["n"]
        for facet in LIST_FACETS:
            rows = self.conn.execute(
                f"""SELECT j.value AS value, COUNT(*) AS n
                    FROM shots s{join}, json_each(s.{facet}_json) j
                    WHERE s.workspace_id = ? AND s.status IN ('indexed','needs_review'){only}
                    GROUP BY j.value""",
                (self.workspace_id, *extra),
            ).fetchall()
            for row in rows:
                counts[(facet, row["value"])] = row["n"]
        return counts

    def facet_pair_counts(self, primary: str, secondary: str) -> dict[tuple[str, str], int]:
        """Co-occurrence counts, used to pick the third folder level."""
        counts: dict[tuple[str, str], int] = {}
        for shot in self.list_shots():
            for a in _facet_values(shot, primary):
                for b in _facet_values(shot, secondary):
                    counts[(a, b)] = counts.get((a, b), 0) + 1
        return counts

    def all_facets(self) -> list[ShotFacets]:
        return [ShotFacets.from_shot(s) for s in self.list_shots()]

    # -- drive shortcuts ----------------------------------------------------

    def record_shortcut(self, source_id: str, shot_id: str, folder_path: str,
                        folder_id: str | None, name: str, shortcut_id: str,
                        target_id: str) -> None:
        self.conn.execute(
            """INSERT INTO drive_shortcuts (workspace_id, source_id, shot_id,
                   folder_path, folder_id, name, shortcut_id, target_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (workspace_id, folder_path, name) DO UPDATE SET
                   source_id = excluded.source_id, shot_id = excluded.shot_id,
                   folder_id = excluded.folder_id,
                   shortcut_id = excluded.shortcut_id,
                   target_id = excluded.target_id""",
            (self.workspace_id, source_id, shot_id, folder_path, folder_id, name,
             shortcut_id, target_id),
        )

    def shortcuts_for_source(self, source_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM drive_shortcuts WHERE workspace_id = ? AND source_id = ?",
            (self.workspace_id, source_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def forget_shortcut(self, folder_path: str, name: str) -> None:
        self.conn.execute(
            "DELETE FROM drive_shortcuts WHERE workspace_id = ? AND folder_path = ?"
            " AND name = ?",
            (self.workspace_id, folder_path, name),
        )

    def forget_shortcuts_for_source(self, source_id: str) -> None:
        self.conn.execute(
            "DELETE FROM drive_shortcuts WHERE workspace_id = ? AND source_id = ?",
            (self.workspace_id, source_id),
        )

    # -- files that need a person -------------------------------------------

    def flag_attention(
        self,
        *,
        kind: str,
        key: str,
        filename: str,
        origin: str = "drive",
        drive_file_id: str | None = None,
        origin_path: str | None = None,
        link: str | None = None,
        size_bytes: int | None = None,
        duration_s: float | None = None,
        detail: str | None = None,
    ) -> int:
        """Put a file on the "Needs attention" list, or refresh it if it is already there.

        One row per file (`key`). A file a person dismissed stays dismissed; one they asked to have
        indexed (`requeued`) comes back as `open` if it is turned away again, so it is never lost.
        """
        self.conn.execute(
            """INSERT INTO attention (workspace_id, kind, key, filename, origin, drive_file_id,
                                      origin_path, link, size_bytes, duration_s, detail)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (workspace_id, key) DO UPDATE SET
                   kind = excluded.kind, filename = excluded.filename,
                   link = COALESCE(excluded.link, attention.link),
                   size_bytes = COALESCE(excluded.size_bytes, attention.size_bytes),
                   duration_s = COALESCE(excluded.duration_s, attention.duration_s),
                   detail = excluded.detail, last_seen = datetime('now'),
                   status = CASE WHEN attention.status IN ('requeued', 'resolved') THEN 'open'
                                 ELSE attention.status END""",
            (self.workspace_id, kind, key, filename, origin, drive_file_id, origin_path, link,
             size_bytes, duration_s, detail),
        )
        return int(self.conn.execute(
            "SELECT id FROM attention WHERE workspace_id = ? AND key = ?",
            (self.workspace_id, key),
        ).fetchone()["id"])

    def dismissed_attention_keys(self) -> set[str]:
        return {
            r["key"] for r in self.conn.execute(
                "SELECT key FROM attention WHERE workspace_id = ? AND status = 'dismissed'",
                (self.workspace_id,),
            ).fetchall()
        }

    def list_attention(self, status: str | None = "open", limit: int = 500) -> list[dict[str, Any]]:
        sql = "SELECT * FROM attention WHERE workspace_id = ?"
        params: list[Any] = [self.workspace_id]
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY kind, filename LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def get_attention(self, item_id: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM attention WHERE workspace_id = ? AND id = ?",
            (self.workspace_id, item_id),
        ).fetchone()
        return dict(row) if row else None

    def set_attention(self, item_id: int, **fields: Any) -> None:
        if not fields:
            return
        assignments = ", ".join(f"{k} = ?" for k in fields)
        self.conn.execute(
            f"UPDATE attention SET {assignments} WHERE workspace_id = ? AND id = ?",
            (*fields.values(), self.workspace_id, item_id),
        )

    def resolve_attention(self, key: str) -> None:
        """A file that was on the list has now been indexed: take it off."""
        self.conn.execute(
            "UPDATE attention SET status = 'resolved', last_seen = datetime('now')"
            " WHERE workspace_id = ? AND key = ? AND status != 'dismissed'",
            (self.workspace_id, key),
        )

    def attention_counts(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT kind, COUNT(*) AS n FROM attention WHERE workspace_id = ? AND status = 'open'"
            " GROUP BY kind",
            (self.workspace_id,),
        ).fetchall()
        return {r["kind"]: r["n"] for r in rows}

    # -- folders the model suggested ----------------------------------------

    def record_folder_proposal(self, path: str, note: str | None, shot_id: str) -> int:
        self.conn.execute(
            """INSERT INTO folder_proposals (workspace_id, path, note) VALUES (?, ?, ?)
               ON CONFLICT (workspace_id, path) DO NOTHING""",
            (self.workspace_id, path, note),
        )
        row = self.conn.execute(
            "SELECT id FROM folder_proposals WHERE workspace_id = ? AND path = ?",
            (self.workspace_id, path),
        ).fetchone()
        self.conn.execute(
            "INSERT OR IGNORE INTO folder_proposal_shots (proposal_id, shot_id) VALUES (?, ?)",
            (row["id"], shot_id),
        )
        return int(row["id"])

    def list_folder_proposals(self, status: str | None = "open") -> list[dict[str, Any]]:
        sql = "SELECT * FROM folder_proposals WHERE workspace_id = ?"
        params: list[Any] = [self.workspace_id]
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY created_at, id"
        out = []
        for row in self.conn.execute(sql, params).fetchall():
            item = dict(row)
            item["shot_ids"] = [
                r["shot_id"] for r in self.conn.execute(
                    """SELECT p.shot_id FROM folder_proposal_shots p
                       JOIN shots s ON s.id = p.shot_id AND s.workspace_id = ?
                       WHERE p.proposal_id = ? ORDER BY p.shot_id""",
                    (self.workspace_id, row["id"]),
                ).fetchall()
            ]
            out.append(item)
        return out

    def get_folder_proposal(self, proposal_id: int) -> dict[str, Any] | None:
        return next(
            (p for p in self.list_folder_proposals(status=None) if p["id"] == proposal_id), None
        )

    def merge_folder_proposals(self, into_id: int, from_ids: list[int]) -> None:
        """Fold near-duplicate suggestions into one: their clips join `into_id`, the rest are closed."""
        for other in from_ids:
            if other == into_id:
                continue
            self.conn.execute(
                "INSERT OR IGNORE INTO folder_proposal_shots (proposal_id, shot_id) "
                "SELECT ?, shot_id FROM folder_proposal_shots WHERE proposal_id = ?",
                (into_id, other),
            )
            self.set_folder_proposal_status(other, "merged")

    def set_folder_proposal_status(self, proposal_id: int, status: str) -> None:
        self.conn.execute(
            "UPDATE folder_proposals SET status = ? WHERE workspace_id = ? AND id = ?",
            (status, self.workspace_id, proposal_id),
        )

    # -- where shots have been used -----------------------------------------

    def record_usage(
        self,
        shot_ids: Iterable[str],
        project: str,
        used_at: str | None = None,
        beats: Mapping[str, str] | None = None,
        replace: bool = False,
    ) -> dict[str, Any]:
        """Remember that these shots are cut into `project` (one video).

        One row per shot per video, so choosing the same shot again for the
        same video moves its date and beat instead of counting it twice.
        `replace` makes this the video's whole list: an editor who re-chooses
        B-roll sends every shot the video now uses, and the ones it dropped stop
        counting against the client's next videos.

        Returns {"recorded": [...], "unknown": [...], "released": n}. Ids that
        are not shots in this library are reported, never stored.
        """
        wanted = list(dict.fromkeys(s for s in shot_ids if s))
        when = canonical_time(used_at)
        beats = beats or {}
        known: set[str] = set()
        if wanted:
            placeholders = ", ".join("?" for _ in wanted)
            known = {
                r["id"] for r in self.conn.execute(
                    f"SELECT id FROM shots WHERE workspace_id = ? AND id IN ({placeholders})",
                    (self.workspace_id, *wanted),
                )
            }
        recorded = [s for s in wanted if s in known]
        released = 0
        with self.transaction() as conn:
            if replace:
                keep = ", ".join("?" for _ in recorded)
                released = conn.execute(
                    "DELETE FROM shot_usage WHERE workspace_id = ? AND project = ?"
                    + (f" AND shot_id NOT IN ({keep})" if recorded else ""),
                    (self.workspace_id, project, *recorded),
                ).rowcount
            for shot_id in recorded:
                conn.execute(
                    """INSERT INTO shot_usage (workspace_id, shot_id, project, beat, used_at)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT (workspace_id, shot_id, project) DO UPDATE SET
                           beat = excluded.beat, used_at = excluded.used_at""",
                    (self.workspace_id, shot_id, project, beats.get(shot_id), when),
                )
        return {
            "recorded": recorded,
            "unknown": [s for s in wanted if s not in known],
            "released": released,
        }

    def shots_used_in_project(self, project: str, except_beat: str | None = None) -> set[str]:
        """Shots a video already uses - except those on `except_beat`.

        The exception is what lets an editor re-run B-roll for a video: a beat
        still sees the shot it already has, while every other beat is kept from
        repeating it. A shot recorded without a beat counts for every beat,
        because nothing says where it sits.
        """
        sql = "SELECT shot_id FROM shot_usage WHERE workspace_id = ? AND project = ?"
        params: list[Any] = [self.workspace_id, project]
        if except_beat is not None:
            sql += " AND (beat IS NULL OR beat != ?)"
            params.append(except_beat)
        return {r["shot_id"] for r in self.conn.execute(sql, params)}

    def shots_used_since(self, since: str, other_than_project: str | None = None) -> set[str]:
        """Shots cut into any video since `since` - other than `other_than_project`.

        A video's own earlier choices never count against it here: re-running
        B-roll for a video must not hide the shots it is already built on.
        """
        sql = "SELECT DISTINCT shot_id FROM shot_usage WHERE workspace_id = ? AND used_at >= ?"
        params: list[Any] = [self.workspace_id, canonical_time(since)]
        if other_than_project:
            sql += " AND project != ?"
            params.append(other_than_project)
        return {r["shot_id"] for r in self.conn.execute(sql, params)}

    def usage_for(self, shot_ids: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
        """Every video each of these shots is cut into, most recent first."""
        ids = list(dict.fromkeys(shot_ids))
        if not ids:
            return {}
        placeholders = ", ".join("?" for _ in ids)
        out: dict[str, list[dict[str, Any]]] = {}
        for row in self.conn.execute(
            f"""SELECT shot_id, project, beat, used_at FROM shot_usage
                WHERE workspace_id = ? AND shot_id IN ({placeholders})
                ORDER BY used_at DESC, project""",
            (self.workspace_id, *ids),
        ):
            out.setdefault(row["shot_id"], []).append(
                {"project": row["project"], "beat": row["beat"], "used_at": row["used_at"]}
            )
        return out

    def list_usage(
        self, project: str | None = None, shot_id: str | None = None, limit: int = 500
    ) -> list[dict[str, Any]]:
        sql = "SELECT shot_id, project, beat, used_at FROM shot_usage WHERE workspace_id = ?"
        params: list[Any] = [self.workspace_id]
        if project:
            sql += " AND project = ?"
            params.append(project)
        if shot_id:
            sql += " AND shot_id = ?"
            params.append(shot_id)
        sql += " ORDER BY used_at DESC, project, shot_id LIMIT ?"
        params.append(max(1, limit))
        return [dict(r) for r in self.conn.execute(sql, params)]

    # -- jobs ---------------------------------------------------------------

    def enqueue(self, kind: str, payload: Mapping[str, Any] | None = None) -> Job:
        job_id = new_id()
        self.conn.execute(
            "INSERT INTO jobs (id, workspace_id, kind, payload_json) VALUES (?, ?, ?, ?)",
            (job_id, self.workspace_id, kind, json.dumps(dict(payload or {}))),
        )
        return self.get_job(job_id)  # type: ignore[return-value]

    def get_job(self, job_id: str) -> Job | None:
        row = self.conn.execute(
            "SELECT * FROM jobs WHERE workspace_id = ? AND id = ?",
            (self.workspace_id, job_id),
        ).fetchone()
        return Job.from_row(row) if row else None

    def claim_job(self, worker_id: str | None = None) -> Job | None:
        """Atomically take the next runnable job, recording who holds it."""
        with self.transaction() as conn:
            row = conn.execute(
                """SELECT * FROM jobs
                   WHERE workspace_id = ? AND status = 'queued'
                     AND (not_before IS NULL OR not_before <= datetime('now'))
                   ORDER BY created_at LIMIT 1""",
                (self.workspace_id,),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                "UPDATE jobs SET status = 'running', attempts = attempts + 1,"
                " started_at = datetime('now'), claimed_by = ? WHERE id = ?",
                (worker_id or worker_identity(), row["id"]),
            )
        return self.get_job(row["id"])

    def finish_job(self, job_id: str, status: str, error: str | None = None,
                   cost: float = 0.0) -> None:
        self.conn.execute(
            """UPDATE jobs SET status = ?, last_error = ?, finished_at = datetime('now'),
                   cost_estimate_usd = cost_estimate_usd + ?
               WHERE workspace_id = ? AND id = ?""",
            (status, error, cost, self.workspace_id, job_id),
        )

    def retry_job(self, job_id: str, error: str, delay_s: float) -> None:
        self.conn.execute(
            """UPDATE jobs SET status = 'queued', last_error = ?,
                   not_before = datetime('now', ?)
               WHERE workspace_id = ? AND id = ?""",
            (error, f"+{int(delay_s)} seconds", self.workspace_id, job_id),
        )

    def requeue_failed_jobs(self) -> int:
        """Put failed jobs back in the queue with a fresh attempt budget."""
        cursor = self.conn.execute(
            """UPDATE jobs SET status = 'queued', attempts = 0, not_before = NULL
               WHERE workspace_id = ? AND status = 'failed'""",
            (self.workspace_id,),
        )
        return cursor.rowcount

    def cancel_job(self, job_id: str) -> Job | None:
        """Take one waiting job out of the queue. Returns it, or None if it isn't waiting.

        Only a queued job (including one waiting out a retry delay) can be cancelled. A job a
        worker has already started runs to the end: stopping it half way would leave a
        half-indexed file.
        """
        with self.transaction() as conn:
            cursor = conn.execute(
                """UPDATE jobs SET status = 'cancelled', finished_at = datetime('now'),
                       last_error = 'Removed from the queue'
                   WHERE workspace_id = ? AND id = ? AND status = 'queued'""",
                (self.workspace_id, job_id),
            )
            if cursor.rowcount == 0:
                return None
        return self.get_job(job_id)

    def cancel_queued_jobs(self) -> list[Job]:
        """Take every waiting job out of the queue. Jobs already running are left to finish."""
        with self.transaction() as conn:
            rows = conn.execute(
                "SELECT id FROM jobs WHERE workspace_id = ? AND status = 'queued'",
                (self.workspace_id,),
            ).fetchall()
            conn.execute(
                """UPDATE jobs SET status = 'cancelled', finished_at = datetime('now'),
                       last_error = 'Removed from the queue'
                   WHERE workspace_id = ? AND status = 'queued'""",
                (self.workspace_id,),
            )
        return [j for j in (self.get_job(r["id"]) for r in rows) if j is not None]

    def delete_source(self, source_id: str) -> dict[str, Any] | None:
        """Remove one file from the library: its shots, tags, vectors and shortcut records.

        Local only: nothing in Drive is touched. Returns what was removed (the source, and the ids
        and thumbnail paths of its shots, by id) so the caller can clean up files and the dashboard, or
        None if there was no such file.
        """
        source = self.get_source(source_id)
        if source is None:
            return None
        shots = self.conn.execute(
            "SELECT id, thumbnail_path FROM shots WHERE workspace_id = ? AND source_id = ?",
            (self.workspace_id, source_id),
        ).fetchall()
        with self.transaction() as conn:
            for shot in shots:
                self.vectors.delete(shot["id"])
            self.forget_shortcuts_for_source(source_id)
            # shots and shot_tags go with it (ON DELETE CASCADE); the search index follows its trigger.
            conn.execute("DELETE FROM sources WHERE workspace_id = ? AND id = ?", (self.workspace_id, source_id))
        return {
            "source": source,
            "shot_ids": [r["id"] for r in shots],
            "thumbnails": {r["id"]: r["thumbnail_path"] for r in shots},
        }

    def clear_library(self) -> dict[str, int]:
        """Delete every source, shot, job and vector in this workspace.

        Local only: nothing in Drive is touched, so files already filed there
        stay exactly where they are.
        """
        counts = {
            "sources": self.conn.execute(
                "SELECT COUNT(*) FROM sources WHERE workspace_id = ?", (self.workspace_id,)
            ).fetchone()[0],
            "shots": self.conn.execute(
                "SELECT COUNT(*) FROM shots WHERE workspace_id = ?", (self.workspace_id,)
            ).fetchone()[0],
            "jobs": self.conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE workspace_id = ?", (self.workspace_id,)
            ).fetchone()[0],
        }
        for table in ("shot_tags", "drive_shortcuts", "vocabulary_candidates",
                      "shot_usage", "attention", "folder_proposals", "jobs", "shots", "sources"):
            if table == "shot_tags":
                self.conn.execute(
                    """DELETE FROM shot_tags WHERE shot_id IN
                       (SELECT id FROM shots WHERE workspace_id = ?)""",
                    (self.workspace_id,),
                )
                continue
            self.conn.execute(f"DELETE FROM {table} WHERE workspace_id = ?", (self.workspace_id,))
        counts["vectors"] = self.vectors.count()
        self.vectors.rebuild()
        self.conn.commit()
        return counts

    def reset_stale_jobs(self) -> int:
        """Requeue jobs left 'running' by a worker that is no longer alive.

        Called when a worker starts. A running job is only orphaned if its
        owner is gone: another live worker - the web app while a CLI command
        runs, say - may be halfway through it, and requeuing that job runs it
        twice. On this machine the owner's pid is checked directly; for another
        machine, whose processes we cannot see, a two-hour lease applies.
        """
        host = socket.gethostname()
        rows = self.conn.execute(
            "SELECT id, claimed_by, started_at FROM jobs"
            " WHERE workspace_id = ? AND status = 'running'",
            (self.workspace_id,),
        ).fetchall()
        orphaned: list[str] = []
        for row in rows:
            owner_host, _, pid = (row["claimed_by"] or "").rpartition(":")
            if not owner_host or not pid.isdigit():
                orphaned.append(row["id"])  # unowned: a pre-v4 row or a hard kill
            elif owner_host == host:
                # A job claimed by our own pid is left over from a previous run: this worker is only
                # starting now. In a container restart the new process often reuses the old pid.
                if int(pid) == os.getpid() or not _pid_alive(int(pid)):
                    orphaned.append(row["id"])
            elif self.conn.execute(
                "SELECT ? < datetime('now', '-2 hours')", (row["started_at"],)
            ).fetchone()[0]:
                orphaned.append(row["id"])
        for job_id in orphaned:
            self.conn.execute(
                "UPDATE jobs SET status = 'queued', claimed_by = NULL WHERE id = ?", (job_id,)
            )
        return len(orphaned)

    def release_job(self, job_id: str, reason: str) -> None:
        """Hand a job back to the queue untouched - its worker is stopping."""
        self.conn.execute(
            "UPDATE jobs SET status = 'queued', claimed_by = NULL, last_error = ?,"
            " attempts = MAX(attempts - 1, 0) WHERE workspace_id = ? AND id = ?",
            (reason, self.workspace_id, job_id),
        )

    def job_counts(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT status, COUNT(*) AS n FROM jobs WHERE workspace_id = ? GROUP BY status",
            (self.workspace_id,),
        ).fetchall()
        return {r["status"]: r["n"] for r in rows}

    def recent_jobs(self, limit: int = 40) -> list[dict[str, Any]]:
        """Running first, then waiting, then the latest finished - a queue panel's rows."""
        rows = self.conn.execute(
            """SELECT id, status, attempts, last_error, cost_estimate_usd,
                      json_extract(payload_json, '$.filename') AS filename,
                      created_at, started_at, finished_at, not_before
               FROM jobs WHERE workspace_id = ? AND status != 'cancelled'
               ORDER BY CASE status WHEN 'running' THEN 0 WHEN 'queued' THEN 1 ELSE 2 END,
                        created_at DESC
               LIMIT ?""",
            (self.workspace_id, max(1, limit)),
        ).fetchall()
        return [dict(r) for r in rows]

    def waiting_errors(self) -> list[str]:
        """The last error of every job waiting to be retried."""
        rows = self.conn.execute(
            "SELECT last_error FROM jobs WHERE workspace_id = ? AND status = 'queued'"
            " AND last_error IS NOT NULL AND last_error != ''",
            (self.workspace_id,),
        ).fetchall()
        return [r["last_error"] for r in rows]

    def total_cost(self) -> float:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(cost_estimate_usd), 0) FROM jobs WHERE workspace_id = ?",
            (self.workspace_id,),
        ).fetchone()
        return float(row[0])


def _bool_or_none(value: bool | None) -> int | None:
    return None if value is None else int(value)


def _facet_values(shot: Shot, facet: str) -> list[str]:
    value = getattr(shot, facet, None)
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [str(value)]
