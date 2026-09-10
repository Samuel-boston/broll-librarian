"""All SQL lives here. Nothing else in the codebase touches SQLite, with one
exception: db/vectors.py owns the vector table, because its DDL depends on which
backend the interpreter can support.

Two databases: the registry (workspace list) and one library.db per workspace.
Every workspace query filters on workspace_id even though each workspace has
its own file today - see the note at the top of schema.sql.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import WorkspaceConfig, broll_home, registry_path, workspace_dir
from .migrations import migrate, migrate_registry
from .models import Job, Shot, ShotFacets, Source, Workspace
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


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


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

    @property
    def vectors(self) -> VectorIndex:
        """Lazily opened so a workspace that never searches never builds one."""
        if self._vectors is None:
            self._vectors = get_vector_index(self.conn, self.workspace_id, self._dimensions)
        return self._vectors

    @classmethod
    def for_config(cls, config: WorkspaceConfig) -> "Store":
        return cls(config.id, config.db_path, config.embedder.dimensions)

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
        self.update_source(source_id, **fields)
        return status

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

    def count_shots(self, status: str | None = None) -> int:
        sql = "SELECT COUNT(*) FROM shots WHERE workspace_id = ?"
        params: list[Any] = [self.workspace_id]
        if status:
            sql += " AND status = ?"
            params.append(status)
        return int(self.conn.execute(sql, params).fetchone()[0])

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
        """The one place search_text is built. FTS5 indexes only this column."""
        row = self.conn.execute(
            "SELECT * FROM shots WHERE workspace_id = ? AND id = ?",
            (self.workspace_id, shot_id),
        ).fetchone()
        if row is None:
            return ""
        parts: list[str] = []
        if row["caption"]:
            parts.append(row["caption"])
        for field in ("action", "setting", "setting_detail", "shot_type",
                      "camera_movement", "time_of_day", "colour_profile",
                      "people_count", "pace"):
            value = row[field]
            if value:
                parts.append(str(value).replace("_", " "))
        for field in ("subjects_json", "mood_json", "emotions_json", "usable_for_json"):
            parts.extend(json.loads(row[field] or "[]"))
        categories = [row["category"], *json.loads(row["secondary_categories_json"] or "[]")]
        parts.extend(c.split("/")[-1] for c in categories if c)
        parts.extend(self.shot_tags(shot_id))
        text = ", ".join(str(p) for p in parts if p)
        self.conn.execute(
            "UPDATE shots SET search_text = ? WHERE workspace_id = ? AND id = ?",
            (text, self.workspace_id, shot_id),
        )
        return text

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

    def facet_counts(self) -> dict[tuple[str, str], int]:
        """How many indexed shots carry each facet value, keyed (facet, value)."""
        counts: dict[tuple[str, str], int] = {}
        for facet in SCALAR_FACETS:
            rows = self.conn.execute(
                f"SELECT {facet} AS value, COUNT(*) AS n FROM shots"
                " WHERE workspace_id = ? AND status IN ('indexed','needs_review')"
                f" AND {facet} IS NOT NULL GROUP BY {facet}",
                (self.workspace_id,),
            ).fetchall()
            for row in rows:
                counts[(facet, row["value"])] = row["n"]
        for facet in LIST_FACETS:
            rows = self.conn.execute(
                f"""SELECT j.value AS value, COUNT(*) AS n
                    FROM shots s, json_each(s.{facet}_json) j
                    WHERE s.workspace_id = ? AND s.status IN ('indexed','needs_review')
                    GROUP BY j.value""",
                (self.workspace_id,),
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

    def claim_job(self) -> Job | None:
        """Atomically take the next runnable job. Safe across processes."""
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
                " started_at = datetime('now') WHERE id = ?",
                (row["id"],),
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

    def reset_stale_jobs(self) -> int:
        """Requeue jobs left 'running' by a killed worker. Called at startup."""
        cur = self.conn.execute(
            "UPDATE jobs SET status = 'queued' WHERE workspace_id = ? AND status = 'running'",
            (self.workspace_id,),
        )
        return cur.rowcount

    def job_counts(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT status, COUNT(*) AS n FROM jobs WHERE workspace_id = ? GROUP BY status",
            (self.workspace_id,),
        ).fetchall()
        return {r["status"]: r["n"] for r in rows}

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
