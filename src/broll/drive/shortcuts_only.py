"""Shortcuts-only organise: a browsable tree of native Drive shortcuts, originals untouched.

The originals stay exactly where they are under their own names. This builds a NEW root folder
and, in it, the folder tree plus one shortcut per placement (the primary shot's home folder
included) pointing at the original's file id. The only Drive writes it can make are
``create_folder`` (inside the new root, or the root itself once) and ``create_shortcut``:
the client is wrapped so any other method raises, and never reads file bytes.

Idempotent and resumable: each shortcut is recorded in ``drive_shortcuts`` the moment it is
created, a finished run plans zero writes, and a stopped run resumes where it left off.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import WorkspaceConfig
from ..db.models import ShotFacets
from ..db.store import Store
from .client import FOLDER_MIME, SHORTCUT_MIME, DriveError, DriveStorageFullError, _reason
from .organizer import ORGANISABLE_STATUSES
from .taxonomy import (
    library_folder,
    plan_filename,
    plan_shortcuts,
    plan_tree,
    primary_shot,
    shortcut_name,
)

log = logging.getLogger(__name__)

#: Drive reasons that mean "stop now": retrying or carrying on only digs the hole deeper.
FATAL_REASONS = {
    "downloadQuotaExceeded", "dailyLimitExceeded", "quotaExceeded", "storageQuotaExceeded",
    "sharingRateLimitExceeded", "domainPolicy", "appNotAuthorizedToFile",
}

#: The only client methods this mode may call.
ALLOWED_CLIENT_METHODS = frozenset({"list_children", "create_folder", "create_shortcut"})


class ShortcutsOnlyViolation(RuntimeError):
    """Something tried to call a Drive method this mode must never use."""


class _GuardedClient:
    """Exposes only the read + create methods. Move, rename, update, delete, upload,
    copy and download raise, so no plan can touch an original."""

    def __init__(self, client):
        self._client = client

    def __getattr__(self, name: str):
        if name in ALLOWED_CLIENT_METHODS:
            return getattr(self._client, name)
        raise ShortcutsOnlyViolation(
            f"shortcuts_only mode may not call client.{name}: it never moves, renames, copies, "
            "deletes or updates anything."
        )


@dataclass
class PlannedShortcut:
    source_id: str
    shot_id: str
    folder_path: str       # inside the new root, e.g. "Videos/Beach/Aerial"
    name: str
    target_id: str         # the ORIGINAL's Drive id


@dataclass
class ShortcutsReport:
    root_exists: bool = False
    folders_to_create: list[str] = field(default_factory=list)
    shortcuts: list[PlannedShortcut] = field(default_factory=list)  # still to create
    already_done: int = 0
    sources: int = 0
    skipped: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    created_folders: int = 0
    created_shortcuts: int = 0
    aborted: bool = False
    dry_run: bool = False

    @property
    def root_to_create(self) -> int:
        return 0 if self.root_exists else 1

    @property
    def write_calls(self) -> int:
        return self.root_to_create + len(self.folders_to_create) + len(self.shortcuts)

    def per_folder(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.shortcuts:
            counts[item.folder_path] = counts.get(item.folder_path, 0) + 1
        return counts

    def summary(self) -> str:
        if self.dry_run:
            return (
                f"{self.sources} source(s) planned, {len(self.skipped)} skipped. Would create: "
                f"{self.root_to_create} root, {len(self.folders_to_create)} folder(s), "
                f"{len(self.shortcuts)} shortcut(s) = {self.write_calls} write call(s); "
                f"{self.already_done} shortcut(s) already in place."
            )
        return (
            f"{self.sources} source(s), {len(self.skipped)} skipped. Created "
            f"{self.created_folders} folder(s) and {self.created_shortcuts} shortcut(s); "
            f"{self.already_done} already in place"
            + (", ABORTED" if self.aborted else "") + "."
        )


class ShortcutsOnlyOrganizer:
    def __init__(
        self,
        config: WorkspaceConfig,
        store: Store,
        client,
        dry_run: bool = False,
        sleep=time.sleep,
        limit: int | None = None,
    ):
        self.config = config
        self.store = store
        self.client = _GuardedClient(client)
        self.dry_run = dry_run
        self._sleep = sleep
        self.limit = limit  # stop after this many write calls (to run in slices)
        self._interval = 1.0 / max(config.ingest.shortcuts_writes_per_second, 0.1)
        self._last_write = 0.0
        self._writes = 0
        self._consecutive_errors = 0
        self._folder_ids: dict[str, str] = {}
        self._root_id: str | None = None

    # -- planning (reads only) -------------------------------------------------

    @property
    def root_name(self) -> str:
        return self.config.ingest.shortcuts_root_name

    def _find_root(self) -> str | None:
        existing = self.client.list_children(self.config.ingest.shortcuts_parent_id).get(self.root_name)
        if existing and existing.mime_type == FOLDER_MIME:
            return existing.id
        return None

    def _planned_folders(self, plans: list[PlannedShortcut]) -> list[str]:
        """Every folder path (parents first) that should exist: the client's whole tree, so editors
        see where things go, plus any folder a shortcut lands in."""
        taxonomy = self.config.taxonomy
        paths: set[tuple[str, ...]] = set()
        if taxonomy.mode == "tree":
            for prefix in taxonomy.media_prefixes():
                for parts in taxonomy.tree_folders():
                    paths.add((*prefix, *parts))
                if taxonomy.top_picks_folder:
                    paths.add((*prefix, taxonomy.top_picks_folder))
        for plan in plans:
            paths.add(tuple(plan.folder_path.split("/")))
        every: set[tuple[str, ...]] = set()
        for parts in paths:
            for i in range(1, len(parts) + 1):
                every.add(parts[:i])
        return ["/".join(p) for p in sorted(every, key=lambda p: (len(p), p))]

    def _desired(self, report: ShortcutsReport) -> list[PlannedShortcut]:
        taxonomy = self.config.taxonomy
        pair_counts = None
        facet_counts = None
        desired: list[PlannedShortcut] = []
        taken: dict[tuple[str, str], str] = {}  # (folder, name) -> target, to rename true duplicates
        sources = [
            s for s in self.store.list_sources(limit=1_000_000) if s.status in ORGANISABLE_STATUSES
        ]
        for source in sorted(sources, key=lambda s: s.id):
            shots = [
                s for s in self.store.shots_for_source(source.id)
                if s.status in ORGANISABLE_STATUSES and "analysis_failed" not in s.review_reasons
            ]
            if not shots:
                continue
            if not source.drive_file_id:
                report.skipped.append(f"{source.original_filename}: no drive_file_id")
                continue
            facets = [ShotFacets.from_shot(s) for s in shots]
            primary = primary_shot(facets)
            extension = Path(source.original_filename).suffix or ".mp4"
            filename = plan_filename(primary, source.content_hash, extension, taxonomy)
            if taxonomy.mode == "tree":
                prefix = taxonomy.media_prefix(source.media_kind)
                home, plans = plan_tree(facets, filename, taxonomy, prefix)
            else:
                if pair_counts is None:
                    from .organizer import Organizer
                    pair_counts = Organizer(self.config, self.store, None)._pair_counts()
                    facet_counts = self.store.facet_counts()
                home = library_folder(_ingest_month(source), taxonomy)
                plans = plan_shortcuts(facets, filename, facet_counts, taxonomy, pair_counts)
            # The original is not placed in its home folder (that would be a move), so the home gets a
            # shortcut too, named for the primary shot.
            placements = [(str(home), shortcut_name(filename, primary), primary.shot_id)]
            placements += [(str(p.folder), p.name, p.shot_id) for p in plans]
            for folder, name, shot_id in placements:
                key = (folder, name)
                if key in taken and taken[key] != source.drive_file_id:
                    stem, dot, ext = name.rpartition(".")
                    n = 2
                    while (folder, f"{stem}_dup{n}{dot}{ext}") in taken:
                        n += 1
                    name = f"{stem}_dup{n}{dot}{ext}"
                    key = (folder, name)
                if key in taken:
                    continue
                taken[key] = source.drive_file_id
                desired.append(PlannedShortcut(source.id, shot_id, folder, name, source.drive_file_id))
            report.sources += 1
        return desired

    def plan(self) -> ShortcutsReport:
        """Work out what is left to do. Reads Drive listings only; writes nothing."""
        report = ShortcutsReport(dry_run=self.dry_run)
        desired = self._desired(report)

        recorded: dict[tuple[str, str], str] = {}
        rows = self.store.conn.execute(
            "SELECT folder_path, name, target_id FROM drive_shortcuts WHERE workspace_id = ?",
            (self.store.workspace_id,),
        ).fetchall()
        for row in rows:
            recorded[(row["folder_path"], row["name"])] = row["target_id"]

        pending: list[PlannedShortcut] = []
        for item in desired:
            have = recorded.get((self._db_path(item.folder_path), item.name))
            if have is None:
                pending.append(item)
            elif have == item.target_id:
                report.already_done += 1
            else:
                report.conflicts.append(
                    f"{item.folder_path}/{item.name}: recorded for another file; left alone"
                )

        report.root_exists = False
        self._root_id = self._find_root()
        report.root_exists = self._root_id is not None

        if pending or not report.root_exists:
            folder_paths = self._planned_folders(pending)
        else:
            folder_paths = []
        # Resolve which folders exist (listing each parent once) and adopt shortcuts already there.
        existing_children: dict[str, dict] = {}
        pending = self._reconcile_with_drive(pending, folder_paths, report, existing_children)
        report.shortcuts = pending
        return report

    def _db_path(self, folder_path: str) -> str:
        return f"{self.root_name}/{folder_path}"

    def _reconcile_with_drive(self, pending, folder_paths, report, children_cache) -> list[PlannedShortcut]:
        to_create: list[str] = []
        ids: dict[str, str] = {}
        if self._root_id:
            ids[""] = self._root_id
        for path in folder_paths:
            parent_path, _, name = path.rpartition("/")
            parent_id = ids.get(parent_path)
            if parent_id is None:
                to_create.append(path)
                continue
            children = children_cache.get(parent_id)
            if children is None:
                children = children_cache[parent_id] = self.client.list_children(parent_id)
            entry = children.get(name)
            if entry and entry.mime_type == FOLDER_MIME:
                ids[path] = entry.id
            else:
                to_create.append(path)
        report.folders_to_create = to_create
        self._folder_ids = {p: i for p, i in ids.items() if p}

        # Shortcuts already in Drive (made by an earlier run whose DB write was lost): adopt, don't duplicate.
        remaining: list[PlannedShortcut] = []
        for item in pending:
            folder_id = ids.get(item.folder_path)
            if folder_id is not None:
                children = children_cache.get(folder_id)
                if children is None:
                    children = children_cache[folder_id] = self.client.list_children(folder_id)
                entry = children.get(item.name)
                if entry is not None:
                    if entry.mime_type == SHORTCUT_MIME and entry.shortcut_target_id == item.target_id:
                        report.already_done += 1
                        if not self.dry_run:
                            self.store.record_shortcut(
                                item.source_id, item.shot_id, self._db_path(item.folder_path),
                                folder_id, item.name, entry.id, item.target_id)
                    else:
                        report.conflicts.append(
                            f"{item.folder_path}/{item.name}: something else is already there; left alone")
                    continue
            remaining.append(item)
        return remaining

    # -- applying --------------------------------------------------------------

    def run(self) -> ShortcutsReport:
        report = self.plan()
        if self.dry_run:
            return report
        try:
            self._apply(report)
        except _Abort as exc:
            report.aborted = True
            report.errors.append(str(exc))
        self.store.conn.commit()
        return report

    def _throttle(self) -> None:
        wait = self._last_write + self._interval - time.monotonic()
        if wait > 0:
            self._sleep(wait)
        self._last_write = time.monotonic()

    def _write(self, label: str, call):
        """One rate-limited write. Raises _Abort on a fatal reason or too many failures in a row."""
        if self.limit is not None and self._writes >= self.limit:
            raise _Abort(f"Stopped after {self._writes} write(s) (--limit); run it again to continue.")
        self._throttle()
        self._writes += 1
        try:
            result = call()
        except DriveStorageFullError as exc:
            raise _Abort(f"{label}: {exc}") from exc
        except Exception as exc:
            if _reason(exc) in FATAL_REASONS or "quota" in str(exc).lower():
                raise _Abort(f"{label}: Drive quota/limit hit, stopping at once: {exc}") from exc
            self._consecutive_errors += 1
            if self._consecutive_errors >= self.config.ingest.shortcuts_max_consecutive_errors:
                raise _Abort(
                    f"Stopped after {self._consecutive_errors} failed writes in a row (last: {label}: {exc}). "
                    "Run it again once the cause is fixed; it resumes."
                ) from exc
            raise _Skip(f"{label}: {type(exc).__name__}: {exc}") from exc
        self._consecutive_errors = 0
        return result

    def _apply(self, report: ShortcutsReport) -> None:
        if self._root_id is None:
            root = self._write(
                f"root {self.root_name}",
                lambda: self.client.create_folder(self.root_name, self.config.ingest.shortcuts_parent_id),
            )
            self._root_id = root.id
            report.created_folders += 1
        ids = {"": self._root_id, **self._folder_ids}
        failed_folders: set[str] = set()
        for path in report.folders_to_create:
            parent_path, _, name = path.rpartition("/")
            if parent_path in failed_folders or parent_path not in ids:
                failed_folders.add(path)
                continue
            try:
                created = self._write(path, lambda: self.client.create_folder(name, ids[parent_path]))
            except _Skip as exc:
                report.errors.append(str(exc))
                failed_folders.add(path)
                continue
            ids[path] = created.id
            report.created_folders += 1
        for item in report.shortcuts:
            folder_id = ids.get(item.folder_path)
            if folder_id is None:
                continue  # its folder failed; the next run retries
            try:
                created = self._write(
                    f"{item.folder_path}/{item.name}",
                    lambda: self.client.create_shortcut(item.target_id, item.name, folder_id),
                )
            except _Skip as exc:
                report.errors.append(str(exc))
                continue
            self.store.record_shortcut(
                item.source_id, item.shot_id, self._db_path(item.folder_path), folder_id,
                item.name, created.id, item.target_id)
            self.store.conn.commit()
            report.created_shortcuts += 1


class _Abort(RuntimeError):
    pass


class _Skip(RuntimeError):
    pass


def _ingest_month(source) -> str:
    from .organizer import _ingest_month as month
    return month(source)
