"""Applies the taxonomy to Drive: uploads, renames, moves and shortcuts.

Idempotent by construction: it reconciles desired state against actual state
rather than blindly creating, so running it twice performs zero writes. The
database is the source of truth and Drive is a rendering of it, which is what
makes ``broll reorganise`` a safe escape hatch when the taxonomy changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ..config import WorkspaceConfig
from ..db.models import ShotFacets, Source
from ..db.store import Store
from .client import (
    FOLDER_MIME,
    SHORTCUT_MIME,
    DriveClient,
    DriveError,
    DriveStorageFullError,
)
from ..analysis.schema import EMOTIONS
from .taxonomy import (
    ShortcutPlan,
    library_folder,
    plan_filename,
    plan_shortcuts,
    plan_tree,
    primary_shot,
    render_guide,
)

GUIDE_NAME = "READ ME FIRST - how this library works"

log = logging.getLogger(__name__)

ORGANISABLE_STATUSES = ("indexed", "needs_review")


@dataclass
class Action:
    kind: str      # create_folder | upload | rename | move | shortcut | delete_shortcut
    path: str
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.kind:<15} {self.path}" + (f"  ({self.detail})" if self.detail else "")


@dataclass
class OrganiseReport:
    actions: list[Action] = field(default_factory=list)
    sources: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)
    aborted: bool = False

    @property
    def writes(self) -> int:
        return len(self.actions)

    def record(self, kind: str, path: str, detail: str = "") -> None:
        self.actions.append(Action(kind=kind, path=path, detail=detail))

    def summary(self) -> str:
        by_kind: dict[str, int] = {}
        for action in self.actions:
            by_kind[action.kind] = by_kind.get(action.kind, 0) + 1
        parts = ", ".join(f"{k}={v}" for k, v in sorted(by_kind.items())) or "no changes"
        return f"{self.sources} source(s), {self.skipped} skipped: {parts}"


class Organizer:
    def __init__(
        self,
        config: WorkspaceConfig,
        store: Store,
        client: DriveClient,
        dry_run: bool = False,
        session=None,
    ):
        self.config = config
        self.store = store
        self.client = client
        self.dry_run = dry_run
        # A DriveSession, when filing clip after clip: its folder-id map is
        # shared, so the second clip does not re-walk the client's tree. Never
        # shared in a dry run, whose placeholder ids must not leak.
        self.session = session if not dry_run else None
        self._root_id: str | None = self.session.root_id if self.session else None
        self._folder_ids: dict[str, str] = self.session.folder_ids if self.session else {}
        self._tree_ready = self.session.tree_ready if self.session else False

    # -- root ---------------------------------------------------------------

    def root_id(self, report: OrganiseReport | None = None) -> str:
        if self._root_id:
            return self._root_id
        if self.config.drive_root_folder_id:
            self._root_id = self.config.drive_root_folder_id
            return self._root_id

        name = self.config.drive_root_folder_name
        existing = self.client.list_children("root").get(name)
        if existing and existing.mime_type == FOLDER_MIME:
            self._root_id = existing.id
        else:
            if self.dry_run:
                if report:
                    report.record("create_folder", name, "workspace root")
                self._root_id = f"dry-run-root:{name}"
                return self._root_id
            self._root_id = self.client.create_folder(name, "root").id
            if report:
                report.record("create_folder", name, "workspace root")
        return self._root_id

    def _folder_id(self, parts: tuple[str, ...], report: OrganiseReport) -> str:
        key = "/".join(parts)
        if key in self._folder_ids:
            return self._folder_ids[key]

        current = self.root_id(report)
        if self.session:
            self.session.root_id = current
        walked: list[str] = []
        for part in parts:
            walked.append(part)
            sub_key = "/".join(walked)
            if sub_key in self._folder_ids:
                current = self._folder_ids[sub_key]
                continue
            # A placeholder parent only exists in this dry run's imagination,
            # so there is nothing to look up: everything beneath it is new.
            existing = (
                None if current.startswith("dry-run")
                else self.client.list_children(current).get(part)
            )
            if existing and existing.mime_type == FOLDER_MIME:
                current = existing.id
            else:
                report.record("create_folder", sub_key)
                if self.dry_run:
                    current = f"dry-run:{sub_key}"
                    self._folder_ids[sub_key] = current
                    continue
                current = self.client.create_folder(part, current).id
            self._folder_ids[sub_key] = current
        return current

    # -- one source ---------------------------------------------------------

    def organise_source(self, source_id: str, report: OrganiseReport | None = None) -> OrganiseReport:
        report = report or OrganiseReport()
        source = self.store.get_source(source_id)
        if source is None:
            report.errors.append(f"no source {source_id}")
            return report

        shots = [
            s for s in self.store.shots_for_source(source_id)
            if s.status in ORGANISABLE_STATUSES
        ]
        if not shots:
            report.skipped += 1
            return report

        facets = [ShotFacets.from_shot(s) for s in shots]
        primary = primary_shot(facets)
        extension = Path(source.original_filename).suffix or ".mp4"
        filename = plan_filename(primary, source.content_hash, extension, self.config.taxonomy)

        if self.config.taxonomy.mode == "tree":
            self._ensure_tree(report)
            home, desired = plan_tree(facets, filename, self.config.taxonomy)
            target = home.parts
        else:
            target = library_folder(_ingest_month(source), self.config.taxonomy).parts
            desired = plan_shortcuts(
                facets,
                filename,
                self.store.facet_counts(),
                self.config.taxonomy,
                self._pair_counts(),
            )

        file_id = self._ensure_canonical_file(source, filename, target, report)
        if file_id is None:
            return report
        self._reconcile_shortcuts(source.id, file_id, desired, report)
        report.sources += 1
        return report

    def _ensure_canonical_file(
        self, source: Source, filename: str, target: tuple[str, ...],
        report: OrganiseReport,
    ) -> str | None:
        """One real copy, in its target folder, named for its primary shot.

        Faceted mode keeps it in _Library/<ingest month>; tree mode keeps it in
        the client's best-fit folder. Either way a change of folder is a move
        and a change of name is a rename - never a second upload.
        """
        library_id = self._folder_id(tuple(target), report)
        drive_path = "/".join(target) + f"/{filename}"

        if source.drive_file_id:
            entry = self.client.get(source.drive_file_id)
            if entry is None:
                report.errors.append(
                    f"{source.original_filename}: drive_file_id "
                    f"{source.drive_file_id} no longer exists - clear it to re-upload"
                )
                return None
            if entry.name != filename:
                report.record("rename", drive_path, f"was {entry.name}")
                if not self.dry_run:
                    self.client.rename(entry.id, filename)
            if library_id not in entry.parents and not library_id.startswith("dry-run"):
                report.record("move", drive_path, f"from {entry.parents}")
                if not self.dry_run:
                    self.client.move(entry.id, library_id, entry.parents)
            if not self.dry_run:
                self.store.update_source(
                    source.id,
                    drive_path=drive_path,
                    drive_web_link=entry.web_view_link or source.drive_web_link,
                )
            return entry.id

        local = Path(source.origin_path) if source.origin_path else None
        if local is None or not local.exists():
            report.errors.append(
                f"{source.original_filename}: no local file to upload "
                f"({source.origin_path or 'no path recorded'})"
            )
            return None

        report.record("upload", drive_path, f"{source.filesize_bytes or 0} bytes")
        if self.dry_run:
            return f"dry-run-file:{source.id}"

        uploaded = self.client.upload(local, filename, library_id)
        self.store.update_source(
            source.id,
            drive_file_id=uploaded.id,
            drive_web_link=uploaded.web_view_link,
            drive_path=drive_path,
        )
        return uploaded.id

    def _ensure_tree(self, report: OrganiseReport) -> None:
        """Create the client's whole folder structure, empty folders included.

        Editors browse this structure, so every folder should exist before
        anything is in it - that is how they learn where things go.
        """
        if self._tree_ready:
            return
        taxonomy = self.config.taxonomy
        for parts in taxonomy.tree_folders():
            self._folder_id(parts, report)
        if taxonomy.top_picks_folder:
            self._folder_id((taxonomy.top_picks_folder,), report)
        if taxonomy.guide_folder:
            guide_id = self._folder_id((taxonomy.guide_folder,), report)
            self._ensure_guide(guide_id, report)
        self._tree_ready = True
        if self.session:
            self.session.tree_ready = True

    def _ensure_guide(self, folder_id: str, report: OrganiseReport) -> None:
        path = f"{self.config.taxonomy.guide_folder}/{GUIDE_NAME}"
        if not folder_id.startswith("dry-run"):
            if GUIDE_NAME in self.client.list_children(folder_id):
                return  # created once; editing it by hand is fine
        report.record("create_doc", path)
        if self.dry_run:
            return
        text = render_guide(
            self.config.taxonomy,
            list(self.config.client.emotions) or list(EMOTIONS),
            self.config.client.name,
        )
        self.client.create_doc(GUIDE_NAME, text, folder_id)

    def _reconcile_shortcuts(
        self, source_id: str, file_id: str, desired: list[ShortcutPlan],
        report: OrganiseReport,
    ) -> None:
        wanted: dict[str, set[str]] = {}
        desired_keys = {(str(plan.folder), plan.name) for plan in desired}
        for plan in desired:
            folder_id = self._folder_id(plan.folder.parts, report)
            wanted.setdefault(folder_id, set()).add(plan.name)

            children = {} if folder_id.startswith("dry-run") else self.client.list_children(folder_id)
            existing = children.get(plan.name)
            if existing is not None:
                if existing.mime_type == SHORTCUT_MIME and existing.shortcut_target_id == file_id:
                    if not self.dry_run:
                        self.store.record_shortcut(
                            source_id, plan.shot_id, str(plan.folder), folder_id,
                            plan.name, existing.id, file_id,
                        )
                    continue  # already correct - the idempotent path
                if existing.mime_type != SHORTCUT_MIME:
                    report.errors.append(
                        f"{plan.path} exists and is not a shortcut - leaving it alone"
                    )
                    continue
                report.record("delete_shortcut", plan.path, "wrong target")
                if not self.dry_run:
                    self.client.delete_shortcut(existing.id)

            report.record("shortcut", plan.path)
            if not self.dry_run:
                created = self.client.create_shortcut(file_id, plan.name, folder_id)
                self.store.record_shortcut(
                    source_id, plan.shot_id, str(plan.folder), folder_id,
                    plan.name, created.id, file_id,
                )

        self._remove_stale(source_id, file_id, desired_keys, wanted, report)

    def _remove_stale(
        self, source_id: str, file_id: str, desired_keys: set[tuple[str, str]],
        wanted: dict[str, set[str]], report: OrganiseReport,
    ) -> None:
        """Shortcuts to this file that the taxonomy no longer justifies.

        Two passes: what we recorded when we created them (which catches
        folders the source is no longer planned into at all), and a sweep of
        the folders it still is planned into (which catches anything created
        outside this tool).
        """
        for record in self.store.shortcuts_for_source(source_id):
            key = (record["folder_path"], record["name"])
            if key in desired_keys:
                continue
            report.record("delete_shortcut", f"{key[0]}/{key[1]}", "no longer justified")
            if not self.dry_run:
                try:
                    self.client.delete_shortcut(record["shortcut_id"])
                except DriveError as exc:
                    report.errors.append(f"{key[0]}/{key[1]}: {exc}")
                self.store.forget_shortcut(*key)

        for folder_id, names in wanted.items():
            if folder_id.startswith("dry-run"):
                continue
            for name, entry in list(self.client.list_children(folder_id).items()):
                if entry.mime_type != SHORTCUT_MIME or entry.shortcut_target_id != file_id:
                    continue
                if name not in names:
                    report.record("delete_shortcut", name, "untracked, no longer justified")
                    if not self.dry_run:
                        self.client.delete_shortcut(entry.id)

    # -- whole library ------------------------------------------------------

    def reorganise(self, source_ids: list[str] | None = None) -> OrganiseReport:
        """Rebuild the tree from the database. The escape hatch when the
        taxonomy changes: the database is the source of truth."""
        report = OrganiseReport()
        ids = source_ids or [
            s.id for s in self.store.list_sources(limit=1_000_000)
            if s.status in ORGANISABLE_STATUSES
        ]
        for source_id in ids:
            try:
                self.organise_source(source_id, report)
            except DriveStorageFullError as exc:
                report.errors.append(str(exc))
                report.aborted = True
                break  # every remaining upload would fail the same way
            except Exception as exc:  # HttpError and friends: report, keep going
                report.errors.append(f"{source_id}: {type(exc).__name__}: {exc}")
        return report

    def _pair_counts(self) -> dict[tuple[str, str, str], int]:
        """Co-occurrence counts that decide the third folder level."""
        from .taxonomy import SECONDARY_FACET

        counts: dict[tuple[str, str, str], int] = {}
        for shot in self.store.list_shots(limit=1_000_000):
            if shot.status not in ORGANISABLE_STATUSES:
                continue
            facets = ShotFacets.from_shot(shot)
            for facet, secondary_facet in SECONDARY_FACET.items():
                for value in _values(facets, facet):
                    for other in _values(facets, secondary_facet):
                        key = (facet, value, other)
                        counts[key] = counts.get(key, 0) + 1
        return counts


def _values(facets: ShotFacets, facet: str) -> list[str]:
    value = getattr(facets, facet, None)
    if value is None:
        return []
    values = list(value) if isinstance(value, list) else [value]
    return [v for v in values if v and v not in ("unknown", "none")]


def _ingest_month(source: Source) -> str:
    stamp = source.created_at or ""
    try:
        return datetime.fromisoformat(stamp.replace("Z", "+00:00")).strftime("%Y-%m")
    except ValueError:
        return datetime.now(UTC).strftime("%Y-%m")
