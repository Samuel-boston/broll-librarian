"""Index files obtained outside Drive, and tie each one to its Drive original.

A file downloaded from Drive's website, or copied from a teammate, is byte-for-byte the Drive original.
Drive reports an md5 for every original, so a local file whose md5 and size agree with a listed Drive file
*is* that file. It is indexed with that file's `drive_file_id` and link, exactly as if it had been read from
Drive, so cards link to the original, filing later renames and moves the original (no upload, no copy), and
the dashboard sync sees an ordinary Drive-indexed source.

Nothing here reads a byte from Drive: the only Drive call is a metadata listing (names, ids, md5, size),
cached on disk so it is made once.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import WorkspaceConfig, broll_home
from ..db.store import Store
from .hashing import content_hash
from .scanner import DiscoveredFile, is_junk, media_kind

FILE_URL = "https://drive.google.com/file/d/{id}/view"
READ_CHUNK = 8 * 1024 * 1024


def inbox_dir() -> Path:
    """The drop-in folder: files copied here can be matched and indexed (`/data/inbox` on the server)."""
    return broll_home() / "inbox"


# -- the Drive listing ------------------------------------------------------


@dataclass
class DriveItem:
    id: str
    name: str
    size: int | None
    md5: str | None
    path: str = ""
    duration_s: float | None = None


def _cache_path(config: WorkspaceConfig, folder_id: str) -> Path:
    return config.dir / f"drive_inventory_{folder_id}.json"


def load_inventory(
    config: WorkspaceConfig, folder_id: str, client=None, refresh: bool = False,
) -> tuple[list[DriveItem], bool]:
    """Every media file under the Drive folder, from the cache when there is one.

    Returns (items, listed_now). Listing is metadata only, a request per folder. `client` is needed only
    when there is no cache yet (or `refresh`).
    """
    cache = _cache_path(config, folder_id)
    if cache.exists() and not refresh:
        data = json.loads(cache.read_text())
        return [DriveItem(**row) for row in data["items"]], False
    if client is None:
        raise RuntimeError("No saved Drive listing yet, and no Drive client to make one.")
    items: list[DriveItem] = []
    stack = [(folder_id, "")]
    seen: set[str] = set()
    while stack:
        current, prefix = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        for entry in client.list_all(current):
            if entry.is_folder:
                stack.append((entry.id, f"{prefix}/{entry.name}" if prefix else entry.name))
            elif entry.is_shortcut or is_junk(entry.name) or media_kind(entry.name) is None:
                continue
            else:
                items.append(DriveItem(entry.id, entry.name, entry.size, entry.md5, prefix, entry.duration_s))
    config.dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"folder": folder_id, "listed_at": time.time(),
                                 "items": [vars(i) for i in items]}))
    return items, True


# -- local md5, remembered ---------------------------------------------------


class Md5Cache:
    """md5 of a local file, remembered by (path, size, mtime) so a second run does not re-read gigabytes."""

    def __init__(self, path: Path):
        self.path = path
        try:
            self.data: dict[str, str] = json.loads(path.read_text())
        except (OSError, ValueError):
            self.data = {}
        self.dirty = False

    def md5(self, file: Path) -> str:
        stat = file.stat()
        key = f"{file.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
        if key not in self.data:
            digest = hashlib.md5()  # noqa: S324 - Drive's own checksum, not a security use
            with file.open("rb") as fh:
                while chunk := fh.read(READ_CHUNK):
                    digest.update(chunk)
            self.data[key] = digest.hexdigest()
            self.dirty = True
        return self.data[key]

    def save(self) -> None:
        if self.dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data))
            self.dirty = False


# -- matching ---------------------------------------------------------------


@dataclass
class MatchReport:
    matched: list[DiscoveredFile] = field(default_factory=list)
    already_indexed: list[str] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    not_media: int = 0
    # Informational: matched, but the local name differs from Drive's, or Drive holds the same bytes twice.
    renamed: list[str] = field(default_factory=list)
    duplicate_in_drive: list[str] = field(default_factory=list)
    duplicate_local: list[str] = field(default_factory=list)
    listed_now: bool = False

    def summary(self) -> str:
        parts = [f"{len(self.matched)} matched to Drive", f"{len(self.already_indexed)} already indexed",
                 f"{len(self.unmatched)} not found in Drive"]
        if self.duplicate_local:
            parts.append(f"{len(self.duplicate_local)} repeated copies")
        return ", ".join(parts)


def indexed_ids(store: Store) -> set[str]:
    rows = store.conn.execute(
        "SELECT drive_file_id FROM sources WHERE workspace_id = ? AND drive_file_id IS NOT NULL",
        (store.workspace_id,),
    ).fetchall()
    return {r[0] for r in rows}


def match_local_files(
    files: list[Path], items: list[DriveItem], store: Store, cache: Md5Cache, force: bool = False,
) -> MatchReport:
    """Match each local file to the Drive original with the same md5 and size.

    Files not found in Drive are reported and left out: this never creates a source that is not tied to
    a Drive file. Files already in the library (same Drive id, or same content) are skipped.
    """
    report = MatchReport()
    by_key: dict[tuple[str, int], list[DriveItem]] = {}
    for item in items:
        if item.md5 and item.size:
            by_key.setdefault((item.md5.lower(), item.size), []).append(item)
    indexed = set() if force else indexed_ids(store)
    claimed: set[str] = set()

    for path in files:
        if not path.is_file() or is_junk(path.name) or media_kind(path) is None:
            report.not_media += 1
            continue
        size = path.stat().st_size
        candidates = by_key.get((cache.md5(path).lower(), size))
        if not candidates:
            report.unmatched.append(path.name)
            continue
        named = [c for c in candidates if c.name.lower() == path.name.lower()]
        chosen = sorted(named or candidates, key=lambda c: c.id)[0]
        if len(candidates) > 1:
            report.duplicate_in_drive.append(f"{path.name}: {len(candidates)} identical files in Drive, using {chosen.id}")
        if chosen.id in claimed:
            report.duplicate_local.append(path.name)
            continue
        claimed.add(chosen.id)
        if chosen.id in indexed or (
            not force and store.source_by_hash(content_hash(path)) is not None
        ):
            report.already_indexed.append(path.name)
            continue
        if chosen.name != path.name:
            report.renamed.append(f"{path.name} is {chosen.name} in Drive")
        report.matched.append(DiscoveredFile(
            origin="local", path=path.resolve(), filename=chosen.name,
            drive_file_id=chosen.id, origin_path=str(path.resolve()),
            size_bytes=size, link=FILE_URL.format(id=chosen.id),
        ))
    cache.save()
    return report


def scan_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file())


def attach_drive_ids(config: WorkspaceConfig, indexed: set[str], files: list[DiscoveredFile]) -> int:
    """For the Upload screen: tie uploaded files that are Drive originals to their Drive file.

    Uses only the saved Drive listing (never calls Drive). With no listing, or no
    `drive_source_folder_id`, it does nothing and the uploads are ordinary new footage. A file that is not
    in the listing stays an ordinary upload too: it is new footage and will be filed into Drive as such.
    `indexed` is the Drive ids already in the library (`indexed_ids(store)`, read on the caller's thread).
    Returns how many were matched.
    """
    folder_id = config.drive_source_folder_id
    if not folder_id or not _cache_path(config, folder_id).exists() or not files:
        return 0
    items, _ = load_inventory(config, folder_id)
    by_key: dict[tuple[str, int], DriveItem] = {}
    for item in sorted(items, key=lambda i: i.id):
        if item.md5 and item.size:
            by_key.setdefault((item.md5.lower(), item.size), item)
    cache = Md5Cache(config.dir / "local_md5.json")
    indexed = set(indexed)
    matched = 0
    for discovered in files:
        if discovered.path is None or not discovered.path.is_file():
            continue
        item = by_key.get((cache.md5(discovered.path).lower(), discovered.path.stat().st_size))
        if item is None or item.id in indexed:
            continue
        discovered.drive_file_id = item.id
        discovered.filename = item.name
        discovered.link = FILE_URL.format(id=item.id)
        discovered.size_bytes = discovered.path.stat().st_size
        indexed.add(item.id)
        matched += 1
    cache.save()
    return matched


# -- copies made inside Drive -----------------------------------------------


@dataclass
class CopyReport:
    matched: list[DiscoveredFile] = field(default_factory=list)
    already_indexed: list[str] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    duplicate_in_drive: list[str] = field(default_factory=list)
    duplicate_copies: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"{len(self.matched)} copies matched to their originals, "
                f"{len(self.already_indexed)} already indexed, {len(self.unmatched)} with no original")


def _strip_copy_prefix(name: str) -> str:
    for prefix in ("Copy of ", "Copy of  "):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def match_copies(copies: list[DriveItem], originals: list[DriveItem], store: Store,
                 force: bool = False) -> CopyReport:
    """Tie each file in a folder of Drive copies to its original, from two metadata listings.

    A Drive "Make a copy" has the original's md5 and size. The result is indexed in place from the copy
    (bytes streamed from it) but is identified as the original: its `drive_file_id` and link are the
    original's, the copy's id is only the byte source. A copy with no original is reported and left out.
    """
    report = CopyReport()
    by_key: dict[tuple[str, int], list[DriveItem]] = {}
    for item in originals:
        if item.md5 and item.size:
            by_key.setdefault((item.md5.lower(), item.size), []).append(item)
    indexed = set() if force else indexed_ids(store)
    claimed: set[str] = set()
    for copy in sorted(copies, key=lambda c: (c.name, c.id)):
        candidates = by_key.get((copy.md5.lower(), copy.size)) if copy.md5 and copy.size else None
        if not candidates:
            report.unmatched.append(copy.name)
            continue
        wanted = _strip_copy_prefix(copy.name).lower()
        named = [c for c in candidates if c.name.lower() == wanted]
        chosen = sorted(named or candidates, key=lambda c: c.id)[0]
        if len(candidates) > 1:
            report.duplicate_in_drive.append(
                f"{copy.name}: {len(candidates)} identical originals, using {chosen.id}")
        if chosen.id in claimed:
            report.duplicate_copies.append(copy.name)
            continue
        claimed.add(chosen.id)
        if chosen.id in indexed:
            report.already_indexed.append(copy.name)
            continue
        report.matched.append(DiscoveredFile(
            origin="drive", path=None, filename=chosen.name, drive_file_id=chosen.id,
            origin_path=f"drive:{chosen.id}", size_bytes=chosen.size,
            duration_s=copy.duration_s or chosen.duration_s,
            link=FILE_URL.format(id=chosen.id), byte_source_id=copy.id,
        ))
    return report
