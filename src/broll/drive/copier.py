"""Copy a Drive folder, subfolders included, into a new folder, leaving the original untouched.

Why: the organiser moves and renames the files it files. To organise an archive without touching the
original, copy it first, then index and organise the copy.

* The copy is made by Drive itself (`files.copy`): nothing is downloaded or uploaded, so it is fast
  and uses no bandwidth here. The copies belong to whoever is signed in and count against *their*
  storage, so sign in as the person who should own the archive.
* The source is only ever read. This module never moves, renames, edits or deletes anything in it.
* Only photos and video are copied by default (a footage folder often has stray documents).
  Shortcuts are skipped, and counted.
* It remembers what it has copied (a small state file), so a run that is stopped, or repeated after
  new footage arrives, copies only what is missing and never makes a second copy of anything.
* `dry_run` reads and counts everything and writes nothing: the size to expect, before committing.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .client import DriveError, DriveFile, DriveStorageFullError

MEDIA_PREFIXES = ("video/", "image/")
PROVENANCE_KEY = "broll_copy_of"
FOLDER_ID_RE = re.compile(r"/folders/([A-Za-z0-9_-]{10,})")
BARE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{10,}$")


def parse_folder_id(text: str) -> str:
    """A folder id from a Drive link, or the id itself."""
    text = text.strip()
    match = FOLDER_ID_RE.search(text)
    if match:
        return match.group(1)
    if BARE_ID_RE.match(text):
        return text
    raise ValueError(f"That doesn't look like a Drive folder link or id: {text!r}")


def human_size(n: int | float) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


@dataclass
class CopyReport:
    dry_run: bool
    source_name: str = ""
    dest_root_id: str | None = None
    folders: int = 0
    media_files: int = 0
    media_bytes: int = 0
    other_files: int = 0
    shortcuts: int = 0
    copied: int = 0
    copied_bytes: int = 0
    already_copied: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)
    storage_limit: int | None = None
    storage_used: int = 0

    @property
    def to_copy(self) -> int:
        return self.media_files - self.already_copied

    def summary(self) -> str:
        lines = [
            f"{self.source_name or 'Source'}: {self.media_files} photo/video file(s) in "
            f"{self.folders} folder(s), {human_size(self.media_bytes)} in total."
        ]
        if self.other_files or self.shortcuts:
            lines.append(f"Left out: {self.other_files} non-media file(s), {self.shortcuts} shortcut(s).")
        if self.storage_limit:
            free = max(0, self.storage_limit - self.storage_used)
            lines.append(f"Drive storage: {human_size(self.storage_used)} used of "
                         f"{human_size(self.storage_limit)} ({human_size(free)} free).")
        elif self.storage_used:
            lines.append(f"Drive storage in use: {human_size(self.storage_used)} (no limit reported).")
        if self.dry_run:
            lines.append(f"A copy would add about {human_size(self.media_bytes)} to the Drive. Nothing was copied.")
        else:
            lines.append(f"Copied {self.copied} file(s) ({human_size(self.copied_bytes)}); "
                         f"{self.already_copied} were already copied.")
            if self.dest_root_id:
                lines.append(f"The copy is in folder {self.dest_root_id}.")
        if self.failed:
            lines.append(f"{len(self.failed)} file(s) could not be copied:")
            lines.extend(f"  - {name}: {why}" for name, why in self.failed[:20])
        return "\n".join(lines)


def _load_state(path: Path, source_id: str) -> dict:
    try:
        all_state = json.loads(path.read_text())
        state = all_state.get(source_id)
        if isinstance(state, dict) and isinstance(state.get("files"), dict) and isinstance(state.get("folders"), dict):
            return state
    except (OSError, ValueError):
        pass
    return {"dest_root": None, "folders": {}, "files": {}}


def _save_state(path: Path, source_id: str, state: dict) -> None:
    try:
        all_state = json.loads(path.read_text())
        if not isinstance(all_state, dict):
            all_state = {}
    except (OSError, ValueError):
        all_state = {}
    all_state[source_id] = state
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(all_state))
    os.replace(tmp, path)  # swapped in whole: a crash never leaves half a file


def _ancestors(client, file_id: str, limit: int = 50) -> set[str]:
    """Ids of every folder above `file_id`, so a destination inside the source can be refused."""
    seen: set[str] = set()
    frontier = [file_id]
    for _ in range(limit):
        if not frontier:
            break
        entry = client.get(frontier.pop())
        if entry is None:
            continue
        for parent in entry.parents:
            if parent not in seen:
                seen.add(parent)
                frontier.append(parent)
    return seen


def copy_folder(
    client,
    source_id: str,
    dest_parent_id: str = "root",
    dest_name: str | None = None,
    *,
    state_path: Path,
    media_only: bool = True,
    dry_run: bool = False,
    progress: Callable[[str], None] | None = None,
) -> CopyReport:
    say = progress or (lambda _msg: None)
    source = client.get(source_id)
    if source is None or not source.is_folder:
        raise DriveError("That isn't a folder this Drive login can open. Is it shared with the account you signed in with?")

    report = CopyReport(dry_run=dry_run, source_name=source.name)
    try:
        report.storage_limit, report.storage_used = client.storage_quota()
    except Exception:  # the size report must never stop a copy
        pass

    # Never copy a folder into itself or anything under it: it would copy its own copy forever.
    if dest_parent_id == source_id or source_id in _ancestors(client, dest_parent_id):
        raise DriveError("The destination is inside the folder being copied. Choose a different place for the copy.")

    state = _load_state(state_path, source_id)
    dest_name = dest_name or f"{source.name} (working copy)"

    # 1. Read the whole source tree. Read-only, and it gives the size before anything is written.
    tree: list[tuple[DriveFile, str, list[DriveFile]]] = []  # (folder, path, media files in it)
    stack: list[tuple[DriveFile, str]] = [(source, "")]
    while stack:
        folder, path = stack.pop()
        files: list[DriveFile] = []
        for child in client.list_all(folder.id):
            if child.is_folder:
                stack.append((child, f"{path}/{child.name}" if path else child.name))
            elif child.is_shortcut:
                report.shortcuts += 1
            elif media_only and not child.mime_type.startswith(MEDIA_PREFIXES):
                report.other_files += 1
            else:
                files.append(child)
        report.folders += 1
        tree.append((folder, path, files))
        report.media_files += len(files)
        report.media_bytes += sum(f.size or 0 for f in files)
        say(f"read {report.folders} folder(s), {report.media_files} file(s)...")

    report.already_copied = sum(1 for _f, _p, files in tree for f in files if f.id in state["files"])
    if dry_run:
        return report

    # 2. Mirror the folders (before the files, so every file has somewhere to go).
    if state["dest_root"] is None:
        state["dest_root"] = client.ensure_folder(dest_name, dest_parent_id).id
        _save_state(state_path, source_id, state)
    report.dest_root_id = state["dest_root"]
    state["folders"][source_id] = state["dest_root"]

    for folder, _path, _files in sorted(tree, key=lambda t: t[1].count("/") if t[1] else -1):
        if folder.id in state["folders"]:
            continue
        dest_parent = next((state["folders"][pid] for pid in folder.parents if pid in state["folders"]),
                           state["dest_root"])
        state["folders"][folder.id] = client.ensure_folder(folder.name, dest_parent).id
    _save_state(state_path, source_id, state)

    # Trust Drive over the state file: every copy carries a stamp saying what it was copied from, so
    # anything copied but not yet written down (a run killed part way, a lost state file) is found
    # here and is never copied twice.
    for dest_folder_id in set(state["folders"].values()):
        for existing in client.list_all(dest_folder_id):
            origin = (existing.app_properties or {}).get(PROVENANCE_KEY)
            if origin and origin not in state["files"]:
                state["files"][origin] = existing.id
    report.already_copied = sum(1 for _f, _p, files in tree for f in files if f.id in state["files"])
    _save_state(state_path, source_id, state)

    # 3. Copy the files, remembering each one as it lands.
    done_since_save = 0
    for folder, _path, files in tree:
        dest_folder = state["folders"][folder.id]
        for entry in files:
            if entry.id in state["files"]:
                continue
            try:
                copy = client.copy_file(entry.id, entry.name, dest_folder, {PROVENANCE_KEY: entry.id})
            except DriveStorageFullError:
                _save_state(state_path, source_id, state)
                raise  # nothing else can succeed; what is done is kept, and a re-run resumes
            except Exception as exc:  # one unreadable file must not stop the rest
                report.failed.append((entry.name, str(exc)[:160]))
                continue
            state["files"][entry.id] = copy.id
            report.copied += 1
            report.copied_bytes += entry.size or 0
            done_since_save += 1
            if done_since_save >= 10:
                _save_state(state_path, source_id, state)
                done_since_save = 0
                say(f"copied {report.copied} of {report.media_files - report.already_copied}...")
    _save_state(state_path, source_id, state)
    return report
