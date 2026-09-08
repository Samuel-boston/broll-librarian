"""An in-memory Drive stand-in with the same surface as DriveClient.

Every write is counted, which is how the idempotency test asserts that a second
organise pass performs zero writes.
"""

from __future__ import annotations

import itertools
from pathlib import Path

from broll.drive.client import FOLDER_MIME, SHORTCUT_MIME, DriveError, DriveFile


class FakeDriveClient:
    def __init__(self):
        self._files: dict[str, DriveFile] = {
            "root": DriveFile(id="root", name="My Drive", mime_type=FOLDER_MIME)
        }
        self._ids = itertools.count(1)
        self.writes = 0
        self.calls: list[str] = []

    # -- helpers ------------------------------------------------------------

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}{next(self._ids)}"

    def _record(self, name: str) -> None:
        self.writes += 1
        self.calls.append(name)

    def tree(self) -> set[str]:
        """Every path in the fake Drive, for comparing before and after."""
        paths: set[str] = set()

        def walk(parent_id: str, prefix: str) -> None:
            for child in self._files.values():
                if parent_id in child.parents:
                    path = f"{prefix}/{child.name}" if prefix else child.name
                    paths.add(path)
                    if child.is_folder:
                        walk(child.id, path)

        walk("root", "")
        return paths

    def shortcuts(self) -> list[DriveFile]:
        return [f for f in self._files.values() if f.is_shortcut]

    # -- reads --------------------------------------------------------------

    def get(self, file_id: str) -> DriveFile | None:
        return self._files.get(file_id)

    def list_children(self, parent_id: str, refresh: bool = False) -> dict[str, DriveFile]:
        return {
            f.name: f for f in self._files.values() if parent_id in f.parents
        }

    # -- writes -------------------------------------------------------------

    def create_folder(self, name: str, parent_id: str) -> DriveFile:
        self._record(f"create_folder:{name}")
        entry = DriveFile(
            id=self._new_id("folder"), name=name, mime_type=FOLDER_MIME, parents=[parent_id]
        )
        self._files[entry.id] = entry
        return entry

    def ensure_folder(self, name: str, parent_id: str) -> DriveFile:
        existing = self.list_children(parent_id).get(name)
        if existing and existing.is_folder:
            return existing
        return self.create_folder(name, parent_id)

    def resolve_path(self, parts, root_id: str) -> str:
        current = root_id
        for part in parts:
            current = self.ensure_folder(part, current).id
        return current

    def upload(self, path: Path, name: str, parent_id: str) -> DriveFile:
        self._record(f"upload:{name}")
        entry = DriveFile(
            id=self._new_id("file"),
            name=name,
            mime_type="video/mp4",
            parents=[parent_id],
            web_view_link=f"https://drive.google.com/file/d/{name}/view",
        )
        self._files[entry.id] = entry
        return entry

    def download(self, file_id: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fake video bytes")
        return destination

    def create_shortcut(self, target_id: str, name: str, parent_id: str) -> DriveFile:
        self._record(f"shortcut:{name}")
        entry = DriveFile(
            id=self._new_id("shortcut"),
            name=name,
            mime_type=SHORTCUT_MIME,
            parents=[parent_id],
            shortcut_target_id=target_id,
        )
        self._files[entry.id] = entry
        return entry

    def rename(self, file_id: str, name: str) -> DriveFile:
        self._record(f"rename:{name}")
        self._files[file_id].name = name
        return self._files[file_id]

    def move(self, file_id: str, add_parent: str, remove_parents) -> DriveFile:
        self._record(f"move:{file_id}")
        entry = self._files[file_id]
        entry.parents = [p for p in entry.parents if p not in set(remove_parents)]
        entry.parents.append(add_parent)
        return entry

    def delete_shortcut(self, file_id: str) -> None:
        entry = self._files.get(file_id)
        if entry is None:
            return
        if not entry.is_shortcut:
            raise DriveError(f"refusing to delete {entry.name!r}: it is not a shortcut.")
        self._record(f"delete_shortcut:{entry.name}")
        del self._files[file_id]

    def invalidate(self) -> None:
        pass
