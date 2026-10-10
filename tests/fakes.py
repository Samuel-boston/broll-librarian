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
        if parent_id not in self._files:
            # Match the real API: listing an unknown id is a 404, not an empty list.
            raise DriveError(f"File not found: {parent_id}")
        return {
            f.name: f for f in self._files.values() if parent_id in f.parents
        }

    def list_all(self, parent_id: str) -> list[DriveFile]:
        if parent_id not in self._files:
            raise DriveError(f"File not found: {parent_id}")
        return [f for f in self._files.values() if parent_id in f.parents]

    def storage_quota(self) -> tuple[int | None, int]:
        return getattr(self, "quota", (None, 0))

    def add_file(self, name: str, parent_id: str, mime_type: str = "video/mp4",
                 size: int | None = 1000) -> DriveFile:
        """Seed a file without counting it as a write (it is the starting state of the test)."""
        entry = DriveFile(id=self._new_id("src"), name=name, mime_type=mime_type,
                          parents=[parent_id], size=size)
        self._files[entry.id] = entry
        return entry

    def add_folder(self, name: str, parent_id: str) -> DriveFile:
        entry = DriveFile(id=self._new_id("srcfolder"), name=name, mime_type=FOLDER_MIME,
                          parents=[parent_id])
        self._files[entry.id] = entry
        return entry

    # -- writes -------------------------------------------------------------

    def copy_file(self, file_id: str, name: str, parent_id: str, app_properties=None) -> DriveFile:
        self._record(f"copy:{name}")
        original = self._files[file_id]
        entry = DriveFile(id=self._new_id("copy"), name=name, mime_type=original.mime_type,
                          parents=[parent_id], size=original.size, md5=original.md5)
        entry.app_properties = dict(app_properties or {}) or None
        self._files[entry.id] = entry
        return entry

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

    def create_doc(self, name: str, text: str, parent_id: str) -> DriveFile:
        self._record(f"create_doc:{name}")
        entry = DriveFile(
            id=self._new_id("doc"), name=name,
            mime_type="application/vnd.google-apps.document", parents=[parent_id],
        )
        entry.text = text  # type: ignore[attr-defined]
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


class FakeTextProvider:
    """A text model that files clips by a word in their caption. No network.

    `rules` maps a caption word to (folder, confidence, secondary folders). Counts its calls.
    """

    name = "fake"

    def __init__(self, rules: dict, default=None):
        self.rules = rules
        self.default = default
        self.prompts: list[str] = []
        self.fail_first = 0

    async def complete(self, prompt, schema):
        import re

        self.prompts.append(prompt)
        if self.fail_first:
            self.fail_first -= 1
            return schema.model_validate({"items": [{"clip": 1, "category_confidence": 7}]})
        items = []
        for number, block in re.findall(r"CLIP (\d+) \(.*?\)\n((?:  .*\n?)*)", prompt):
            caption = re.search(r"caption: (.*)", block).group(1).lower()
            pick = next((v for k, v in self.rules.items() if k in caption), self.default)
            if pick is None:
                continue
            folder, confidence, *rest = pick
            items.append({"clip": int(number), "category": folder, "category_confidence": confidence,
                          "secondary_categories": list(rest[0]) if rest else []})
        return schema.model_validate({"items": items})

    def estimate_cost(self, prompt):
        return 0.001
