"""Discovering files to index.

Three entry points all produce the same job payload: an uploaded file staged by
the web UI, a local path, or a Drive folder.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

VIDEO_SUFFIXES = {
    ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".mts", ".m2ts", ".mxf", ".wmv",
}

Origin = Literal["upload", "local", "drive"]


@dataclass
class DiscoveredFile:
    origin: Origin
    path: Path | None          # local bytes, if we already have them
    filename: str
    drive_file_id: str | None = None
    origin_path: str | None = None

    def payload(self) -> dict:
        return {
            "origin": self.origin,
            "path": str(self.path) if self.path else None,
            "filename": self.filename,
            "drive_file_id": self.drive_file_id,
            "origin_path": self.origin_path,
        }


def is_video(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES


def scan_local(path: Path, recursive: bool = True) -> list[DiscoveredFile]:
    """Videos at a local path. Files here are never moved, copied or deleted."""
    if path.is_file():
        candidates: Iterator[Path] = iter([path])
    else:
        candidates = path.rglob("*") if recursive else path.glob("*")
    return [
        DiscoveredFile(
            origin="local", path=p.resolve(), filename=p.name, origin_path=str(p.resolve())
        )
        for p in sorted(candidates)
        if is_video(p)
    ]


def stage_upload(src: Path, staging_dir: Path) -> DiscoveredFile:
    """Copy an uploaded file into the workspace staging area.

    Only files under staging/ and tmp/ are ever deleted by cleanup.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    target = staging_dir / src.name
    counter = 1
    while target.exists():
        target = staging_dir / f"{src.stem}_{counter}{src.suffix}"
        counter += 1
    shutil.copy2(src, target)
    return DiscoveredFile(
        origin="upload", path=target, filename=src.name, origin_path=str(target)
    )


def scan_drive_folder(folder_id: str) -> list[DiscoveredFile]:
    """Videos already in a Drive folder. Implemented in M3."""
    raise NotImplementedError(
        "Indexing an existing Drive folder arrives with M3 (Drive integration). "
        "Until then, point `broll index` at a local path."
    )
