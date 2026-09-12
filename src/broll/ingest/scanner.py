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
# Stills are footage too: the model already only ever sees extracted frames, so
# a photograph is the same pipeline with the shot detection removed. HEIC is
# read through ffmpeg (HEVC in a HEIF container), so no extra dependency.
IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tif", ".tiff", ".avif", ".bmp",
}
MEDIA_SUFFIXES = VIDEO_SUFFIXES | IMAGE_SUFFIXES

Origin = Literal["upload", "local", "drive"]
MediaKind = Literal["video", "image"]


def media_kind(name: str | Path) -> MediaKind | None:
    """'video', 'image', or None for a file this library does not index."""
    suffix = Path(name).suffix.lower()
    if suffix in VIDEO_SUFFIXES:
        return "video"
    if suffix in IMAGE_SUFFIXES:
        return "image"
    return None


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


def is_media(path: Path) -> bool:
    return path.is_file() and media_kind(path) is not None


def scan_local(path: Path, recursive: bool = True) -> list[DiscoveredFile]:
    """Clips and stills at a local path. Nothing here is moved, copied or deleted."""
    if path.is_file():
        candidates: Iterator[Path] = iter([path])
    else:
        candidates = path.rglob("*") if recursive else path.glob("*")
    return [
        DiscoveredFile(
            origin="local", path=p.resolve(), filename=p.name, origin_path=str(p.resolve())
        )
        for p in sorted(candidates)
        if is_media(p)
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


def scan_drive_folder(client, folder_id: str, recursive: bool = True) -> list[DiscoveredFile]:
    """Clips and stills already in a Drive folder.

    Nothing here is downloaded: the pipeline fetches bytes only when it reaches
    a file it actually has to analyse.
    """
    found: list[DiscoveredFile] = []
    stack = [folder_id]
    seen: set[str] = set()

    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        for entry in client.list_children(current).values():
            if entry.is_folder:
                if recursive:
                    stack.append(entry.id)
                continue
            if entry.is_shortcut:
                continue  # a shortcut is another view of a file we already saw
            if media_kind(entry.name) is None:
                continue
            found.append(
                DiscoveredFile(
                    origin="drive",
                    path=None,
                    filename=entry.name,
                    drive_file_id=entry.id,
                    origin_path=f"drive:{entry.id}",
                )
            )
    return sorted(found, key=lambda f: f.filename)
