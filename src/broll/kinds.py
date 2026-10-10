"""Videos and images are two libraries side by side, never one mixed list.

A clip is cut into an edit by its time; a photograph is held for as long as the edit needs. They are
filed in separate folder trees in Drive ("Videos/..." and "Images/...") and every screen keeps them
apart the same way: pick one first, then everything you see is of that kind.
"""

from __future__ import annotations

KINDS = ("video", "image")
LABELS = {"video": "Videos", "image": "Images"}
SINGULAR = {"video": "video", "image": "image"}


def clean(value: str | None) -> str:
    """"video" or "image", or "" for anything else."""
    value = (value or "").strip().lower()
    return value if value in KINDS else ""


def other(kind: str) -> str:
    return "image" if kind == "video" else "video"


def noun(kind: str, count: int) -> str:
    """"1 video", "3 videos", "0 images"."""
    word = SINGULAR.get(kind, "clip")
    return f"{count} {word}" + ("" if count == 1 else "s")


def of_file(filename: str | None) -> str:
    """Which side a file belongs on, from its extension. Anything unknown is treated as footage."""
    from .ingest.scanner import media_kind

    return media_kind(filename or "") or "video"
