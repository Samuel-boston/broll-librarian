"""Browsing the library by its folders. Pure functions over shot rows.

A clip counts in its own folder, every folder above it, and every folder it is
a secondary match for - the same places it appears in Drive, where secondary
folders hold a shortcut to it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_NUMBER = re.compile(r"^(\d+)[_\s.-]+")


def folder_label(name: str) -> tuple[str | None, str]:
    """'01_Nervous System Practices' -> ('01', 'Nervous System Practices')."""
    match = _NUMBER.match(name)
    return (match.group(1), name[match.end():]) if match else (None, name)


def ancestors(path: str) -> list[str]:
    parts = path.split("/")
    return ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]


def children(path: str, folder_paths: list[str]) -> list[str]:
    """Folders directly under ``path`` ("" for the top level), in tree order."""
    return [p for p in folder_paths if p.rpartition("/")[0] == path]


@dataclass
class FolderSummary:
    path: str
    number: str | None
    label: str
    count: int = 0
    thumbnails: list[str] = field(default_factory=list)
    has_children: bool = False


def summarise_folders(
    rows: list[dict], folder_paths: list[str], thumbnails_per_folder: int = 4
) -> dict[str, FolderSummary]:
    """Clip counts and preview thumbnails (newest first) for every folder."""
    summaries: dict[str, FolderSummary] = {}
    for path in folder_paths:
        number, label = folder_label(path.split("/")[-1])
        summaries[path] = FolderSummary(path=path, number=number, label=label)
    for path in folder_paths:
        parent = path.rpartition("/")[0]
        if parent in summaries:
            summaries[parent].has_children = True

    members: dict[str, set[str]] = {path: set() for path in folder_paths}
    for row in rows:  # newest first, so the first thumbnails are the latest
        touched: set[str] = set()
        for category in (row.get("category"), *row.get("secondary", [])):
            if category:
                touched.update(ancestors(category))
        for path in touched:
            if path not in summaries or row["id"] in members[path]:
                continue
            members[path].add(row["id"])
            summary = summaries[path]
            summary.count += 1
            if row.get("has_thumbnail") and len(summary.thumbnails) < thumbnails_per_folder:
                summary.thumbnails.append(row["id"])
    return summaries
