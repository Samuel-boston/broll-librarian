"""Metadata -> Drive folder paths and filenames.

Pure functions only: no Drive calls, no I/O, no database. Everything the rules
need is passed in, which is what makes this the most heavily tested module in
the project.

The shape of the tree is one canonical copy of each file under ``_Library``
plus a faceted tree of *shortcuts*. A shortcut is a native Drive object that
points at a file without duplicating its bytes, so one clip can appear in a
dozen browsable places at zero storage cost.

Two rules stop the tree exploding:

* a third-level folder is only created when at least ``min_clips_for_subfolder``
  clips share that combination - otherwise clips sit at the second level;
* no level holds more than ``max_folders_per_level`` folders - the long tail is
  grouped under ``Other``.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..config import TaxonomyConfig
from ..db.models import ShotFacets

FacetKey = tuple[str, str]  # (facet, value)

# Which facet becomes the top-level browse folder, and which facet earns the
# third level under it when enough clips share the combination.
FACET_FOLDERS: dict[str, str] = {
    "subjects": "By Subject",
    "action": "By Action",
    "setting": "By Setting",
    "mood": "By Mood",
    "shot_type": "By Shot Type",
    "camera_movement": "By Camera Movement",
    "time_of_day": "By Time of Day",
    "colour_profile": "By Colour",
    "usable_for": "By Use",
}

SECONDARY_FACET: dict[str, str] = {
    "subjects": "setting",
    "action": "setting",
    "setting": "time_of_day",
    "mood": "setting",
    "shot_type": "camera_movement",
    "camera_movement": "setting",
    "time_of_day": "setting",
    "colour_profile": "mood",
    "usable_for": "setting",
}

LIST_FACETS = ("subjects", "mood", "usable_for")
UNINFORMATIVE = {"unknown", "none", "", None}


@dataclass(frozen=True)
class FolderPath:
    parts: tuple[str, ...]

    def __str__(self) -> str:
        return "/".join(self.parts)

    def __len__(self) -> int:
        return len(self.parts)


@dataclass(frozen=True)
class ShortcutPlan:
    folder: FolderPath
    name: str
    shot_id: str

    @property
    def path(self) -> str:
        return f"{self.folder}/{self.name}"


# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------


def folder_label(value: str) -> str:
    """'golden_hour' -> 'Golden Hour', 'open-plan office' -> 'Open-Plan Office'."""
    text = str(value).replace("_", " ").strip()
    return re.sub(
        r"[A-Za-z']+",
        lambda m: m.group(0)[0].upper() + m.group(0)[1:],
        text,
    )


def _ascii_slug(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def plan_filename(
    primary: ShotFacets,
    content_hash: str,
    ext: str,
    config: TaxonomyConfig | None = None,
) -> str:
    """Deterministic, descriptive, sortable: setting_action_time_shot_hash8.ext.

    The 8-character hash suffix guarantees uniqueness and makes the file
    traceable back to its database row.
    """
    config = config or TaxonomyConfig()
    parts = [
        _ascii_slug(primary.setting or ""),
        _ascii_slug(primary.action or ""),
        _ascii_slug(primary.time_of_day or ""),
        _ascii_slug(primary.shot_type or ""),
    ]
    parts = [p for p in parts if p and p not in ("unknown", "none")]
    suffix = content_hash[:8]
    extension = ext if ext.startswith(".") else f".{ext}"
    extension = _ascii_slug(extension) and f".{_ascii_slug(extension)}" or ".mp4"

    budget = config.filename_max_length - len(suffix) - len(extension) - 1
    stem = "_".join(parts) or "clip"
    if len(stem) > budget:
        stem = stem[:budget].rstrip("_")
    return f"{stem}_{suffix}{extension}"


def shortcut_name(filename: str, shot: ShotFacets) -> str:
    """Non-primary shots carry their timecode, so a browsing editor knows where
    to look inside the file."""
    if shot.is_primary:
        return filename
    stem, _, ext = filename.rpartition(".")
    minutes, seconds = divmod(int(shot.start_s), 60)
    return f"{stem}_at_{minutes}m{seconds:02d}s.{ext}"


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------


def library_folder(ingest_month: str, config: TaxonomyConfig | None = None) -> FolderPath:
    """The real files, sharded by ingest month to keep folders under ~500 items."""
    config = config or TaxonomyConfig()
    return FolderPath((config.library_folder_name, ingest_month))


def review_folder(config: TaxonomyConfig | None = None) -> FolderPath:
    config = config or TaxonomyConfig()
    return FolderPath((config.review_folder_name,))


def _facet_values(shot: ShotFacets, facet: str) -> list[str]:
    value = getattr(shot, facet, None)
    if value is None:
        return []
    values = list(value) if isinstance(value, list) else [value]
    return [v for v in values if v not in UNINFORMATIVE]


def _top_values(
    facet: str, facet_counts: Mapping[FacetKey, int], config: TaxonomyConfig
) -> set[str] | None:
    """The values that get their own folder, or None when no cap is needed.

    Returning None rather than "every value seen so far" matters: a value the
    counts have not caught up with yet must still get its own folder, not be
    swept into Other.
    """
    values = [(v, n) for (f, v), n in facet_counts.items() if f == facet]
    if len(values) <= config.max_folders_per_level:
        return None
    values.sort(key=lambda kv: (-kv[1], kv[0]))
    return {v for v, _ in values[: config.max_folders_per_level]}


def plan_paths(
    shots: Sequence[ShotFacets],
    facet_counts: Mapping[FacetKey, int],
    config: TaxonomyConfig | None = None,
    pair_counts: Mapping[tuple[str, str, str], int] | None = None,
) -> list[FolderPath]:
    """Folder paths a source belongs in - the union over all of its shots.

    ``facet_counts`` maps (facet, value) to how many shots in the library carry
    it; ``pair_counts`` maps (facet, value, secondary_value) the same way. Both
    come from the store - keeping them arguments is what keeps this pure.
    """
    config = config or TaxonomyConfig()
    pair_counts = pair_counts or {}
    paths: list[FolderPath] = []
    seen: set[tuple[str, ...]] = set()

    def add(parts: tuple[str, ...]) -> None:
        if parts not in seen:
            seen.add(parts)
            paths.append(FolderPath(parts))

    for facet, top_folder in FACET_FOLDERS.items():
        allowed = _top_values(facet, facet_counts, config)
        for shot in shots:
            for value in _facet_values(shot, facet):
                if allowed is None or value in allowed:
                    second = folder_label(value)
                else:
                    # The long tail is grouped rather than given its own folder.
                    add((top_folder, config.other_folder_name))
                    continue

                third = _third_level(facet, value, shot, pair_counts, config)
                if third:
                    add((top_folder, second, third))
                else:
                    add((top_folder, second))

    if any(shot.needs_review for shot in shots):
        add(review_folder(config).parts)

    return paths


def _third_level(
    facet: str,
    value: str,
    shot: ShotFacets,
    pair_counts: Mapping[tuple[str, str, str], int],
    config: TaxonomyConfig,
) -> str | None:
    """A third level only when the combination has earned it."""
    secondary_facet = SECONDARY_FACET.get(facet)
    if not secondary_facet:
        return None
    candidates = _facet_values(shot, secondary_facet)
    best: tuple[int, str] | None = None
    for candidate in candidates:
        count = pair_counts.get((facet, value, candidate), 0)
        if count >= config.min_clips_for_subfolder and (best is None or count > best[0]):
            best = (count, candidate)
    return folder_label(best[1]) if best else None


def plan_shortcuts(
    shots: Sequence[ShotFacets],
    filename: str,
    facet_counts: Mapping[FacetKey, int],
    config: TaxonomyConfig | None = None,
    pair_counts: Mapping[tuple[str, str, str], int] | None = None,
) -> list[ShortcutPlan]:
    """One shortcut per (folder, shot) the source justifies.

    A multi-shot source appears under the union of its shots' facets; each
    shortcut is named for the shot that put it there, so a file containing both
    a beach shot and an office shot shows up under both with the right timecode.
    """
    config = config or TaxonomyConfig()
    pair_counts = pair_counts or {}
    plans: list[ShortcutPlan] = []
    seen: set[str] = set()

    for shot in shots:
        for path in plan_paths([shot], facet_counts, config, pair_counts):
            name = shortcut_name(filename, shot)
            key = f"{path}/{name}"
            if key in seen:
                continue
            seen.add(key)
            plans.append(ShortcutPlan(folder=path, name=name, shot_id=shot.shot_id))
    return plans


def primary_shot(shots: Sequence[ShotFacets]) -> ShotFacets | None:
    for shot in shots:
        if shot.is_primary:
            return shot
    return shots[0] if shots else None
