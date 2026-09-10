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


FILENAME_TOKENS = {
    "setting": lambda f: f.setting,
    "action": lambda f: f.action,
    "time_of_day": lambda f: f.time_of_day,
    "shot_type": lambda f: f.shot_type,
    "leaf": lambda f: f.category.split("/")[-1] if f.category else None,
    "emotion": lambda f: f.emotions[0] if f.emotions else None,
    "mood": lambda f: f.mood[0] if f.mood else None,
    "subject": lambda f: f.subjects[0] if f.subjects else None,
}


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
    tokens = re.findall(r"\{(\w+)\}", config.filename_template)
    parts = [
        _ascii_slug(FILENAME_TOKENS[token](primary) or "")
        for token in tokens
        if token in FILENAME_TOKENS
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
            values = _facet_values(shot, facet)
            if facet in LIST_FACETS:
                values = values[: config.max_list_values_per_shot]
            for value in values:
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


# --------------------------------------------------------------------------
# A client's own folder structure ("tree" mode)
# --------------------------------------------------------------------------


def plan_tree(
    shots: Sequence[ShotFacets],
    filename: str,
    config: TaxonomyConfig,
) -> tuple[FolderPath, list[ShortcutPlan]]:
    """Where the real file lives, and which shortcuts point at it.

    The file itself goes in the primary shot's category - the folder an editor
    browsing the client's structure will look in first. Shortcuts cover the
    rest: other shots' categories, secondary categories, and Top Picks for a
    starred shot. Nothing is ever shortcut into the folder the file is in.
    """
    leaves = {path for path, _ in config.category_leaves()}
    primary = primary_shot(shots)
    home_category = primary.category if primary and primary.category in leaves else None
    home = (
        FolderPath(tuple(home_category.split("/")))
        if home_category
        else FolderPath((config.unsorted_folder,))
    )

    plans: list[ShortcutPlan] = []
    seen: set[tuple[str, str]] = set()

    def add(parts, shot: ShotFacets) -> None:
        name = shortcut_name(filename, shot)
        key = ("/".join(parts), name)
        if key not in seen:
            seen.add(key)
            plans.append(ShortcutPlan(folder=FolderPath(tuple(parts)), name=name,
                                      shot_id=shot.shot_id))

    for shot in shots:
        for category in (shot.category, *shot.secondary_categories):
            if category and category in leaves and category != home_category:
                add(category.split("/"), shot)
        if shot.top_pick and config.top_picks_folder:
            add((config.top_picks_folder,), shot)
    return home, plans


_TOKEN_LABELS = {
    "setting": "setting", "action": "action", "time_of_day": "time-of-day",
    "shot_type": "shot-type", "leaf": "folder", "emotion": "emotion",
    "mood": "mood", "subject": "subject",
}
_TOKEN_EXAMPLES = {
    "setting": "beach", "action": "meditating", "time_of_day": "golden_hour",
    "shot_type": "wide", "leaf": "meditation_stillness", "emotion": "calm",
    "mood": "serene", "subject": "man",
}


def render_guide(
    config: TaxonomyConfig,
    emotions: Sequence[str],
    client_name: str | None = None,
) -> str:
    """The text of the 00_START HERE guide. Pure, so it is testable."""
    tokens = [t for t in re.findall(r"\{(\w+)\}", config.filename_template) if t in _TOKEN_LABELS]
    pattern = "_".join(_TOKEN_LABELS[t] for t in tokens) + "_<id>.<ext>"
    example = "_".join(_TOKEN_EXAMPLES[t] for t in tokens) + "_f54362f4.mov"

    lines = [
        f"{client_name + ' - ' if client_name else ''}B-ROLL LIBRARY: HOW IT WORKS",
        "",
        "Every clip lives in exactly one folder - the one that fits it best. If it also "
        "fits somewhere else, there is a shortcut to it there too (the icon has a small "
        "arrow). Shortcuts take up no space and open the same file.",
        "",
    ]
    if config.top_picks_folder:
        lines.append(f"{config.top_picks_folder} holds shortcuts to the strongest clips, chosen by hand.")
    lines += [
        f"{config.unsorted_folder} holds anything that has not clearly fitted a folder yet.",
        "",
        "FILE NAMES",
        f"Files are renamed to: {pattern}",
        f"For example: {example}",
        "The id at the end is how the library finds the file again - please don't remove it.",
        "If a clip holds more than one shot, shortcuts to the later shots end in "
        "_at_1m23s, so you know where to scrub to.",
        "",
        "FOLDERS",
    ]
    for path, note in config.category_leaves():
        lines.append(f"{path.replace('/', ' > ')}" + (f" - {note}" if note else ""))
    lines += [
        "",
        "EMOTIONS",
        "Every clip is tagged with the emotions it carries, so you can search by feeling:",
        ", ".join(emotions),
    ]
    return "\n".join(lines) + "\n"
