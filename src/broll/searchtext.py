"""How a shot becomes searchable text.

One place, so keyword search, the vector index and the dashboard all see the same words.

Two tiers, because they deserve different trust:

* primary - what is in the picture: the caption, the subjects, the action, the place, the visible
  tags. A word here means the thing is on screen.
* concept - what the footage stands for: the client's themes, the mood and emotions, the folder it
  was filed in. A word here is an interpretation, so a match counts for less.

Deliberately left out of both: shot type, camera movement, colour profile, people count, pace and
"usable for". They stay as filters. As words they only add noise: "static" sits on most clips, so
a script line about "static electricity" or "a slow morning" would pull in unrelated footage.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence

# Time-of-day values that carry no information as a word.
_UNINFORMATIVE_TIME = {"", "unknown", "indoor_artificial"}


def time_words(time_of_day: str | None) -> str | None:
    if not time_of_day or time_of_day in _UNINFORMATIVE_TIME:
        return None
    return time_of_day.replace("_", " ")


def _terms(values: Iterable[str | None]) -> list[str]:
    return [str(v).strip() for v in values if v and str(v).strip()]


_NUMBER_PREFIX = re.compile(r"^\d+[_ -]+")


def category_terms(categories: Iterable[str | None]) -> list[str]:
    """Every folder name on the way to each folder a shot is filed in, leaf first.

    "03_Work & Speaking/Speaking/Keynotes" is three words worth searching: a clip in Keynotes is
    also a clip about speaking. The "03_" that keeps folders in order is not a word.
    """
    terms: list[str] = []
    for category in categories:
        if not category:
            continue
        for segment in reversed(category.split("/")):
            name = _NUMBER_PREFIX.sub("", segment).strip()
            if name and name not in terms:
                terms.append(name)
    return terms


# Kept for callers that only want the last folder.
def category_leaves(categories: Iterable[str | None]) -> list[str]:
    """The folder names (last path segment) of the folders a shot is filed in."""
    return [_NUMBER_PREFIX.sub("", c.split("/")[-1]).strip() for c in categories if c]


def primary_terms(
    *,
    caption: str | None,
    action: str | None,
    setting: str | None,
    setting_detail: str | None,
    subjects: Sequence[str],
    tags: Sequence[str],
    time_of_day: str | None,
    body_language: Sequence[str] = (),
) -> list[str]:
    return _terms([
        caption,
        None if action in (None, "none") else action,
        setting,
        setting_detail,
        *subjects,
        *body_language,
        *tags,
        time_words(time_of_day),
    ])


def concept_terms(
    *,
    themes: Sequence[str],
    mood: Sequence[str],
    emotions: Sequence[str],
    categories: Iterable[str | None],
    phrases: Sequence[str] = (),
) -> list[str]:
    """What the clip stands for, and how an editor would put it when looking for it."""
    return _terms([*themes, *mood, *emotions, *phrases, *category_terms(categories)])


def join_terms(terms: Iterable[str]) -> str:
    return ", ".join(terms)


def embedding_text(
    *,
    caption: str | None,
    action: str | None,
    setting: str | None,
    setting_detail: str | None,
    subjects: Sequence[str],
    tags: Sequence[str],
    time_of_day: str | None,
    body_language: Sequence[str] = (),
    themes: Sequence[str] = (),
    mood: Sequence[str],
    emotions: Sequence[str],
    categories: Sequence[str | None],
    folder_notes: Mapping[str, str] | None = None,
    phrases: Sequence[str] = (),
) -> str:
    """What the embedder reads: the picture first, then what it stands for, each term once.

    A clip is also described by what its folder is for. The client's own note on the folder ("low
    points: sad, crying, staring blankly, head in hands...") is the vocabulary they think in, so a search
    for any word of it finds every clip filed there, whichever word the model happened to use.
    """
    seen: set[str] = set()
    out: list[str] = []
    for term in (
        *primary_terms(
            caption=caption, action=action, setting=setting, setting_detail=setting_detail,
            subjects=subjects, tags=tags, time_of_day=time_of_day, body_language=body_language,
        ),
        *concept_terms(themes=themes, mood=mood, emotions=emotions, categories=categories, phrases=phrases),
    ):
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(term)
    text = join_terms(out)
    if folder_notes:
        first = next((c for c in categories if c), None)
        note = (folder_notes or {}).get(first or "")
        if note:
            text += ". Filed under " + category_terms([first])[0] + ": " + note.strip()
    return text
