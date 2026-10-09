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

from collections.abc import Iterable, Sequence

# Time-of-day values that carry no information as a word.
_UNINFORMATIVE_TIME = {"", "unknown", "indoor_artificial"}


def time_words(time_of_day: str | None) -> str | None:
    if not time_of_day or time_of_day in _UNINFORMATIVE_TIME:
        return None
    return time_of_day.replace("_", " ")


def _terms(values: Iterable[str | None]) -> list[str]:
    return [str(v).strip() for v in values if v and str(v).strip()]


def category_leaves(categories: Iterable[str | None]) -> list[str]:
    """The folder names (last path segment) of the folders a shot is filed in."""
    return [c.split("/")[-1] for c in categories if c]


def primary_terms(
    *,
    caption: str | None,
    action: str | None,
    setting: str | None,
    setting_detail: str | None,
    subjects: Sequence[str],
    tags: Sequence[str],
    time_of_day: str | None,
) -> list[str]:
    return _terms([
        caption,
        None if action in (None, "none") else action,
        setting,
        setting_detail,
        *subjects,
        *tags,
        time_words(time_of_day),
    ])


def concept_terms(
    *,
    themes: Sequence[str],
    mood: Sequence[str],
    emotions: Sequence[str],
    categories: Iterable[str | None],
) -> list[str]:
    return _terms([*themes, *mood, *emotions, *category_leaves(categories)])


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
    themes: Sequence[str],
    mood: Sequence[str],
    emotions: Sequence[str],
    categories: Iterable[str | None],
) -> str:
    """What the embedder reads: the picture first, then what it stands for, each term once."""
    seen: set[str] = set()
    out: list[str] = []
    for term in (
        *primary_terms(
            caption=caption, action=action, setting=setting, setting_detail=setting_detail,
            subjects=subjects, tags=tags, time_of_day=time_of_day,
        ),
        *concept_terms(themes=themes, mood=mood, emotions=emotions, categories=categories),
    ):
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(term)
    return join_terms(out)
