"""Typo tolerance for search.

One wrong character used to mean no results: "meditaing" matched nothing.

The rule is deliberately narrow, so a correction never makes a search worse:

* only a word the library has never seen is corrected (a word it knows, in
  any stemmed form, is left alone), and never an ordinary English word: a small
  library has not seen "band" or "waking" or "wide", and turning them into "sand",
  "walking" and "side" would change what was asked for, and
* only to a word the library *has*, within one or two keystrokes, and
* never a very short word - at three letters, one edit usually lands on a
  different real word.

The caller shows every correction ("showing results for...") with a way back,
so a wrong guess is visible and undone in one click.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path


def max_edits(word: str) -> int:
    """How many keystrokes a word may be off by and still be corrected."""
    length = len(word)
    if length <= 3:
        return 0
    if length <= 7:
        return 1
    return 2


def edit_distance(a: str, b: str, limit: int) -> int:
    """Optimal string alignment distance (a swapped pair counts as one edit).

    Returns ``limit + 1`` as soon as the distance must exceed ``limit``.
    """
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    before_previous: list[int] | None = None
    previous = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        current = [i] + [0] * len(b)
        row_best = current[0]
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            if (
                before_previous is not None and j > 1
                and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]
            ):
                current[j] = min(current[j], before_previous[j - 2] + 1)
            row_best = min(row_best, current[j])
        if row_best > limit:
            return limit + 1
        before_previous, previous = previous, current
    return previous[-1]


def suggest(word: str, vocabulary: Counter) -> str | None:
    """The library word this was probably meant to be, or None.

    Nearest by edits; ties go to the word used on more clips, then
    alphabetically, so the answer is stable.
    """
    limit = max_edits(word)
    if not limit or word in vocabulary:
        return None
    best: tuple[tuple[int, int, str], str] | None = None
    for candidate, uses in vocabulary.items():
        if abs(len(candidate) - len(word)) > limit:
            continue
        distance = edit_distance(word, candidate, limit)
        if distance > limit:
            continue
        key = (distance, -uses, candidate)
        if best is None or key < best[0]:
            best = (key, candidate)
    return best[1] if best else None


# A few thousand ordinary English words, so a real word the library simply has not seen yet is never mistaken for
# a typo. Derived from the whole words of the BERT (uncased) vocabulary, Apache License 2.0, Google.
_COMMON_WORDS = Path(__file__).with_name("common_words.txt")
# Words a B-roll search is likely to use that an ordinary word list lacks.
_EXTRA_WORDS = frozenset({
    "handheld", "aerial", "drone", "drones", "macro", "pov", "timelapse", "gimbal", "tripod", "bokeh", "vlog",
    "broll", "firefighter", "firefighters", "sauna", "meditating", "meditate", "meditation", "mindfulness",
    "breathwork", "journaling", "workout", "workouts", "sandboarding", "snowboarding", "skateboarding",
    "kayaking", "kayak", "skydiving", "surfing", "paddleboard", "podcast", "podcasts", "livestream",
})


def _inflections(word: str) -> set[str]:
    forms = {word, word + "s", word + "es", word + "ed", word + "ing", word + "er", word + "ers", word + "ly",
             word + "d", word + "y"}
    if word.endswith("e"):
        forms |= {word[:-1] + "ing", word + "r", word + "rs"}
    if word.endswith("y"):
        forms |= {word[:-1] + "ies", word[:-1] + "ied", word[:-1] + "ier"}
    if len(word) > 2 and word[-1] not in "aeiouwy" and word[-2] in "aeiou" and word[-3] not in "aeiou":
        forms |= {word + word[-1] + "ing", word + word[-1] + "ed", word + word[-1] + "er"}
    return forms


@lru_cache(maxsize=1)
def common_words() -> frozenset[str]:
    """Ordinary English words, with their plurals and -ed/-ing forms. Empty if the list is missing."""
    try:
        base = _COMMON_WORDS.read_text(encoding="utf-8").split()
    except OSError:
        return frozenset(_EXTRA_WORDS)
    words: set[str] = set(_EXTRA_WORDS)
    for word in base:
        words |= _inflections(word)
    return frozenset(words)


def is_common_word(word: str) -> bool:
    return word.lower() in common_words()
