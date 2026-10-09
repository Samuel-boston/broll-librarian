"""Is the tagging any good? Measured on the library itself, no model calls.

The first thing to go wrong with AI tagging is not a wrong tag on one clip: it is the same tag on
too many clips. "Calm" or "nervous system" on every other shot turns a search for it into a dump of
half the library. This reads the index and reports the signs of that, plus how sure the model said it
was, so the settings can be tuned on evidence.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .analysis.analyzer import _mentioned
from .db.store import Store

#: A tag on more than this share of the library does not tell clips apart.
COMMON_SHARE = 0.12
#: Words for who is in the shot. A library of one man's life has a man in most of it; that is true, not noise,
#: and a search for one of these should return a lot.
PEOPLE_WORDS = frozenset({"man", "men", "woman", "women", "person", "people", "group", "child", "children",
                          "boy", "girl", "baby", "crowd"})
#: ...but only judged once the library is big enough for the share to mean something.
MIN_SHOTS = 25

BUCKETS = ((0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01))


def _one_line(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "")


def _histogram(values: list[float]) -> dict[str, int]:
    out = {f"{lo:.1f}-{min(hi, 1.0):.1f}": 0 for lo, hi in BUCKETS}
    for v in values:
        for lo, hi in BUCKETS:
            if lo <= v < hi:
                out[f"{lo:.1f}-{min(hi, 1.0):.1f}"] += 1
                break
    return out


def tag_audit(store: Store, top: int = 25, examples: int = 15) -> dict[str, Any]:
    shots = store.list_shots(limit=1_000_000)
    total = len(shots)
    report: dict[str, Any] = {"shots": total}
    if not total:
        return report

    tag_counts: Counter = Counter()
    theme_counts: Counter = Counter()
    emotion_counts: Counter = Counter()
    for shot in shots:
        tag_counts.update(set(shot.tags))
        theme_counts.update(set(shot.themes))
        emotion_counts.update(set(shot.emotions))

    judged = total >= MIN_SHOTS
    common = [
        {"tag": t, "shots": n, "share": round(n / total, 3)}
        for t, n in tag_counts.most_common()
        if judged and n / total > COMMON_SHARE and t not in PEOPLE_WORDS
    ]
    unsupported: list[dict[str, Any]] = []
    for shot in shots:
        seen = " ".join([
            shot.caption or "", *shot.observations, *shot.subjects, shot.action or "",
            shot.setting or "", shot.setting_detail or "",
        ])
        loose = [t for t in shot.tags if not _mentioned(t, seen)]
        # A tag the shot's own description never mentions is not necessarily wrong (a synonym is
        # fine), but a clip made mostly of them is one to look at.
        if shot.tags and len(loose) / len(shot.tags) > 0.5:
            unsupported.append({"shot": shot.id, "caption": shot.caption, "tags_not_in_description": loose})

    reasons: Counter = Counter()
    for shot in shots:
        reasons.update(shot.review_reasons)

    confidences = [s.confidence for s in shots if s.confidence is not None]
    cat_confidences = [s.category_confidence for s in shots if s.category_confidence is not None]
    report.update(
        {
            "tags_per_shot": round(sum(len(s.tags) for s in shots) / total, 1),
            "distinct_tags": len(tag_counts),
            "tags_used_once": sum(1 for n in tag_counts.values() if n == 1),
            "top_tags": [
                {"tag": t, "shots": n, "share": round(n / total, 3)} for t, n in tag_counts.most_common(top)
            ],
            "too_common_tags": common,
            "common_judged": judged,
            "themes": dict(theme_counts.most_common()),
            "themes_per_shot": round(sum(len(s.themes) for s in shots) / total, 2),
            "shots_with_no_theme": sum(1 for s in shots if not s.themes),
            "top_emotions": dict(emotion_counts.most_common(top)),
            "confidence": _histogram(confidences),
            "category_confidence": _histogram(cat_confidences),
            "review_reasons": dict(reasons.most_common()),
            "needs_review": sum(1 for s in shots if s.status == "needs_review"),
            "tags_barely_in_description": unsupported[:examples],
            "count_tags_barely_in_description": len(unsupported),
        }
    )
    return report


def format_report(report: dict[str, Any]) -> str:
    if not report.get("shots"):
        return "No shots indexed yet."
    lines = [
        f"{report['shots']} shots · {report['distinct_tags']} distinct tags · "
        f"{report['tags_per_shot']} tags a shot · {report['tags_used_once']} tags used only once",
        "",
    ]
    if report["too_common_tags"]:
        lines.append("TAGS ON TOO MANY CLIPS (they will pull the same clips into every search):")
        lines += [f"  {t['tag']:<28} {t['shots']:>5} shots  {t['share'] * 100:>4.0f}%" for t in report["too_common_tags"]]
    elif report["common_judged"]:
        lines.append("No tag is on more than %d%% of the library." % int(COMMON_SHARE * 100))
    else:
        lines.append(f"(Too few shots to judge over-used tags: wait for {MIN_SHOTS}.)")
    lines += ["", "MOST USED TAGS:"]
    lines += [f"  {t['tag']:<28} {t['shots']:>5}  {t['share'] * 100:>4.0f}%" for t in report["top_tags"]]
    lines += [
        "",
        f"THEMES ({report['themes_per_shot']} a shot; {report['shots_with_no_theme']} shots have none):",
    ]
    lines += [f"  {t:<28} {n:>5}" for t, n in report["themes"].items()] or ["  none used"]
    lines += ["", "HOW SURE THE MODEL WAS (overall):"]
    lines += [f"  {k}: {v}" for k, v in report["confidence"].items()]
    if any(report["category_confidence"].values()):
        lines += ["HOW SURE IT WAS OF THE FOLDER:"] + [f"  {k}: {v}" for k, v in report["category_confidence"].items()]
    lines += ["", f"{report['needs_review']} shots are waiting for review. Why:"]
    lines += [f"  {k}: {v}" for k, v in report["review_reasons"].items()] or ["  (none)"]
    if report["count_tags_barely_in_description"]:
        lines += ["", f"{report['count_tags_barely_in_description']} shots whose tags are mostly absent from "
                  "their own description (look at a few):"]
        for entry in report["tags_barely_in_description"][:8]:
            lines.append(f"  {entry['shot']}: {_one_line(entry['caption'])[:70]}")
            lines.append(f"      tags: {', '.join(entry['tags_not_in_description'])}")
    return "\n".join(lines)
