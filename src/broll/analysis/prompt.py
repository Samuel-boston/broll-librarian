"""The single shared analysis prompt.

One template, all providers. Bump PROMPT_VERSION whenever the wording changes
in a way that would change output; it is stored on every shot as
``analysis_version`` so stale rows can be found and re-analysed.

What varies per workspace - the client brief, their emotion vocabulary, their
folder list - arrives on the ShotContext, so providers never need to know what
a workspace is.
"""

from __future__ import annotations

from collections.abc import Sequence

from .schema import (
    ACTIONS,
    EMOTIONS,
    MOODS,
    QUALITY_FLAGS,
    SETTINGS,
    SUBJECTS,
    USABLE_FOR,
    CameraMove,
    ColourProfile,
    Pace,
    PeopleCount,
    ShotContext,
    ShotType,
    TimeOfDay,
)

PROMPT_VERSION = "2.0.0"

SYSTEM_PROMPT = """\
You are a video editor's assistant cataloguing raw B-roll footage. You are shown still \
frames sampled in order from one stretch of footage, with the time of each frame. \
Describe that stretch as a whole, not the individual frames.

Your output does two jobs: it is written into a search index, and it decides which folder \
the clip is filed in. A wrong tag is worse than a missing one: it makes the clip turn up in \
searches it has nothing to do with, and then editors stop trusting the library. So be \
literal and specific, and prefer the controlled vocabulary.

Work in this order. The output fields follow it.

1. observations - first, list what you can literally see: people and what they wear and \
do, objects, the place, the light, any movement. Short and factual, 4-10 items, no \
interpretation. Everything below must be grounded in this list.
2. caption - one sentence, the way an editor would describe the clip out loud to a \
colleague. No preamble, no "this video shows".
3. subjects, action, setting, mood, emotions, usable_for, quality_flags - use the \
controlled vocabularies exactly as written. If nothing fits, use the closest term; only \
invent one when there is genuinely no near miss.
4. tags - search keywords for things that are visibly present or plainly happening: 6-12, \
lowercase, singular. Each tag must be one of your observations or a common synonym of one \
("sofa" for "couch", "ocean" for "sea"). Never put in tags: feelings, ideas, themes, what \
the clip "represents", camera or shot terms, or anything you cannot see. A man sitting \
still on a beach is tagged man, sitting, beach, sand, ocean, sunrise - not "mindfulness", \
"wellness" or "breathwork". An activity is visible only when the pose, equipment or \
setting makes it unmistakable: eyes closed in a meditation posture is meditation; a \
person who is merely sitting is not.
5. themes - only from the client's THEMES list, when one is given, copied exactly. Include \
a theme only if the footage clearly shows it or is a direct visual stand-in for it. Most \
clips match none or one; an empty list is normal and correct. Never add a theme just \
because the client is about it.
6. emotions matter more to an editor than anything except the caption, because editors \
cut to feeling. Give 2-5: what the person on screen is feeling, and what the shot makes a \
viewer feel. Be honest about it - someone yawning and falling back asleep is tired or \
drained, not cosy. mood is different: mood is how the shot looks as an image (cinematic, \
moody, clean, warm).
7. setting_detail is free text and is where specifics go ("rocky Cornish coastline", \
"open-plan office with exposed brick").
8. Infer camera movement from the differences between frames. If the framing is identical \
across frames, it is static.
9. quality_flags describe technical problems only. An intentionally shallow depth of field \
is not out_of_focus. Leave the list empty when the clip is clean.
10. confidence is your honest confidence that this analysis is right, 0 to 1. Use the whole \
range: 0.9 or more only when everything is clear; 0.7 to 0.9 when it is mostly clear; \
below 0.7 when the frames are dark or ambiguous, or you are guessing at the action.

If the first or last frames show the camera being set up or handled, someone walking in or \
out of shot, a slate or countdown, or a focus or exposure adjustment, leave them out of \
your description: describe the shot itself, not the preparation for it.

If you are given client context, write for that client, but never force anything onto \
footage that does not show it.
"""

CATEGORY_HEADER = (
    "CATEGORIES - this client files footage into their own folders. Set `category` "
    "to the single best-fit folder, copying the path exactly as written before the "
    '" — ". Prefer the most specific folder that fits and follow each folder\'s '
    "note. Use secondary_categories for up to 2 other folders the clip also clearly "
    "belongs in, and leave it empty otherwise. Set category_confidence to how sure you "
    "are of `category` alone: below 0.6 when two folders fit about equally well or none "
    "fits well.\n"
    "Only if NONE of these folders genuinely fits, propose one new folder: set "
    "new_category to 'Existing Folder/New Name', placing it under the most fitting "
    "existing folder, with a short Title Case name in the same style as its "
    "neighbours, and set new_category_note to one line on what belongs in it. Still "
    "set `category` to the closest existing folder. Use this sparingly - a close "
    "existing folder is always better than a new one."
)


def _vocab_block(name: str, terms: Sequence[str]) -> str:
    return f"{name}: {', '.join(terms)}"


def _enum_block(name: str, enum_cls) -> str:
    return f"{name}: {', '.join(m.value for m in enum_cls)}"


def vocabulary_block(emotions: Sequence[str] = EMOTIONS) -> str:
    return "\n\n".join(
        [
            "CONTROLLED VOCABULARIES - use these exact terms.",
            _vocab_block("subjects (choose all that apply, most important first)", SUBJECTS),
            _vocab_block("action (choose exactly one, or 'none')", ACTIONS),
            _vocab_block("setting (choose exactly one)", SETTINGS),
            _vocab_block("mood (choose up to 3)", MOODS),
            _vocab_block("emotions (choose 2-5, most important first)", emotions),
            _vocab_block("usable_for (choose 1-4 editorial uses)", USABLE_FOR),
            _vocab_block("quality_flags (only if genuinely present)", QUALITY_FLAGS),
            "\n".join(
                [
                    "FIXED ENUMS - one value each, exactly as spelled:",
                    _enum_block("shot_type", ShotType),
                    _enum_block("camera_movement", CameraMove),
                    _enum_block("time_of_day", TimeOfDay),
                    _enum_block("colour_profile", ColourProfile),
                    _enum_block("people_count", PeopleCount),
                    _enum_block("pace", Pace),
                ]
            ),
        ]
    )


VOCABULARY_BLOCK = vocabulary_block()


def render_client_context(profile) -> str:
    """The client brief, as a block of prompt text. Empty for a generic library."""
    if profile is None or not (
        profile.name or profile.brief or profile.featured_person or profile.themes
    ):
        return ""
    lines = ["CLIENT CONTEXT - every clip in this library belongs to one client."]
    if profile.name:
        lines.append(f"Client: {profile.name}")
    if profile.brief:
        lines.append(profile.brief.strip())
    if profile.themes:
        lines.append(
            "THEMES - the only values allowed in the `themes` field, copied exactly. Use one "
            "only when the footage clearly shows it or is a direct visual stand-in for it; "
            "most clips match none: " + "; ".join(profile.themes) + "."
        )
    if profile.featured_person:
        first = profile.featured_person.split()[0]
        who = profile.featured_person_description or "one person"
        lines.append(
            f"Featured person: much of this footage shows {profile.featured_person}. "
            f"When {who} is clearly the featured subject of the shot, call them "
            f'"{first}" in the caption (for example "{first} meditates on the beach at '
            f'sunrise") and set featured_person_in_shot to true. Never name anyone else '
            f"- describe other people generically. In group shots only name {first} if "
            f"they are unmistakably the focus. If you cannot tell, do not use the name "
            f"and set featured_person_in_shot to false."
        )
    return "\n".join(lines)


def build_user_prompt(context: ShotContext, retry_error: str | None = None) -> str:
    """The per-shot user turn. Frames are attached separately by the provider."""
    parts = [vocabulary_block(context.emotion_vocab or EMOTIONS)]
    if context.client_context:
        parts += ["", context.client_context]
    if context.category_options:
        parts += ["", CATEGORY_HEADER, *(f"- {line}" for line in context.category_options)]
    parts += [
        "",
        "SHOT METADATA",
        context.describe(),
        "",
        "The attached frames are sampled evenly across this shot, in order. "
        "Analyse the shot as a whole and return the structured result.",
    ]
    if retry_error:
        parts += [
            "",
            "YOUR PREVIOUS ANSWER FAILED VALIDATION. Fix exactly this and return "
            "the whole result again:",
            retry_error,
        ]
    return "\n".join(parts)
