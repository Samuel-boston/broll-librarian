"""The single shared analysis prompt.

One template, all providers. Bump PROMPT_VERSION whenever the wording changes
in a way that would change output; it is stored on every shot as
``analysis_version`` so stale rows can be found and re-analysed.
"""

from __future__ import annotations

from .schema import (
    ACTIONS,
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

PROMPT_VERSION = "1.0.0"

SYSTEM_PROMPT = """\
You are a video editor's assistant cataloguing B-roll footage. You are shown a \
handful of still frames sampled from a single continuous shot, in chronological \
order. Describe the shot as a whole, not the individual frames.

Your output is used two ways: it is written into a searchable index, and it \
decides which folders the clip is filed under. So be specific, be consistent, \
and prefer the controlled vocabulary terms you are given over your own wording.

Rules that matter:
- The caption is one vivid sentence, the way an editor would describe the clip \
out loud to a colleague. No preamble, no "this video shows".
- Use the controlled vocabularies exactly as written for subjects, action, \
setting, mood, usable_for and quality_flags. If nothing fits, use the closest \
term rather than inventing one; only invent when there is genuinely no near miss.
- setting_detail is free text and is where specifics go ("rocky Cornish \
coastline", "open-plan office with exposed brick").
- tags is the open field, and it is what keyword search hits. Include 5-15 \
lowercase singular tags, and deliberately include synonyms and adjacent \
concepts an editor might type. A meditation clip should carry meditation, \
mindfulness, wellness, calm, yoga, sunrise, beach, ocean and solitude - not \
just "meditation".
- Infer camera movement from the differences between frames. If the framing is \
identical across frames, it is static.
- quality_flags describe technical problems only. An intentionally shallow \
depth of field is not out_of_focus. Leave the list empty when the clip is clean.
- confidence is your own honest confidence in this analysis, 0 to 1. Lower it \
when the frames are ambiguous, dark, or you are guessing at the action.
"""


def _vocab_block(name: str, terms: tuple[str, ...]) -> str:
    return f"{name}: {', '.join(terms)}"


def _enum_block(name: str, enum_cls) -> str:
    return f"{name}: {', '.join(m.value for m in enum_cls)}"


VOCABULARY_BLOCK = "\n\n".join(
    [
        "CONTROLLED VOCABULARIES - use these exact terms.",
        _vocab_block("subjects (choose all that apply, most important first)", SUBJECTS),
        _vocab_block("action (choose exactly one, or 'none')", ACTIONS),
        _vocab_block("setting (choose exactly one)", SETTINGS),
        _vocab_block("mood (choose up to 3)", MOODS),
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


def build_user_prompt(context: ShotContext, retry_error: str | None = None) -> str:
    """The per-shot user turn. Frames are attached separately by the provider."""
    parts = [
        VOCABULARY_BLOCK,
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
