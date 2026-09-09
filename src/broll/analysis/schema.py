"""The analysis contract: controlled vocabularies + the AnalysisResult model.

This is the heart of the product. Search quality is entirely determined by how
good and how consistent this output is, so everything here is deliberately
narrow: enums where a closed set is possible, controlled vocabularies where a
closed set is desirable, and exactly one open field (``tags``) where synonyms
and specifics belong.

Out-of-vocabulary terms are *kept* on the row (they are often useful) but are
also recorded as vocabulary candidates so the operator can promote them. See
``find_oov``.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Iterable

from pydantic import BaseModel, Field, field_validator

# --------------------------------------------------------------------------
# Enums (closed sets - a model returning something else is a validation error)
# --------------------------------------------------------------------------


class ShotType(str, Enum):
    extreme_wide = "extreme_wide"
    wide = "wide"
    medium_wide = "medium_wide"
    medium = "medium"
    medium_close_up = "medium_close_up"
    close_up = "close_up"
    extreme_close_up = "extreme_close_up"
    macro = "macro"
    two_shot = "two_shot"
    over_the_shoulder = "over_the_shoulder"
    pov = "pov"
    aerial = "aerial"
    top_down = "top_down"
    insert = "insert"
    unknown = "unknown"


class CameraMove(str, Enum):
    static = "static"
    handheld = "handheld"
    pan_left = "pan_left"
    pan_right = "pan_right"
    tilt_up = "tilt_up"
    tilt_down = "tilt_down"
    push_in = "push_in"
    pull_out = "pull_out"
    dolly = "dolly"
    truck = "truck"
    crane = "crane"
    orbit = "orbit"
    zoom_in = "zoom_in"
    zoom_out = "zoom_out"
    tracking = "tracking"
    gimbal_walk = "gimbal_walk"
    drone_fly_over = "drone_fly_over"
    whip_pan = "whip_pan"
    unknown = "unknown"


class TimeOfDay(str, Enum):
    dawn = "dawn"
    morning = "morning"
    midday = "midday"
    afternoon = "afternoon"
    golden_hour = "golden_hour"
    dusk = "dusk"
    blue_hour = "blue_hour"
    night = "night"
    indoor_artificial = "indoor_artificial"
    unknown = "unknown"


class ColourProfile(str, Enum):
    warm = "warm"
    cool = "cool"
    neutral = "neutral"
    vibrant = "vibrant"
    desaturated = "desaturated"
    high_contrast = "high_contrast"
    low_contrast = "low_contrast"
    monochrome = "monochrome"
    pastel = "pastel"
    dark_moody = "dark_moody"
    bright_airy = "bright_airy"


class PeopleCount(str, Enum):
    none = "none"
    one = "one"
    two = "two"
    small_group = "small_group"
    crowd = "crowd"


class Pace(str, Enum):
    still = "still"
    slow = "slow"
    moderate = "moderate"
    fast = "fast"


# Aliases the models reach for often enough to be worth absorbing rather than
# bouncing back as a validation failure. Keys are normalised (lowercase, spaces
# and hyphens collapsed to underscores) before lookup.
ENUM_ALIASES: dict[str, dict[str, str]] = {
    "shot_type": {
        "wide_shot": "wide", "ws": "wide", "long_shot": "wide",
        "extreme_wide_shot": "extreme_wide", "ews": "extreme_wide",
        "establishing": "extreme_wide", "establishing_shot": "extreme_wide",
        "medium_shot": "medium", "ms": "medium", "mid_shot": "medium",
        "medium_long_shot": "medium_wide", "cowboy_shot": "medium_wide",
        "close": "close_up", "closeup": "close_up", "cu": "close_up",
        "extreme_close": "extreme_close_up", "ecu": "extreme_close_up",
        "extreme_closeup": "extreme_close_up",
        "medium_closeup": "medium_close_up", "mcu": "medium_close_up",
        "bird_eye": "top_down", "birds_eye": "top_down", "overhead": "top_down",
        "drone": "aerial", "point_of_view": "pov", "ots": "over_the_shoulder",
        "detail": "insert", "cutaway": "insert",
    },
    "camera_movement": {
        "none": "static", "still": "static", "locked_off": "static",
        "locked": "static", "tripod": "static", "fixed": "static",
        "pan": "pan_right", "tilt": "tilt_up", "zoom": "zoom_in",
        "dolly_in": "push_in", "dolly_out": "pull_out",
        "push": "push_in", "pull": "pull_out", "pullback": "pull_out",
        "slow_push_in": "push_in", "slow_pull_out": "pull_out",
        "track": "tracking", "tracking_shot": "tracking", "follow": "tracking",
        "shaky": "handheld", "hand_held": "handheld", "gimbal": "gimbal_walk",
        "orbiting": "orbit", "arc": "orbit", "flyover": "drone_fly_over",
        "aerial": "drone_fly_over", "steadicam": "gimbal_walk",
    },
    "time_of_day": {
        "sunrise": "dawn", "sunset": "golden_hour", "magic_hour": "golden_hour",
        "twilight": "blue_hour", "evening": "dusk", "noon": "midday",
        "day": "midday", "daytime": "midday", "nighttime": "night",
        "indoor": "indoor_artificial", "indoors": "indoor_artificial",
        "interior": "indoor_artificial", "artificial": "indoor_artificial",
        "n_a": "unknown", "none": "unknown",
    },
    "colour_profile": {
        "color_warm": "warm", "warm_tones": "warm", "cool_tones": "cool",
        "cold": "cool", "muted": "desaturated", "washed_out": "desaturated",
        "saturated": "vibrant", "colourful": "vibrant", "colorful": "vibrant",
        "contrasty": "high_contrast", "flat": "low_contrast",
        "black_and_white": "monochrome", "bw": "monochrome",
        "greyscale": "monochrome", "grayscale": "monochrome",
        "moody": "dark_moody", "dark": "dark_moody", "low_key": "dark_moody",
        "bright": "bright_airy", "high_key": "bright_airy", "airy": "bright_airy",
    },
    "people_count": {
        "0": "none", "1": "one", "2": "two", "zero": "none", "single": "one",
        "one_person": "one", "two_people": "two", "couple": "two", "pair": "two",
        "few": "small_group", "several": "small_group", "group": "small_group",
        "many": "crowd", "large_group": "crowd", "many_people": "crowd",
    },
    "pace": {
        "static": "still", "none": "still", "very_slow": "slow",
        "medium": "moderate", "normal": "moderate", "quick": "fast",
        "very_fast": "fast", "frantic": "fast",
    },
}


# --------------------------------------------------------------------------
# Controlled vocabularies (open at the row level, closed at the folder level)
# --------------------------------------------------------------------------

SUBJECTS: tuple[str, ...] = (
    "person", "man", "woman", "child", "baby", "teenager", "elderly person",
    "couple", "family", "group of people", "crowd", "athlete", "dancer",
    "musician", "chef", "doctor", "nurse", "teacher", "student", "office worker",
    "construction worker", "farmer", "artist", "photographer", "hands", "face",
    "eyes", "feet", "silhouette", "pet", "dog", "cat", "horse", "bird", "fish",
    "insect", "wildlife", "livestock", "tree", "forest", "flower", "plant",
    "grass", "leaves", "mountain", "hill", "cliff", "rock", "beach", "sand",
    "ocean", "wave", "lake", "river", "waterfall", "sky", "clouds", "sun",
    "moon", "stars", "rain", "snow", "fog", "fire", "smoke", "city", "skyline",
    "street", "road", "highway", "bridge", "building", "skyscraper", "house",
    "apartment", "office", "desk", "computer", "laptop", "phone", "screen",
    "keyboard", "camera", "book", "notebook", "pen", "coffee", "tea", "food",
    "meal", "fruit", "vegetable", "bread", "restaurant", "kitchen", "cafe",
    "shop", "market", "car", "bus", "train", "bicycle", "motorcycle", "boat",
    "airplane", "drone", "machinery", "tools", "factory", "warehouse",
    "gym equipment", "yoga mat", "money", "documents", "packaging", "product",
    "sign", "window", "door", "stairs", "furniture", "chair", "table", "bed",
    "park", "studio", "valley", "path", "field",
    "artwork", "texture", "pattern", "light", "shadow", "abstract shapes",
)

ACTIONS: tuple[str, ...] = (
    "walking", "running", "jogging", "hiking", "climbing", "cycling", "driving",
    "riding", "swimming", "surfing", "skiing", "skateboarding", "dancing",
    "jumping", "stretching", "exercising", "lifting weights", "doing yoga",
    "meditating", "breathing", "sitting", "standing", "lying down", "sleeping",
    "waking up", "resting", "relaxing", "talking", "laughing", "smiling",
    "crying", "thinking", "listening", "presenting", "teaching", "learning",
    "studying", "reading", "writing", "typing", "coding", "drawing", "painting",
    "filming", "photographing", "cooking", "baking", "eating", "drinking",
    "pouring", "chopping", "serving", "shopping", "paying", "working",
    "building", "repairing", "assembling", "cleaning", "gardening", "farming",
    "harvesting", "welding", "packing", "shipping", "waiting", "queuing",
    "travelling", "boarding", "arriving", "departing", "celebrating",
    "cheering", "hugging", "holding hands", "playing", "playing music",
    "performing", "browsing phone", "scrolling", "video calling", "meeting",
    "collaborating", "brainstorming", "shaking hands", "pointing", "gesturing",
    "watching", "observing", "searching", "opening", "closing", "carrying",
    "throwing", "catching", "falling", "flying", "floating", "flowing",
    "growing", "burning", "glowing", "raining", "snowing", "rising", "setting",
    "none",
)

SETTINGS: tuple[str, ...] = (
    "beach", "coastline", "cliff", "desert", "forest", "jungle", "woodland",
    "meadow", "field", "farm", "orchard", "vineyard", "mountain", "valley",
    "canyon", "lake", "river", "waterfall", "wetland", "glacier", "snowfield",
    "park", "garden", "backyard", "rooftop", "city street", "alley",
    "downtown", "suburb", "village", "town square", "highway", "parking lot",
    "bridge", "tunnel", "construction site", "industrial area", "factory floor",
    "warehouse", "port", "airport", "train station", "bus stop", "subway",
    "office", "open-plan office", "meeting room", "home office",
    "coworking space", "classroom", "lecture hall", "library", "laboratory",
    "hospital", "clinic", "studio", "workshop", "gym", "yoga studio",
    "sports field", "stadium", "swimming pool", "kitchen", "dining room",
    "living room", "bedroom", "bathroom", "hallway", "staircase", "basement",
    "cafe", "restaurant", "bar", "hotel", "lobby", "shop", "supermarket",
    "market", "mall", "gallery", "museum", "theatre", "concert venue",
    "place of worship", "playground", "campsite", "boat", "car interior",
    "plane interior", "train interior", "elevator", "studio backdrop",
    "white background", "black background", "outdoors", "indoors",
    "underwater", "aerial", "abstract",
)

MOODS: tuple[str, ...] = (
    "calm", "peaceful", "serene", "meditative", "hopeful", "uplifting",
    "joyful", "happy", "playful", "energetic", "lively", "vibrant", "exciting",
    "dynamic", "urgent", "tense", "dramatic", "intense", "powerful", "bold",
    "confident", "determined", "focused", "professional", "corporate",
    "clinical", "minimal", "clean", "modern", "futuristic", "technical",
    "industrial", "gritty", "raw", "rugged", "natural", "organic", "earthy",
    "rustic", "cosy", "warm", "intimate", "romantic", "nostalgic", "dreamy",
    "ethereal", "mysterious", "moody", "melancholy", "sad", "lonely",
    "isolated", "sombre", "dark", "ominous", "cold", "harsh", "chaotic",
    "busy", "quiet", "still", "contemplative", "reflective", "inspiring",
    "aspirational", "luxurious", "elegant", "refined", "wholesome",
    "community", "friendly", "welcoming", "celebratory", "festive",
    "adventurous", "free", "expansive", "epic", "cinematic", "whimsical",
    "quirky", "humorous",
)

USABLE_FOR: tuple[str, ...] = (
    "establishing shot", "transition", "talking-head cutaway",
    "b-roll under narration", "product hero", "mood setter", "opener",
    "closer", "lower-third background", "text background", "loop",
    "time-lapse insert", "detail insert", "reaction shot", "montage element",
    "tutorial step", "testimonial backdrop", "social vertical crop",
    "ad hook", "scene setter", "atmosphere filler",
)

QUALITY_FLAGS: tuple[str, ...] = (
    "out_of_focus", "shaky", "overexposed", "underexposed", "watermarked",
    "low_resolution", "contains_logo", "motion_blur", "noisy",
    "compression_artifacts", "obstructed", "colour_cast", "empty_frame",
)

# Flags that mean the footage is technically defective, as opposed to merely
# worth knowing about (a logo or on-screen text matters for licensing, not for
# whether the shot is usable).
DEFECT_FLAGS: frozenset[str] = frozenset({
    "out_of_focus", "shaky", "overexposed", "underexposed", "motion_blur",
    "noisy", "compression_artifacts", "empty_frame", "low_resolution",
})

VOCABULARIES: dict[str, tuple[str, ...]] = {
    "subjects": SUBJECTS,
    "action": ACTIONS,
    "setting": SETTINGS,
    "mood": MOODS,
    "usable_for": USABLE_FOR,
    "quality_flags": QUALITY_FLAGS,
}


# --------------------------------------------------------------------------
# Normalisation helpers
# --------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def normalise_term(value: str) -> str:
    """Lowercase, collapse whitespace, strip trailing punctuation."""
    return _WS.sub(" ", value.strip().lower()).strip(" .,;:")


def normalise_enum_key(value: str) -> str:
    """Key used for alias lookup: lowercase with spaces/hyphens as underscores."""
    return _WS.sub("_", value.strip().lower().replace("-", "_")).strip("_")


def coerce_enum(field: str, value: str, enum_cls: type[Enum]) -> str:
    """Map a loose model answer onto an enum member, or return it unchanged.

    Returning it unchanged means Pydantic raises, which is what we want: an
    unmappable enum value is a real validation failure worth a retry.
    """
    key = normalise_enum_key(value)
    if key in {m.value for m in enum_cls}:
        return key
    return ENUM_ALIASES.get(field, {}).get(key, value)


# --------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------


class AnalysisResult(BaseModel):
    """What every vision provider must return for a single shot."""

    caption: str = Field(description="One vivid sentence: what an editor would say out loud.")
    subjects: list[str] = Field(default_factory=list, description="What is in frame.")
    action: str | None = Field(default=None, description="The primary verb.")
    setting: str = Field(description="Where it is.")
    setting_detail: str | None = Field(default=None, description="Free text, e.g. 'rocky coastline'.")
    shot_type: ShotType
    camera_movement: CameraMove
    time_of_day: TimeOfDay
    mood: list[str] = Field(default_factory=list, description="Up to 3 mood terms.")
    colour_profile: ColourProfile
    people_count: PeopleCount
    has_recognisable_faces: bool = False
    has_text_on_screen: bool = False
    pace: Pace
    tags: list[str] = Field(default_factory=list, description="Open vocabulary, 5-15 terms.")
    usable_for: list[str] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # -- coercion -----------------------------------------------------------

    @field_validator("shot_type", mode="before")
    @classmethod
    def _shot_type(cls, v):
        return coerce_enum("shot_type", v, ShotType) if isinstance(v, str) else v

    @field_validator("camera_movement", mode="before")
    @classmethod
    def _camera_movement(cls, v):
        return coerce_enum("camera_movement", v, CameraMove) if isinstance(v, str) else v

    @field_validator("time_of_day", mode="before")
    @classmethod
    def _time_of_day(cls, v):
        return coerce_enum("time_of_day", v, TimeOfDay) if isinstance(v, str) else v

    @field_validator("colour_profile", mode="before")
    @classmethod
    def _colour_profile(cls, v):
        return coerce_enum("colour_profile", v, ColourProfile) if isinstance(v, str) else v

    @field_validator("people_count", mode="before")
    @classmethod
    def _people_count(cls, v):
        if isinstance(v, int):
            v = str(v)
        return coerce_enum("people_count", v, PeopleCount) if isinstance(v, str) else v

    @field_validator("pace", mode="before")
    @classmethod
    def _pace(cls, v):
        return coerce_enum("pace", v, Pace) if isinstance(v, str) else v

    @field_validator("subjects", "mood", "tags", "usable_for", "quality_flags", mode="before")
    @classmethod
    def _clean_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            v = [v]
        out: list[str] = []
        for item in v:
            if not isinstance(item, str):
                continue
            term = normalise_term(item)
            if term and term not in out:
                out.append(term)
        return out

    @field_validator("quality_flags", mode="after")
    @classmethod
    def _flag_shape(cls, v: list[str]) -> list[str]:
        return [normalise_enum_key(f) for f in v]

    @field_validator("action", "setting", mode="before")
    @classmethod
    def _clean_term(cls, v):
        if v is None:
            return None
        return normalise_term(v) if isinstance(v, str) else v

    @field_validator("mood", mode="after")
    @classmethod
    def _cap_mood(cls, v: list[str]) -> list[str]:
        return v[:3]

    @field_validator("tags", mode="after")
    @classmethod
    def _cap_tags(cls, v: list[str]) -> list[str]:
        return v[:15]

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.5
        return min(1.0, max(0.0, f))

    # -- derived ------------------------------------------------------------

    def embedding_text(self) -> str:
        """Text handed to the embedder. Caption first: it carries the most signal."""
        parts = [self.caption]
        if self.action:
            parts.append(self.action)
        parts.append(self.setting)
        if self.setting_detail:
            parts.append(self.setting_detail)
        parts.extend(self.subjects)
        parts.extend(self.mood)
        parts.extend(self.usable_for)
        parts.extend(self.tags)
        parts.extend([
            self.shot_type.value.replace("_", " "),
            self.camera_movement.value.replace("_", " "),
            self.time_of_day.value.replace("_", " "),
            self.colour_profile.value.replace("_", " "),
            f"{self.pace.value} pace",
        ])
        return ", ".join(p for p in parts if p)

    def search_text(self) -> str:
        """Denormalised FTS5 payload. See store.recompute_search_text."""
        return self.embedding_text()


class ShotContext(BaseModel):
    """What the provider is told about the shot beyond the pixels."""

    source_filename: str
    duration_s: float
    width: int
    height: int
    fps: float | None = None
    shot_index: int = 0
    shot_count: int = 1
    start_s: float = 0.0
    end_s: float | None = None

    def describe(self) -> str:
        pos = (
            f"shot {self.shot_index + 1} of {self.shot_count} "
            f"(starts {self.start_s:.1f}s into the file)"
            if self.shot_count > 1
            else "the whole file is a single continuous shot"
        )
        fps = f"{self.fps:.2f}" if self.fps else "unknown"
        return (
            f"Original filename: {self.source_filename}\n"
            f"Shot duration: {self.duration_s:.2f}s\n"
            f"Resolution: {self.width}x{self.height} at {fps} fps\n"
            f"Position: {pos}"
        )


# --------------------------------------------------------------------------
# Vocabulary checking
# --------------------------------------------------------------------------

_VOCAB_FIELDS = {
    "subjects": "subjects",
    "action": "action",
    "setting": "setting",
    "mood": "mood",
    "usable_for": "usable_for",
    "quality_flags": "quality_flags",
}


def find_oov(
    result: AnalysisResult,
    overrides: dict[str, Iterable[str]] | None = None,
) -> list[tuple[str, str]]:
    """Return (field, term) pairs the model used that are not in vocabulary.

    These are kept on the row but recorded in ``vocabulary_candidates`` so the
    operator can promote the ones that keep coming up.
    """
    vocab = {k: set(v) for k, v in VOCABULARIES.items()}
    for field, extra in (overrides or {}).items():
        vocab.setdefault(field, set()).update(normalise_term(t) for t in extra)

    oov: list[tuple[str, str]] = []
    for field in _VOCAB_FIELDS:
        value = getattr(result, field)
        terms = [value] if isinstance(value, str) else list(value or [])
        for term in terms:
            if term and term not in vocab[field]:
                oov.append((field, term))
    return oov


def in_vocabulary(result: AnalysisResult, overrides=None) -> bool:
    return not find_oov(result, overrides)
