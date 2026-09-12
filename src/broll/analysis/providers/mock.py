"""A deterministic provider for tests and dry runs.

Two modes:
  * replay - if a recorded response exists for the frame set, return it. This
    is how provider tests run without live API calls (see tests/fixtures).
  * synthesise - derive a stable, schema-valid result from the filename, so the
    whole pipeline can be exercised end to end with no API key at all.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from pydantic import BaseModel

from ..schema import (
    ACTIONS,
    MOODS,
    SETTINGS,
    SUBJECTS,
    USABLE_FOR,
    AnalysisResult,
    CameraMove,
    ColourProfile,
    Pace,
    PeopleCount,
    ShotContext,
    ShotType,
    TimeOfDay,
)
from .base import Pricing, TextProvider, VisionProvider

FIXTURE_ENV = "BROLL_MOCK_FIXTURES"


def _pick(sequence, seed: int, offset: int = 0):
    return sequence[(seed + offset) % len(sequence)]


class MockVisionProvider(VisionProvider):
    name = "mock"
    pricing = Pricing(input_per_m=0.0, output_per_m=0.0, tokens_per_image=0)

    def __init__(self, model: str = "mock-1", fixtures_dir: Path | None = None):
        self.model = model
        env_dir = os.environ.get(FIXTURE_ENV)
        self.fixtures_dir = fixtures_dir or (Path(env_dir) if env_dir else None)

    def _fixture(self, context: ShotContext) -> AnalysisResult | None:
        if not self.fixtures_dir:
            return None
        stem = Path(context.source_filename).stem
        candidate = self.fixtures_dir / f"{stem}.{context.shot_index}.json"
        if not candidate.exists():
            candidate = self.fixtures_dir / f"{stem}.json"
        if candidate.exists():
            return AnalysisResult.model_validate_json(candidate.read_text())
        return None

    async def analyse(self, frames, context, retry_error=None) -> AnalysisResult:
        recorded = self._fixture(context)
        if recorded is not None:
            return recorded

        seed = int(hashlib.sha256(context.source_filename.encode()).hexdigest()[:8], 16)
        seed += context.shot_index
        setting = _pick(SETTINGS, seed)
        action = _pick(ACTIONS, seed, 7)
        subjects = [_pick(SUBJECTS, seed, i) for i in (0, 3, 11)]
        mood = [_pick(MOODS, seed, i) for i in (0, 5)]
        shot_type = list(ShotType)[seed % len(ShotType)]
        return AnalysisResult(
            caption=(
                f"A {shot_type.value.replace('_', ' ')} of {subjects[0]} "
                f"{action} at a {setting}."
            ),
            subjects=subjects,
            action=action,
            setting=setting,
            setting_detail=f"mock detail for {Path(context.source_filename).stem}",
            shot_type=shot_type,
            camera_movement=list(CameraMove)[seed % len(CameraMove)],
            time_of_day=list(TimeOfDay)[seed % len(TimeOfDay)],
            mood=mood,
            colour_profile=list(ColourProfile)[seed % len(ColourProfile)],
            people_count=list(PeopleCount)[seed % len(PeopleCount)],
            has_recognisable_faces=bool(seed % 2),
            has_text_on_screen=bool(seed % 3 == 0),
            pace=list(Pace)[seed % len(Pace)],
            tags=sorted({*subjects, *mood, setting, action, "mock"}),
            usable_for=[_pick(USABLE_FOR, seed), _pick(USABLE_FOR, seed, 4)],
            quality_flags=[],
            confidence=0.80 + (seed % 20) / 100,
        )

    def estimate_cost(self, frames) -> float:
        return 0.0


class MockTextProvider(TextProvider):
    name = "mock"

    def __init__(self, model: str = "mock-1", responses: dict[str, dict] | None = None):
        self.model = model
        self.responses = responses or {}

    async def complete(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        key = hashlib.sha256(prompt.encode()).hexdigest()[:12]
        if key in self.responses:
            return schema.model_validate(self.responses[key])
        # Build the emptiest valid instance the schema allows.
        return schema.model_validate(json.loads("{}"))

    def estimate_cost(self, prompt: str) -> float:
        return 0.0
