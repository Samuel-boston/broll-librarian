"""End-to-end ingest against the mock provider: no API key, no network."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from broll.analysis.analyzer import Analyzer
from broll.analysis.prompt import PROMPT_VERSION
from broll.analysis.schema import AnalysisResult, ShotContext
from broll.cli import app
from broll.db.models import Shot, Source
from broll.db.store import Store, new_id
from broll.ingest.probe import probe

runner = CliRunner()


async def test_analyzer_produces_a_valid_result(workspace, clips, tmp_path):
    clip = clips["single_static_bars"]
    meta = probe(clip)
    analyzer = Analyzer(workspace)
    context = ShotContext(
        source_filename=clip.name, duration_s=meta.duration_s,
        width=meta.width, height=meta.height, fps=meta.fps,
    )
    outcome = await analyzer.analyse_shot(clip, context, tmp_path / "frames")
    assert isinstance(outcome.result, AnalysisResult)
    assert len(outcome.frames) == 3
    assert outcome.analysis_version == PROMPT_VERSION


def test_store_roundtrip_and_derived_source_status(store, workspace):
    source = store.insert_source(
        Source(
            id=new_id(), workspace_id=workspace.id, content_hash="abc123",
            original_filename="a.mp4", origin="local", duration_s=4.0,
        )
    )
    shot = Shot(
        id=f"{source.id}-0", workspace_id=workspace.id, source_id=source.id,
        shot_index=0, duration_s=4.0, end_s=4.0, caption="A calm beach at dawn.",
        setting="beach", action="meditating", mood=["calm"], subjects=["person"],
        tags=["meditation", "sunrise"], status="indexed",
    )
    store.insert_shot(shot)

    stored = store.get_shot(shot.id)
    assert stored.caption == "A calm beach at dawn."
    assert stored.tags == ["meditation", "sunrise"]
    assert "beach" in stored.search_text and "meditation" in stored.search_text

    assert store.recompute_source_status(source.id) == "indexed"
    store.set_shot_fields(shot.id, status="needs_review")
    assert store.recompute_source_status(source.id) == "needs_review"


def test_fts_index_tracks_search_text(store, workspace):
    source = store.insert_source(
        Source(id=new_id(), workspace_id=workspace.id, content_hash="h",
               original_filename="a.mp4", origin="local")
    )
    shot = Shot(id=f"{source.id}-0", workspace_id=workspace.id, source_id=source.id,
                caption="Sunrise over a quiet harbour.", setting="beach", status="indexed")
    store.insert_shot(shot)
    rows = store.conn.execute(
        "SELECT rowid FROM shots_fts WHERE shots_fts MATCH ?", ("harbour",)
    ).fetchall()
    assert len(rows) == 1


def test_facet_counts_include_scalar_and_list_facets(store, workspace):
    for i, setting in enumerate(["beach", "beach", "office"]):
        source = store.insert_source(
            Source(id=new_id(), workspace_id=workspace.id, content_hash=f"h{i}",
                   original_filename=f"{i}.mp4", origin="local")
        )
        store.insert_shot(
            Shot(id=f"{source.id}-0", workspace_id=workspace.id, source_id=source.id,
                 setting=setting, mood=["calm"], status="indexed")
        )
    counts = store.facet_counts()
    assert counts[("setting", "beach")] == 2
    assert counts[("setting", "office")] == 1
    assert counts[("mood", "calm")] == 3


def test_vocabulary_candidates_accumulate(store):
    store.record_vocabulary_candidates([("subjects", "hydrofoil")])
    store.record_vocabulary_candidates([("subjects", "hydrofoil")])
    candidates = store.vocabulary_candidates()
    assert candidates[0]["term"] == "hydrofoil"
    assert candidates[0]["count"] == 2


# -- CLI ------------------------------------------------------------------


def _init_mock_workspace(broll_home):
    result = runner.invoke(app, ["init", "--name", "Fixtures", "--provider", "mock"])
    assert result.exit_code == 0, result.output
    return "fixtures"


def test_cli_init_analyse_index_status(broll_home, clips, monkeypatch):
    monkeypatch.setenv("BROLL_HOME", str(broll_home))
    _init_mock_workspace(broll_home)

    clip = clips["single_static_bars"]
    result = runner.invoke(app, ["analyse", str(clip)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    AnalysisResult.model_validate(payload["analysis"])
    assert payload["frames_used"] == 3
    assert payload["analysis_version"] == PROMPT_VERSION

    result = runner.invoke(app, ["index", str(clip)])
    assert result.exit_code == 0, result.output
    assert "indexed" in result.output

    # Second run must dedupe on content hash rather than re-analyse.
    result = runner.invoke(app, ["index", str(clip)])
    assert "already indexed" in result.output

    result = runner.invoke(app, ["status"])
    assert "shots:     1 total" in result.output


def test_cli_index_writes_a_thumbnail(broll_home, clips, monkeypatch):
    monkeypatch.setenv("BROLL_HOME", str(broll_home))
    _init_mock_workspace(broll_home)
    runner.invoke(app, ["index", str(clips["moving_push_in"])])

    from broll.config import load_workspace_config

    config = load_workspace_config("fixtures")
    store = Store.for_config(config)
    try:
        shots = store.list_shots()
        assert len(shots) == 1
        assert shots[0].thumbnail_path
        from pathlib import Path

        assert Path(shots[0].thumbnail_path).exists()
    finally:
        store.close()


async def test_recorded_responses_are_replayed(workspace, clips, tmp_path):
    """Provider tests replay recorded responses rather than calling a live API."""
    from pathlib import Path

    from broll.analysis.providers.mock import MockVisionProvider

    responses = Path(__file__).parent / "fixtures" / "responses"
    provider = MockVisionProvider(fixtures_dir=responses)
    analyzer = Analyzer(workspace, provider=provider)

    clip = clips["out_of_focus"]
    meta = probe(clip)
    context = ShotContext(
        source_filename=clip.name, duration_s=meta.duration_s,
        width=meta.width, height=meta.height, fps=meta.fps,
    )
    outcome = await analyzer.analyse_shot(clip, context, tmp_path / "frames")
    assert outcome.result.quality_flags == ["out_of_focus"]
    assert "defocused" in outcome.result.caption
    # quality_flags present -> the shot is routed to the review queue
    assert outcome.status == "needs_review"


async def test_a_transient_provider_error_is_raised_for_the_queue_to_retry(workspace, tmp_path):
    """A 503 while Google is overloaded must not park a shot in the review queue."""
    import pytest

    from broll.analysis.providers.base import TransientProviderError, classify_error

    assert classify_error("503 UNAVAILABLE high demand") is TransientProviderError
    assert classify_error("400 invalid schema") is not TransientProviderError

    class Flaky:
        name = "flaky"

        async def analyse(self, frames, context, retry_error=None):
            raise TransientProviderError("gemini request failed: 503 UNAVAILABLE")

        def estimate_cost(self, frames):
            return 0.0

    analyzer = Analyzer(workspace, provider=Flaky())
    context = ShotContext(source_filename="a.mp4", duration_s=3.0, width=640, height=360)
    with pytest.raises(TransientProviderError):
        await analyzer.analyse_frames([tmp_path / "frame.jpg"], context)


async def test_only_real_defects_route_a_shot_to_review(workspace, tmp_path):
    """A logo is worth recording; it is not a reason to make a human look."""
    from broll.analysis.schema import AnalysisResult

    base = {
        "caption": "A fire truck raises its ladder.", "setting": "industrial area",
        "shot_type": "medium_wide", "camera_movement": "tilt_up", "time_of_day": "afternoon",
        "colour_profile": "vibrant", "people_count": "one", "pace": "slow", "confidence": 0.95,
    }

    class Fixed:
        name = "fixed"

        def __init__(self, flags):
            self.flags = flags

        async def analyse(self, frames, context, retry_error=None):
            return AnalysisResult.model_validate({**base, "quality_flags": self.flags})

        def estimate_cost(self, frames):
            return 0.0

    context = ShotContext(source_filename="a.mp4", duration_s=3.0, width=640, height=360)
    frames = [tmp_path / "frame.jpg"]

    logo = await Analyzer(workspace, provider=Fixed(["contains_logo"])).analyse_frames(frames, context)
    assert logo.status == "indexed"

    blurry = await Analyzer(workspace, provider=Fixed(["out_of_focus"])).analyse_frames(frames, context)
    assert blurry.status == "needs_review"
