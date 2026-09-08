"""Search: FTS5 escaping, filters, fusion, and the M2 relevance gate.

The relevance gate was written before the search code, per the build spec. It
runs against a fixture library of recorded analyses (tests/fixtures/library.json)
rather than live model output, so it is deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from broll.analysis.embedder import LocalEmbedder, local_embeddings_available
from broll.analysis.prompt import PROMPT_VERSION
from broll.analysis.schema import AnalysisResult
from broll.db.models import Shot, Source
from broll.db.store import new_id
from broll.search.filters import SearchFilters
from broll.search.query import SearchEngine, escape_fts_query

LIBRARY = json.loads((Path(__file__).parent / "fixtures" / "library.json").read_text())

# The M2 acceptance gate: eight fixed queries, expected clip in the top 3.
RELEVANCE_QUERIES = [
    ("peaceful sunrise by the sea", "beach_meditation_sunrise.mp4"),
    ("team brainstorming in an open plan office", "office_meeting_brainstorm.mp4"),
    ("close up of coffee being poured", "coffee_pour_closeup.mp4"),
    ("aerial shot flying over mountains", "mountain_drone_flyover.mp4"),
    ("busy city street at night", "city_timelapse_night.mp4"),
    ("someone working out with heavy weights", "gym_weights_lifting.mp4"),
    ("hands typing at a desk", "laptop_typing_desk.mp4"),
    ("rain on a window, moody and quiet", "rain_window_moody.mp4"),
]


def seed_library(store, embedder=None) -> dict[str, str]:
    """Insert the fixture library through the normal store API."""
    by_filename: dict[str, str] = {}
    for record in LIBRARY:
        analysis = AnalysisResult.model_validate(record["analysis"])
        source = store.insert_source(
            Source(
                id=new_id(),
                workspace_id=store.workspace_id,
                content_hash=record["filename"],
                original_filename=record["filename"],
                origin="local",
                origin_path=f"/library/{record['filename']}",
                duration_s=record["duration_s"],
                width=record["width"],
                height=record["height"],
                fps=record["fps"],
                drive_web_link=f"https://drive.google.com/file/d/{record['filename']}/view",
            )
        )
        shot = Shot(
            id=f"{source.id}-0",
            workspace_id=store.workspace_id,
            source_id=source.id,
            shot_index=0,
            is_primary=True,
            start_s=0.0,
            end_s=record["duration_s"],
            duration_s=record["duration_s"],
            status="indexed",
        ).apply_analysis(analysis, PROMPT_VERSION)
        store.insert_shot(shot)
        store.recompute_source_status(source.id)
        by_filename[record["filename"]] = shot.id

    if embedder is not None:
        shots = store.list_shots()
        texts = [s.search_text for s in shots]
        for shot, vector in zip(shots, embedder.embed(texts)):
            store.vectors.upsert(shot.id, vector)
    return by_filename


# -- FTS5 escaping ---------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "person's wide-shot",
        'a "quoted" phrase',
        "NEAR(beach ocean)",
        "beach AND NOT office",
        "wide * shot",
        "colon: caret^ dash-dash",
        "OR",
        "((()))",
        "emoji 🌊 clip",
        "  ",
    ],
)
def test_fts_queries_never_raise(store, query):
    """An apostrophe or an operator in user input is text, not syntax."""
    seed_library(store)
    engine = SearchEngine(store, embedder=None)
    results = engine.search(query, SearchFilters(), limit=5)
    assert isinstance(results, list)


def test_escape_fts_query_quotes_every_token():
    assert escape_fts_query("wide-shot") == '"wide" OR "shot"'
    assert escape_fts_query("") == ""


# -- keyword, filters, fusion ---------------------------------------------


def test_keyword_search_finds_an_exact_tag(store):
    ids = seed_library(store)
    engine = SearchEngine(store, embedder=None)
    results = engine.search("barbell", SearchFilters(), limit=5)
    assert results and results[0].shot.id == ids["gym_weights_lifting.mp4"]
    assert results[0].matched == ["keyword"]


def test_filters_narrow_results(store):
    seed_library(store)
    engine = SearchEngine(store, embedder=None)

    aerial = engine.search("", SearchFilters(shot_type=["aerial"]), limit=20)
    assert [r.source.original_filename for r in aerial] == ["mountain_drone_flyover.mp4"]

    crowded = engine.search("", SearchFilters(people_count=["crowd"]), limit=20)
    assert [r.source.original_filename for r in crowded] == ["city_timelapse_night.mp4"]

    calm = engine.search("", SearchFilters(mood=["calm"]), limit=20)
    names = {r.source.original_filename for r in calm}
    assert {"beach_meditation_sunrise.mp4", "yoga_studio_stretching.mp4"} <= names
    # mood is a facet, not a tag: rain_window_moody carries the *tag* "calm"
    # but its moods are melancholy/quiet/moody, so it must not match here.
    assert "rain_window_moody.mp4" not in names

    short = engine.search("", SearchFilters(duration_max_s=7.0), limit=20)
    assert all(r.shot.duration_s <= 7.0 for r in short)


def test_drive_link_carries_the_shot_timecode(store):
    ids = seed_library(store)
    engine = SearchEngine(store, embedder=None)
    shot_id = ids["gym_weights_lifting.mp4"]
    store.set_shot_fields(shot_id, start_s=42.0)
    result = next(r for r in engine.search("barbell", limit=5) if r.shot.id == shot_id)
    assert result.drive_link.endswith("#t=42")
    assert result.timecode == "0m42s"


def test_vector_search_matches_without_shared_words(store):
    """The point of the vector side: no lexical overlap with the caption."""
    if not local_embeddings_available():
        pytest.skip("sentence-transformers not installed")
    from broll.config import EmbedderConfig

    embedder = LocalEmbedder(EmbedderConfig())
    ids = seed_library(store, embedder)
    engine = SearchEngine(store, embedder)

    hits = engine.vector_search("automated manufacturing", SearchFilters(), limit=3)
    assert ids["factory_robot_arm.mp4"] in [shot_id for shot_id, _ in hits]


# -- the M2 acceptance gate -----------------------------------------------


def test_relevance_smoke_test(store, capsys):
    """Eight fixed queries, expected clip in the top 3. The M2 gate."""
    if not local_embeddings_available():
        pytest.skip("sentence-transformers not installed - hybrid search needs it")
    from broll.config import EmbedderConfig

    embedder = LocalEmbedder(EmbedderConfig())
    ids = seed_library(store, embedder)
    engine = SearchEngine(store, embedder)

    lines, failures = [], []
    for query, expected in RELEVANCE_QUERIES:
        results = engine.search(query, SearchFilters(), limit=10)
        names = [r.source.original_filename for r in results]
        position = names.index(expected) + 1 if expected in names else None
        lines.append(f"  {'PASS' if position and position <= 3 else 'FAIL'} "
                     f"#{position or '-'} {query!r} -> {names[:3]}")
        if position is None or position > 3:
            failures.append(f"{query!r}: expected {expected} in top 3, got {names[:3]}")

    with capsys.disabled():
        print("\nM2 relevance smoke test")
        for line in lines:
            print(line)

    assert not failures, "\n".join(failures)
