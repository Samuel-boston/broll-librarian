"""Search must return only what is relevant.

Regression, from the client: "I need a shot of him meditating" returned every
clip in a three-clip library, meditation first. Vector search always returns
its nearest neighbours however far away, and the keyword side matched "a".
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from broll.analysis.embedder import LocalEmbedder, local_embeddings_available
from broll.config import EmbedderConfig
from broll.search.query import SearchEngine, meaningful_tokens
from tests.test_search import RELEVANCE_QUERIES, seed_library

needs_embeddings = pytest.mark.skipif(
    not local_embeddings_available(), reason="sentence-transformers not installed"
)

_EMBEDDER = None


def embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = LocalEmbedder(EmbedderConfig())
    return _EMBEDDER


def names(results):
    return [r.source.original_filename for r in results]


def test_a_request_is_reduced_to_what_it_describes():
    assert meaningful_tokens("I need a shot of him meditating") == ["meditating"]
    assert meaningful_tokens("can you find me some footage of a calm beach") == ["calm", "beach"]
    assert meaningful_tokens("a shot of him") == []


def test_stemming_finds_other_forms_of_a_word(store):
    seed_library(store)
    assert "beach_meditation_sunrise.mp4" in names(SearchEngine(store).search("meditate"))


def test_without_embeddings_filler_words_still_match_nothing(store):
    seed_library(store)
    results = SearchEngine(store).search("I need a shot of him meditating")
    assert names(results) == ["beach_meditation_sunrise.mp4"]


@needs_embeddings
def test_the_clients_query_returns_only_the_meditation_clip(store):
    seed_library(store, embedder())
    engine = SearchEngine(store, embedder())
    results = engine.search("I need a shot of him meditating", limit=20)
    assert names(results) == ["beach_meditation_sunrise.mp4"]
    assert engine.hidden_count > 0, "the rest are hidden, not lost"


@needs_embeddings
def test_a_theme_tag_is_found_even_when_the_meaning_is_distant(store):
    """'nervous system' scores low semantically against a meditation clip, but
    when the clip is literally tagged with it, it must still come back."""
    seed_library(store, embedder())
    shot = next(s for s in store.list_shots()
                if store.get_source(s.source_id).original_filename == "beach_meditation_sunrise.mp4")
    store.set_shot_tags(shot.id, shot.tags + ["nervous system"])
    store.recompute_search_text(shot.id)
    results = SearchEngine(store, embedder()).search("nervous system regulation")
    assert "beach_meditation_sunrise.mp4" in names(results)


@needs_embeddings
def test_nothing_relevant_means_nothing_shown(store):
    seed_library(store, embedder())
    engine = SearchEngine(store, embedder())
    assert engine.search("a mariachi band playing trumpets at a wedding") == []
    assert engine.hidden_count > 0


@needs_embeddings
def test_every_relevance_query_is_precise_as_well_as_right(store, capsys):
    """The M2 queries: the expected clip first, and little else."""
    seed_library(store, embedder())
    engine = SearchEngine(store, embedder())
    report = []
    for query, expected in RELEVANCE_QUERIES:
        found = names(engine.search(query, limit=20))
        report.append(f"  {len(found):>2} result(s)  {query!r} -> {found}")
        assert found and found[0] == expected, (query, found)
        assert len(found) <= 3, (query, found)
    with capsys.disabled():
        print("\nprecision:\n" + "\n".join(report))


@needs_embeddings
def test_loose_mode_returns_what_strict_mode_hid(store):
    seed_library(store, embedder())
    engine = SearchEngine(store, embedder())
    strict = engine.search("I need a shot of him meditating", limit=50)
    hidden = engine.hidden_count
    loose = engine.search("I need a shot of him meditating", limit=50, strict=False)
    assert len(loose) == len(strict) + hidden


@needs_embeddings
def test_the_search_page_hides_loose_matches_behind_a_link(workspace, store):
    from broll.web.app import create_app

    seed_library(store, embedder())
    store.close()
    app = create_app(workspace, run_worker=False)
    app.state.broll.embedder = embedder()
    with TestClient(app) as client:
        strict = client.get("/search", params={"q": "I need a shot of him meditating"}).text
        loose = client.get("/search", params={"q": "I need a shot of him meditating",
                                              "loose": "true"}).text
    assert strict.count("<article") == 1
    assert "loosely related clip" in strict and "Show them" in strict
    assert loose.count("<article") > 1
