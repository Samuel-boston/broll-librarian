"""Typo tolerance, and browsing the library by its folders.

From the client: one wrong character meant the clip he wanted didn't appear;
and a grid of six arbitrary clips is no way to see what a big library holds.
"""

from __future__ import annotations

from collections import Counter

import pytest
from fastapi.testclient import TestClient

from broll.db.models import Shot, Source
from broll.db.store import new_id
from broll.search.browse import children, folder_label, summarise_folders
from broll.search.filters import SearchFilters
from broll.search.query import SearchEngine
from broll.search.spelling import edit_distance, max_edits, suggest
from tests.test_person_search import ADAM, adam_library
from tests.test_relevance import names
from tests.test_search import seed_library

# -- spelling --------------------------------------------------------------------


@pytest.mark.parametrize("a,b,expected", [
    ("meditating", "meditating", 0),
    ("meditaing", "meditating", 1),     # a dropped letter
    ("meditaitng", "meditating", 1),    # a swapped pair counts once
    ("firefihgting", "firefighting", 1),
    ("tird", "tired", 1),
])
def test_edit_distance(a, b, expected):
    assert edit_distance(a, b, 3) == expected


def test_edit_distance_gives_up_early():
    assert edit_distance("meditation", "firefighter", 2) == 3


def test_short_words_are_never_corrected():
    assert max_edits("bed") == 0
    assert suggest("bde", Counter({"bed": 5})) is None


def test_suggest_prefers_the_nearest_then_the_commonest():
    vocabulary = Counter({"meditating": 3, "meditation": 9, "mediating": 1})
    assert suggest("meditaing", vocabulary) == "meditating"
    assert suggest("tird", Counter({"tired": 4, "third": 1})) == "tired"
    assert suggest("helicopter", Counter({"meditating": 3})) is None


def test_a_known_word_is_left_alone():
    assert suggest("meditation", Counter({"meditation": 1, "meditating": 5})) is None


def test_a_typo_still_finds_the_clip(store):
    seed_library(store)
    engine = SearchEngine(store)
    assert "beach_meditation_sunrise.mp4" in names(engine.search("meditaing on the beach"))
    assert ("meditaing", "meditating") in engine.corrections
    assert engine.corrected_query.startswith("meditating")


def test_search_as_typed_turns_correction_off(store):
    seed_library(store)
    engine = SearchEngine(store)
    assert engine.search("meditaing", correct=False) == []
    assert engine.corrections == []


def test_a_word_the_library_knows_in_another_form_is_not_corrected(store):
    seed_library(store)
    engine = SearchEngine(store)
    engine.search("meditate")  # "meditating" is there; the stemmed index knows it
    assert engine.corrections == []


def test_a_misspelled_name_is_still_the_person(store):
    adam_library(store, with_embeddings=False)
    engine = SearchEngine(store, featured_person=ADAM)
    assert names(engine.search("Adma meditating")) == ["meditation.mov"]
    assert ("adma", "adam") in engine.corrections


# -- browsing ------------------------------------------------------------------------


FOLDERS = [
    "01_Nervous System Practices",
    "01_Nervous System Practices/Breathwork",
    "01_Nervous System Practices/Meditation & Stillness",
    "05_Travel & Adventure",
    "05_Travel & Adventure/Beach & Water",
]


def test_folder_label_splits_the_number():
    assert folder_label("01_Nervous System Practices") == ("01", "Nervous System Practices")
    assert folder_label("Breathwork") == (None, "Breathwork")


def test_children_lists_one_level():
    assert children("", FOLDERS) == ["01_Nervous System Practices", "05_Travel & Adventure"]
    assert children("01_Nervous System Practices", FOLDERS) == [
        "01_Nervous System Practices/Breathwork",
        "01_Nervous System Practices/Meditation & Stillness",
    ]


def test_counts_roll_up_and_include_secondary_folders():
    rows = [
        {"id": "a", "category": "01_Nervous System Practices/Meditation & Stillness",
         "secondary": ["05_Travel & Adventure/Beach & Water"], "has_thumbnail": True},
        {"id": "b", "category": "01_Nervous System Practices/Breathwork",
         "secondary": [], "has_thumbnail": True},
    ]
    summary = summarise_folders(rows, FOLDERS)
    assert summary["01_Nervous System Practices"].count == 2
    assert summary["01_Nervous System Practices/Breathwork"].count == 1
    assert summary["05_Travel & Adventure/Beach & Water"].count == 1   # via secondary
    assert summary["05_Travel & Adventure"].count == 1
    assert summary["01_Nervous System Practices"].thumbnails == ["a", "b"]  # newest first
    assert summary["01_Nervous System Practices"].has_children


def _tree_workspace(workspace, store):
    from tests.test_client_tree import tree_config

    workspace.taxonomy = tree_config()
    workspace.client.featured_person = ADAM
    for filename, category, secondary, emotions in [
        ("meditation.mov", "01_Nervous System Practices/Meditation & Stillness",
         ["05_Travel & Adventure/Beach & Water"], ["calm", "grounded"]),
        ("breath.mov", "01_Nervous System Practices/Breathwork", [], ["calm"]),
        ("stray.mov", None, [], ["curious"]),
    ]:
        source = store.insert_source(Source(id=new_id(), workspace_id=workspace.id,
                                            content_hash=filename, original_filename=filename,
                                            origin="local"))
        store.insert_shot(Shot(id=f"{source.id}-0", workspace_id=workspace.id,
                               source_id=source.id, caption=f"A clip called {filename}.",
                               category=category, secondary_categories=secondary,
                               emotions=emotions, status="indexed"))


def test_the_folder_filter_includes_secondary_matches(workspace, store):
    _tree_workspace(workspace, store)
    engine = SearchEngine(store)
    beach = names(engine.browse(SearchFilters(category=["05_Travel & Adventure/Beach & Water"]), 50))
    assert beach == ["meditation.mov"]
    assert sorted(names(engine.browse(SearchFilters(category=["01_Nervous System Practices"]), 50))) == [
        "breath.mov", "meditation.mov"]
    assert names(engine.browse(SearchFilters(uncategorised=True), 50)) == ["stray.mov"]


def test_the_library_home_shows_folders_feelings_and_recent_clips(workspace, store):
    from broll.web.app import create_app

    _tree_workspace(workspace, store)
    store.close()
    with TestClient(create_app(workspace, run_worker=False)) as client:
        home = client.get("/").text  # redirects to /library
        # The page (rightly) escapes "&" as "&amp;".
        assert "Nervous System Practices" in home and "Travel &amp; Adventure" in home
        assert "Browse by feeling" in home and "calm" in home
        assert "Recently added" in home
        assert "Unsorted · 1" in home
        assert "START HERE" not in home and "Top Picks</span>" not in home

        folder = client.get("/library", params={"path": "01_Nervous System Practices"}).text
        assert "Breathwork" in folder and "Meditation &amp; Stillness" in folder  # subfolders
        assert folder.count("<article") == 2                                       # both clips
        assert "2 clips in Nervous System Practices" in folder

        feeling = client.get("/library", params={"emotion": "grounded"}).text
        assert feeling.count("<article") == 1 and "Feeling grounded" in feeling


def test_the_search_page_offers_the_way_back_from_a_correction(workspace, store):
    from broll.web.app import create_app

    seed_library(store)
    store.close()
    with TestClient(create_app(workspace, run_worker=False)) as client:
        html = client.get("/search", params={"q": "meditaing"}).text
        assert "Showing results for" in html and "Search instead for" in html
        exact = client.get("/search", params={"q": "meditaing", "exact": "true"}).text
        assert "Showing results for" not in exact
