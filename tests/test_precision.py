"""Tags that are true, and searches that only find what they should.

The failure these guard against is not one wrong tag: it is the same tag on too many clips, and
camera-setting words ("static", "wide", "slow") sitting in the search text of most of the library.
"""

from __future__ import annotations

import hashlib
import math
import re
import sqlite3

import pytest

from broll.analysis.analyzer import Analyzer
from broll.analysis.embedder import Embedder
from broll.analysis.schema import AnalysisResult, ShotContext
from broll.config import ClientProfile
from broll.db import migrations
from broll.db.models import Shot, Source
from broll.db.store import new_id
from broll.search.filters import SearchFilters
from broll.search.query import SearchEngine, parse_structure

BASE = {
    "caption": "A man sits cross-legged on a sandy beach at sunrise with his eyes closed.",
    "observations": ["man sitting cross-legged", "sand", "ocean", "orange sky", "eyes closed"],
    "subjects": ["man", "beach"], "action": "meditating", "setting": "beach",
    "shot_type": "wide", "camera_movement": "static", "time_of_day": "dawn",
    "colour_profile": "warm", "people_count": "one", "pace": "slow",
}


def analyzer_for(workspace, **result_fields):
    workspace.client = ClientProfile(
        name="Adam Kunder", featured_person="Adam Kunder",
        themes=["nervous system", "breathwork", "stillness", "ice bath"],
    )

    class Fixed:
        name = "fixed"

        async def analyse(self, frames, context, retry_error=None):
            return AnalysisResult.model_validate({**BASE, **result_fields})

        def estimate_cost(self, frames):
            return 0.0

    return Analyzer(workspace, provider=Fixed())


async def analyse(analyzer, tmp_path):
    ctx = ShotContext(source_filename="a.mp4", duration_s=5, width=1920, height=1080)
    return await analyzer.analyse_frames([tmp_path / "f.jpg"], ctx)


# -- tags are held to what is visible ---------------------------------------------------------


async def test_a_feeling_or_idea_is_not_a_visible_tag(workspace, tmp_path):
    out = await analyse(analyzer_for(workspace, tags=["man", "beach", "sand", "calm", "peaceful", "mindfulness"]), tmp_path)
    assert "calm" not in out.result.tags and "peaceful" not in out.result.tags
    assert out.result.tags == ["man", "beach", "sand", "mindfulness"]  # mindfulness is no mood word: kept


async def test_a_mood_word_that_the_caption_really_says_is_kept(workspace, tmp_path):
    out = await analyse(analyzer_for(
        workspace, caption="A calm sea at dawn.", observations=["calm sea"], tags=["sea", "calm"]), tmp_path)
    assert out.result.tags == ["sea", "calm"]


async def test_a_theme_put_in_tags_is_moved_to_themes(workspace, tmp_path):
    out = await analyse(analyzer_for(workspace, tags=["man", "beach", "breathwork"], themes=[]), tmp_path)
    assert "breathwork" not in out.result.tags and out.result.themes == ["breathwork"]


async def test_themes_can_only_come_from_the_clients_list(workspace, tmp_path):
    out = await analyse(analyzer_for(workspace, themes=["Stillness", "wellness", "ice bath", "vibes"]), tmp_path)
    assert out.result.themes == ["stillness", "ice bath"]


async def test_themes_are_empty_for_a_library_with_no_client_themes(workspace, tmp_path):
    analyzer = analyzer_for(workspace, themes=["stillness"])
    analyzer._themes = {}
    out = await analyse(analyzer, tmp_path)
    assert out.result.themes == []


async def test_filler_and_camera_words_are_never_tags(workspace, tmp_path):
    out = await analyse(analyzer_for(
        workspace, tags=["video", "footage", "cinematic", "close up", "static", "beach", "adam", "kunder"]), tmp_path)
    assert out.result.tags == ["beach"]


async def test_the_clients_name_is_a_filter_not_a_tag(workspace, tmp_path):
    out = await analyse(analyzer_for(workspace, tags=["adam kunder", "adam", "beach"]), tmp_path)
    assert out.result.tags == ["beach"]


# -- when a person should look -----------------------------------------------------------------


async def test_review_reasons_say_why(workspace, tmp_path):
    a = analyzer_for(workspace, confidence=0.5, quality_flags=["shaky"])
    out = await analyse(a, tmp_path)
    assert out.status == "needs_review" and out.review_reasons == ["low_confidence", "quality_defect"]


async def test_a_clean_confident_clip_has_no_reasons(workspace, tmp_path):
    out = await analyse(analyzer_for(workspace, confidence=0.93), tmp_path)
    assert out.status == "indexed" and out.review_reasons == []


async def test_unsure_which_folder_is_a_reason_even_when_sure_of_the_clip(workspace, tmp_path):
    from tests.test_client_tree import tree_config

    workspace.taxonomy = tree_config()
    a = analyzer_for(workspace, confidence=0.95, category="01_Nervous System Practices/Breathwork",
                     category_confidence=0.4)
    workspace.taxonomy = tree_config()
    out = await analyse(a, tmp_path)
    assert out.status == "needs_review" and "low_category_confidence" in out.review_reasons


async def test_a_folder_that_is_not_the_clients_is_a_reason(workspace, tmp_path):
    from tests.test_client_tree import tree_config

    workspace.taxonomy = tree_config()
    a = analyzer_for(workspace, confidence=0.95, category="Totally/Made Up")
    workspace.taxonomy = tree_config()
    out = await analyse(a, tmp_path)
    assert "category_unmatched" in out.review_reasons and out.result.category is None


# -- what is searchable ---------------------------------------------------------------------


def add_shot(store, name, **fields):
    source = store.insert_source(Source(
        id=new_id(), workspace_id=store.workspace_id, content_hash=name, original_filename=name, origin="local"))
    fields.setdefault("status", "indexed")
    shot = Shot(id=f"{source.id}-0", workspace_id=store.workspace_id, source_id=source.id, **fields)
    store.insert_shot(shot)
    return shot.id


def test_camera_settings_are_not_words_in_the_index(store):
    add_shot(store, "a.mp4", caption="A man on a beach.", setting="beach", shot_type="wide",
             camera_movement="static", pace="slow", people_count="one", colour_profile="warm",
             time_of_day="dawn", subjects=["man"], tags=["sand"], usable_for=["opener"])
    row = store.conn.execute("SELECT search_text, concept_text FROM shots").fetchone()
    words = set(re.findall(r"[a-z]+", (row["search_text"] + " " + row["concept_text"]).lower()))
    for noise in ("wide", "static", "slow", "one", "warm", "opener", "pace"):
        assert noise not in words, noise
    assert {"beach", "man", "sand", "dawn"} <= words
    engine = SearchEngine(store)
    for query in ("wide", "static", "slow", "moderate", "close"):
        assert engine.search(query) == [], query


def test_picture_words_and_concept_words_go_in_different_columns(store):
    add_shot(store, "a.mp4", caption="A man on a beach.", setting="beach", subjects=["man"], tags=["sand"],
             themes=["stillness"], mood=["serene"], emotions=["calm"],
             category="01_Nervous System Practices/Meditation & Stillness")
    row = store.conn.execute("SELECT search_text, concept_text FROM shots").fetchone()
    assert "sand" in row["search_text"] and "stillness" not in row["search_text"]
    assert all(w in row["concept_text"] for w in ("stillness", "serene", "calm", "Meditation & Stillness"))


def test_a_match_on_what_is_in_the_picture_outweighs_one_on_a_concept(store):
    a = add_shot(store, "a.mp4", caption="A man doing a headstand on a beach.", setting="beach", subjects=["man"])
    b = add_shot(store, "b.mp4", caption="A woman in an office.", setting="office", themes=["beach"])
    coverage = SearchEngine(store)._coverage(["beach"], [a, b])
    assert coverage[a] == pytest.approx(1.0) and coverage[b] == pytest.approx(0.7)
    assert [r.shot.id for r in SearchEngine(store).search("beach")][0] == a


def test_a_clip_that_matches_only_on_concepts_ranks_below_one_that_matches_on_the_picture(store):
    on_screen = add_shot(store, "a.mp4", caption="A man meditating on a beach.", setting="beach",
                         subjects=["man"], emotions=["calm"])
    only_concepts = add_shot(store, "b.mp4", caption="A woman types at a desk.", setting="office",
                             subjects=["woman"], emotions=["calm"], themes=["beach"])
    engine = SearchEngine(store)
    coverage = engine._coverage(["calm", "beach"], [on_screen, only_concepts])
    assert coverage[on_screen] == pytest.approx((0.7 + 1.0) / 2)
    assert coverage[only_concepts] == pytest.approx(0.7)
    assert [r.shot.id for r in engine.search("calm beach")] == [on_screen, only_concepts]


# -- asking for a kind of shot -------------------------------------------------------------------


def test_asking_for_a_shot_type_is_read_as_a_request_not_a_word():
    assert parse_structure("close up of coffee being poured") == (
        "of coffee being poured", {"shot_type": {"close_up", "extreme_close_up", "macro"}})
    text, wanted = parse_structure("handheld walk through a market")
    assert text == "walk through a market" and wanted == {"camera_movement": {"handheld"}}


def test_words_that_are_also_content_stay_in_the_query():
    text, wanted = parse_structure("aerial shot of mountains")
    assert "aerial" in text and wanted["shot_type"] == {"aerial", "top_down"}


def test_ordinary_uses_of_the_same_words_are_left_alone():
    for phrase in ("a wide range of options", "static electricity", "a close friend", "slow down"):
        assert parse_structure(phrase) == (phrase, {}), phrase


def test_the_shot_type_asked_for_nudges_the_ranking_but_never_filters(store):
    wide = add_shot(store, "a.mp4", caption="Coffee pouring into a cup on a table.", subjects=["coffee"],
                    shot_type="wide")
    close = add_shot(store, "b.mp4", caption="Coffee pouring into a cup on a table.", subjects=["coffee"],
                     shot_type="close_up")
    ids = [r.shot.id for r in SearchEngine(store).search("close up of coffee being poured")]
    assert ids == [close, wide], "the close-up rises, the other is still offered"


def test_a_query_that_is_only_a_kind_of_shot_lists_those_shots(store):
    close = add_shot(store, "a.mp4", caption="Coffee.", subjects=["coffee"], shot_type="close_up")
    add_shot(store, "b.mp4", caption="Beach.", subjects=["beach"], shot_type="wide")
    assert [r.shot.id for r in SearchEngine(store).search("close up")] == [close]


# -- the relevance gate adapts to the embedding model --------------------------------------------


class HashEmbedder(Embedder):
    """Bag-of-words vectors, plus an optional constant every vector shares.

    The shared part is what a real model like Gemini does: unrelated texts still score ~0.5, so a
    fixed floor tuned on another model lets everything through.
    """

    name = "hash"
    model = "hash"

    def __init__(self, dims=64, common=0.0):
        self.dimensions, self.common = dims, common

    def _vector(self, text):
        v = [0.0] * self.dimensions
        for word in re.findall(r"[a-z]+", text.lower()):
            digest = hashlib.sha256(word.encode()).digest()
            for k in range(3):
                v[digest[k] % self.dimensions] += 1.0 if digest[k + 3] % 2 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        v = [x / norm for x in v]
        if self.common:
            v = [x * (1 - self.common) + self.common * (1.0 if i == 0 else 0.0) for i, x in enumerate(v)]
            norm = math.sqrt(sum(x * x for x in v))
            v = [x / norm for x in v]
        return v

    def embed(self, texts):
        return [self._vector(t) for t in texts]


TOPICS = ["beach surf", "office desk laptop", "forest hiking trail", "city night traffic", "kitchen cooking pasta",
          "gym weights barbell", "mountain snow ski", "child playing park", "yoga studio stretch", "river kayak paddle"]


def big_library(store, embedder, n=120):
    for i in range(n):
        topic = TOPICS[i % len(TOPICS)]
        sid = add_shot(store, f"c{i}.mp4", caption=f"A {topic} scene number {i}.", subjects=topic.split()[:2])
        store.vectors.upsert(sid, embedder.embed_documents([store.embedding_text(sid)])[0])


@pytest.mark.parametrize("common", [0.0, 0.7])
def test_the_floor_is_measured_against_the_library_not_a_fixed_number(store, common):
    embedder = HashEmbedder(common=common)
    big_library(store, embedder)
    engine = SearchEngine(store, embedder)
    on_topic = engine.search("surf on the beach", limit=50)
    assert len(on_topic) >= 8 and all("beach" in r.shot.caption for r in on_topic)
    assert engine.search("a mariachi band playing trumpets at a wedding") == []


def test_a_library_too_small_to_sample_uses_the_fixed_numbers_for_its_model(store):
    embedder = HashEmbedder()
    big_library(store, embedder, n=20)
    engine = SearchEngine(store, embedder)
    floor, gap, near_floor, near_gap = engine._thresholds(embedder.embed_query("beach"))
    assert (floor, gap) == (0.25, 0.12) and near_floor < floor  # the default for an unnamed model


def test_the_measured_floor_sits_above_the_background(store):
    embedder = HashEmbedder(common=0.7)
    big_library(store, embedder)
    engine = SearchEngine(store, embedder)
    q = embedder.embed_query("a mariachi band")
    floor, gap, near_floor, _ = engine._thresholds(q)
    background = list(engine._similarities(q, store.sample_shot_ids(400)).values())
    mean = sum(background) / len(background)
    assert floor > mean and floor > 0.6, "with a shared component everything scores high; the floor rose with it"
    assert near_floor < floor


# -- upgrading a library that was indexed under the old rules ------------------------------------


def test_an_old_library_gets_clean_search_text_when_it_is_upgraded(tmp_path):
    conn = sqlite3.connect(tmp_path / "old.db", isolation_level=None)
    conn.row_factory = sqlite3.Row
    for version in range(1, 8):
        migrations.MIGRATIONS[version](conn)
        conn.execute(f"PRAGMA user_version = {version}")
    conn.execute("INSERT INTO sources (id, workspace_id, content_hash, original_filename, origin)"
                 " VALUES ('s', 'w', 'h', 'a.mp4', 'local')")
    conn.execute(
        "INSERT INTO shots (id, workspace_id, source_id, shot_index, caption, setting, shot_type,"
        " camera_movement, pace, people_count, subjects_json, mood_json, emotions_json, search_text)"
        " VALUES ('s-0', 'w', 's', 0, 'A man on a beach.', 'beach', 'wide', 'static', 'slow', 'one',"
        " '[\"man\"]', '[\"serene\"]', '[\"calm\"]', 'A man on a beach., beach, wide, static, slow, one, man')")
    migrations.migrate(conn)

    row = conn.execute("SELECT search_text, concept_text FROM shots").fetchone()
    assert "static" not in row["search_text"] and "serene" in row["concept_text"]
    hits = conn.execute("SELECT rowid FROM shots_fts WHERE shots_fts MATCH 'beach'").fetchall()
    assert len(hits) == 1
    assert conn.execute("SELECT rowid FROM shots_fts WHERE shots_fts MATCH 'static'").fetchall() == []
    assert conn.execute("PRAGMA user_version").fetchone()[0] == migrations.SCHEMA_VERSION
