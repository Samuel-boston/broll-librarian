"""Searching a client's library by what happens, not by who is in it.

Regression, from the client: "I need a clip of Adam meditating" returned the
meditation clip *and* "Adam yawns in bed", and called a firefighter clip
"loosely related". "Adam" is on nearly every clip in Adam's library, so as a
search word it matched half the query everywhere.
"""

from __future__ import annotations

from broll.analysis.analyzer import Analyzer
from broll.analysis.schema import AnalysisResult, ShotContext
from broll.db.models import Shot, Source
from broll.db.store import new_id
from broll.search.query import SearchEngine
from tests.test_relevance import embedder, names, needs_embeddings

ADAM = "Adam Kunder"

CLIPS = [
    ("meditation.mov", "Adam sits cross-legged meditating on a sunny beach by the ocean.",
     True, "meditating", ["meditation", "mindfulness", "calm", "beach", "ocean", "stillness"]),
    ("bed.mov", "Adam yawns in bed and pulls the covers up to go back to sleep.",
     True, "sleeping", ["bed", "sleep", "tired", "yawn", "morning", "rest"]),
    ("firefighter.mp4", "A firefighter operates the controls of a red fire engine.",
     False, "operating", ["firefighter", "fire engine", "work", "emergency", "ladder"]),
]


def adam_library(store, with_embeddings: bool):
    for filename, caption, featured, action, tags in CLIPS:
        source = store.insert_source(Source(
            id=new_id(), workspace_id=store.workspace_id, content_hash=filename,
            original_filename=filename, origin="local"))
        store.insert_shot(Shot(
            id=f"{source.id}-0", workspace_id=store.workspace_id, source_id=source.id,
            caption=caption, featured_person=featured, action=action, tags=tags,
            status="indexed"))
    if with_embeddings:
        for shot in store.list_shots():
            store.vectors.upsert(shot.id, embedder().embed_one(shot.search_text))


def test_the_name_comes_out_of_the_query():
    engine = SearchEngine(store=None, featured_person=ADAM)  # type: ignore[arg-type]
    assert engine.split_person("I need a clip of Adam meditating") == ("I need a clip of meditating", True)
    assert engine.split_person("Adam Kunder's morning") == ("morning", True)
    assert engine.split_person("adam's partner") == ("partner", True)
    assert engine.split_person("a madam at a desk") == ("a madam at a desk", False)  # whole words only


@needs_embeddings
def test_a_clip_of_adam_meditating_is_only_the_meditation_clip(store):
    adam_library(store, with_embeddings=True)
    engine = SearchEngine(store, embedder(), featured_person=ADAM)
    assert names(engine.search("I need a clip of Adam meditating")) == ["meditation.mov"]
    # A clip without Adam in it is not even a near match for a search about Adam.
    loose = names(engine.search("I need a clip of Adam meditating", strict=False))
    assert "firefighter.mp4" not in loose


def test_adam_alone_lists_his_clips(store):
    adam_library(store, with_embeddings=False)
    results = SearchEngine(store, featured_person=ADAM).search("clips of Adam")
    assert sorted(names(results)) == ["bed.mov", "meditation.mov"]


def test_without_embeddings_the_name_still_does_not_match_everything(store):
    adam_library(store, with_embeddings=False)
    results = SearchEngine(store, featured_person=ADAM).search("I need a clip of Adam meditating")
    assert names(results) == ["meditation.mov"]


def test_transcripts_drop_the_name_but_do_not_require_it(store):
    """Narration says "Adam" over footage with no Adam in it."""
    adam_library(store, with_embeddings=False)
    engine = SearchEngine(store, featured_person=ADAM)
    found = names(engine.search("Adam used to work as a firefighter", strict=False, person_filter=False))
    assert "firefighter.mp4" in found


def test_a_word_on_most_clips_carries_little_weight(store):
    """With "calm" on every clip, "calm meditating" must not return calm-only clips."""
    for index in range(12):
        tags = ["calm", "meditation"] if index == 0 else ["calm", "office"]
        source = store.insert_source(Source(
            id=new_id(), workspace_id=store.workspace_id, content_hash=f"h{index}",
            original_filename=f"clip{index}.mov", origin="local"))
        store.insert_shot(Shot(
            id=f"{source.id}-0", workspace_id=store.workspace_id, source_id=source.id,
            caption="A clip.", tags=tags, action="meditating" if index == 0 else "typing",
            status="indexed"))
    assert names(SearchEngine(store).search("calm meditating")) == ["clip0.mov"]


def test_a_word_nothing_contains_does_not_sink_a_strong_match(store):
    """2 of 3 specific words present ("regulation" is on no clip) is still a match."""
    adam_library(store, with_embeddings=False)
    shot = next(s for s in store.list_shots() if "meditating" in (s.caption or ""))
    store.set_shot_tags(shot.id, shot.tags + ["nervous system"])
    store.recompute_search_text(shot.id)
    assert names(SearchEngine(store).search("nervous system regulation")) == ["meditation.mov"]


async def test_the_featured_name_is_never_stored_as_a_tag(workspace, tmp_path):
    workspace.client.featured_person = ADAM

    class Tagged:
        name = "tagged"

        async def analyse(self, frames, context, retry_error=None):
            return AnalysisResult.model_validate({
                "caption": "Adam meditates on the beach.", "setting": "beach",
                "shot_type": "wide", "camera_movement": "static", "time_of_day": "dawn",
                "colour_profile": "warm", "people_count": "one", "pace": "still",
                "tags": ["adam", "adam kunder", "kunder", "meditation", "beach"],
            })

        def estimate_cost(self, frames):
            return 0.0

    outcome = await Analyzer(workspace, provider=Tagged()).analyse_frames(
        [tmp_path / "f.jpg"], ShotContext(source_filename="a.mov", duration_s=5, width=1920, height=1080))
    assert outcome.result.tags == ["meditation", "beach"]
