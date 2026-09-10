"""A client's own folder tree, client-aware analysis, and emotions."""

from __future__ import annotations

import pytest

from broll.analysis.analyzer import Analyzer, match_category
from broll.analysis.prompt import build_user_prompt, render_client_context
from broll.analysis.schema import AnalysisResult, ShotContext
from broll.config import CategoryNode, ClientProfile, TaxonomyConfig
from broll.db.models import ShotFacets
from broll.drive.taxonomy import plan_filename, plan_tree, render_guide

TREE = [
    CategoryNode(name="00_START HERE", destination=False),
    CategoryNode(name="01_Nervous System Practices", children=[
        CategoryNode(name="Breathwork", description="Breathing exercises."),
        CategoryNode(name="Meditation & Stillness", description="Meditation and stillness."),
    ]),
    CategoryNode(name="02_Gym & Training", description="Gym sessions."),
    CategoryNode(name="05_Travel & Adventure", description="Split by activity", children=[
        CategoryNode(name="Travel", children=[
            CategoryNode(name="Flights", description="On a plane."),
            CategoryNode(name="In the Car", description="Driving."),
        ]),
        CategoryNode(name="Beach & Water", description="Beaches and water sports."),
    ]),
    CategoryNode(name="07_Cinematic & Mood", description="Only for shots with no clear activity",
                 children=[CategoryNode(name="Reflective", description="Quiet moments.")]),
    CategoryNode(name="★ Top Picks", destination=False),
]


def tree_config(**overrides) -> TaxonomyConfig:
    return TaxonomyConfig(mode="tree", tree=[n.model_copy(deep=True) for n in TREE],
                          guide_folder="00_START HERE", top_picks_folder="★ Top Picks",
                          filename_template="{leaf}_{action}_{emotion}", **overrides)


def facets(**kw) -> ShotFacets:
    base = dict(shot_id="s0", source_id="src", is_primary=True, action="meditating",
                emotions=["calm"], category="01_Nervous System Practices/Meditation & Stillness")
    base.update(kw)
    return ShotFacets(**base)


# -- the tree itself -----------------------------------------------------------


def test_only_real_destinations_are_offered():
    leaves = dict(tree_config().category_leaves())
    assert "01_Nervous System Practices/Meditation & Stillness" in leaves
    assert "02_Gym & Training" in leaves                      # a leaf at the top level
    assert "05_Travel & Adventure/Travel/Flights" in leaves  # three levels deep
    assert "01_Nervous System Practices" not in leaves        # a container
    assert "00_START HERE" not in leaves and "★ Top Picks" not in leaves


def test_a_parent_rule_reaches_its_children():
    leaves = dict(tree_config().category_leaves())
    assert "Only for shots with no clear activity" in leaves["07_Cinematic & Mood/Reflective"]


def test_adding_a_folder_under_a_leaf_keeps_the_leaf_a_destination():
    config = tree_config()
    path = config.add_folder("02_Gym & Training", "Boxing", "Boxing and pad work.")
    leaves = dict(config.category_leaves())
    assert path == "02_Gym & Training/Boxing" and path in leaves
    assert "02_Gym & Training" in leaves, "clips already filed there must not be orphaned"


# -- filing ---------------------------------------------------------------------


def test_the_file_lives_in_its_best_fit_folder_with_shortcuts_elsewhere():
    shot = facets(secondary_categories=["05_Travel & Adventure/Beach & Water"])
    home, shortcuts = plan_tree([shot], "clip.mov", tree_config())
    assert str(home) == "01_Nervous System Practices/Meditation & Stillness"
    assert [str(s.folder) for s in shortcuts] == ["05_Travel & Adventure/Beach & Water"]


def test_no_shortcut_is_made_into_the_files_own_folder():
    a = facets(shot_id="s0")
    b = facets(shot_id="s1", is_primary=False, start_s=12.0)  # same folder as the file
    _, shortcuts = plan_tree([a, b], "clip.mov", tree_config())
    assert shortcuts == []


def test_a_starred_shot_gets_a_top_picks_shortcut():
    _, shortcuts = plan_tree([facets(top_pick=True)], "clip.mov", tree_config())
    assert [str(s.folder) for s in shortcuts] == ["★ Top Picks"]


def test_an_unfiled_clip_goes_to_unsorted():
    home, _ = plan_tree([facets(category=None)], "clip.mov", tree_config())
    assert str(home) == "_Unsorted"


def test_filenames_follow_the_template():
    name = plan_filename(facets(), "f54362f4aaaa", ".mov", tree_config())
    assert name == "meditation_stillness_meditating_calm_f54362f4.mov"


def test_the_guide_explains_names_folders_and_emotions():
    text = render_guide(tree_config(), ["calm", "grounded"], "Adam Kunder")
    assert "folder_action_emotion_<id>" in text
    assert "01_Nervous System Practices > Meditation & Stillness" in text
    assert "calm, grounded" in text


# -- matching and creating folders ----------------------------------------------


@pytest.mark.parametrize("given,expected", [
    ("01_Nervous System Practices/Meditation & Stillness", "01_Nervous System Practices/Meditation & Stillness"),
    ("01_nervous system practices/meditation and stillness", None),  # "and" is not "&"
    ("01_Nervous System Practices / Meditation & Stillness", "01_Nervous System Practices/Meditation & Stillness"),
    ("Meditation & Stillness", "01_Nervous System Practices/Meditation & Stillness"),
    ("Flights", "05_Travel & Adventure/Travel/Flights"),
    ("Something Else", None),
])
def test_match_category(given, expected):
    leaves = [p for p, _ in tree_config().category_leaves()]
    assert match_category(given, leaves) == expected


def _analyzer(workspace, taxonomy, proposal):
    workspace.taxonomy = taxonomy

    class Fixed:
        name = "fixed"

        async def analyse(self, frames, context, retry_error=None):
            return AnalysisResult.model_validate({
                "caption": "Adam rides a jet ski across a bay.", "setting": "coastline",
                "shot_type": "wide", "camera_movement": "tracking", "time_of_day": "afternoon",
                "colour_profile": "vibrant", "people_count": "one", "pace": "fast",
                "category": "05_Travel & Adventure/Beach & Water", **proposal,
            })

        def estimate_cost(self, frames):
            return 0.0

    return Analyzer(workspace, provider=Fixed())


async def test_a_proposed_folder_is_created_under_its_parent_and_persisted(workspace, tmp_path):
    from broll.config import load_workspace_config

    analyzer = _analyzer(workspace, tree_config(), {
        "new_category": "05_Travel & Adventure/wakeboarding",
        "new_category_note": "Wakeboarding and towed water sports.",
    })
    context = ShotContext(source_filename="a.mp4", duration_s=5, width=1920, height=1080)
    outcome = await analyzer.analyse_frames([tmp_path / "f.jpg"], context)

    assert outcome.result.category == "05_Travel & Adventure/Wakeboarding"
    assert "05_Travel & Adventure/Wakeboarding" in dict(load_workspace_config(workspace.id).taxonomy.category_leaves())
    # ...and the next clip is offered it.
    assert any(o.startswith("05_Travel & Adventure/Wakeboarding") for o in analyzer.category_options)


async def test_a_near_duplicate_folder_reuses_the_existing_one(workspace, tmp_path):
    analyzer = _analyzer(workspace, tree_config(), {"new_category": "05_Travel & Adventure/Beach and Water"})
    context = ShotContext(source_filename="a.mp4", duration_s=5, width=1920, height=1080)
    outcome = await analyzer.analyse_frames([tmp_path / "f.jpg"], context)
    assert outcome.result.category == "05_Travel & Adventure/Beach & Water"
    assert analyzer.new_folders == []


async def test_a_folder_needs_a_real_parent(workspace, tmp_path):
    analyzer = _analyzer(workspace, tree_config(), {"new_category": "Hobbies/Jet Skiing"})
    context = ShotContext(source_filename="a.mp4", duration_s=5, width=1920, height=1080)
    outcome = await analyzer.analyse_frames([tmp_path / "f.jpg"], context)
    assert outcome.result.category == "05_Travel & Adventure/Beach & Water"  # fell back
    assert analyzer.new_folders == []


async def test_new_folders_can_be_switched_off(workspace, tmp_path):
    analyzer = _analyzer(workspace, tree_config(allow_new_folders=False),
                         {"new_category": "05_Travel & Adventure/Wakeboarding"})
    context = ShotContext(source_filename="a.mp4", duration_s=5, width=1920, height=1080)
    outcome = await analyzer.analyse_frames([tmp_path / "f.jpg"], context)
    assert outcome.result.category == "05_Travel & Adventure/Beach & Water"


# -- client-aware prompt --------------------------------------------------------


def test_the_client_block_names_the_person_and_their_themes():
    block = render_client_context(ClientProfile(
        name="Adam Kunder", featured_person="Adam Kunder", featured_person_description="a man",
        brief="Adam is a mindset coach.", themes=["nervous system", "breathwork"]))
    assert "Adam is a mindset coach." in block
    assert "nervous system, breathwork" in block
    assert 'call them "Adam"' in block and "Never name anyone else" in block


def test_a_generic_library_gets_no_client_block():
    assert render_client_context(ClientProfile()) == ""


def test_the_prompt_carries_client_emotions_and_folders():
    prompt = build_user_prompt(ShotContext(
        source_filename="a.mp4", duration_s=5, width=1920, height=1080,
        client_context="CLIENT CONTEXT - x", emotion_vocab=["regulated", "grounded"],
        category_options=["02_Gym & Training — Gym sessions."]))
    assert "emotions (choose 2-5, most important first): regulated, grounded" in prompt
    assert "CLIENT CONTEXT - x" in prompt
    assert "- 02_Gym & Training — Gym sessions." in prompt
    assert "new_category" in prompt


# -- emotions end to end --------------------------------------------------------


def test_emotions_are_stored_searchable_and_filterable(store, workspace):
    from broll.db.models import Shot, Source
    from broll.db.store import new_id
    from broll.search.filters import SearchFilters
    from broll.search.query import SearchEngine

    source = store.insert_source(Source(id=new_id(), workspace_id=workspace.id, content_hash="h",
                                        original_filename="bed.mov", origin="local"))
    store.insert_shot(Shot(id=f"{source.id}-0", workspace_id=workspace.id, source_id=source.id,
                           caption="Adam yawns and falls back asleep.", emotions=["tired", "drained"],
                           category="04_Daily Rituals/Sleep & Rest", featured_person=True,
                           status="indexed"))
    engine = SearchEngine(store, embedder=None)

    assert engine.search("drained", SearchFilters(), 5), "emotions are in the keyword index"
    assert engine.search("", SearchFilters(emotions=["tired"]), 5)
    assert not engine.search("", SearchFilters(emotions=["joyful"]), 5)
    assert engine.search("", SearchFilters(category=["04_Daily Rituals"]), 5), "a parent folder matches"
    assert engine.search("", SearchFilters(featured_person=True), 5)
    assert not engine.search("", SearchFilters(top_pick=True), 5)
