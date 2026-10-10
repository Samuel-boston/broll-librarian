"""`broll refile` and `broll reanalyse` on a library whose folder tree changed after indexing."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from broll import cli, refile
from broll.cli import app
from broll.db.models import Source
from tests.fakes import FakeTextProvider
from tests.test_client_tree import tree_config
from tests.test_precision import add_shot

runner = CliRunner()
MED = "01_Nervous System Practices/Meditation & Stillness"
GYM = "02_Gym & Training"
FLIGHTS = "05_Travel & Adventure/Travel/Flights"


@pytest.fixture()
def tree(workspace, store):
    workspace.taxonomy = tree_config()
    workspace.save()
    return workspace


@pytest.fixture()
def cli_ws(broll_home):
    from broll.config import load_workspace_config
    from broll.db.store import Store

    assert runner.invoke(app, ["init", "--name", "Fixtures", "--provider", "mock"]).exit_code == 0
    ws = load_workspace_config("fixtures")
    ws.taxonomy = tree_config()
    ws.save()
    store = Store.for_config(ws)
    yield ws, store
    try:
        store.close()
    except Exception:  # noqa: BLE001 - a test may have closed it already
        pass


def _plan(tree, store, provider, **kw):
    shots = refile.select_shots(store, **{k: kw.pop(k) for k in ("folders", "statuses", "source_ids") if k in kw})
    return refile.plan(tree, store, provider, shots, **kw)


def test_a_clip_moves_to_the_better_folder_and_the_rest_stay(tree, store):
    a = add_shot(store, "a.mp4", caption="A man boards a plane", category=GYM, category_confidence=0.9)
    b = add_shot(store, "b.mp4", caption="Lifting weights", category=GYM, category_confidence=0.9)
    provider = FakeTextProvider({"plane": (FLIGHTS, 0.9), "weights": (GYM, 0.9)})
    report = _plan(tree, store, provider)
    assert [(m.shot_id, m.after) for m in report.moves] == [(a, FLIGHTS)]
    assert report.unchanged == 1
    assert report.matrix() == [(GYM, FLIGHTS, 1)]
    assert store.get_shot(a).category == GYM, "planning writes nothing"
    refile.apply(store, report)
    assert store.get_shot(a).category == FLIGHTS and store.get_shot(b).category == GYM


def test_search_text_follows_the_new_folder(tree, store):
    a = add_shot(store, "a.mp4", caption="A man boards a plane", category=GYM)
    report = _plan(tree, store, FakeTextProvider({"plane": (FLIGHTS, 0.9)}))
    refile.apply(store, report)
    row = store.conn.execute("SELECT concept_text FROM shots WHERE id = ?", (a,)).fetchone()
    assert "Flights" in row["concept_text"] and "Gym" not in row["concept_text"]


def test_a_tie_keeps_the_clip_where_it_is(tree, store):
    a = add_shot(store, "a.mp4", caption="A man sits quietly", category=MED)
    provider = FakeTextProvider({"sits": (GYM, 0.4, [MED])})
    report = _plan(tree, store, provider)
    assert report.moves == [] and report.kept_on_tie == 1
    assert store.get_shot(a).category == MED


def test_a_confident_move_still_happens_and_unsure_new_folder_is_flagged(tree, store):
    a = add_shot(store, "a.mp4", caption="A man sits quietly", category=MED)
    report = _plan(tree, store, FakeTextProvider({"sits": (GYM, 0.4)}))
    refile.apply(store, report)
    shot = store.get_shot(a)
    assert shot.category == GYM and shot.status == "needs_review"
    assert shot.review_reasons == ["low_category_confidence"]


def test_a_sure_move_clears_the_old_folder_reason_but_keeps_others(tree, store):
    a = add_shot(store, "a.mp4", caption="A man boards a plane", category=GYM, status="needs_review",
                 review_reasons=["low_category_confidence", "quality_defect"])
    refile.apply(store, _plan(tree, store, FakeTextProvider({"plane": (FLIGHTS, 0.95)})))
    shot = store.get_shot(a)
    assert shot.review_reasons == ["quality_defect"] and shot.status == "needs_review"


def test_corrected_clips_stay_unless_asked(tree, store):
    a = add_shot(store, "a.mp4", caption="A man boards a plane", category=GYM,
                 raw_analysis={"corrected_by_operator": True})
    provider = FakeTextProvider({"plane": (FLIGHTS, 0.9)})
    report = _plan(tree, store, provider)
    assert report.moves == [] and report.skipped_corrected == 1 and provider.prompts == []
    report = _plan(tree, store, provider, include_corrected=True)
    assert [m.shot_id for m in report.moves] == [a]


def test_folder_filter_includes_sub_folders_only(tree, store):
    add_shot(store, "a.mp4", caption="plane", category=GYM)
    inside = add_shot(store, "b.mp4", caption="plane", category=MED)
    shots = refile.select_shots(store, folders=["01_Nervous System Practices"])
    assert [s.id for s in shots] == [inside]


def test_an_unknown_folder_from_the_model_changes_nothing(tree, store):
    a = add_shot(store, "a.mp4", caption="plane", category=GYM)
    report = _plan(tree, store, FakeTextProvider({"plane": ("Not/A Folder", 0.9)}))
    assert report.moves == [] and store.get_shot(a).category == GYM


def test_a_clip_in_a_removed_folder_is_filed_afresh_or_flagged(tree, store):
    a = add_shot(store, "a.mp4", caption="plane", category="Gone/Folder")
    b = add_shot(store, "b.mp4", caption="mystery", category="Gone/Folder")
    report = _plan(tree, store, FakeTextProvider({"plane": (FLIGHTS, 0.9), "mystery": ("nope", 0.9)}))
    refile.apply(store, report)
    assert store.get_shot(a).category == FLIGHTS
    assert store.get_shot(b).category is None and "category_unmatched" in store.get_shot(b).review_reasons


def test_a_bad_answer_is_retried_once(tree, store):
    a = add_shot(store, "a.mp4", caption="plane", category=GYM)
    provider = FakeTextProvider({"plane": (FLIGHTS, 0.9)})
    provider.fail_first = 1
    report = _plan(tree, store, provider)
    assert [m.shot_id for m in report.moves] == [a] and len(provider.prompts) == 2
    assert "FAILED VALIDATION" in provider.prompts[1]


def test_clips_are_sent_twenty_to_a_call(tree, store):
    for i in range(45):
        add_shot(store, f"{i}.mp4", caption="plane", category=GYM)
    provider = FakeTextProvider({"plane": (FLIGHTS, 0.9)})
    report = _plan(tree, store, provider)
    assert len(provider.prompts) == 3 and len(report.moves) == 45


def test_the_prompt_carries_the_wording_rules_and_the_stored_description(tree, store):
    from broll.analysis.prompt import CATEGORY_HEADER

    add_shot(store, "a.mp4", caption="plane", category=GYM, observations=["a man in a queue"],
             tags=["airport"], body_language=["walking alone"])
    provider = FakeTextProvider({})
    _plan(tree, store, provider)
    prompt = provider.prompts[0]
    for rule in ("File a clip by what it is *for*", "An object in frame never decides the folder"):
        assert rule in CATEGORY_HEADER and rule in prompt
    assert "a man in a queue" in prompt and "airport" in prompt and "walking alone" in prompt
    assert FLIGHTS in prompt


def test_cli_dry_run_writes_nothing_and_apply_moves(cli_ws, monkeypatch):
    tree, store = cli_ws
    a = add_shot(store, "a.mp4", caption="A man boards a plane", category=GYM)
    store.close()
    monkeypatch.setattr(cli, "get_text_provider", lambda config: FakeTextProvider({"plane": (FLIGHTS, 0.9)}))
    monkeypatch.setattr(cli, "check_credentials", lambda config: None)
    monkeypatch.setattr(cli, "_load_embedder", lambda config, required=False: None)
    base = ["refile", "-w", tree.id]

    out = runner.invoke(app, [*base, "--dry-run"])
    assert out.exit_code != 0, "needs --all or a filter"
    out = runner.invoke(app, [*base, "--all", "--dry-run"])
    assert out.exit_code == 0 and "1 would move" in out.output and f"{GYM}  ->  {FLIGHTS}" in out.output
    from broll.db.store import Store

    check = Store.for_config(tree)
    assert check.get_shot(a).category == GYM
    check.close()

    out = runner.invoke(app, [*base, "--all"])
    assert out.exit_code == 0 and "Moved 1" in out.output
    check = Store.for_config(tree)
    assert check.get_shot(a).category == FLIGHTS
    check.close()


# -- reanalyse for Drive sources ------------------------------------------------


def _drive_source(store, name="d.mp4", **kw):
    source = store.insert_source(Source(
        id=name, workspace_id=store.workspace_id, content_hash=name, original_filename=name,
        origin="drive", drive_file_id=f"id-{name}", origin_path=f"drive:id-{name}",
        filesize_bytes=3_000_000_000, duration_s=42.0, **kw))
    add_shot_for = add_shot(store, name + "x", caption="c", category=GYM)
    store.set_shot_fields(add_shot_for, source_id=source.id)
    return source


def test_reanalyse_requeues_drive_sources_with_their_file_id(cli_ws, monkeypatch):
    tree, store = cli_ws
    _drive_source(store)
    store.close()
    out = runner.invoke(app, ["reanalyse", "-w", tree.id, "--all", "--no-wait"])
    assert out.exit_code == 0, out.output
    assert "Estimated cost" in out.output and "Queued 1 job" in out.output
    from broll.db.store import Store

    check = Store.for_config(tree)
    job = check.conn.execute("SELECT payload_json FROM jobs").fetchone()
    check.close()
    import json

    payload = json.loads(job["payload_json"])
    assert payload["origin"] == "drive" and payload["drive_file_id"] == "id-d.mp4"
    assert payload["force"] is True and payload["size_bytes"] == 3_000_000_000


def test_reanalyse_asks_for_yes_above_a_dollar(cli_ws):
    tree, store = cli_ws
    for i in range(160):
        _drive_source(store, f"d{i}.mp4")
    store.close()
    out = runner.invoke(app, ["reanalyse", "-w", tree.id, "--all", "--no-wait"])
    assert out.exit_code != 0 and "--yes" in out.output
    out = runner.invoke(app, ["reanalyse", "-w", tree.id, "--all", "--no-wait", "--yes"])
    assert out.exit_code == 0 and "Queued 160" in out.output


def test_reanalyse_folder_limits_the_sources(cli_ws):
    tree, store = cli_ws
    _drive_source(store, "a.mp4")
    other = _drive_source(store, "b.mp4")
    store.set_shot_fields(store.shots_for_source(other.id)[0].id, category=MED)
    store.close()
    out = runner.invoke(app, ["reanalyse", "-w", tree.id, "--all", "--folder", MED, "--dry-run"])
    assert out.exit_code == 0 and "1 source(s)" in out.output and "b.mp4" in out.output
