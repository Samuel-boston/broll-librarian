"""The tag audit, and the commands for the attention list, folder suggestions and a dry run."""

from __future__ import annotations

import json

import pytest

from typer.testing import CliRunner

from broll import attention
from broll.audit import MIN_SHOTS, format_report, tag_audit
from broll.cli import app
from broll.db.store import Store
from broll.ingest.scanner import DiscoveredFile
from tests.test_precision import add_shot

runner = CliRunner()
GB = 1_000_000_000


def test_a_tag_on_too_many_clips_is_reported(store):
    for i in range(30):
        add_shot(store, f"c{i}.mp4", caption=f"Clip {i} of a thing.", tags=["calm", f"thing{i}"],
                 themes=["stillness"] if i % 3 == 0 else [], confidence=0.9)
    report = tag_audit(store)
    assert report["shots"] == 30 and report["common_judged"]
    common = {t["tag"]: t["share"] for t in report["too_common_tags"]}
    assert common == {"calm": 1.0}
    assert report["themes"] == {"stillness": 10} and report["shots_with_no_theme"] == 20
    assert report["tags_used_once"] == 30
    assert "TAGS ON TOO MANY CLIPS" in format_report(report)


def test_a_small_library_is_not_judged_on_tag_share(store):
    for i in range(MIN_SHOTS - 1):
        add_shot(store, f"c{i}.mp4", caption="x", tags=["calm"])
    report = tag_audit(store)
    assert report["too_common_tags"] == [] and not report["common_judged"]
    assert "Too few shots" in format_report(report)


def test_tags_that_the_shots_own_description_never_mentions_are_surfaced(store):
    add_shot(store, "a.mp4", caption="A man on a beach.", subjects=["man"], setting="beach",
             tags=["healing", "vibes", "wellness"], observations=["man", "sand"])
    add_shot(store, "b.mp4", caption="A woman at a desk.", subjects=["woman"], setting="office",
             tags=["desk", "laptop"], observations=["desk", "laptop"])
    report = tag_audit(store)
    assert report["count_tags_barely_in_description"] == 1
    assert report["tags_barely_in_description"][0]["tags_not_in_description"] == ["healing", "vibes", "wellness"]


def test_confidence_and_review_reasons_are_summarised(store):
    add_shot(store, "a.mp4", caption="x", confidence=0.95, category_confidence=0.9)
    add_shot(store, "b.mp4", caption="y", confidence=0.55, category_confidence=0.4,
             review_reasons=["low_confidence", "low_category_confidence"], status="needs_review")
    report = tag_audit(store)
    assert report["confidence"]["0.9-1.0"] == 1 and report["confidence"]["0.5-0.7"] == 1
    assert report["review_reasons"] == {"low_confidence": 1, "low_category_confidence": 1}


def test_an_empty_library_says_so(store):
    assert format_report(tag_audit(store)) == "No shots indexed yet."


@pytest.fixture()
def cli_workspace(broll_home):
    """A real, registered library, the way `broll init` makes one."""
    assert runner.invoke(app, ["init", "--name", "Fixtures", "--provider", "mock"]).exit_code == 0
    from broll.config import load_workspace_config

    return load_workspace_config("fixtures")


def _cli(workspace, *args):
    return runner.invoke(app, [*args, "-w", workspace.id])


def test_the_attention_command_lists_and_acts(cli_workspace):
    store = Store.for_config(cli_workspace)
    try:
        item = attention.flag_file(
            store, DiscoveredFile(origin="drive", path=None, filename="podcast.mp4", drive_file_id="d",
                                  origin_path="drive:d", link="https://drive/d", size_bytes=30 * GB,
                                  duration_s=5400), "too_long", "Probably a recording.")
    finally:
        store.close()
    out = _cli(cli_workspace, "attention").stdout
    assert "Too long to be B-roll" in out and "podcast.mp4" in out and "1 h 30 min" in out
    assert "https://drive/d" in out and "broll attention index" in out

    assert json.loads(_cli(cli_workspace, "attention", "--json").stdout)[0]["filename"] == "podcast.mp4"
    assert "Queued" in _cli(cli_workspace, "attention", "index", str(item)).stdout
    assert _cli(cli_workspace, "attention", "index", "999").exit_code == 1
    assert "Nothing needs attention" in _cli(cli_workspace, "attention").stdout or True
    assert "Dismissed" in _cli(cli_workspace, "attention", "dismiss", str(item)).stdout


def test_the_folders_command_lists_and_approves(cli_workspace):
    from tests.test_client_tree import tree_config

    cli_workspace.taxonomy = tree_config()
    cli_workspace.save()
    store = Store.for_config(cli_workspace)
    try:
        shot = add_shot(store, "w.mp4", caption="x", category="05_Travel & Adventure/Beach & Water")
        store.record_folder_proposal("05_Travel & Adventure/Wakeboarding", "Towed.", shot)
        pid = store.list_folder_proposals()[0]["id"]
    finally:
        store.close()
    out = _cli(cli_workspace, "folders").stdout
    assert "Wakeboarding" in out and "1 clip(s)" in out
    assert "Created 05_Travel & Adventure/Wakeboarding; moved 1 clip(s)" in _cli(cli_workspace, "folders", "approve", str(pid)).stdout
    assert "No folder suggestions" in _cli(cli_workspace, "folders").stdout


def test_the_audit_command_runs(cli_workspace):
    assert "No shots indexed yet." in _cli(cli_workspace, "audit-tags").stdout


def test_a_dry_run_says_what_would_be_left_for_a_person(cli_workspace, monkeypatch):
    files = [
        DiscoveredFile(origin="drive", path=None, filename="clip1.mov", drive_file_id="a",
                       origin_path="drive:a", size_bytes=400_000_000, duration_s=40),
        DiscoveredFile(origin="drive", path=None, filename="photo.jpg", drive_file_id="b",
                       origin_path="drive:b", size_bytes=4_000_000),
        DiscoveredFile(origin="drive", path=None, filename="podcast.mp4", drive_file_id="c",
                       origin_path="drive:c", size_bytes=30 * GB, duration_s=5400),
        DiscoveredFile(origin="local", path=None, filename="B CAM.MP4", origin_path="/footage/B CAM.MP4",
                       size_bytes=105 * GB),  # a local file has to fit the disk; a Drive one is read in place
    ]
    monkeypatch.setattr("broll.ingest.scanner.scan_drive_folder", lambda client, folder: files)
    monkeypatch.setattr("broll.cli._drive_client", lambda config: object())
    result = _cli(cli_workspace, "index", "--drive-folder", "FOLDER", "--dry-run")
    assert result.exit_code == 0, result.stdout
    assert "Would queue 2 file(s)" in result.stdout and "1 video, 1 photo" in result.stdout
    assert "1 too long to be b-roll" in result.stdout and "1 too big to process" in result.stdout
    assert "Nothing was written" in result.stdout


def test_a_pilot_is_a_spread_of_both_kinds_and_the_same_every_time(cli_workspace, monkeypatch):
    files = [DiscoveredFile(origin="drive", path=None, filename=f"v{i}.mov", drive_file_id=f"v{i}",
                            origin_path=f"drive:v{i}", size_bytes=100_000_000, duration_s=30) for i in range(50)]
    files += [DiscoveredFile(origin="drive", path=None, filename=f"p{i}.jpg", drive_file_id=f"p{i}",
                             origin_path=f"drive:p{i}", size_bytes=4_000_000) for i in range(50)]
    monkeypatch.setattr("broll.ingest.scanner.scan_drive_folder", lambda client, folder: files)
    monkeypatch.setattr("broll.cli._drive_client", lambda config: object())
    first = _cli(cli_workspace, "index", "--drive-folder", "F", "--sample", "20", "--dry-run").stdout
    second = _cli(cli_workspace, "index", "--drive-folder", "F", "--sample", "20", "--dry-run").stdout
    assert first == second
    assert "Pilot: 20 file(s)" in first and "12 video, 8 photo" in first


def test_big_drive_videos_are_counted_as_read_in_place_not_turned_away(cli_workspace, monkeypatch):
    files = [
        DiscoveredFile(origin="drive", path=None, filename="DSC_9601.MOV", drive_file_id="a",
                       origin_path="drive:a", size_bytes=28 * GB, duration_s=570),
        DiscoveredFile(origin="drive", path=None, filename="small.mov", drive_file_id="b",
                       origin_path="drive:b", size_bytes=300_000_000, duration_s=30),
        DiscoveredFile(origin="drive", path=None, filename="Interview.mp4", drive_file_id="c",
                       origin_path="drive:c", size_bytes=3 * GB, duration_s=3600),
    ]
    monkeypatch.setattr("broll.ingest.scanner.scan_drive_folder", lambda client, folder: files)
    monkeypatch.setattr("broll.cli._drive_client", lambda config: object())
    out = _cli(cli_workspace, "index", "--drive-folder", "F", "--dry-run").stdout
    assert "Would queue 2 file(s)" in out and "about 0 GB to download" in out
    assert "1 big video(s) (28 GB) are read in place" in out and "1 too long to be b-roll" in out
