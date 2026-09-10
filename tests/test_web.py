"""Web UI: the M4 acceptance criteria.

50 files dropped through the UI must all reach `indexed` with no CLI
intervention, the queue view must reflect live progress, and killing the worker
mid-run must resume cleanly on restart.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from broll.db.store import Store
from broll.jobs.queue import queue_stats
from broll.web.app import create_app

FILE_COUNT = 50


@pytest.fixture(scope="session")
def many_clips(tmp_path_factory) -> list[Path]:
    """50 tiny, distinct videos. Distinct bytes matter: dedupe is by content."""
    directory = tmp_path_factory.mktemp("many")
    clips = []
    for index in range(FILE_COUNT):
        target = directory / f"clip_{index:03d}.mp4"
        subprocess.run(
            [
                "ffmpeg", "-f", "lavfi",
                "-i", f"testsrc2=size=160x120:rate=10:duration=0.6",
                "-vf", f"hue=h={index * 7}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast",
                "-y", "-loglevel", "error", str(target),
            ],
            check=True,
        )
        clips.append(target)
    return clips


def _client(workspace, run_worker: bool = True) -> TestClient:
    return TestClient(create_app(workspace, run_worker=run_worker))


def _upload(client: TestClient, clips: list[Path]):
    files = [("files", (c.name, c.read_bytes(), "video/mp4")) for c in clips]
    return client.post("/ingest/upload", files=files)


def _drain(workspace, timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        store = Store.for_config(workspace)
        try:
            if queue_stats(store).outstanding == 0:
                return
        finally:
            store.close()
        time.sleep(0.5)
    raise AssertionError("the queue did not drain in time")


def test_pages_render(workspace):
    with _client(workspace, run_worker=False) as client:
        assert client.get("/search").status_code == 200
        assert client.get("/ingest").status_code == 200
        assert client.get("/ingest/queue").status_code == 200
        assert client.get("/").status_code == 200
        assert client.get("/no-such-page").status_code == 404


def test_uploading_a_non_video_is_rejected_not_queued(workspace, tmp_path):
    junk = tmp_path / "notes.txt"
    junk.write_text("not a video")
    with _client(workspace, run_worker=False) as client:
        response = client.post(
            "/ingest/upload",
            files=[("files", (junk.name, junk.read_bytes(), "text/plain"))],
        )
    assert response.status_code == 200
    assert "Skipped 1 non-video" in response.text

    store = Store.for_config(workspace)
    try:
        assert queue_stats(store).total == 0
    finally:
        store.close()


def test_indexing_a_folder_in_place_queues_without_copying(workspace, clips):
    folder = next(iter(clips.values())).parent
    with _client(workspace, run_worker=False) as client:
        response = client.post("/ingest/path", data={"path": str(folder)})
    assert response.status_code == 200
    assert "Queued 7 file(s)" in response.text
    assert not list(workspace.staging_dir.glob("*.mp4")), "in-place indexing must not copy"


def test_search_page_shows_indexed_shots(workspace, store):
    from tests.test_search import seed_library

    seed_library(store)
    store.close()
    with _client(workspace, run_worker=False) as client:
        html = client.get("/search?q=coffee").text
    assert "coffee" in html.lower()
    assert "shots indexed" in html


@pytest.mark.slow
def test_fifty_files_through_the_ui_all_reach_indexed(workspace, many_clips):
    """The M4 acceptance criterion."""
    with _client(workspace) as client:
        response = _upload(client, many_clips)
        assert response.status_code == 200
        assert f"Queued {FILE_COUNT} file(s)" in response.text

        # The queue view reflects live progress while work is outstanding.
        partial = client.get("/ingest/queue").text
        assert "hx-trigger=\"every 2s\"" in partial
        assert "worker running" in partial

        _drain(workspace)
        final = client.get("/ingest/queue").text

    assert "Failed" in final
    store = Store.for_config(workspace)
    try:
        stats = queue_stats(store)
        assert stats.done == FILE_COUNT
        assert stats.failed == 0
        sources = store.list_sources(limit=1000)
        assert len(sources) == FILE_COUNT
        assert {s.status for s in sources} == {"indexed"}
        assert store.count_shots() == FILE_COUNT
        assert store.vectors.count() == FILE_COUNT
    finally:
        store.close()


@pytest.mark.slow
def test_killing_the_web_worker_mid_run_resumes_on_restart(workspace, many_clips):
    """Shutting the process down mid-run must not lose or duplicate work."""
    subset = many_clips[:20]

    with _client(workspace) as client:
        _upload(client, subset)
        # Let a few finish, then leave the context: lifespan shutdown stops the
        # worker exactly as a Ctrl-C would.
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            store = Store.for_config(workspace)
            try:
                if queue_stats(store).done >= 3:
                    break
            finally:
                store.close()
            time.sleep(0.2)

    store = Store.for_config(workspace)
    try:
        mid = queue_stats(store)
        assert mid.outstanding > 0, "nothing was left to resume"
    finally:
        store.close()

    with _client(workspace):
        _drain(workspace)

    store = Store.for_config(workspace)
    try:
        assert queue_stats(store).done == len(subset)
        sources = store.list_sources(limit=1000)
        assert len(sources) == len(subset)
        hashes = [s.content_hash for s in sources]
        assert len(set(hashes)) == len(hashes)
        assert {s.status for s in sources} == {"indexed"}
    finally:
        store.close()


def test_empty_filter_values_do_not_error(workspace):
    """An untouched number input posts "" - that must not be a 422."""
    with _client(workspace, run_worker=False) as client:
        response = client.get("/search", params={"q": "beach", "min_duration": "",
                                                 "max_duration": ""})
    assert response.status_code == 200


def test_filters_narrow_the_rendered_grid(workspace, store):
    from tests.test_search import seed_library

    seed_library(store)
    store.close()
    with _client(workspace, run_worker=False) as client:
        everything = client.get("/search").text
        aerial = client.get("/search", params={"shot_type": "aerial"}).text
    assert everything.count("<article") == 14
    assert aerial.count("<article") == 1
    assert "mountain_drone_flyover.mp4" in aerial


def test_every_search_control_carries_its_own_verb(workspace):
    """htmx inherits hx-target and hx-include, but not the verb: a control
    without its own hx-get silently never fires a request."""
    with _client(workspace, run_worker=False) as client:
        html = client.get("/search").text
    import re

    controls = re.findall(r"<input[^>]*hx-trigger[^>]*>", html)
    assert controls, "no htmx-driven controls found"
    assert all('hx-get="/search"' in control for control in controls)


def test_transcript_screen_matches_swaps_and_exports(workspace, store):
    """Runs with embeddings, as the real app does. Narration rarely shares a
    meaningful word with a caption, so keyword-only matching finds nothing for
    it - it used to "match" on filler words like "the", which was worse."""
    from broll.analysis.embedder import LocalEmbedder, local_embeddings_available
    from broll.config import EmbedderConfig
    from tests.test_search import seed_library

    if not local_embeddings_available():
        pytest.skip("sentence-transformers not installed")
    embedder = LocalEmbedder(EmbedderConfig())
    seed_library(store, embedder)
    store.close()

    srt = ("1\n00:00:00,000 --> 00:00:07,000\n"
           "Most of us start the day already behind.\n\n"
           "2\n00:00:07,000 --> 00:00:15,000\n"
           "So we built something that keeps up with your team.\n")

    app = create_app(workspace, run_worker=False)
    app.state.broll.embedder = embedder
    with TestClient(app) as client:
        assert client.get("/transcript").status_code == 200

        response = client.post("/transcript", data={"text": srt, "filename": "demo.srt",
                                                    "rerank": "false"})
        assert response.status_code == 200
        assert "on timeline" in response.text
        assert "FCP7 XML" in response.text

        import re

        run_id = re.search(r"/transcript/([0-9a-f]+)/export/xml", response.text).group(1)

        swap = re.search(
            r'name="beat" value="(\d+)">\s*<input type="hidden" name="shot_id" value="([^"]+)"',
            response.text,
        )
        assert swap, "no alternative offered to swap to"
        swapped = client.post(f"/transcript/{run_id}/swap",
                              data={"beat": swap.group(1), "shot_id": swap.group(2)})
        assert swapped.status_code == 200

        xml = client.get(f"/transcript/{run_id}/export/xml")
        assert xml.status_code == 200
        assert xml.text.startswith("<?xml")
        assert "attachment" in xml.headers["content-disposition"]

        assert client.get(f"/transcript/{run_id}/export/edl").text.startswith("TITLE:")
        assert "beat,beat_start" in client.get(f"/transcript/{run_id}/export/csv").text
        assert client.get("/transcript/deadbeef/export/xml").status_code == 404


def test_transcript_requires_some_text(workspace):
    with _client(workspace, run_worker=False) as client:
        assert client.post("/transcript", data={"text": "  "}).status_code == 400
