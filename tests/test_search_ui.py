"""The search page: Videos and Photos as two lists that are always there, and every match reachable."""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from broll.db.models import Shot, Source
from broll.db.store import Store, new_id
from broll.web.app import create_app


def seed(workspace, videos: int = 0, photos: int = 0, caption="Adam speaks into a microphone while presenting."):
    store = Store.for_config(workspace)
    try:
        for kind, count in (("video", videos), ("image", photos)):
            for i in range(count):
                source = store.insert_source(Source(
                    id=new_id(), workspace_id=workspace.id, content_hash=f"{kind}{i}",
                    original_filename=f"{kind}{i}.{'mov' if kind == 'video' else 'jpg'}", origin="local",
                    media_kind=kind))
                store.insert_shot(Shot(id=f"{source.id}-0", workspace_id=workspace.id, source_id=source.id,
                                       caption=f"{caption} {kind} {i}", subjects=["man", "microphone"],
                                       action="speaking", status="indexed", duration_s=0 if kind == "image" else 5))
    finally:
        store.close()


def page(workspace, **params):
    client = TestClient(create_app(workspace, run_worker=False))
    return client.get("/search", params=params).text


def tab(html, label):
    match = re.search(rf"{label}\s*<span[^>]*>(\d+\+?)</span>", html)
    return match.group(1) if match else None


def test_videos_and_photos_are_two_lists_with_their_own_counts(workspace):
    seed(workspace, videos=3, photos=2)
    html = page(workspace, q="presenting")
    assert tab(html, "Videos") == "3" and tab(html, "Photos") == "2"


def test_both_lists_are_there_even_when_one_is_empty(workspace):
    seed(workspace, videos=3, photos=0)
    html = page(workspace, q="presenting")
    assert tab(html, "Videos") == "3" and tab(html, "Photos") == "0"
    photos = page(workspace, q="presenting", view="image")
    assert "No photos match." in photos and "video0" not in photos


def test_the_photos_list_shows_only_photos_and_the_videos_list_only_videos(workspace):
    seed(workspace, videos=2, photos=2)
    videos = page(workspace, q="presenting", view="video")
    photos = page(workspace, q="presenting", view="image")
    assert "video0.mov" in videos and "image0.jpg" not in videos
    assert "image0.jpg" in photos and "video0.mov" not in photos


def test_with_only_photos_the_page_opens_on_them(workspace):
    seed(workspace, videos=0, photos=2)
    html = page(workspace, q="presenting")
    assert "image0.jpg" in html and tab(html, "Videos") == "0"


def test_an_empty_search_still_shows_both_lists(workspace):
    html = page(workspace, q="")
    assert tab(html, "Videos") == "0" and tab(html, "Photos") == "0"


def test_every_match_can_be_reached_a_page_at_a_time(workspace):
    seed(workspace, videos=130)
    first = page(workspace, q="presenting")
    assert first.count("<article") == 48 and tab(first, "Videos") == "48+" and "Show 48 more" in first
    second = page(workspace, q="presenting", view="video", limit=96)
    assert second.count("<article") == 96 and "Show 48 more" in second
    third = page(workspace, q="presenting", view="video", limit=144)
    assert third.count("<article") == 130 and tab(third, "Videos") == "130" and "Show 48 more" not in third


def test_the_old_clips_and_photos_checkboxes_are_gone(workspace):
    seed(workspace, videos=1)
    assert 'name="media"' not in page(workspace, q="presenting")
