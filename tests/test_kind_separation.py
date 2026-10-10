"""Videos and images are kept apart on every screen: pick a side first, and everything after is that side.

Also: the links on a page are relative. Behind the HTTPS proxy the app sees plain http, and a link built
from the request's own address pointed at http:// on an https:// page, which the browser blocks. That is
why clicking the Photos tab did nothing on the live site.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

from fastapi.testclient import TestClient

from broll import attention, kinds
from broll.db.models import Shot, Source
from broll.db.store import Store, new_id
from broll.ingest.scanner import DiscoveredFile
from broll.web.app import create_app, url_with
from tests.fakes import FakeDriveClient
from tests.test_client_tree import tree_config

FOLDER = "01_Nervous System Practices/Meditation & Stillness"


def _client(workspace) -> TestClient:
    return TestClient(create_app(workspace, run_worker=False))


def add(store, workspace, name, kind, category=None, status="indexed", caption=None, emotions=("calm",), **fields):
    source = store.insert_source(Source(
        id=new_id(), workspace_id=workspace.id, content_hash=name, original_filename=name,
        origin="local", media_kind=kind))
    shot = Shot(id=f"{source.id}-0", workspace_id=workspace.id, source_id=source.id,
                caption=caption or f"A man sits calmly, {name}.", category=category, status=status,
                emotions=list(emotions), duration_s=0 if kind == "image" else 6, **fields)
    store.insert_shot(shot)
    return shot.id


def two_sides(workspace, videos=3, images=2, status="indexed"):
    workspace.taxonomy = tree_config()
    workspace.save()
    store = Store.for_config(workspace)
    try:
        ids = {"video": [], "image": []}
        for i in range(videos):
            ids["video"].append(add(store, workspace, f"clip{i}.mov", "video", FOLDER, status))
        for i in range(images):
            ids["image"].append(add(store, workspace, f"still{i}.jpg", "image", FOLDER, status))
    finally:
        store.close()
    return ids


# ---- links are relative -----------------------------------------------------------------------


def test_a_link_to_this_page_with_one_change_is_relative_and_keeps_the_rest():
    request = SimpleNamespace(
        url=SimpleNamespace(path="/search"),
        query_params=SimpleNamespace(multi_items=lambda: [("q", "work"), ("mood", "a"), ("mood", "b"), ("limit", "96")]),
    )
    link = url_with(request, view="image", limit=None)
    assert link.startswith("/search?") and "http" not in link
    assert "q=work" in link and "mood=a" in link and "mood=b" in link and "view=image" in link
    assert "limit" not in link


def test_no_page_builds_an_absolute_link_to_itself(workspace):
    """The test client reports http://testserver. If any link were built from the request address it would show."""
    two_sides(workspace)
    store = Store.for_config(workspace)
    try:
        for n in range(60):  # enough clips for a "Show more" link
            add(store, workspace, f"more{n}.mov", "video", FOLDER, caption=f"A man presenting {n}.")
        shot = add(store, workspace, "flag.mov", "video", FOLDER, "needs_review")
        store.record_folder_proposal("01_Nervous System Practices/New", None, shot)
    finally:
        store.close()
    client = _client(workspace)
    pages = [
        "/library", "/library?kind=video", "/library?kind=image", f"/library?kind=video&path={FOLDER}",
        "/library?kind=video&view=unsorted", "/search?q=presenting", "/search?q=presenting&view=image",
        "/search?q=presentng", "/review", "/review?kind=image", "/attention", "/ingest", "/transcript",
    ]  # not /settings: it shows the Google redirect address on purpose, built from the forwarded scheme
    for page in pages:
        html = client.get(page).text
        assert "http://testserver" not in html, page


def test_the_search_tabs_are_plain_links_to_the_other_side_and_keep_the_query(workspace):
    two_sides(workspace)
    html = _client(workspace).get("/search", params={"q": "calmly", "view": "video"}).text
    tabs = re.findall(r'<a role="tab"[^>]*href="([^"]+)"', html)
    assert len(tabs) == 2 and all(t.startswith("/search?") for t in tabs)
    image_tab = next(t for t in tabs if "view=image" in t)
    assert "q=calmly" in image_tab
    # Plain links: the whole page loads, so the filters on the left change with the side.
    assert not re.search(r'role="tab"[^>]*hx-get', html)


# ---- the home page and each side -------------------------------------------------------------


def test_the_home_page_is_two_doors_and_nothing_else(workspace):
    two_sides(workspace)
    html = _client(workspace).get("/library").text
    assert 'href="/library?kind=video"' in html and 'href="/library?kind=image"' in html
    assert "Videos" in html and "Images" in html
    for gone in ("Delete all", "Recently added", "Browse by feeling", "Top Picks", 'name="q"', "Folders"):
        assert gone not in html, gone
    assert "<article" not in html  # no clips on the front door


def test_each_door_shows_only_its_own_kind_in_the_same_folders(workspace):
    ids = two_sides(workspace, videos=3, images=2)
    client = _client(workspace)
    videos = client.get("/library", params={"kind": "video", "path": FOLDER}).text
    images = client.get("/library", params={"kind": "image", "path": FOLDER}).text
    assert videos.count("<article") == 3 and images.count("<article") == 2
    assert "clip0.mov" in videos and "still0.jpg" not in videos
    assert "still0.jpg" in images and "clip0.mov" not in images
    # The folders are the same tree, counted for one side at a time.
    video_front = client.get("/library", params={"kind": "video"}).text
    image_front = client.get("/library", params={"kind": "image"}).text
    assert re.search(r"Nervous System Practices</span>\s*<span[^>]*>3", video_front)
    assert re.search(r"Nervous System Practices</span>\s*<span[^>]*>2", image_front)
    assert set(ids) == {"video", "image"}


def test_the_home_tiles_count_each_side(workspace):
    two_sides(workspace, videos=3, images=2)
    html = _client(workspace).get("/library").text
    assert re.search(r"Videos</span>\s*<span[^>]*>3<", html) and re.search(r"Images</span>\s*<span[^>]*>2<", html)


def test_a_side_offers_the_way_across_to_the_same_folder_with_its_count(workspace):
    two_sides(workspace, videos=3, images=2)
    html = _client(workspace).get("/library", params={"kind": "video", "path": FOLDER}).text
    across = re.search(r'<a role="tab"[^>]*href="([^"]+)"[^>]*>\s*Images\s*<span[^>]*>(\d+)<', html)
    assert across and across.group(2) == "2"
    assert across.group(1).startswith("/library?") and "kind=image" in across.group(1) and "path=" in across.group(1)
    assert "http" not in across.group(1)


def test_a_side_searches_that_side(workspace):
    two_sides(workspace)
    html = _client(workspace).get("/library", params={"kind": "image"}).text
    assert '<input type="hidden" name="view" value="image">' in html


def test_old_links_land_on_a_side_not_on_a_mixed_list(workspace):
    two_sides(workspace)
    client = _client(workspace)
    old = client.get("/library", params={"path": FOLDER}, follow_redirects=False)
    assert old.status_code == 303 and old.headers["location"].startswith("/library?")
    assert "kind=video" in old.headers["location"] and "path=" in old.headers["location"]
    photos = client.get("/library", params={"view": "photos"}, follow_redirects=False)
    assert photos.headers["location"] == "/library?kind=image"
    assert client.get("/library", params={"path": FOLDER}).status_code == 200  # and the redirect is followed


def test_the_quick_views_are_scoped_to_the_side(workspace):
    ids = two_sides(workspace, videos=2, images=2)
    store = Store.for_config(workspace)
    try:
        store.set_shot_fields(ids["video"][0], top_pick=1)
        store.set_shot_fields(ids["image"][0], top_pick=1)
    finally:
        store.close()
    client = _client(workspace)
    picks = client.get("/library", params={"kind": "image", "view": "top-picks"}).text
    assert picks.count("<article") == 1 and "still0.jpg" in picks and "clip0.mov" not in picks


# ---- search ----------------------------------------------------------------------------------


def test_the_filters_on_the_left_describe_the_side_being_looked_at(workspace):
    workspace.taxonomy = tree_config()
    workspace.save()
    store = Store.for_config(workspace)
    try:
        add(store, workspace, "a.mov", "video", FOLDER, caption="A man presenting.", emotions=["calm"])
        add(store, workspace, "b.jpg", "image", FOLDER, caption="A man presenting.", emotions=["joyful"])
    finally:
        store.close()
    client = _client(workspace)
    videos = client.get("/search", params={"q": "presenting", "view": "video"}).text
    images = client.get("/search", params={"q": "presenting", "view": "image"}).text
    assert 'value="calm"' in videos and 'value="joyful"' not in videos
    assert 'value="joyful"' in images and 'value="calm"' not in images


def test_the_store_counts_one_side_at_a_time(workspace):
    ids = two_sides(workspace, videos=3, images=2)
    store = Store.for_config(workspace)
    try:
        assert store.count_shots(media_kind="video") == 3 and store.count_shots(media_kind="image") == 2
        assert store.count_shots() == 5
        assert store.facet_counts("video")[("emotions", "calm")] == 3
        assert store.facet_counts("image")[("emotions", "calm")] == 2
        assert store.facet_counts()[("emotions", "calm")] == 5
        assert {r["media_kind"] for r in store.browse_rows("image")} == {"image"}
        assert len(store.browse_rows()) == 5
        assert store.shot_kinds([ids["video"][0], ids["image"][0], "nope"]) == {
            ids["video"][0]: "video", ids["image"][0]: "image"}
    finally:
        store.close()


# ---- review ----------------------------------------------------------------------------------


def test_review_keeps_videos_and_images_apart(workspace):
    two_sides(workspace, videos=2, images=1, status="needs_review")
    client = _client(workspace)
    videos = client.get("/review", params={"kind": "video"}).text
    images = client.get("/review", params={"kind": "image"}).text
    assert "clip0.mov" in videos and "still0.jpg" not in videos
    assert "still0.jpg" in images and "clip0.mov" not in images
    assert re.search(r"Videos\s*<span[^>]*>2<", videos) and re.search(r"Images\s*<span[^>]*>1<", videos)
    # A correction is saved on the side it was made from, and the page stays there.
    shot = re.search(r'hx-post="/review/([^"?]+)\?kind=image"', images).group(1)
    saved = client.post(f"/review/{shot}?kind=image", data={"caption": "Fixed.", "status": "indexed"}).text
    assert "No images need review" in saved and 'href="/review?kind=video"' in saved


def test_the_review_page_opens_on_the_side_that_has_something_waiting(workspace):
    two_sides(workspace, videos=0, images=2, status="needs_review")
    html = _client(workspace).get("/review").text
    assert "still0.jpg" in html


def test_suggested_folders_are_listed_on_the_side_whose_clips_they_hold(workspace):
    ids = two_sides(workspace, videos=1, images=1, status="needs_review")
    store = Store.for_config(workspace)
    try:
        store.record_folder_proposal("01_Nervous System Practices/Hush", None, ids["video"][0])
        store.record_folder_proposal("01_Nervous System Practices/Portraits", None, ids["image"][0])
    finally:
        store.close()
    client = _client(workspace)
    videos = client.get("/review", params={"kind": "video"}).text
    images = client.get("/review", params={"kind": "image"}).text
    assert "Hush" in videos and "Portraits" not in videos
    assert "Portraits" in images and "Hush" not in images


# ---- needs attention -------------------------------------------------------------------------


def _flag(workspace, name, kind="too_long"):
    store = Store.for_config(workspace)
    try:
        return attention.flag_file(
            store, DiscoveredFile(origin="drive", path=None, filename=name, drive_file_id=f"id-{name}",
                                  origin_path=f"drive:id-{name}", link="https://drive/x"), kind, "why")
    finally:
        store.close()


def test_needs_attention_keeps_videos_and_images_apart(workspace):
    _flag(workspace, "podcast.mp4")
    _flag(workspace, "huge.nef", kind="unreadable")
    client = _client(workspace)
    videos = client.get("/attention", params={"kind": "video"}).text
    images = client.get("/attention", params={"kind": "image"}).text
    assert "podcast.mp4" in videos and "huge.nef" not in videos
    assert "huge.nef" in images and "podcast.mp4" not in images
    assert re.search(r"Videos\s*<span[^>]*>1<", videos) and re.search(r"Images\s*<span[^>]*>1<", videos)
    # The buttons keep to the side they were pressed on.
    item = attention_id(workspace, "huge.nef")
    assert f"/attention/{item}/dismiss?kind=image" in images
    after = client.post(f"/attention/{item}/dismiss", params={"kind": "image"}).text
    assert "huge.nef" not in after and "No images need attention" in after


def attention_id(workspace, name):
    store = Store.for_config(workspace)
    try:
        return next(i["id"] for i in store.list_attention("open") if i["filename"] == name)
    finally:
        store.close()


def test_the_api_says_which_side_each_listed_file_is_on(workspace):
    _flag(workspace, "podcast.mp4")
    _flag(workspace, "huge.nef", kind="unreadable")
    items = _client(workspace).get("/api/attention").json()["items"]
    assert {i["filename"]: i["media"] for i in items} == {"podcast.mp4": "video", "huge.nef": "image"}


def test_the_drive_mirror_keeps_videos_and_images_in_separate_folders(workspace, store):
    workspace.taxonomy = tree_config(media_split=True)
    client = FakeDriveClient()
    for name in ("a.mp4", "b.jpg"):
        entry = client.add_file(name, "root")
        attention.flag_file(
            store, DiscoveredFile(origin="drive", path=None, filename=name, drive_file_id=entry.id,
                                  origin_path=f"drive:{entry.id}"), "too_long", "why")
    assert attention.sync_drive_shortcuts(workspace, store, client) == 2
    tree = client.tree()
    assert any(p.endswith("_Needs Attention/Videos/Too long to be B-roll/a.mp4") for p in tree)
    assert any(p.endswith("_Needs Attention/Images/Too long to be B-roll/b.jpg") for p in tree)


# ---- the upload queue ------------------------------------------------------------------------


def test_the_queue_counts_videos_and_images_separately_and_tags_each_row(workspace):
    from broll.jobs.queue import enqueue_files

    store = Store.for_config(workspace)
    try:
        enqueue_files(store, [
            DiscoveredFile(origin="local", path=None, filename="a.mov", origin_path="/x/a.mov"),
            DiscoveredFile(origin="local", path=None, filename="b.mov", origin_path="/x/b.mov"),
            DiscoveredFile(origin="local", path=None, filename="c.jpg", origin_path="/x/c.jpg"),
        ])
    finally:
        store.close()
    html = _client(workspace).get("/ingest/queue").text
    queued = re.search(r"Queued</td>\s*<td[^>]*>(\d+)</td>\s*<td[^>]*>(\d+)</td>", html)
    assert queued and queued.groups() == ("2", "1")  # videos first, images second
    assert re.search(r'<span class="tag shrink-0">image</span>\s*<span[^>]*>c.jpg', html)
    assert re.search(r'<span class="tag shrink-0">video</span>\s*<span[^>]*>a.mov', html)


# ---- the script screen -----------------------------------------------------------------------


def test_the_script_screen_matches_one_side_at_a_time(workspace, monkeypatch):
    seen = []

    class Matcher:
        def __init__(self, config, engine, text_provider=None, filters=None):
            seen.append(filters.media_kind)

        async def match(self, beats):
            return []

    monkeypatch.setattr("broll.web.routes.transcript.TranscriptMatcher", Matcher)
    client = _client(workspace)
    assert 'name="media" value="image"' in client.get("/transcript").text
    for choice in ("image", "video", "nonsense"):
        client.post("/transcript", data={"text": "He walks along the beach.", "media": choice, "rerank": ""})
    assert seen == [["image"], ["video"], ["video"]]


# ---- the delete-everything control moved to Settings -----------------------------------------


def test_delete_all_lives_on_settings_not_on_the_library(workspace):
    client = _client(workspace)
    assert "Delete all" in client.get("/settings").text
    assert "Delete all" not in client.get("/library").text
    assert "Delete all" not in client.get("/library", params={"kind": "video"}).text


def test_the_nav_has_a_search_link(workspace):
    html = _client(workspace).get("/library").text
    assert 'href="/search"' in html


def test_helpers():
    assert kinds.clean("Video ") == "video" and kinds.clean("photos") == "" and kinds.clean(None) == ""
    assert kinds.other("video") == "image" and kinds.other("image") == "video"
    assert kinds.noun("image", 1) == "1 image" and kinds.noun("video", 0) == "0 videos"
    assert kinds.of_file("a.NEF") == "image" and kinds.of_file("a.mov") == "video" and kinds.of_file("") == "video"


def test_status_counts_videos_and_images_apart(workspace):
    two_sides(workspace, videos=3, images=2)
    body = _client(workspace).get("/api/status").json()
    assert (body["shots"], body["videos"], body["images"]) == (5, 3, 2)


def test_a_suggested_folder_can_be_renamed_moved_and_described_before_it_is_created(workspace):
    ids = two_sides(workspace, videos=1, images=0, status="needs_review")
    store = Store.for_config(workspace)
    try:
        store.record_folder_proposal("01_Nervous System Practices/Red Light", "Red light beds.", ids["video"][0])
        proposal = store.list_folder_proposals()[0]["id"]
    finally:
        store.close()
    client = _client(workspace)
    page = client.get("/review").text
    assert 'name="name" value="Red Light"' in page and "Red light beds." in page
    after = client.post(f"/review/folders/{proposal}/approve?kind=video", data={
        "name": "Red Light Therapy", "parent": "01_Nervous System Practices", "note": "Infrared panels."}).text
    assert "Created 01_Nervous System Practices/Red Light Therapy" in after
    store = Store.for_config(workspace)
    try:
        assert store.get_shot(ids["video"][0]).category == "01_Nervous System Practices/Red Light Therapy"
    finally:
        store.close()
    node = workspace.__class__.model_validate(workspace.model_dump()).taxonomy.find_node(
        "01_Nervous System Practices/Red Light Therapy")
