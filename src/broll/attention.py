"""Files the library did not index, and why - so none of them disappears quietly.

Anything the pipeline turns away (too long to be B-roll, too big for the disk) or cannot finish (a
download that stopped short, a file nothing can read) is listed here with a link to it. A person
can then open the file, dismiss it, or tell the library to index it anyway.

When Drive is connected the list is mirrored there too: a "_Needs Attention" folder with one
shortcut per file, grouped by reason. A shortcut points at the file where it already is. Nothing is
moved, renamed or deleted. Videos and images are listed apart, as everywhere else.
"""

from __future__ import annotations

import logging
from typing import Any

from . import kinds
from .config import WorkspaceConfig
from .db.store import Store
from .ingest.scanner import DiscoveredFile

log = logging.getLogger(__name__)

#: kind -> (heading, what to do about it)
KINDS: dict[str, tuple[str, str]] = {
    "too_long": (
        "Too long to be B-roll",
        "Probably a recording (a podcast, an event). Open it to check. If parts of it are usable, "
        "choose \"Index anyway\" and the library will pick out the usable stretches.",
    ),
    "too_big": (
        "Too big to process",
        "The file is larger than this server can handle. Shorten it, or index it on a bigger machine.",
    ),
    "download_incomplete": (
        "Download didn't finish",
        "Google Drive stopped part way through. It is tried again automatically; if it keeps "
        "failing, open the file to check it plays.",
    ),
    "unreadable": (
        "Couldn't be read",
        "The file is damaged or in a format that can't be opened. Open it in Drive to check.",
    ),
    "analysis_failed": (
        "Couldn't be analysed",
        "The model could not describe it after several tries. Use \"Try again\", or open it to check.",
    ),
    "unsupported": (
        "Unsupported file type",
        "This library can't read this kind of file.",
    ),
}

#: Where in Drive the mirrored shortcuts go.
DRIVE_FOLDER = "_Needs Attention"


def heading(kind: str) -> str:
    return KINDS.get(kind, (kind.replace("_", " ").capitalize(), ""))[0]


def advice(kind: str) -> str:
    return KINDS.get(kind, ("", ""))[1]


def key_for(discovered: DiscoveredFile) -> str:
    return discovered.origin_path or (
        f"drive:{discovered.drive_file_id}" if discovered.drive_file_id else discovered.filename
    )


def flag_file(
    store: Store,
    discovered: DiscoveredFile,
    kind: str,
    detail: str,
    *,
    size_bytes: int | None = None,
    duration_s: float | None = None,
) -> int:
    return store.flag_attention(
        kind=kind,
        key=key_for(discovered),
        filename=discovered.filename,
        origin=discovered.origin,
        drive_file_id=discovered.drive_file_id,
        origin_path=discovered.origin_path,
        link=discovered.link,
        size_bytes=size_bytes if size_bytes is not None else discovered.size_bytes,
        duration_s=duration_s if duration_s is not None else discovered.duration_s,
        detail=detail,
    )


def classify_failure(error: str) -> str:
    """Which list a permanently failed file belongs on, from its last error."""
    lowered = (error or "").lower()
    if "free disk" in lowered:
        return "too_big"
    if "download" in lowered and ("stopped" in lowered or "empty" in lowered or "incomplete" in lowered):
        return "download_incomplete"
    if any(m in lowered for m in ("ffprobe could not read", "contains no image or video", "could not decode",
                                  "could not read", "moov atom", "invalid data")):
        return "unreadable"
    return "analysis_failed"


def requeue(config: WorkspaceConfig, store: Store, item_id: int, forced: bool = True) -> bool:
    """Queue a flagged file for indexing again. `forced` lifts the length limit for it."""
    from .jobs.queue import KIND_INDEX_SOURCE

    item = store.get_attention(item_id)
    if item is None:
        return False
    key = item["origin_path"] or item["key"]
    already = store.conn.execute(
        """SELECT 1 FROM jobs WHERE workspace_id = ? AND kind = ? AND status IN ('queued', 'running')
           AND COALESCE(json_extract(payload_json, '$.origin_path'), json_extract(payload_json, '$.filename')) = ?""",
        (store.workspace_id, KIND_INDEX_SOURCE, key),
    ).fetchone()
    if already:
        store.set_attention(item_id, status="requeued")  # it is on its way; a second click adds nothing
        return True
    payload: dict[str, Any] = {
        "origin": item["origin"],
        "path": item["origin_path"] if item["origin"] != "drive" else None,
        "filename": item["filename"],
        "drive_file_id": item["drive_file_id"],
        "origin_path": item["origin_path"],
        "size_bytes": item["size_bytes"],
        "duration_s": item["duration_s"],
        "link": item["link"],
        "force": False,
        "overwrite_corrections": False,
        "allow_long": bool(forced),
    }
    store.enqueue(KIND_INDEX_SOURCE, payload)
    store.set_attention(item_id, status="requeued")
    return True


def dismiss(store: Store, item_id: int, config: WorkspaceConfig | None = None) -> bool:
    """Say a listed file is not wanted: it is never queued again.

    An upload that was turned away is sitting in the server's staging folder, and nothing else would
    ever free it. Dismissing it deletes that copy (the original is on whoever uploaded it). A file the
    library only looked at in place, or in Drive, is never touched.
    """
    item = store.get_attention(item_id)
    if item is None:
        return False
    store.set_attention(item_id, status="dismissed")
    if config is not None and item["origin"] == "upload" and item["origin_path"]:
        from .library_admin import _unlink_inside

        _unlink_inside(item["origin_path"], config.staging_dir)
    return True


# --------------------------------------------------------------------------
# Mirroring the list into Drive
# --------------------------------------------------------------------------


def sync_drive_shortcuts(config: WorkspaceConfig, store: Store, client) -> int:
    """Keep `_Needs Attention` in Drive in step with the list. Returns how many shortcuts were added.

    * a shortcut is added for each open Drive file that has none (reusing one that is already there, so
      running it twice, or after the library was cleared and rescanned, never doubles up);
    * a shortcut is removed once its file is indexed or dismissed (the file itself is never touched:
      only shortcuts are ever deleted);
    * one file that cannot be shortcut does not stop the rest.

    Best effort: the caller treats a failure as a courtesy lost, not a reason to stop indexing.
    """
    from .drive.organizer import OrganiseReport, Organizer

    for item in store.list_attention(None):
        if item["shortcut_id"] and item["status"] in ("resolved", "dismissed", "requeued"):
            try:
                client.delete_shortcut(item["shortcut_id"])
            except Exception as exc:  # noqa: BLE001 - gone already, or not ours to remove: leave it
                log.warning("could not remove the shortcut for %s: %s", item["filename"], exc)
            store.set_attention(item["id"], shortcut_id=None)

    pending = [i for i in store.list_attention("open") if i["drive_file_id"] and not i["shortcut_id"]]
    if not pending:
        return 0
    organizer = Organizer(config, store, client)
    report = OrganiseReport()
    created = 0
    for item in pending:
        try:
            # Videos and images are kept apart here too: _Needs Attention/Videos/<reason>, .../Images/<reason>.
            side = config.taxonomy.media_prefix(kinds.of_file(item["filename"]))
            folder = organizer._folder_id((DRIVE_FOLDER, *side, heading(item["kind"])), report)
            existing = next(
                (e for e in client.list_all(folder)
                 if e.is_shortcut and e.shortcut_target_id == item["drive_file_id"]),
                None,
            )
            shortcut = existing or client.create_shortcut(item["drive_file_id"], item["filename"], folder)
            store.set_attention(item["id"], shortcut_id=shortcut.id)
            created += 0 if existing else 1
        except Exception as exc:  # noqa: BLE001
            log.warning("could not add a shortcut for %s: %s", item["filename"], exc)
    return created
