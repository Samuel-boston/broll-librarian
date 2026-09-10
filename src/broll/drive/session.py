"""One Drive session shared by every clip filed in a run.

Filing a clip needs a Drive client and the ids of the folders it goes into.
Doing that from scratch for each clip meant a new client and a fresh walk of
the client's whole folder tree every time - 53 folders for Adam, so ~160 Drive
calls to file three clips, and 464 seconds. The session keeps the client and
the folder-id map across clips.

It lives as long as the web server, which can be days, so the caches expire:
after ``ttl_s`` the session starts clean and re-checks Drive. A lock serialises
use, because clips are filed from worker threads and httplib2 is not
thread-safe.
"""

from __future__ import annotations

import threading
import time

from ..config import WorkspaceConfig


class DriveSession:
    def __init__(self, config: WorkspaceConfig, ttl_s: float = 600.0):
        self.config = config
        self.ttl_s = ttl_s
        self._lock = threading.Lock()
        self._client = None
        self._started = time.monotonic()
        # Shared with each Organizer so one clip's lookups help the next.
        self.folder_ids: dict[str, str] = {}
        self.root_id: str | None = None
        self.tree_ready = False

    def _expire_if_stale(self) -> None:
        if time.monotonic() - self._started > self.ttl_s:
            self._client = None
            self.folder_ids.clear()
            self.root_id = None
            self.tree_ready = False
            self._started = time.monotonic()

    def client(self):
        if self._client is None:
            from .auth import load_credentials
            from .client import DriveClient

            credentials = load_credentials(self.config)
            if credentials is None:
                raise RuntimeError("Drive is not connected. Run `broll drive login`.")
            self._client = DriveClient(credentials)
        return self._client

    def organise(self, source_id: str):
        """File one source. Safe to call from any thread."""
        from ..db.store import Store
        from .organizer import Organizer

        with self._lock:
            self._expire_if_stale()
            # A sqlite3 connection may not cross threads, so each call opens
            # its own; only the Drive-side state is shared.
            store = Store.for_config(self.config)
            try:
                organizer = Organizer(self.config, store, self.client(), session=self)
                report = organizer.organise_source(source_id)
            finally:
                store.close()
        if report.errors:
            raise RuntimeError("; ".join(report.errors))
        return report
