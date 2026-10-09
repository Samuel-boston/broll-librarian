"""Reading a video straight out of Google Drive, without downloading it.

A 30 GB camera file does not fit a small server, and downloading it to pick out a dozen frames would be
absurd. ffmpeg can open an HTTPS address and ask only for the bytes it needs (the index at the end of
the file, then the little run of data around each frame), so that is what is done for large files: Drive
is asked for a few megabytes per frame instead of the whole file.

`RemoteDriveVideo` is what the rest of the pipeline sees where it would otherwise see a local path.
"""

from __future__ import annotations

import hashlib
import time
import urllib.error
import urllib.request

CHUNK = 1024 * 1024
API = "https://www.googleapis.com/drive/v3/files/{id}?alt=media&supportsAllDrives=true"


class RemoteDriveVideo:
    """A video that lives in Drive. Anything that opens a video with ffmpeg can open this."""

    is_remote = True

    def __init__(self, client, file_id: str, name: str, size: int | None = None, url: str | None = None):
        self.client = client
        self.file_id = file_id
        self.name = name
        self.size = size
        # Overridable so a test can point it at a local server.
        self.url = url or API.format(id=file_id)

    def __str__(self) -> str:
        return f"drive:{self.file_id}"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.client.access_token()}"}

    def ffmpeg_input(self) -> list[str]:
        """The ffmpeg arguments that open it. The token is fetched fresh each time: a long run outlives one."""
        return [
            "-headers", f"Authorization: Bearer {self.client.access_token()}\r\n",
            "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
            "-rw_timeout", "30000000",   # give up on a connection that has gone quiet for 30 s
            "-i", self.url,
        ]

    def _range(self, first: int, last: int) -> bytes:
        """Bytes `first`..`last` of the file. Never more than asked for, and retried when Drive hiccups."""
        want = last - first + 1
        delay = 1.0
        for attempt in range(4):
            try:
                request = urllib.request.Request(
                    self.url, headers={**self._headers(), "Range": f"bytes={first}-{last}"}
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status == 200 and first > 0:
                        # The server ignored the Range header and is sending the whole file.
                        raise RuntimeError("the server ignored a byte-range request")
                    return response.read(want + 1)[:want]
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 3:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == 3:
                    raise
            time.sleep(delay)
            delay *= 2
        raise RuntimeError("unreachable")  # pragma: no cover

    def content_hash(self) -> str:
        """The same fingerprint `ingest.hashing.content_hash` gives the downloaded file: its size plus its
        first and last megabyte. So a file is recognised whether it was streamed or fetched."""
        if not self.size:
            raise ValueError("the size of a remote file is needed to fingerprint it")
        digest = hashlib.sha256()
        digest.update(str(self.size).encode())
        digest.update(self._range(0, min(CHUNK, self.size) - 1))
        if self.size > CHUNK * 2:
            digest.update(self._range(self.size - CHUNK, self.size - 1))
        return digest.hexdigest()
