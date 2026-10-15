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
                if exc.code == 403:
                    # Drive's daily download quota answers a plain 403 here; say so, so the worker waits
                    # for it to reset instead of marking the file as failed.
                    try:
                        body = exc.read(2000).decode("utf-8", "replace").lower()
                    except Exception:  # noqa: BLE001
                        body = ""
                    if "quota" in body:
                        raise RuntimeError(
                            "Drive's download quota for this account is used up (downloadQuotaExceeded)"
                        ) from exc
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 3:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == 3:
                    raise
            time.sleep(delay)
            delay *= 2
        raise RuntimeError("unreachable")  # pragma: no cover

    def quota_error(self) -> str | None:
        """Ask Drive for a megabyte, to find out whether it is refusing this account's downloads.

        ffmpeg says nothing useful when Drive answers 403, so a read that came back empty is checked here.
        Returns the quota message, or None when Drive answers (the clip is the problem, or nothing is).
        A megabyte, not a byte: measured on the real quota, a 1 KB read passes while a 512 KB one is refused.
        """
        try:
            self._range(0, CHUNK - 1)
        except RuntimeError as exc:
            return str(exc) if "quota" in str(exc).lower() else None
        except Exception:  # noqa: BLE001 - this is only a question; any other answer means "not the quota"
            return None
        return None

    def can_read(self, nbytes: int = 512 * 1024) -> bool | None:
        """Is Drive letting a real piece of this file through right now?

        True when it does, False when it refuses with the download quota, None when it cannot be told (the
        file is gone, the network is down). A piece of half a megabyte, because a 1 KB read passes even
        while the quota is refusing everything bigger (measured 15 Oct). It reads a range, never the file.
        """
        try:
            self._range(0, nbytes - 1)
        except RuntimeError as exc:
            return False if "quota" in str(exc).lower() else None
        except Exception:  # noqa: BLE001
            return None
        return True

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
