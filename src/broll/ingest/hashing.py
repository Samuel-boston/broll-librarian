"""Content hashing for dedupe.

SHA256 of (filesize + first 1MB + last 1MB). Fast on large files and good
enough: two different videos colliding on all three would be remarkable.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

CHUNK = 1024 * 1024


def content_hash(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode())
    with path.open("rb") as fh:
        digest.update(fh.read(CHUNK))
        if size > CHUNK * 2:
            fh.seek(-CHUNK, 2)
            digest.update(fh.read(CHUNK))
    return digest.hexdigest()


def short_hash(content_hash_value: str) -> str:
    """The 8-character suffix used in Drive filenames."""
    return content_hash_value[:8]
