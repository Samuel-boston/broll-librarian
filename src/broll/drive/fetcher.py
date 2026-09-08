"""Downloading Drive-only sources into the workspace temp directory.

Probing, shot detection and frame extraction all need local bytes, so a source
that lives only in Drive is fetched first. Uploaded and local-path sources skip
this entirely. Fetched files land in the workspace temp directory, which is the
only place cleanup is ever allowed to delete from.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..config import WorkspaceConfig
from .auth import load_credentials
from .client import DriveClient

log = logging.getLogger(__name__)


def fetch_drive_file(
    config: WorkspaceConfig,
    file_id: str,
    filename: str,
    client: DriveClient | None = None,
) -> Path:
    if client is None:
        credentials = load_credentials(config)
        if credentials is None:
            raise RuntimeError(
                "Drive is not connected for this workspace. Run `broll drive login`."
            )
        client = DriveClient(credentials)

    config.ensure_dirs()
    destination = config.temp_dir / f"drive-{file_id}-{Path(filename).name}"
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    log.info("downloading %s from Drive", filename)
    return client.download(file_id, destination)
