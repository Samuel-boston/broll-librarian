"""Downloading Drive-only sources into the workspace temp directory.

Probing, shot detection and frame extraction all need local bytes, so a source
that lives only in Drive is fetched before the pipeline runs. Implemented in M3.
"""

from __future__ import annotations

from pathlib import Path

from ..config import WorkspaceConfig


def fetch_drive_file(config: WorkspaceConfig, file_id: str, filename: str) -> Path:
    raise NotImplementedError(
        "Fetching from Drive arrives with M3. Until then, index from a local path."
    )
