from __future__ import annotations

import os
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"
CLIP_DIR = FIXTURE_DIR / "clips"


@pytest.fixture()
def broll_home(tmp_path, monkeypatch) -> Path:
    """Isolate ~/.broll for every test."""
    home = tmp_path / "broll-home"
    monkeypatch.setenv("BROLL_HOME", str(home))
    return home


@pytest.fixture()
def workspace(broll_home):
    from broll.config import WorkspaceConfig

    config = WorkspaceConfig(id="test", name="Test Workspace")
    config.provider.vision = "mock"
    config.save()
    config.ensure_dirs()
    return config


@pytest.fixture()
def store(workspace):
    from broll.db.store import Store

    s = Store.for_config(workspace)
    yield s
    s.close()


@pytest.fixture()
def clips() -> dict[str, Path]:
    return {p.stem: p for p in sorted(CLIP_DIR.glob("*.mp4"))}
