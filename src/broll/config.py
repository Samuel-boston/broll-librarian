"""Workspace configuration.

Non-secret settings live in ``config.yaml`` inside the workspace directory.
Secrets (API keys, OAuth tokens) come from the environment or a ``.env`` file
and are never written to the config file or the database.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

BROLL_HOME_ENV = "BROLL_HOME"
DEFAULT_HOME = Path.home() / ".broll"

# Which environment variable holds each provider's key. Deliberately the
# conventional names so an existing shell environment just works.
PROVIDER_KEY_ENV = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "mock": (),
}

DEFAULT_MODELS = {
    # 2.5-flash is closed to new API users (404 pointing at 3.6), so this is
    # the current floor for bulk indexing.
    "gemini": "gemini-3.6-flash",
    # Richest structured descriptions; see README for the cost trade-off.
    "anthropic": "claude-opus-5",
    "openai": "gpt-4.1-mini",
    "mock": "mock-1",
}


def broll_home() -> Path:
    return Path(os.environ.get(BROLL_HOME_ENV, DEFAULT_HOME)).expanduser()


def registry_path() -> Path:
    return broll_home() / "registry.db"


def load_env() -> None:
    """Load .env from the current directory and from BROLL_HOME, if present."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv is a hard dependency
        return
    load_dotenv(Path.cwd() / ".env", override=False)
    load_dotenv(broll_home() / ".env", override=False)


# --------------------------------------------------------------------------
# Config models
# --------------------------------------------------------------------------


class TaxonomyConfig(BaseModel):
    """Thresholds that stop the Drive tree exploding into one-item folders."""

    min_clips_for_subfolder: int = 5
    max_folders_per_level: int = 40
    other_folder_name: str = "Other"
    library_folder_name: str = "_Library"
    review_folder_name: str = "_Needs Review"
    filename_max_length: int = 100


class ProviderConfig(BaseModel):
    vision: str = "gemini"
    vision_model: str | None = None
    text: str | None = None  # defaults to the vision provider
    text_model: str | None = None

    def resolved_vision_model(self) -> str:
        return self.vision_model or DEFAULT_MODELS.get(self.vision, "")

    def resolved_text_provider(self) -> str:
        return self.text or self.vision

    def resolved_text_model(self) -> str:
        return self.text_model or DEFAULT_MODELS.get(self.resolved_text_provider(), "")


class EmbedderConfig(BaseModel):
    """Local by default so search quality does not depend on the vision provider."""

    kind: str = "local"  # local | provider
    model: str = "all-MiniLM-L6-v2"
    dimensions: int = 384


class IngestConfig(BaseModel):
    frames_per_shot: int = 3
    frame_max_edge: int = 768
    thumbnail_max_edge: int = 640
    min_shot_length_s: float = 1.5
    # If shot detection returns shots averaging under this, treat the file as
    # one shot: over-splitting short B-roll is worse than under-splitting.
    min_average_shot_length_s: float = 2.0
    concurrency: int = 4
    drive_concurrency: int = 2


class TranscriptConfig(BaseModel):
    words_per_minute: int = 150
    beat_min_s: float = 3.0
    beat_max_s: float = 15.0
    candidates_per_beat: int = 8
    suggestions_per_beat: int = 3
    sequence_fps: float | None = None  # None = modal fps of the suggested clips


class WorkspaceConfig(BaseModel):
    id: str
    name: str
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    taxonomy: TaxonomyConfig = Field(default_factory=TaxonomyConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    transcript: TranscriptConfig = Field(default_factory=TranscriptConfig)

    drive_root_folder_id: str | None = None
    drive_root_folder_name: str = "B-Roll"
    # Path to the Google Drive for Desktop mount, used to turn Drive file IDs
    # into local paths an NLE can actually link. See README.
    drive_local_mount_path: str | None = None

    # Operator additions to the controlled vocabularies, promoted from
    # vocabulary_candidates.
    vocabulary_overrides: dict[str, list[str]] = Field(default_factory=dict)

    # -- paths --------------------------------------------------------------

    @property
    def dir(self) -> Path:
        return workspace_dir(self.id)

    @property
    def db_path(self) -> Path:
        return self.dir / "library.db"

    @property
    def config_path(self) -> Path:
        return self.dir / "config.yaml"

    @property
    def temp_dir(self) -> Path:
        return self.dir / "tmp"

    @property
    def thumbnails_dir(self) -> Path:
        return self.dir / "thumbnails"

    @property
    def staging_dir(self) -> Path:
        return self.dir / "staging"

    @property
    def drive_token_path(self) -> Path:
        return self.dir / "drive_token.json"

    def ensure_dirs(self) -> None:
        for p in (self.dir, self.temp_dir, self.thumbnails_dir, self.staging_dir):
            p.mkdir(parents=True, exist_ok=True)

    # -- persistence --------------------------------------------------------

    def save(self) -> Path:
        self.ensure_dirs()
        data = self.model_dump(mode="json", exclude={"id"})
        self.config_path.write_text(yaml.safe_dump(data, sort_keys=False))
        return self.config_path

    def api_key(self, provider: str | None = None) -> str | None:
        provider = provider or self.provider.vision
        for var in PROVIDER_KEY_ENV.get(provider, ()):
            value = os.environ.get(var)
            if value:
                return value
        return None


def workspace_dir(workspace_id: str) -> Path:
    return broll_home() / "workspaces" / workspace_id


def load_workspace_config(workspace_id: str) -> WorkspaceConfig:
    path = workspace_dir(workspace_id) / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"No config for workspace {workspace_id!r} at {path}. "
            "Run `broll init` to create a workspace."
        )
    data: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    data["id"] = workspace_id
    return WorkspaceConfig.model_validate(data)


_SLUG = re.compile(r"[^a-z0-9]+")


def slugify_id(name: str) -> str:
    slug = _SLUG.sub("-", name.strip().lower()).strip("-")
    return slug or "workspace"


# --------------------------------------------------------------------------
# System dependency checks
# --------------------------------------------------------------------------


class DependencyError(RuntimeError):
    pass


FFMPEG_INSTALL_HINT = (
    "ffmpeg and ffprobe are required.\n"
    "  macOS:   brew install ffmpeg\n"
    "  Debian:  sudo apt install ffmpeg\n"
    "  Windows: winget install Gyan.FFmpeg\n"
)


def check_ffmpeg() -> tuple[str, str]:
    """Return (ffmpeg_path, ffprobe_path) or raise with an install message."""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        missing = ", ".join(n for n, p in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe)) if not p)
        raise DependencyError(f"Missing: {missing}.\n{FFMPEG_INSTALL_HINT}")
    return ffmpeg, ffprobe


def ffmpeg_version() -> str:
    ffmpeg, _ = check_ffmpeg()
    out = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True)
    return out.stdout.splitlines()[0] if out.stdout else "unknown"
