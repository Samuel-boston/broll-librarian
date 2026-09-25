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


def write_env_var(name: str, value: str) -> Path:
    """Set NAME=value in BROLL_HOME/.env (0600) and in this process."""
    if any(c in value for c in "\r\n\x00") or any(c in name for c in "\r\n=\x00 "):
        raise ValueError("Values can't contain line breaks.")
    path = broll_home() / ".env"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text().splitlines() if path.exists() else []
    for index, line in enumerate(lines):
        if line.split("=", 1)[0].strip() == name:
            lines[index] = f"{name}={value}"
            break
    else:
        lines.append(f"{name}={value}")
    # Written owner-only from the start and swapped in whole, so a crash can't truncate the
    # file that holds every key, and it is never readable by others, even for a moment.
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(tmp, path)
    os.environ[name] = value
    return path


# --------------------------------------------------------------------------
# Config models
# --------------------------------------------------------------------------


class CategoryNode(BaseModel):
    """One folder in a client's own Drive structure."""

    name: str
    description: str = ""
    children: list["CategoryNode"] = Field(default_factory=list)
    # Whether clips can be filed here. Unset means "only if it has no
    # subfolders". False for a guide folder or a shortcuts-only Top Picks;
    # forced True when a subfolder is added to what used to be a leaf, so
    # clips already filed there stay put.
    destination: bool | None = None

    def accepts_clips(self) -> bool:
        return self.destination if self.destination is not None else not self.children

    def walk(self, prefix: tuple[str, ...] = ()):
        path = (*prefix, self.name)
        yield path, self
        for child in self.children:
            yield from child.walk(path)


CategoryNode.model_rebuild()


class TaxonomyConfig(BaseModel):
    """Thresholds that stop the Drive tree exploding into one-item folders."""

    min_clips_for_subfolder: int = 5
    max_folders_per_level: int = 40
    # Subjects, moods and uses are ranked most-important-first by the model.
    # Filing a clip under all of them makes a thicket of near-empty folders;
    # search still sees every value regardless.
    max_list_values_per_shot: int = 2
    other_folder_name: str = "Other"
    library_folder_name: str = "_Library"
    review_folder_name: str = "_Needs Review"
    filename_max_length: int = 100

    # "faceted" builds the generic By Subject / By Mood / ... tree. "tree" uses a
    # client's own folder structure: each file lives in its best-fit folder,
    # with shortcuts wherever else it belongs.
    mode: str = "faceted"
    tree: list[CategoryNode] = Field(default_factory=list)
    top_picks_folder: str | None = "\u2605 Top Picks"
    guide_folder: str | None = None
    unsorted_folder: str = "_Unsorted"
    # Tokens: {setting} {action} {time_of_day} {shot_type} {leaf} {emotion}
    # {mood} {subject}. The 8-character content hash is always appended.
    filename_template: str = "{setting}_{action}_{time_of_day}_{shot_type}"
    # Let the model add a folder when nothing in the tree genuinely fits.
    allow_new_folders: bool = True
    # Split the whole tree by media at the top level, so photographs and clips
    # never sit in the same folder: Videos/<tree> and Images/<tree>. The tree
    # itself is written once and used for both.
    media_split: bool = False
    video_folder_name: str = "Videos"
    image_folder_name: str = "Images"

    def media_prefix(self, media_kind: str) -> tuple[str, ...]:
        """The folder a source of this kind hangs under. Empty when not split."""
        if not self.media_split:
            return ()
        name = self.image_folder_name if media_kind == "image" else self.video_folder_name
        return (name,)

    def media_prefixes(self) -> list[tuple[str, ...]]:
        """Every prefix the tree has to exist under."""
        if not self.media_split:
            return [()]
        return [(self.video_folder_name,), (self.image_folder_name,)]

    def tree_folders(self) -> list[tuple[str, ...]]:
        return [path for node in self.tree for path, _ in node.walk()]

    def category_leaves(self) -> list[tuple[str, str]]:
        """(path, note) for every folder a clip can be filed in.

        A folder's note carries its parents' notes too, so a rule written on a
        parent ("only for shots with no clear activity") reaches the model.
        """
        leaves: list[tuple[str, str]] = []

        def visit(node: CategoryNode, prefix: tuple[str, ...], inherited: list[str]) -> None:
            path = (*prefix, node.name)
            notes = inherited + ([f"{node.name}: {node.description}"] if node.children and node.description else [])
            if node.accepts_clips():
                own = (node.description if not node.children else "").rstrip(" .")
                note = " ".join(([own + "."] if own else []) + [f"({n})" for n in inherited])
                leaves.append(("/".join(path), note))
            for child in node.children:
                visit(child, path, notes)

        for node in self.tree:
            visit(node, (), [])
        return leaves

    def parent_folders(self) -> list[str]:
        """Folders a new subfolder may be created under (not Top Picks or a guide)."""
        return [
            "/".join(path) for node in self.tree for path, item in node.walk()
            if item.destination is not False
        ]

    def find_node(self, path: str) -> CategoryNode | None:
        nodes, node = self.tree, None
        for part in [p for p in path.split("/") if p]:
            node = next((n for n in nodes if n.name == part), None)
            if node is None:
                return None
            nodes = node.children
        return node

    def add_folder(self, parent_path: str, name: str, note: str = "") -> str:
        """Add a folder under an existing one and return its path."""
        parent = self.find_node(parent_path)
        if parent is None:
            raise KeyError(parent_path)
        if parent.destination is None and not parent.children:
            parent.destination = True  # it was a leaf: keep what is filed there
        parent.children.append(CategoryNode(name=name, description=note))
        return f"{parent_path}/{name}"


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
    # Cap on vision requests per minute, 0 for no cap. Free Gemini keys allow
    # about 5 a minute; without a cap the workers are answered with 429s and
    # 503s that look like the model failing.
    requests_per_minute: float = 0.0
    # Send a shot to the review queue when the model's own confidence is below
    # this. Clean footage comes back at 0.95+, so anything under this really is
    # the model hedging - and a wrong caption is worse than a minute of review.
    review_below_confidence: float = 0.7
    # File each source into Drive as soon as it is indexed, when Drive is
    # connected. This has to happen inside the pipeline: an uploaded file's
    # only copy is in staging, and cleanup would otherwise delete it first.
    auto_organise: bool = True


class TranscriptConfig(BaseModel):
    words_per_minute: int = 150
    beat_min_s: float = 3.0
    beat_max_s: float = 15.0
    candidates_per_beat: int = 8
    suggestions_per_beat: int = 3
    sequence_fps: float | None = None  # None = modal fps of the suggested clips


class ClientProfile(BaseModel):
    """Who this library belongs to. Shapes captions, tags and emotions."""

    name: str | None = None
    # The recurring person to name in captions when they are clearly the
    # subject. An instruction to the model, not face recognition: it names
    # whoever is the lone featured person, so the review queue matters.
    featured_person: str | None = None
    # Helps the model tell the featured person from others, e.g. "a man".
    featured_person_description: str | None = None
    brief: str = ""
    themes: list[str] = Field(default_factory=list)
    # Replaces the default emotion vocabulary when set.
    emotions: list[str] = Field(default_factory=list)


class DashboardConfig(BaseModel):
    """Optional link to the Content Ops dashboard's Supabase project.

    The service key is a secret, so it lives in the environment / .env
    (DASHBOARD_SUPABASE_KEY), never here.
    """

    enabled: bool = False
    supabase_url: str | None = None
    # How often the running app pushes changes, in seconds.
    interval_s: int = 60


class WorkspaceConfig(BaseModel):
    id: str
    name: str
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    client: ClientProfile = Field(default_factory=ClientProfile)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    taxonomy: TaxonomyConfig = Field(default_factory=TaxonomyConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    transcript: TranscriptConfig = Field(default_factory=TranscriptConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)

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
