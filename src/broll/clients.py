"""Creating a client library - for `broll init` and for an app that creates
one whenever it gains a client.

Every client gets a library, whether or not it ever holds footage: an empty
library is a normal state (search finds nothing, a shortlist says the library
is empty, and the editing agent uses each beat's fallback), and it is ready the
day the first clip is dropped in.
"""

from __future__ import annotations

import re

from .config import WorkspaceConfig, slugify_id
from .db.models import Workspace
from .db.store import Registry, Store


def _normalised(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip()).lower()


def register(config: WorkspaceConfig, registry: Registry) -> None:
    """Write a new workspace's config and database, and list it in the registry."""
    config.save()
    Store.for_config(config).close()  # creates and migrates library.db
    registry.create(
        Workspace(
            id=config.id,
            name=config.name,
            provider=config.provider.vision,
            db_path=str(config.db_path),
            drive_root_folder_id=config.drive_root_folder_id,
        )
    )


def find_library(registry: Registry, name: str, workspace_id: str | None = None) -> Workspace | None:
    """The existing library for this client: by id if given, else by name.

    Matching by name (ignoring case and spacing) is what makes creation
    idempotent for an app that only knows its client's name: asking twice, or
    asking for a client whose library was made by hand under another id, never
    makes a second library for the same client.
    """
    if workspace_id:
        return registry.get(workspace_id)
    wanted = _normalised(name)
    for workspace in registry.list():
        if _normalised(workspace.name) == wanted:
            return workspace
    return registry.get(slugify_id(name))


def create_library(
    name: str,
    workspace_id: str | None = None,
    like: WorkspaceConfig | None = None,
    registry: Registry | None = None,
) -> tuple[Workspace, bool]:
    """(the library, created). Returns the existing one untouched if there is one.

    ``like`` is another library whose provider, embedder and request-rate cap
    the new one copies: libraries on one server share one API key and one
    loaded embedding model, so a new library must use the same ones, and a
    free key's rate cap has to hold for it too. Nothing else is copied - the
    client profile and folder tree are the new client's own.
    """
    name = re.sub(r"\s+", " ", (name or "").strip())
    if not name:
        raise ValueError("A client library needs a name.")
    own = registry is None
    registry = registry or Registry()
    try:
        existing = find_library(registry, name, workspace_id)
        if existing is not None:
            return existing, False
        config = WorkspaceConfig(id=workspace_id or slugify_id(name), name=name)
        if like is not None:
            config.provider = like.provider.model_copy(deep=True)
            config.embedder = like.embedder.model_copy(deep=True)
            config.ingest.requests_per_minute = like.ingest.requests_per_minute
            config.ingest.concurrency = like.ingest.concurrency
        register(config, registry)
        return registry.get(config.id), True  # type: ignore[return-value]
    finally:
        if own:
            registry.close()
