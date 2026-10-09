"""Problems with a file that are about the file, not the model."""

from __future__ import annotations

from ..analysis.providers.base import TransientProviderError


class IncompleteDownloadError(RuntimeError):
    """A download stopped short. The partial file is deleted; it is never mistaken for the clip."""


class DiskSpaceError(TransientProviderError):
    """Not enough free disk to download this file without risking the server.

    Transient on purpose: other files finishing frees space, so the queue retries it with backoff
    instead of giving up.
    """
