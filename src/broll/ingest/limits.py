"""Which files are not B-roll, or not worth the server's time.

A folder of footage always holds things that are not clips: a two-hour podcast recording, a camera
left running at an event. Indexing one is slow, can fill the disk, and yields a handful of tags for
two hours of video. So files past a length or a size are not downloaded or analysed. They are put on the
"Needs attention" list with a link, where a person can still choose to index one anyway.

The decision is made from what Drive already reports (length and size), so nothing is downloaded to
find out.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import IngestConfig

GB = 1_000_000_000


@dataclass
class Verdict:
    kind: str     # too_long | too_big
    detail: str


def human_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    if minutes:
        return f"{minutes} min {secs:02d} s"
    return f"{secs} s"


def check_limits(
    cfg: IngestConfig,
    *,
    size_bytes: int | None,
    duration_s: float | None,
    is_image: bool = False,
    forced: bool = False,
    free_bytes: int | None = None,
) -> Verdict | None:
    """None if the file may go ahead, otherwise why it may not.

    `forced` is a person saying "index this one anyway": the length limit rises to
    `max_forced_duration_s`, and the size limit becomes whatever the disk can actually take.
    """
    if not is_image and duration_s:
        limit = cfg.max_forced_duration_s if forced else cfg.max_duration_s
        if limit and duration_s > limit:
            return Verdict(
                "too_long",
                f"{human_duration(duration_s)} long; the limit is {human_duration(limit)}. "
                "Probably a recording rather than B-roll.",
            )
    if size_bytes:
        if forced:
            if free_bytes is not None and size_bytes > free_bytes - cfg.disk_headroom_gb * GB:
                return Verdict(
                    "too_big",
                    f"{size_bytes / GB:.1f} GB is more than this server can hold at once "
                    f"({max(0.0, free_bytes / GB - cfg.disk_headroom_gb):.1f} GB free to use).",
                )
        elif cfg.max_file_gb and size_bytes > cfg.max_file_gb * GB:
            return Verdict(
                "too_big",
                f"{size_bytes / GB:.1f} GB; the limit is {cfg.max_file_gb:g} GB.",
            )
    return None
