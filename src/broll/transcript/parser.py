"""Transcript parsing and beat segmentation.

SRT and VTT carry real timecodes. Plain text does not, so timings are estimated
from a configurable words-per-minute (default 150) - which is honest guesswork,
and is flagged as estimated on the way out.

A beat is a coherent 3-15 second span of narration: the unit an editor would cut
a single piece of B-roll against. Sentences are the natural boundary, then long
sentences are divided and short ones merged to land inside that window.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

SRT_TIME = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
TAG = re.compile(r"</?[^>]+>")
SPEAKER = re.compile(r"^\s*[A-Z][A-Za-z .'-]{0,24}:\s*")


@dataclass
class Cue:
    start_s: float
    end_s: float
    text: str


@dataclass
class Beat:
    index: int
    start_s: float
    end_s: float
    text: str
    estimated_timing: bool = False
    words: int = field(default=0)

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    @property
    def timecode(self) -> str:
        minutes, seconds = divmod(int(self.start_s), 60)
        return f"{minutes}m{seconds:02d}s"


class TranscriptError(ValueError):
    pass


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def _clean(text: str) -> str:
    text = TAG.sub("", text)
    text = SPEAKER.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def parse_srt(text: str) -> list[Cue]:
    cues: list[Cue] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [line for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        match = None
        body_start = 0
        for position, line in enumerate(lines[:3]):
            match = SRT_TIME.search(line)
            if match:
                body_start = position + 1
                break
        if not match:
            continue
        body = _clean(" ".join(lines[body_start:]))
        if not body:
            continue
        groups = match.groups()
        cues.append(
            Cue(
                start_s=_to_seconds(*groups[:4]),
                end_s=_to_seconds(*groups[4:]),
                text=body,
            )
        )
    return cues


def parse_vtt(text: str) -> list[Cue]:
    body = re.sub(r"^\s*WEBVTT.*?(\n\n|\Z)", "", text, flags=re.DOTALL)
    body = re.sub(r"^\s*(NOTE|STYLE|REGION)\b.*?(\n\n|\Z)", "", body,
                  flags=re.DOTALL | re.MULTILINE)
    return parse_srt(body)


def parse_plain_text(text: str, words_per_minute: int = 150) -> list[Cue]:
    """No timecodes, so estimate them from a reading speed."""
    sentences = [s.strip() for s in SENTENCE_END.split(text.strip()) if s.strip()]
    cues: list[Cue] = []
    cursor = 0.0
    seconds_per_word = 60.0 / max(1, words_per_minute)
    for sentence in sentences:
        cleaned = _clean(sentence)
        if not cleaned:
            continue
        duration = max(1.0, len(cleaned.split()) * seconds_per_word)
        cues.append(Cue(start_s=cursor, end_s=cursor + duration, text=cleaned))
        cursor += duration
    return cues


def parse(text: str, filename: str | None = None, words_per_minute: int = 150) -> tuple[list[Cue], bool]:
    """Return (cues, timings_were_estimated)."""
    suffix = Path(filename).suffix.lower() if filename else ""
    stripped = text.strip()
    if suffix == ".vtt" or stripped.upper().startswith("WEBVTT"):
        return parse_vtt(text), False
    if suffix == ".srt" or SRT_TIME.search(stripped[:2000] or ""):
        cues = parse_srt(text)
        if cues:
            return cues, False
    return parse_plain_text(text, words_per_minute), True


def parse_file(path: Path, words_per_minute: int = 150) -> tuple[list[Cue], bool]:
    if not path.exists():
        raise TranscriptError(f"No transcript at {path}")
    return parse(path.read_text(encoding="utf-8", errors="replace"), path.name, words_per_minute)


# --------------------------------------------------------------------------
# Beats
# --------------------------------------------------------------------------


def segment(
    cues: list[Cue],
    beat_min_s: float = 3.0,
    beat_max_s: float = 15.0,
    estimated: bool = False,
) -> list[Beat]:
    """Cues -> beats: split on sentence boundaries, then merge and divide."""
    sentences = _sentences(cues)
    merged = _merge_short(sentences, beat_min_s, beat_max_s)
    divided: list[Cue] = []
    for cue in merged:
        divided.extend(_divide_long(cue, beat_max_s))

    return [
        Beat(
            index=index,
            start_s=round(cue.start_s, 3),
            end_s=round(cue.end_s, 3),
            text=cue.text,
            estimated_timing=estimated,
            words=len(cue.text.split()),
        )
        for index, cue in enumerate(divided)
    ]


def _sentences(cues: list[Cue]) -> list[Cue]:
    """Re-cut cues on sentence boundaries, interpolating times by word count."""
    out: list[Cue] = []
    for cue in cues:
        parts = [p.strip() for p in SENTENCE_END.split(cue.text) if p.strip()]
        if len(parts) <= 1:
            out.append(cue)
            continue
        total_words = sum(len(p.split()) for p in parts) or 1
        cursor = cue.start_s
        span = cue.end_s - cue.start_s
        for part in parts:
            share = span * (len(part.split()) / total_words)
            out.append(Cue(start_s=cursor, end_s=cursor + share, text=part))
            cursor += share
    return out


def _merge_short(cues: list[Cue], minimum: float, maximum: float) -> list[Cue]:
    merged: list[Cue] = []
    for cue in cues:
        if merged:
            previous = merged[-1]
            combined = cue.end_s - previous.start_s
            if (previous.end_s - previous.start_s) < minimum and combined <= maximum:
                merged[-1] = Cue(previous.start_s, cue.end_s, f"{previous.text} {cue.text}")
                continue
        merged.append(cue)
    # A trailing runt has no successor to absorb it; fold it backwards.
    if len(merged) > 1 and (merged[-1].end_s - merged[-1].start_s) < minimum:
        last = merged.pop()
        previous = merged[-1]
        merged[-1] = Cue(previous.start_s, last.end_s, f"{previous.text} {last.text}")
    return merged


def _divide_long(cue: Cue, maximum: float) -> list[Cue]:
    span = cue.end_s - cue.start_s
    if span <= maximum:
        return [cue]

    pieces = int(span // maximum) + 1
    words = cue.text.split()
    per_piece = max(1, len(words) // pieces)
    out: list[Cue] = []
    cursor = cue.start_s
    for index in range(pieces):
        chunk = words[index * per_piece: None if index == pieces - 1 else (index + 1) * per_piece]
        if not chunk:
            continue
        share = span * (len(chunk) / len(words))
        out.append(Cue(cursor, cursor + share, " ".join(chunk)))
        cursor += share
    return out or [cue]


def parse_and_segment(
    text: str,
    filename: str | None = None,
    words_per_minute: int = 150,
    beat_min_s: float = 3.0,
    beat_max_s: float = 15.0,
) -> list[Beat]:
    cues, estimated = parse(text, filename, words_per_minute)
    return segment(cues, beat_min_s, beat_max_s, estimated)
