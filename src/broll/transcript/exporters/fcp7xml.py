"""FCP7 XML (xmeml v5) export.

The highest-value export: it imports into both Premiere Pro and DaVinci Resolve.
Gaps are simply the absence of a clipitem - the timeline positions are absolute,
so nothing needs to be emitted for the empty stretch.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

from .base import Timeline, frames_to_timecode, ntsc_timebase, path_to_url

XML_DECLARATION = '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n'


def _rate(parent: ET.Element, fps: float) -> ET.Element:
    timebase, ntsc = ntsc_timebase(fps)
    rate = ET.SubElement(parent, "rate")
    ET.SubElement(rate, "timebase").text = str(timebase)
    ET.SubElement(rate, "ntsc").text = "TRUE" if ntsc else "FALSE"
    return rate


def _text(parent: ET.Element, tag: str, value) -> ET.Element:
    element = ET.SubElement(parent, tag)
    element.text = str(value)
    return element


def build(timeline: Timeline) -> str:
    root = ET.Element("xmeml", version="5")
    sequence = ET.SubElement(root, "sequence", id="broll-sequence-1")
    _text(sequence, "name", timeline.name)
    _text(sequence, "duration", timeline.duration_frames)
    _rate(sequence, timeline.fps)

    timecode = ET.SubElement(sequence, "timecode")
    _rate(timecode, timeline.fps)
    _text(timecode, "string", frames_to_timecode(0, timeline.fps))
    _text(timecode, "frame", 0)
    _text(timecode, "displayformat", "NDF")

    media = ET.SubElement(sequence, "media")
    video = ET.SubElement(media, "video")

    fmt = ET.SubElement(video, "format")
    characteristics = ET.SubElement(fmt, "samplecharacteristics")
    _rate(characteristics, timeline.fps)
    _text(characteristics, "width", timeline.width)
    _text(characteristics, "height", timeline.height)

    track = ET.SubElement(video, "track")
    seen_files: dict[str, str] = {}

    for position, item in enumerate(timeline.items, start=1):
        clip = ET.SubElement(track, "clipitem", id=f"clipitem-{position}")
        _text(clip, "name", item.name)
        _text(clip, "enabled", "TRUE")
        _text(clip, "duration", item.duration_frames)
        _rate(clip, timeline.fps)
        _text(clip, "start", item.start_frame)
        _text(clip, "end", item.end_frame)
        _text(clip, "in", item.source_in_frame)
        _text(clip, "out", item.source_out_frame)

        source_key = item.suggestion.source.id
        if source_key in seen_files:
            ET.SubElement(clip, "file", id=seen_files[source_key])
        else:
            file_id = f"file-{len(seen_files) + 1}"
            seen_files[source_key] = file_id
            file_element = ET.SubElement(clip, "file", id=file_id)
            _text(file_element, "name", item.name)
            _text(file_element, "pathurl", path_to_url(item.media_path))
            _rate(file_element, item.source_fps)
            file_media = ET.SubElement(file_element, "media")
            file_video = ET.SubElement(file_media, "video")
            file_characteristics = ET.SubElement(file_video, "samplecharacteristics")
            _rate(file_characteristics, item.source_fps)
            _text(file_characteristics, "width", item.suggestion.source.width or timeline.width)
            _text(file_characteristics, "height", item.suggestion.source.height or timeline.height)

        comments = ET.SubElement(clip, "comments")
        _text(comments, "mastercomment1", item.match.beat.text[:200])
        _text(comments, "mastercomment2", item.suggestion.reason[:200])

    ET.indent(root, space="  ")
    return XML_DECLARATION + ET.tostring(root, encoding="unicode") + "\n"


def write(timeline: Timeline, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build(timeline), encoding="utf-8")
    return path
