"""Parser for Pascal VOC style XML annotations.

IDD Detection stores one XML file per image. Each file lists the image size and one <object> per
labelled item, with a class name and a bounding box in absolute pixel coordinates. This module turns
one such file into plain Python objects and records anything suspicious instead of crashing, so a
single corrupt file can never stop a 40,000-image conversion.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VocObject:
    """One labelled object: its class name and box corners in pixels (top-left, bottom-right)."""

    name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class VocAnnotation:
    """Everything parsed from one XML file."""

    path: Path                 # which file this came from (useful in error messages)
    width: int | None          # image width declared in the XML, None if missing/invalid
    height: int | None         # image height declared in the XML, None if missing/invalid
    # field(default_factory=list) gives every instance its OWN empty list. A plain `= []` would
    # make all instances share one list, a classic Python bug.
    objects: list[VocObject] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)  # human-readable issues found while parsing


def _to_int(text: str | None) -> int | None:
    """Convert XML text to a positive integer, or None if that is impossible.

    The leading underscore marks this as a private helper. It goes through float() first so values
    such as "1920.0" still work. Zero or negative sizes are treated as invalid because an image
    cannot have a non-positive width or height.
    """
    try:
        value = int(float(text))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        # TypeError: text was None (tag missing). ValueError: text was not a number.
        return None
    return value if value > 0 else None


def parse_voc_xml(path: str | Path) -> VocAnnotation:
    """Parse one VOC xml file. Never raises on malformed content; problems are recorded.

    Returns a VocAnnotation whose `problems` list explains everything that was skipped, so the
    caller can log it and carry on.
    """
    path = Path(path)
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as e:
        # ParseError: the file is not valid XML. OSError: the file could not be opened/read.
        # Either way we return an empty annotation that carries the reason.
        return VocAnnotation(path, None, None, [], [f"xml_parse_error: {e}"])

    problems: list[str] = []

    # --- image size block: <size><width>..</width><height>..</height></size> ---
    size = root.find("size")
    width = height = None
    if size is not None:
        width = _to_int(size.findtext("width"))
        height = _to_int(size.findtext("height"))
    if width is None or height is None:
        problems.append("missing_or_invalid_size")

    # --- one <object> element per labelled item ---
    objects: list[VocObject] = []
    for i, obj in enumerate(root.iter("object")):  # iter() finds <object> tags at any depth
        name = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if not name:
            problems.append(f"object_{i}: missing_name")
            continue
        if box is None:
            problems.append(f"object_{i}: missing_bndbox")
            continue
        try:
            # Coordinates are read BY TAG NAME, never by position. This matters for IDD: its XML
            # writes the children in the odd order xmin, ymax, xmax, ymin, so a parser that took
            # them positionally would silently swap the vertical axis.
            coords = [float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]  # type: ignore[arg-type]
        except (TypeError, ValueError):
            # TypeError: a coordinate tag was missing. ValueError: it was not a number.
            problems.append(f"object_{i}: non_numeric_box")
            continue
        objects.append(VocObject(name, *coords))  # *coords unpacks the 4 numbers into the 4 fields
    return VocAnnotation(path, width, height, objects, problems)
