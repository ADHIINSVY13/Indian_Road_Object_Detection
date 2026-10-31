"""Parser for Pascal VOC style XML annotations."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VocObject:
    name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class VocAnnotation:
    path: Path
    width: int | None
    height: int | None
    objects: list[VocObject] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def _to_int(text: str | None) -> int | None:
    try:
        value = int(float(text))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def parse_voc_xml(path: str | Path) -> VocAnnotation:
    """Parse one VOC xml file. Never raises on malformed content; problems are recorded."""
    path = Path(path)
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as e:
        return VocAnnotation(path, None, None, [], [f"xml_parse_error: {e}"])

    problems: list[str] = []
    size = root.find("size")
    width = height = None
    if size is not None:
        width = _to_int(size.findtext("width"))
        height = _to_int(size.findtext("height"))
    if width is None or height is None:
        problems.append("missing_or_invalid_size")

    objects: list[VocObject] = []
    for i, obj in enumerate(root.iter("object")):
        name = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if not name:
            problems.append(f"object_{i}: missing_name")
            continue
        if box is None:
            problems.append(f"object_{i}: missing_bndbox")
            continue
        try:
            coords = [float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]  # type: ignore[arg-type]
        except (TypeError, ValueError):
            problems.append(f"object_{i}: non_numeric_box")
            continue
        objects.append(VocObject(name, *coords))
    return VocAnnotation(path, width, height, objects, problems)
