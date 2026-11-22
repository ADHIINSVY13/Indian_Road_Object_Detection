"""Loading of dataset and class-mapping configuration.

Every other module gets its settings (paths, class list, COCO mapping) through the helpers in this
file, so there is exactly one place that knows how the YAML files are read.
"""
# Lets us write modern type hints such as `str | Path` and `list[str]` even on older Python versions,
# because annotations are stored as text instead of being evaluated immediately.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# Absolute path of the repository root. This file lives at <repo>/src/data/config.py, so
# parents[0] = src/data, parents[1] = src, parents[2] = <repo>. Using the root as a fixed anchor
# means relative paths from the YAML configs work no matter which folder the user runs a script from.
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict:
    """Read a YAML file and return its contents as a plain Python dict.

    safe_load is used on purpose: it only builds simple data (dicts, lists, numbers, strings) and
    refuses to run arbitrary code that a malicious YAML file could otherwise contain.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path, base: Path = REPO_ROOT) -> Path:
    """Turn a config path into an absolute path.

    Absolute paths (for example `C:\\idd_data`) are returned unchanged; relative paths (for example
    `data/yolo`) are interpreted relative to the repository root.
    """
    p = Path(path)
    return p if p.is_absolute() else base / p


@dataclass
class ClassInfo:
    """One IDD class together with everything we know about how it relates to COCO.

    A dataclass is just a tidy container: Python writes __init__ and __repr__ for us.
    """

    id: int               # YOLO class id (0..N-1); decided by the order in class_mapping.yaml
    name: str             # normalised lower-case class name, e.g. "traffic light"
    group: str            # "common", "india_specific" or "ambiguous" (see class_mapping.yaml)
    coco_names: list[str]  # COCO class names the baseline may predict for this class
    coco_ids: list[int]    # matching COCO ids; an empty list means "no COCO equivalent"
    note: str             # free-text explanation of the mapping decision


def normalize_label(name: str) -> str:
    """Canonical form of a class name so spelling variants compare equal.

    Steps: strip outer whitespace, lower-case, turn underscores into spaces, and collapse runs of
    spaces. So "Traffic_Light", " traffic  light " and "traffic light" all become "traffic light".
    """
    return " ".join(name.strip().lower().replace("_", " ").split())


def load_classes(mapping_path: str | Path) -> list[ClassInfo]:
    """Read configs/class_mapping.yaml and build the ordered list of ClassInfo objects.

    The position of a class in the YAML list *is* its YOLO class id, which is why the file warns
    never to reorder it after training has started: that would silently relabel every class.
    """
    cfg = load_yaml(mapping_path)
    classes = []
    for i, c in enumerate(cfg["classes"]):  # enumerate gives the running index = class id
        classes.append(
            ClassInfo(
                id=i,
                name=normalize_label(c["name"]),
                group=c["group"],
                # .get(..., []) makes the optional keys safe: a class with no COCO match simply
                # has empty lists instead of raising a KeyError.
                coco_names=list(c.get("coco_names", [])),
                coco_ids=list(c.get("coco_ids", [])),
                note=c.get("note", ""),
            )
        )
    return classes


def class_index(classes: list[ClassInfo]) -> dict[str, int]:
    """Build a {class name -> YOLO id} lookup used when converting annotation labels to ids."""
    return {c.name: c.id for c in classes}
