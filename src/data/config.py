"""Loading of dataset and class-mapping configuration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path, base: Path = REPO_ROOT) -> Path:
    p = Path(path)
    return p if p.is_absolute() else base / p


@dataclass
class ClassInfo:
    id: int
    name: str
    group: str
    coco_names: list[str]
    coco_ids: list[int]
    note: str


def normalize_label(name: str) -> str:
    return " ".join(name.strip().lower().replace("_", " ").split())


def load_classes(mapping_path: str | Path) -> list[ClassInfo]:
    cfg = load_yaml(mapping_path)
    classes = []
    for i, c in enumerate(cfg["classes"]):
        classes.append(
            ClassInfo(
                id=i,
                name=normalize_label(c["name"]),
                group=c["group"],
                coco_names=list(c.get("coco_names", [])),
                coco_ids=list(c.get("coco_ids", [])),
                note=c.get("note", ""),
            )
        )
    return classes


def class_index(classes: list[ClassInfo]) -> dict[str, int]:
    return {c.name: c.id for c in classes}
