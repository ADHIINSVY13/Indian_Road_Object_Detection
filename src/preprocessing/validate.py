"""Validation of raw boxes and of the converted YOLO dataset.

Issue records are dicts: {split, stem, severity, code, detail}.
severity "error"   -> the box/sample is dropped and logged
severity "warning" -> kept, logged
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def issue(split: str, stem: str, severity: str, code: str, detail: str = "") -> dict:
    return {"split": split, "stem": stem, "severity": severity, "code": code, "detail": detail}


def box_issues(xmin, ymin, xmax, ymax, width, height, min_side: float = 2.0) -> list[tuple[str, str]]:
    """Return (severity, code) pairs for one pixel-space box."""
    out: list[tuple[str, str]] = []
    if xmax <= xmin or ymax <= ymin:
        out.append(("error", "non_positive_size"))
        return out
    if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
        out.append(("warning", "out_of_bounds_clipped"))
    if (xmax - xmin) < min_side or (ymax - ymin) < min_side:
        out.append(("warning", "tiny_box"))
    return out


def clip_box(xmin, ymin, xmax, ymax, width, height):
    return (
        min(max(xmin, 0.0), float(width)),
        min(max(ymin, 0.0), float(height)),
        min(max(xmax, 0.0), float(width)),
        min(max(ymax, 0.0), float(height)),
    )


def validate_yolo_dataset(yolo_root: str | Path, num_classes: int, check_images: bool = False) -> list[dict]:
    """Check a converted dataset: label format, class ids, coordinate ranges,
    duplicates, and image/label pairing. Optionally decode every image."""
    yolo_root = Path(yolo_root)
    issues: list[dict] = []
    for split_dir in sorted((yolo_root / "labels").glob("*")):
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        img_dir = yolo_root / "images" / split
        label_stems = set()
        for lf in split_dir.glob("*.txt"):
            stem = lf.stem
            label_stems.add(stem)
            seen = set()
            text = lf.read_text(encoding="utf-8").strip()
            if not text:
                issues.append(issue(split, stem, "warning", "empty_label_file"))
            for ln, line in enumerate(text.splitlines() if text else [], 1):
                parts = line.split()
                if len(parts) != 5:
                    issues.append(issue(split, stem, "error", "bad_line_format", f"line {ln}: {line!r}"))
                    continue
                try:
                    cls = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])
                except ValueError:
                    issues.append(issue(split, stem, "error", "non_numeric_value", f"line {ln}"))
                    continue
                if not 0 <= cls < num_classes:
                    issues.append(issue(split, stem, "error", "class_id_out_of_range", f"line {ln}: {cls}"))
                if not all(0.0 <= v <= 1.0 for v in (xc, yc, w, h)) or w <= 0 or h <= 0:
                    issues.append(issue(split, stem, "error", "coords_out_of_range", f"line {ln}"))
                if line in seen:
                    issues.append(issue(split, stem, "warning", "duplicate_box", f"line {ln}"))
                seen.add(line)
        image_files = {p.stem: p for p in img_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS} if img_dir.exists() else {}
        for stem in sorted(label_stems - set(image_files)):
            issues.append(issue(split, stem, "error", "label_without_image"))
        for stem in sorted(set(image_files) - label_stems):
            issues.append(issue(split, stem, "error", "image_without_label"))
        if check_images:
            for stem, p in image_files.items():
                try:
                    with Image.open(p) as im:
                        im.verify()
                except Exception as e:  # noqa: BLE001 - any decode failure is a finding
                    issues.append(issue(split, stem, "error", "corrupt_image", str(e)))
    return issues
