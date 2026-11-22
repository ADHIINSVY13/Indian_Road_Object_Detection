"""Validation of raw boxes and of the converted YOLO dataset.

Issue records are dicts: {split, stem, severity, code, detail}.
severity "error"   -> the box/sample is dropped and logged
severity "warning" -> kept, logged

There are two layers of checking:
  1. box_issues / clip_box  - judge a single box while converting (before anything is written).
  2. validate_yolo_dataset  - re-read the finished dataset from disk and verify it independently,
                              so a bug in the converter cannot hide itself.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

# File extensions we treat as images when pairing images with label files.
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def issue(split: str, stem: str, severity: str, code: str, detail: str = "") -> dict:
    """Create one issue record. Keeping the shape in one function guarantees every CSV row has
    the same columns: which split, which image, how serious, a short machine code, free-text detail."""
    return {"split": split, "stem": stem, "severity": severity, "code": code, "detail": detail}


def box_issues(xmin, ymin, xmax, ymax, width, height, min_side: float = 2.0) -> list[tuple[str, str]]:
    """Return (severity, code) pairs for one pixel-space box.

    Possible findings:
      non_positive_size      (error)   box has zero/negative width or height (or corners swapped)
      out_of_bounds_clipped  (warning) box pokes outside the image; the converter will clip it
      tiny_box               (warning) a side shorter than min_side pixels; kept but flagged
    """
    out: list[tuple[str, str]] = []
    # A valid box needs its right edge strictly right of the left edge, and bottom below top.
    if xmax <= xmin or ymax <= ymin:
        out.append(("error", "non_positive_size"))
        # Nothing else is worth checking on a degenerate box, so stop early.
        return out
    if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
        out.append(("warning", "out_of_bounds_clipped"))
    if (xmax - xmin) < min_side or (ymax - ymin) < min_side:
        out.append(("warning", "tiny_box"))
    return out


def clip_box(xmin, ymin, xmax, ymax, width, height):
    """Force a box to lie inside the image.

    max(v, 0) lifts negative coordinates to 0 and min(.., size) pulls overshooting ones back to the
    image edge. The result is always within [0, width] x [0, height].
    """
    return (
        min(max(xmin, 0.0), float(width)),
        min(max(ymin, 0.0), float(height)),
        min(max(xmax, 0.0), float(width)),
        min(max(ymax, 0.0), float(height)),
    )


def validate_yolo_dataset(yolo_root: str | Path, num_classes: int, check_images: bool = False) -> list[dict]:
    """Check a converted dataset: label format, class ids, coordinate ranges,
    duplicates, and image/label pairing. Optionally decode every image.

    Returns a list of issue dicts; an empty list means a perfectly clean dataset.
    """
    yolo_root = Path(yolo_root)
    issues: list[dict] = []

    # Loop over each split folder found under labels/ (train, val, ...).
    for split_dir in sorted((yolo_root / "labels").glob("*")):
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        img_dir = yolo_root / "images" / split
        label_stems = set()  # names of every label file in this split, used for pairing later

        # ---------- check every label file ----------
        for lf in split_dir.glob("*.txt"):
            stem = lf.stem  # file name without ".txt"
            label_stems.add(stem)
            seen = set()    # exact lines already met in this file, to detect duplicate boxes
            text = lf.read_text(encoding="utf-8").strip()
            if not text:
                # An empty label file means "no objects in this image". Legal in YOLO, but worth noting.
                issues.append(issue(split, stem, "warning", "empty_label_file"))
            # enumerate(..., 1) numbers lines from 1 so messages match what an editor shows.
            for ln, line in enumerate(text.splitlines() if text else [], 1):
                parts = line.split()
                # YOLO format is exactly: class x_center y_center width height
                if len(parts) != 5:
                    issues.append(issue(split, stem, "error", "bad_line_format", f"line {ln}: {line!r}"))
                    continue
                try:
                    cls = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])
                except ValueError:
                    issues.append(issue(split, stem, "error", "non_numeric_value", f"line {ln}"))
                    continue
                # Class id must be a real class: 0 .. num_classes-1.
                if not 0 <= cls < num_classes:
                    issues.append(issue(split, stem, "error", "class_id_out_of_range", f"line {ln}: {cls}"))
                # All four numbers are fractions of the image, so they must lie in [0, 1];
                # width and height must also be strictly positive.
                if not all(0.0 <= v <= 1.0 for v in (xc, yc, w, h)) or w <= 0 or h <= 0:
                    issues.append(issue(split, stem, "error", "coords_out_of_range", f"line {ln}"))
                # The very same line twice means the same box was annotated twice.
                if line in seen:
                    issues.append(issue(split, stem, "warning", "duplicate_box", f"line {ln}"))
                seen.add(line)

        # ---------- check image <-> label pairing ----------
        # Map each image's name (without extension) to its path; empty if the folder is absent.
        image_files = {p.stem: p for p in img_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS} if img_dir.exists() else {}
        # Set subtraction finds names present on one side only.
        for stem in sorted(label_stems - set(image_files)):
            issues.append(issue(split, stem, "error", "label_without_image"))
        for stem in sorted(set(image_files) - label_stems):
            issues.append(issue(split, stem, "error", "image_without_label"))

        # ---------- optional: make sure every image can actually be decoded ----------
        if check_images:
            for stem, p in image_files.items():
                try:
                    with Image.open(p) as im:
                        im.verify()  # checks file integrity without fully decoding pixels
                except Exception as e:  # noqa: BLE001 - any decode failure is a finding
                    issues.append(issue(split, stem, "error", "corrupt_image", str(e)))
    return issues
