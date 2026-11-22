"""Helpers for the qualitative comparison: box matching, failure scoring and panel drawing.

Pure functions (no model, no disk) live here so they can be unit-tested; the script that runs the
models is `scripts/qualitative.py`.

"Qualitative" means looking at actual pictures (ground truth vs baseline vs fine-tuned predictions)
rather than only at numbers such as mAP.
"""
from __future__ import annotations

from PIL import Image, ImageDraw


def iou(a, b) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes.

    IoU = overlap area / combined area, from 0 (no overlap) to 1 (identical boxes). It is the
    standard way to decide whether a predicted box "matches" a ground-truth box.
    """
    # The overlap rectangle runs from the larger of the left/top edges to the smaller of the
    # right/bottom edges.
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    # max(0, ...) handles boxes that do not overlap at all (negative width/height -> 0 area).
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    # Union = area A + area B - the part counted twice (the intersection).
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def recall_at_iou(gt: list[tuple[int, tuple]], pred: list[tuple[int, tuple]], thr: float = 0.5) -> float:
    """Fraction of ground-truth boxes found by a same-class prediction with IoU >= thr.

    Each prediction can explain at most one ground-truth box (greedy, best IoU first), so a single
    large box cannot "find" a whole crowd. Items are (class_id, (x1, y1, x2, y2)).
    """
    if not gt:
        return 1.0  # nothing to find, so nothing was missed
    # Build every (ground truth, prediction) pair of the SAME class with its IoU, best first.
    pairs = sorted(
        ((iou(g[1], p[1]), gi, pi) for gi, g in enumerate(gt) for pi, p in enumerate(pred) if g[0] == p[0]),
        reverse=True,
    )
    used_g, used_p, hit = set(), set(), 0  # which boxes are already matched, and the match count
    for score, gi, pi in pairs:
        if score < thr:
            break  # pairs are sorted, so every remaining pair is below the threshold too
        if gi in used_g or pi in used_p:
            continue  # one of the two boxes is already matched to something better
        used_g.add(gi)
        used_p.add(pi)
        hit += 1
    return hit / len(gt)


def read_yolo_labels(path, width: int, height: int) -> list[tuple[int, tuple]]:
    """YOLO label file (normalized cx cy w h) -> [(class_id, (x1, y1, x2, y2) in pixels)].

    Converts back from YOLO's centre/size fractions to pixel corners so boxes can be drawn.
    """
    out = []
    for line in open(path, encoding="utf-8").read().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue  # skip blank or malformed lines
        c, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
        # corner = centre -/+ half the size, then multiply the fractions by the image size.
        out.append((c, ((cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height)))
    return out


def median_box_area_fraction(gt: list[tuple[int, tuple]], width: int, height: int) -> float | None:
    """Median box area as a fraction of the image; small values mean a scene of tiny objects."""
    areas = sorted((b[2] - b[0]) * (b[3] - b[1]) / (width * height) for _, b in gt)
    return areas[len(areas) // 2] if areas else None  # middle element of the sorted list


# One colour per class id (cycled with %), so each class always gets the same colour in every picture.
_PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46a6a6", "#f032e6", "#9a6324",
            "#800000", "#808000", "#000075", "#e6a800", "#008080", "#a0a0a0", "#ff6f91"]


def draw_boxes(image: Image.Image, boxes: list[tuple[int, tuple, float | None]], names, title: str) -> Image.Image:
    """Return a copy with boxes + labels drawn and a title bar. Items are (class_id, xyxy, conf|None)."""
    im = image.convert("RGB").copy()  # copy so the caller's image is never modified
    d = ImageDraw.Draw(im)
    for cid, (x1, y1, x2, y2), conf in boxes:
        color = _PALETTE[cid % len(_PALETTE)]
        d.rectangle([x1, y1, x2, y2], outline=color, width=2)
        # Ground-truth boxes have no confidence (conf is None); predictions show "name 0.87".
        label = names[cid] if conf is None else f"{names[cid]} {conf:.2f}"
        tw = d.textlength(label)  # label width in pixels, to size its background strip
        # Filled strip above the box (kept inside the image) so the white text stays readable.
        d.rectangle([x1, max(0, y1 - 11), x1 + tw + 4, max(11, y1)], fill=color)
        d.text((x1 + 2, max(0, y1 - 11)), label, fill="white")
    # Add a dark 20-pixel title bar on top of the picture (e.g. "Ground truth").
    bar = Image.new("RGB", (im.width, 20), "#202124")
    ImageDraw.Draw(bar).text((6, 4), title, fill="white")
    out = Image.new("RGB", (im.width, im.height + 20))
    out.paste(bar, (0, 0))
    out.paste(im, (0, 20))
    return out


def side_by_side(panels: list[Image.Image]) -> Image.Image:
    """Join several images horizontally into one sheet with a 4-pixel white gap between them."""
    h = max(p.height for p in panels)  # sheet is as tall as the tallest panel
    sheet = Image.new("RGB", (sum(p.width for p in panels) + 4 * (len(panels) - 1), h), "white")
    x = 0
    for p in panels:
        sheet.paste(p, (x, 0))
        x += p.width + 4  # move right by the panel width plus the gap
    return sheet
