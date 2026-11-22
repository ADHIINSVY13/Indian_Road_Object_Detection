"""Helpers for the qualitative comparison: box matching, failure scoring and panel drawing.

Pure functions (no model, no disk) live here so they can be unit-tested; the script that runs the
models is `scripts/qualitative.py`.
"""
from __future__ import annotations

from PIL import Image, ImageDraw


def iou(a, b) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def recall_at_iou(gt: list[tuple[int, tuple]], pred: list[tuple[int, tuple]], thr: float = 0.5) -> float:
    """Fraction of ground-truth boxes found by a same-class prediction with IoU >= thr.

    Each prediction can explain at most one ground-truth box (greedy, best IoU first), so a single
    large box cannot "find" a whole crowd. Items are (class_id, (x1, y1, x2, y2)).
    """
    if not gt:
        return 1.0
    pairs = sorted(
        ((iou(g[1], p[1]), gi, pi) for gi, g in enumerate(gt) for pi, p in enumerate(pred) if g[0] == p[0]),
        reverse=True,
    )
    used_g, used_p, hit = set(), set(), 0
    for score, gi, pi in pairs:
        if score < thr:
            break
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        hit += 1
    return hit / len(gt)


def read_yolo_labels(path, width: int, height: int) -> list[tuple[int, tuple]]:
    """YOLO label file (normalized cx cy w h) -> [(class_id, (x1, y1, x2, y2) in pixels)]."""
    out = []
    for line in open(path, encoding="utf-8").read().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        c, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
        out.append((c, ((cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height)))
    return out


def median_box_area_fraction(gt: list[tuple[int, tuple]], width: int, height: int) -> float | None:
    areas = sorted((b[2] - b[0]) * (b[3] - b[1]) / (width * height) for _, b in gt)
    return areas[len(areas) // 2] if areas else None


_PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46a6a6", "#f032e6", "#9a6324",
            "#800000", "#808000", "#000075", "#e6a800", "#008080", "#a0a0a0", "#ff6f91"]


def draw_boxes(image: Image.Image, boxes: list[tuple[int, tuple, float | None]], names, title: str) -> Image.Image:
    """Return a copy with boxes + labels drawn and a title bar. Items are (class_id, xyxy, conf|None)."""
    im = image.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    for cid, (x1, y1, x2, y2), conf in boxes:
        color = _PALETTE[cid % len(_PALETTE)]
        d.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = names[cid] if conf is None else f"{names[cid]} {conf:.2f}"
        tw = d.textlength(label)
        d.rectangle([x1, max(0, y1 - 11), x1 + tw + 4, max(11, y1)], fill=color)
        d.text((x1 + 2, max(0, y1 - 11)), label, fill="white")
    bar = Image.new("RGB", (im.width, 20), "#202124")
    ImageDraw.Draw(bar).text((6, 4), title, fill="white")
    out = Image.new("RGB", (im.width, im.height + 20))
    out.paste(bar, (0, 0))
    out.paste(im, (0, 20))
    return out


def side_by_side(panels: list[Image.Image]) -> Image.Image:
    h = max(p.height for p in panels)
    sheet = Image.new("RGB", (sum(p.width for p in panels) + 4 * (len(panels) - 1), h), "white")
    x = 0
    for p in panels:
        sheet.paste(p, (x, 0))
        x += p.width + 4
    return sheet
