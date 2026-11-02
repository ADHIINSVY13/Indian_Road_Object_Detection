"""Convert the raw IDD detection release (VOC xml) into an Ultralytics YOLO dataset.

Output layout under yolo.root:
    images/<split>/<flat_stem>.jpg
    labels/<split>/<flat_stem>.txt
    manifest.csv            one row per converted image (split, stem, width, height, n_boxes)
    conversion_issues.csv   every dropped/clipped/flagged item with a reason code
    conversion_report.json  per-split counts
    idd.yaml                dataset file for Ultralytics
"""
from __future__ import annotations

import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

from src.data.config import ClassInfo, class_index, normalize_label
from src.preprocessing.validate import box_issues, clip_box, issue
from src.preprocessing.voc import parse_voc_xml

SPLITS = ("train", "val", "test")


def flat_name(stem: str) -> str:
    return stem.replace("\\", "/").strip("/").replace("/", "__")


def read_split_stems(raw_root: Path, split_file: str | None) -> list[str] | None:
    """Return stems listed in the split file, or None if the file is absent."""
    if not split_file:
        return None
    p = raw_root / split_file
    if not p.exists():
        return None
    stems = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            stems.append(os.path.splitext(line)[0] if line.lower().endswith((".jpg", ".jpeg", ".png", ".xml")) else line)
    return stems


def find_image(raw_root: Path, image_dir: str, stem: str, exts: list[str]) -> Path | None:
    for ext in exts:
        p = raw_root / image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def resize_image(src: Path, dst: Path, max_side: int) -> None:
    """Save a copy whose longer side is at most max_side. Labels are normalized, so they stay valid."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        scale = max_side / max(im.size)
        if scale < 1:
            im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
        im.convert("RGB").save(dst, quality=90)


def place_image(src: Path, dst: Path, mode: str, max_side: int | None = None) -> None:
    if mode == "none" or dst.exists():
        return
    if max_side:
        resize_image(src, dst.with_suffix(".jpg"), max_side)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass  # different volume or unsupported filesystem
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)


def convert_split(split, stems, raw_cfg, yolo_root, classes, image_mode, min_side, progress=None, max_side=None):
    raw_root = Path(raw_cfg["_abs_root"])
    cls_idx = class_index(classes)
    img_out = yolo_root / "images" / split
    lbl_out = yolo_root / "labels" / split
    lbl_out.mkdir(parents=True, exist_ok=True)

    issues, manifest = [], []
    class_counts: Counter = Counter()
    stats = Counter()

    iterator = progress(stems, desc=f"convert {split}") if progress else stems
    for stem in iterator:
        stats["listed"] += 1
        xml_path = raw_root / raw_cfg["annotation_dir"] / f"{stem}.xml"
        img_path = find_image(raw_root, raw_cfg["image_dir"], stem, raw_cfg["image_extensions"])
        if img_path is None:
            issues.append(issue(split, stem, "error", "missing_image", str(raw_root / raw_cfg["image_dir"])))
            stats["skipped_missing_image"] += 1
            continue
        if not xml_path.exists():
            issues.append(issue(split, stem, "error", "missing_annotation"))
            stats["skipped_missing_annotation"] += 1
            continue

        ann = parse_voc_xml(xml_path)
        for p in ann.problems:
            issues.append(issue(split, stem, "error", "annotation_problem", p))

        # Image size: trust the file, flag disagreement with the xml.
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception as e:  # noqa: BLE001
            issues.append(issue(split, stem, "error", "corrupt_image", str(e)))
            stats["skipped_corrupt_image"] += 1
            continue
        if ann.width and ann.height and (ann.width, ann.height) != (w, h):
            issues.append(issue(split, stem, "warning", "size_mismatch_xml_vs_image",
                                f"xml={ann.width}x{ann.height} image={w}x{h}"))
            if ann.width * h != ann.height * w:
                # Different aspect ratio: boxes cannot be trusted for this image.
                issues.append(issue(split, stem, "error", "aspect_mismatch_skipped"))
                stats["skipped_aspect_mismatch"] += 1
                continue

        lines = []
        for obj in ann.objects:
            label = normalize_label(obj.name)
            if label not in cls_idx:
                issues.append(issue(split, stem, "error", "unknown_class", obj.name))
                stats["boxes_dropped_unknown_class"] += 1
                continue
            found = box_issues(obj.xmin, obj.ymin, obj.xmax, obj.ymax, w, h, min_side)
            dropped = False
            for sev, code in found:
                issues.append(issue(split, stem, sev, code, f"{obj.name} {obj.xmin},{obj.ymin},{obj.xmax},{obj.ymax}"))
                dropped |= sev == "error"
            if dropped:
                stats["boxes_dropped_invalid"] += 1
                continue
            x1, y1, x2, y2 = clip_box(obj.xmin, obj.ymin, obj.xmax, obj.ymax, w, h)
            if x2 <= x1 or y2 <= y1:
                issues.append(issue(split, stem, "error", "empty_after_clip", obj.name))
                stats["boxes_dropped_invalid"] += 1
                continue
            lines.append(
                f"{cls_idx[label]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
            )
            class_counts[label] += 1
            stats["boxes_kept"] += 1

        if not lines:
            issues.append(issue(split, stem, "warning", "no_valid_objects"))
            stats["images_without_objects"] += 1
        name = flat_name(stem)
        (lbl_out / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        place_image(img_path, img_out / f"{name}{img_path.suffix.lower()}", image_mode, max_side)
        manifest.append({"split": split, "stem": name, "width": w, "height": h, "n_boxes": len(lines)})
        stats["converted"] += 1

    return issues, manifest, class_counts, stats


def write_dataset_yaml(yolo_root: Path, classes: list[ClassInfo], splits_done: list[str]) -> Path:
    data = {"path": str(yolo_root.resolve()), "names": {c.id: c.name for c in classes}}
    for s in splits_done:
        data[s] = f"images/{s}"
    out = yolo_root / "idd.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def convert_dataset(raw_cfg, yolo_cfg, classes, repo_root: Path, splits=SPLITS, limit: int | None = None, progress=None):
    raw_cfg = dict(raw_cfg)
    raw_root = Path(raw_cfg["root"])
    raw_root = raw_root if raw_root.is_absolute() else repo_root / raw_root
    raw_cfg["_abs_root"] = str(raw_root)
    yolo_root = Path(yolo_cfg["root"])
    yolo_root = yolo_root if yolo_root.is_absolute() else repo_root / yolo_root
    yolo_root.mkdir(parents=True, exist_ok=True)

    all_issues, all_manifest = [], []
    report = {"raw_root": str(raw_root), "splits": {}, "classes": [c.name for c in classes]}
    done = []
    for split in splits:
        stems = read_split_stems(raw_root, raw_cfg.get("split_files", {}).get(split))
        if stems is None:
            report["splits"][split] = {"status": "no split file found, not converted"}
            continue
        if limit:
            stems = stems[:limit]
        issues, manifest, class_counts, stats = convert_split(
            split, stems, raw_cfg, yolo_root, classes,
            yolo_cfg.get("image_mode", "hardlink"), float(yolo_cfg.get("min_box_side_px", 2)), progress,
            yolo_cfg.get("max_side") or None,
        )
        all_issues += issues
        all_manifest += manifest
        report["splits"][split] = {"status": "converted", **dict(stats), "class_counts": dict(class_counts)}
        done.append(split)

    write_csv(yolo_root / "manifest.csv", all_manifest, ["split", "stem", "width", "height", "n_boxes"])
    write_csv(yolo_root / "conversion_issues.csv", all_issues, ["split", "stem", "severity", "code", "detail"])
    (yolo_root / "conversion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_dataset_yaml(yolo_root, classes, done)
    return report, all_issues
