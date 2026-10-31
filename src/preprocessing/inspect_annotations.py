"""Mandatory first step: look at the real IDD files before converting anything.

Reports, from the actual download:
  * directory tree summary and file-extension counts
  * the first N annotation files (raw content) so format/coordinates can be read by eye
  * format detection (VOC xml / json / txt)
  * every class name observed and how often, sampled over the annotation files
  * bounding-box coordinate ranges
  * split-file presence and sample counts
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from src.preprocessing.voc import parse_voc_xml

ANNOTATION_EXTS = {".xml", ".json", ".txt", ".csv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def summarize_tree(root: Path) -> dict:
    ext_counts: Counter = Counter()
    top_dirs: Counter = Counter()
    for p in root.rglob("*"):
        if p.is_file():
            ext_counts[p.suffix.lower()] += 1
            rel = p.relative_to(root)
            top_dirs[rel.parts[0] if len(rel.parts) > 1 else "."] += 1
    return {"extensions": dict(ext_counts), "files_per_top_level_dir": dict(top_dirs)}


def find_annotation_files(root: Path, ann_dir: str | None = None) -> list[Path]:
    base = root / ann_dir if ann_dir and (root / ann_dir).exists() else root
    return sorted(p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in ANNOTATION_EXTS
                  and p.name not in {"train.txt", "val.txt", "test.txt"})


def detect_format(sample: Path) -> str:
    ext = sample.suffix.lower()
    if ext == ".xml":
        head = sample.read_text(encoding="utf-8", errors="replace")[:2000]
        return "voc_xml" if "<annotation" in head and "<bndbox" in sample.read_text(encoding="utf-8", errors="replace") else "xml_other"
    if ext == ".json":
        return "json"
    if ext == ".txt":
        return "txt"
    return ext.lstrip(".") or "unknown"


def inspect_dataset(root: str | Path, ann_dir: str | None = None, n_show: int = 5, n_scan: int = 2000, seed: int = 0) -> dict:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist. Download IDD Detection first (see data/README.md).")
    ann_files = find_annotation_files(root, ann_dir)
    result: dict = {"root": str(root), "tree": summarize_tree(root), "n_annotation_files": len(ann_files)}

    split_info = {}
    for name in ("train.txt", "val.txt", "test.txt"):
        p = root / name
        split_info[name] = (len([l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]) if p.exists() else None)
    result["split_files_line_counts"] = split_info

    if not ann_files:
        result["warning"] = "no annotation files found"
        return result

    result["detected_format"] = detect_format(ann_files[0])
    result["first_files"] = []
    for p in ann_files[:n_show]:
        text = p.read_text(encoding="utf-8", errors="replace")
        result["first_files"].append({"path": str(p.relative_to(root)), "content_head": text[:1500]})

    if result["detected_format"] == "voc_xml":
        rng = random.Random(seed)
        sample = ann_files if len(ann_files) <= n_scan else rng.sample(ann_files, n_scan)
        names: Counter = Counter()
        sizes: Counter = Counter()
        coord_min = [float("inf")] * 4
        coord_max = [float("-inf")] * 4
        parse_problems = 0
        for p in sample:
            a = parse_voc_xml(p)
            parse_problems += bool(a.problems)
            sizes[f"{a.width}x{a.height}"] += 1
            for o in a.objects:
                names[o.name] += 1
                for i, v in enumerate((o.xmin, o.ymin, o.xmax, o.ymax)):
                    coord_min[i] = min(coord_min[i], v)
                    coord_max[i] = max(coord_max[i], v)
        result["scan"] = {
            "files_scanned": len(sample),
            "files_with_parse_problems": parse_problems,
            "class_names_observed": dict(names.most_common()),
            "image_sizes_in_xml": dict(sizes.most_common(10)),
            "box_coord_min_xmin_ymin_xmax_ymax": coord_min,
            "box_coord_max_xmin_ymin_xmax_ymax": coord_max,
            "box_format_note": "VOC convention: absolute pixel corners (xmin, ymin, xmax, ymax)",
        }
    return result


def to_markdown(result: dict) -> str:
    lines = [f"# Annotation inspection\n", f"Root: `{result['root']}`\n"]
    lines.append(f"- Annotation files found: {result['n_annotation_files']}")
    lines.append(f"- Detected format: {result.get('detected_format', 'n/a')}")
    lines.append(f"- Split files (line counts): {json.dumps(result['split_files_line_counts'])}")
    lines.append(f"- File extensions: {json.dumps(result['tree']['extensions'])}\n")
    scan = result.get("scan")
    if scan:
        lines.append("## Class names observed (sampled)\n")
        lines.append("| Class | Count |\n|---|---|")
        for k, v in scan["class_names_observed"].items():
            lines.append(f"| {k} | {v} |")
        lines.append(f"\n- Files scanned: {scan['files_scanned']}, with parse problems: {scan['files_with_parse_problems']}")
        lines.append(f"- Coordinate min: {scan['box_coord_min_xmin_ymin_xmax_ymax']}")
        lines.append(f"- Coordinate max: {scan['box_coord_max_xmin_ymin_xmax_ymax']}")
    lines.append("\n## First annotation files\n")
    for f in result.get("first_files", []):
        lines.append(f"### {f['path']}\n```\n{f['content_head']}\n```")
    return "\n".join(lines) + "\n"
