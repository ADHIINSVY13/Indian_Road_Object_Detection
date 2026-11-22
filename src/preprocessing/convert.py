"""Convert the raw IDD detection release (VOC xml) into an Ultralytics YOLO dataset.

Output layout under yolo.root:
    images/<split>/<flat_stem>.jpg
    labels/<split>/<flat_stem>.txt
    manifest.csv            one row per converted image (split, stem, width, height, n_boxes)
    conversion_issues.csv   every dropped/clipped/flagged item with a reason code
    conversion_report.json  per-split counts
    idd.yaml                dataset file for Ultralytics

Why a conversion is needed at all: IDD gives pixel-corner boxes in XML (xmin, ymin, xmax, ymax),
while YOLO wants one text line per box with the class id and the box centre/size as fractions of the
image: `class x_center y_center width height`.
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

# The three splits the IDD release can contain, in processing order.
SPLITS = ("train", "val", "test")


def flat_name(stem: str) -> str:
    """Turn a nested sample name into a single flat file name.

    IDD stems look like "frontFar/BLR-2018-.../0000060". YOLO expects images and labels in one folder
    per split, so folder separators become double underscores: "frontFar__BLR-2018-...__0000060".
    Backslashes are normalised first so the result is the same on Windows and Linux.

    Input : a stem such as "frontFar/BLR-2018-03-22/0000060" (no extension).
    Output: one flat name with no folder separators.
    Why   : the same frame number (e.g. 0000060) exists in many sequences, so the sequence name must stay
            in the file name or two images would overwrite each other once everything sits in one folder.
    """
    return stem.replace("\\", "/").strip("/").replace("/", "__")


def read_split_stems(raw_root: Path, split_file: str | None) -> list[str] | None:
    """Return stems listed in the split file, or None if the file is absent.

    Returning None (instead of guessing) lets the caller report "this split is unavailable", which is
    exactly what we want for the IDD test split that ships without labels.

    Input : raw_root = folder holding the split files; split_file = e.g. "train.txt" from dataset.yaml.
    Output: a list of stems ("seq/frame"), or None when there is no such file.
    Why   : the official split files decide which image is train and which is val, so we follow them
            instead of making our own random split (that would make results hard to compare).
    """
    if not split_file:
        return None
    p = raw_root / split_file
    if not p.exists():
        return None
    stems = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            # Some releases list "name.jpg" instead of "name": drop a known extension if present.
            stems.append(os.path.splitext(line)[0] if line.lower().endswith((".jpg", ".jpeg", ".png", ".xml")) else line)
    return stems


def find_image(raw_root: Path, image_dir: str, stem: str, exts: list[str]) -> Path | None:
    """Locate the image for a stem by trying each allowed extension; None if none exists.

    Input : the raw folder, the image sub-folder name, the stem, and a list such as [".jpg", ".png"].
    Output: the full path of the first image that exists, or None.
    Why   : the XML only names the sample, not the picture format, so we look for the file ourselves.
    """
    for ext in exts:
        p = raw_root / image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def resize_image(src: Path, dst: Path, max_side: int) -> None:
    """Save a copy whose longer side is at most max_side. Labels are normalized, so they stay valid.

    Boxes are stored as fractions of the image, so shrinking the picture does not move them.

    Input : src = original image, dst = where to save, max_side = largest allowed width/height in pixels.
    Output: nothing returned; a JPEG is written to dst.
    Why   : the raw frames are 1920x1080. YOLO trains at 640 px anyway, so full-size files would only
            waste disk, Drive upload time and data-loading time (the 640 px set is 2.48 GB).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        scale = max_side / max(im.size)  # factor that brings the longer side down to max_side
        if scale < 1:                    # only ever shrink; never upscale small images
            # LANCZOS is a high-quality resampling filter that keeps edges sharp when shrinking.
            im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
        # Convert to RGB (drops any alpha/palette) and save as JPEG at quality 90.
        im.convert("RGB").save(dst, quality=90)


def place_image(src: Path, dst: Path, mode: str, max_side: int | None = None) -> None:
    """Put the image where the YOLO dataset expects it, using the cheapest safe method.

    mode "none"     : write nothing (labels only)
    max_side set    : write a resized JPEG copy (mode is ignored; resizing needs a new file)
    mode "hardlink" : a second name for the same file on disk (no extra space)
    mode "symlink"  : a shortcut pointing at the original
    anything else   : a full copy

    Why several modes: hard links cost no space, but only work inside one drive. Copy always works.
    Re-runs are cheap because an image that already exists at dst is skipped.
    """
    if mode == "none" or dst.exists():  # already there from an earlier run: skip, makes reruns cheap
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
        # Hard links cannot cross drives, so fall back to an ordinary copy.
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)  # copy2 also preserves timestamps


def convert_split(split, stems, raw_cfg, yolo_root, classes, image_mode, min_side, progress=None, max_side=None):
    """Convert every sample of one split. Returns (issues, manifest rows, class counts, statistics).

    Per sample the steps are: find image and XML -> parse XML -> read real image size -> check the XML
    size agrees -> for each object, check the class and box -> write the YOLO label file -> place the image.
    Anything unusable is skipped *and* logged with a reason code; nothing is dropped silently.

    Inputs : split name; stems to convert; raw_cfg (folder names, extensions); yolo_root (output folder);
             classes (ClassInfo list); image_mode (how to place images); min_side (tiny-box limit in px);
             progress (tqdm or None); max_side (resize limit or None).
    Outputs: issues      - list of {split, stem, severity, code, detail} rows for conversion_issues.csv
             manifest    - one row per converted image (size and number of boxes), used later by the EDA
             class_counts- how many boxes of each class were kept
             stats       - counters such as converted, skipped_* and boxes_kept
    Why    : this is the one place where raw annotation turns into training data, so every decision here
             is logged. A reader can then explain exactly why any image or box is missing.
    """
    raw_root = Path(raw_cfg["_abs_root"])  # absolute raw folder, resolved once in convert_dataset
    cls_idx = class_index(classes)  # {"car": 2, ...}; turns an XML class name into a YOLO class id
    img_out = yolo_root / "images" / split
    lbl_out = yolo_root / "labels" / split
    lbl_out.mkdir(parents=True, exist_ok=True)  # the image folder is created later by place_image

    issues, manifest = [], []
    class_counts: Counter = Counter()  # boxes kept, per class name
    stats = Counter()                  # running totals: listed, converted, skipped_*, boxes_*

    # `progress` is normally tqdm, which wraps the list and draws a progress bar.
    iterator = progress(stems, desc=f"convert {split}") if progress else stems
    for stem in iterator:
        stats["listed"] += 1  # every sample named in the split file is counted, converted or not
        xml_path = raw_root / raw_cfg["annotation_dir"] / f"{stem}.xml"
        img_path = find_image(raw_root, raw_cfg["image_dir"], stem, raw_cfg["image_extensions"])

        # ---- 1. both files must exist ----
        # A sample without its image or its XML is useless for training, so it is skipped and logged.
        if img_path is None:
            issues.append(issue(split, stem, "error", "missing_image", str(raw_root / raw_cfg["image_dir"])))
            stats["skipped_missing_image"] += 1
            continue
        if not xml_path.exists():
            issues.append(issue(split, stem, "error", "missing_annotation"))
            stats["skipped_missing_annotation"] += 1
            continue

        # ---- 2. parse the XML; parser problems are logged but do not stop us ----
        ann = parse_voc_xml(xml_path)  # never raises; anything odd comes back in ann.problems
        for p in ann.problems:
            issues.append(issue(split, stem, "error", "annotation_problem", p))

        # Image size: trust the file, flag disagreement with the xml.
        # The size read from the real JPEG is the truth; the XML's declared size can be wrong.
        try:
            # Image.open only reads the file header here, so getting the size is fast.
            with Image.open(img_path) as im:
                w, h = im.size  # real width and height in pixels (all box maths below uses these)
        except Exception as e:  # noqa: BLE001
            issues.append(issue(split, stem, "error", "corrupt_image", str(e)))
            stats["skipped_corrupt_image"] += 1
            continue
        if ann.width and ann.height and (ann.width, ann.height) != (w, h):
            issues.append(issue(split, stem, "warning", "size_mismatch_xml_vs_image",
                                f"xml={ann.width}x{ann.height} image={w}x{h}"))
            # Compare aspect ratios with cross-multiplication (avoids float division):
            # xml_w/xml_h == w/h  <=>  xml_w*h == xml_h*w.
            if ann.width * h != ann.height * w:
                # Different aspect ratio: boxes cannot be trusted for this image.
                issues.append(issue(split, stem, "error", "aspect_mismatch_skipped"))
                stats["skipped_aspect_mismatch"] += 1
                continue

        # ---- 3. convert each object to one YOLO text line ----
        lines = []
        for obj in ann.objects:
            label = normalize_label(obj.name)
            if label not in cls_idx:
                # A class name we have no id for: log it loudly rather than guess.
                issues.append(issue(split, stem, "error", "unknown_class", obj.name))
                stats["boxes_dropped_unknown_class"] += 1
                continue
            # Ask validate.box_issues what is wrong with this box: errors drop it, warnings only log it.
            found = box_issues(obj.xmin, obj.ymin, obj.xmax, obj.ymax, w, h, min_side)
            dropped = False
            for sev, code in found:
                issues.append(issue(split, stem, sev, code, f"{obj.name} {obj.xmin},{obj.ymin},{obj.xmax},{obj.ymax}"))
                dropped |= sev == "error"  # |= keeps True once any finding is an error
            if dropped:
                stats["boxes_dropped_invalid"] += 1
                continue
            # Warnings (out of bounds) are repaired by clipping the box into the image.
            x1, y1, x2, y2 = clip_box(obj.xmin, obj.ymin, obj.xmax, obj.ymax, w, h)
            if x2 <= x1 or y2 <= y1:
                # A box entirely outside the image collapses to nothing after clipping.
                issues.append(issue(split, stem, "error", "empty_after_clip", obj.name))
                stats["boxes_dropped_invalid"] += 1
                continue
            # YOLO line: class, x-centre, y-centre, width, height, all divided by the image size
            # so they are fractions in [0, 1]. Six decimals is more precision than a pixel needs.
            lines.append(
                f"{cls_idx[label]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
            )
            class_counts[label] += 1
            stats["boxes_kept"] += 1

        if not lines:
            # Still written as an empty label file: in YOLO an empty file means "background image".
            issues.append(issue(split, stem, "warning", "no_valid_objects"))
            stats["images_without_objects"] += 1

        # ---- 4. write the label file and place the image under the same flat name ----
        name = flat_name(stem)
        (lbl_out / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        place_image(img_path, img_out / f"{name}{img_path.suffix.lower()}", image_mode, max_side)
        # The manifest records the ORIGINAL image size; the EDA uses it to report resolutions.
        manifest.append({"split": split, "stem": name, "width": w, "height": h, "n_boxes": len(lines)})
        stats["converted"] += 1

    return issues, manifest, class_counts, stats


def write_dataset_yaml(yolo_root: Path, classes: list[ClassInfo], splits_done: list[str]) -> Path:
    """Write idd.yaml, the small file Ultralytics reads to find images and class names.

    Only splits that were actually converted are listed, so training never points at a missing folder.

    Input : yolo_root (dataset folder), classes (ClassInfo list), splits_done (e.g. ["train", "val"]).
    Output: the path of the written idd.yaml.
    Why   : Ultralytics needs exactly this file: the dataset root, one images/<split> path per split, and
            a {id: name} table so the trained model knows that class 7 means "autorickshaw".
    """
    data = {"path": str(yolo_root.resolve()), "names": {c.id: c.name for c in classes}}
    for s in splits_done:
        data[s] = f"images/{s}"
    out = yolo_root / "idd.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    """Write a list of dicts as a CSV file with a header row. newline="" avoids blank lines on Windows.

    Input : output path, rows (each a dict), and the column names in the order they should appear.
    Why   : CSV reports (manifest, issues) can be opened in Excel or pandas without extra tools.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def convert_dataset(raw_cfg, yolo_cfg, classes, repo_root: Path, splits=SPLITS, limit: int | None = None, progress=None):
    """Top-level conversion: run convert_split for each requested split and write all reports.

    `limit` converts only the first N samples per split, for a quick sample run before the full one.
    Returns (report dict, list of all issues).

    Inputs : raw_cfg and yolo_cfg are the "raw" and "yolo" blocks of configs/dataset.yaml; classes is the
             ClassInfo list; repo_root anchors relative paths; splits chooses which splits to convert;
             progress is the optional tqdm wrapper.
    Outputs: files under yolo_root (images, labels, manifest.csv, conversion_issues.csv,
             conversion_report.json, idd.yaml) plus the returned report and issue list.
    Why    : one entry point keeps the whole conversion reproducible from a single config file.
    """
    raw_cfg = dict(raw_cfg)  # copy so we do not modify the caller's config
    raw_root = Path(raw_cfg["root"])
    raw_root = raw_root if raw_root.is_absolute() else repo_root / raw_root
    raw_cfg["_abs_root"] = str(raw_root)  # private key: the resolved absolute raw folder
    yolo_root = Path(yolo_cfg["root"])
    yolo_root = yolo_root if yolo_root.is_absolute() else repo_root / yolo_root
    yolo_root.mkdir(parents=True, exist_ok=True)

    all_issues, all_manifest = [], []
    report = {"raw_root": str(raw_root), "splits": {}, "classes": [c.name for c in classes]}
    done = []  # splits that really got converted; only these go into idd.yaml
    for split in splits:
        stems = read_split_stems(raw_root, raw_cfg.get("split_files", {}).get(split))
        if stems is None:
            # No split file (e.g. test has no labels): say so in the report and move on.
            report["splits"][split] = {"status": "no split file found, not converted"}
            continue
        if limit:
            stems = stems[:limit]
        issues, manifest, class_counts, stats = convert_split(
            split, stems, raw_cfg, yolo_root, classes,
            yolo_cfg.get("image_mode", "hardlink"), float(yolo_cfg.get("min_box_side_px", 2)), progress,
            yolo_cfg.get("max_side") or None,  # null/0 in the config means "keep original size"
        )
        all_issues += issues
        all_manifest += manifest
        report["splits"][split] = {"status": "converted", **dict(stats), "class_counts": dict(class_counts)}
        done.append(split)

    # Reports for the whole run, all written next to the dataset so they travel with it.
    write_csv(yolo_root / "manifest.csv", all_manifest, ["split", "stem", "width", "height", "n_boxes"])
    write_csv(yolo_root / "conversion_issues.csv", all_issues, ["split", "stem", "severity", "code", "detail"])
    (yolo_root / "conversion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_dataset_yaml(yolo_root, classes, done)
    return report, all_issues
