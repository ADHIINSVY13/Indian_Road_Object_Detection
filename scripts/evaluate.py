"""Step 5: evaluate a detector on the IDD val split, and build the COCO baseline number.

    python scripts/evaluate.py --model models/best.pt --tag finetuned
    python scripts/evaluate.py --model yolo11n.pt --tag coco_baseline --coco-baseline

Both modes write `results/eval_<tag>.json` in the same shape, so the two runs diff directly and
the per-group comparison in the report needs no manual transcribing.

Inference only: this runs on CPU, which is what makes the comparison possible without a GPU.

The `--coco-baseline` mode exists because a COCO model cannot be scored on IDD labels as they are.
It predicts 80 COCO classes, so the ground truth is translated into COCO ids first, and only for
the classes the mapping calls a direct match (`group: common` with exactly one `coco_id`).
`india_specific` and `ambiguous` classes are reported as N/A with the reason, never as 0.

How the script works, step by step (useful when explaining it in a viva):
  1. Read the class mapping and decide which IDD classes can be scored for this model.
  2. Build a scratch copy of the val split whose label files use the MODEL's class ids.
  3. Let Ultralytics run the model over every image and compute precision, recall and AP.
  4. Re-attach the right class NAME to each number (this is where the old indexing bug lived).
  5. Write one JSON file with the numbers plus every setting that influenced them.

Vocabulary used below:
  * IoU   - overlap between a predicted box and a true box (intersection / union).
  * AP    - area under one class's precision-recall curve; mAP is the mean over classes.
  * AP50  - AP when a detection counts as correct at IoU >= 0.50.
  * AP50-95 - AP averaged over IoU thresholds 0.50, 0.55, ... 0.95 (stricter, the main metric here).
  * NMS   - non-maximum suppression, which removes duplicate boxes for the same object.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

# Scripts live in <repo>/scripts, so add the repo root to Python's import search path. That lets
# `from src... import ...` work when the script is run directly (python scripts/evaluate.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.config import load_classes, load_yaml, resolve


def parse_args():
    """Define and read the command-line options. RawDescriptionHelpFormatter keeps the docstring's
    line breaks intact in --help output."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Dataset settings file: tells the script where the converted YOLO dataset and class mapping are.
    ap.add_argument("--config", default="configs/dataset.yaml")
    # Which weights file to test (our fine-tuned best.pt, or the stock COCO yolo11n.pt).
    ap.add_argument("--model", required=True, help="checkpoint to evaluate")
    # A short label; it becomes the output file name results/eval_<tag>.json.
    ap.add_argument("--tag", required=True, help="name for results/eval_<tag>.json")
    # Switch on the special scoring mode needed for a COCO model (see the module docstring).
    ap.add_argument("--coco-baseline", action="store_true",
                    help="score a COCO-pretrained model by translating GT into COCO ids")
    # Which part of the dataset to score; "val" is the only labelled split we can evaluate on.
    ap.add_argument("--split", default="val")
    # Image size the model sees; 640 matches the size used in training and in the conversion step.
    ap.add_argument("--imgsz", type=int, default=640)
    # Images per forward pass. Only affects speed and memory, never the metric values.
    ap.add_argument("--batch", type=int, default=16)
    # "cpu" works on any laptop; evaluation is inference only, so no GPU is required.
    ap.add_argument("--device", default="cpu")
    # Quick smoke test: evaluate only the first N images instead of all 10,225.
    ap.add_argument("--limit", type=int, help="evaluate only the first N images of the split")
    # Confidence cut-off. It is kept tiny on purpose: AP is computed by sweeping the threshold, so
    # low-confidence detections must be present or the precision-recall curve is cut short.
    ap.add_argument("--conf", type=float, default=0.001, help="AP needs low confidence predictions")
    # IoU threshold used by NMS to decide two boxes are duplicates of the same object.
    ap.add_argument("--iou", type=float, default=0.7)
    # Scratch folder for the temporary re-labelled copy of the split (safe to delete afterwards).
    ap.add_argument("--work-dir", default="data/_eval", help="scratch dataset built for the run")
    return ap.parse_args()


def build_eval_dir(yolo_root: Path, split: str, id_map: dict[int, int], out_dir: Path, limit: int | None):
    """Mirror the split with labels rewritten into the evaluation class space.

    Images are hardlinked (no disk cost), boxes whose class has no target id are dropped.

    id_map translates an IDD class id into the id the evaluated model understands. For the
    fine-tuned model it is the identity (0->0, 1->1, ...). For the COCO baseline it maps e.g.
    IDD "car" (id 2) to COCO "car" (id 2) and omits classes COCO cannot name.
    Returns (image stems, boxes kept, boxes dropped, images left without any box).

    Why this function exists: Ultralytics scores a model by comparing its predictions with the label
    files next to the images, and the class ids in those files must be the ones the MODEL outputs.
    A COCO model outputs COCO ids (car = 2), our files use IDD ids, so we write a translated copy
    instead of touching the original dataset.

    Why hard links: a hard link is a second file name pointing at the same bytes on disk, so the
    10,225 val images appear in the scratch folder instantly and use no extra space. If the two
    folders are on different drives linking is impossible, so we copy instead.
    """
    # A "stem" is the file name without extension, e.g. "frontNear__BLR-2018-..__000804_r".
    # Sorting makes the order (and therefore --limit) deterministic from run to run.
    stems = sorted(p.stem for p in (yolo_root / "images" / split).glob("*.jpg"))
    if limit:
        stems = stems[:limit]
    if not stems:
        sys.exit(f"No images under {yolo_root / 'images' / split}")

    # Ultralytics expects the layout <root>/images/<split> and <root>/labels/<split>.
    img_dst = out_dir / "images" / split
    lbl_dst = out_dir / "labels" / split
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    # Counters for the report: boxes kept, boxes thrown away (class not scorable), and images
    # that end up with no scorable box at all.
    kept = dropped = empty = 0
    for stem in stems:
        src = yolo_root / "images" / split / f"{stem}.jpg"
        dst = img_dst / f"{stem}.jpg"
        if not dst.exists():  # re-running the script skips images that are already linked
            try:
                os.link(src, dst)           # hard link: a second name for the same bytes, no copy
            except OSError:
                shutil.copy2(src, dst)      # fall back to a real copy if linking is not possible

        # Each label line is "class cx cy w h" (box centre and size as fractions of the image).
        # We keep cx cy w h untouched and only swap the class id for the model's id.
        lines = []
        label_file = yolo_root / "labels" / split / f"{stem}.txt"
        for line in label_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue  # ignore blank lines
            cid, *rest = line.split()       # first token = class id, the rest = box numbers
            target = id_map.get(int(cid))   # None when this class has no id in the eval space
            if target is None:
                dropped += 1
                continue
            lines.append(" ".join([str(target), *rest]))  # same box, class id rewritten
            kept += 1
        # Write the translated label file. An image with zero scorable boxes still gets an (empty)
        # file, because a missing label file would make Ultralytics treat the image as unlabelled.
        (lbl_dst / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if not lines:
            empty += 1

    return stems, kept, dropped, empty


def ap_by_class_id(ap_class_index, maps) -> dict[int, float]:
    """AP50-95 per class id that has ground truth.

    Ultralytics' `box.maps` has one slot per class in the label space, indexed by class id, and
    classes with no ground truth hold the overall mAP. `ap_class_index` lists only the classes
    that do have ground truth. Pairing them by position (`zip`) shifts every value whenever a
    class id is absent, which is the normal case for the 80-class COCO baseline, so look each
    class up by its id.

    Inputs : ap_class_index - ids of the classes that have ground truth, e.g. [0, 1, 5, 7]
             maps           - list with one AP per class id of the whole label space (length 80 for COCO)
    Output : {class_id: AP}, e.g. {0: 0.10, 1: 0.08, 5: 0.35, 7: 0.16}

    The old bug (fixed): the code did zip(ap_class_index, maps), pairing the 1st present class with
    maps[0], the 2nd with maps[1] and so on. For COCO only 8 of 80 ids are present, so "bus" (id 5)
    received the AP of id 1 and the baseline table showed wrong per-class numbers. Looking each
    value up as maps[class_id] removes the problem. tests/test_pipeline.py has a regression test.
    """
    return {int(c): float(maps[int(c)]) for c in ap_class_index}


def compact_by_class_id(ap_class_index, values) -> dict[int, float]:
    """Per-class values that Ultralytics stores *compactly*: `box.p`, `box.r` and `box.ap50` have one
    entry per class that has ground truth, in the order of `ap_class_index` (unlike `box.maps`,
    which is indexed by class id).

    Inputs : ap_class_index - ids of the classes that have ground truth, e.g. [0, 1, 5, 7]
             values         - a COMPACT list with exactly one number per entry of ap_class_index
    Output : {class_id: value}

    Contrast with ap_by_class_id: there `maps` is long (one slot per class id), so we index into it;
    here `values` is short (one slot per present class), so pairing by position with zip is correct.
    Using the wrong pairing style for either array is exactly what caused the earlier bug.
    """
    return {int(c): float(v) for c, v in zip(ap_class_index, values)}


def group_means(per_class: dict[str, float], groups: dict[str, str]) -> dict[str, float]:
    """Average the per-class APs inside each class group (common / ambiguous / india_specific).

    Inputs : per_class - {class name: AP}, only for classes that were actually scored
             groups    - {class name: group name}, taken from configs/class_mapping.yaml
    Output : {group name: mean AP rounded to 4 places}

    Why a group mean: the project question is "does fine-tuning help India-specific classes more
    than common ones?", so the group average is the number that answers it. A group with no scored
    class simply does not appear in the result (it is N/A, not 0).
    """
    buckets: dict[str, list[float]] = {}
    for name, ap in per_class.items():
        # setdefault creates the empty list the first time a group is seen, then we append to it.
        buckets.setdefault(groups.get(name, "ungrouped"), []).append(ap)
    return {g: round(sum(v) / len(v), 4) for g, v in sorted(buckets.items()) if v}


def main():
    """Run one full evaluation: set up classes, build the scratch dataset, validate, write JSON."""
    args = parse_args()
    cfg = load_yaml(resolve(args.config))
    # The 15 IDD classes in id order, each with its group and COCO equivalents (class_mapping.yaml).
    classes = load_classes(resolve(cfg["class_mapping"]))
    groups = {c.name: c.group for c in classes}   # class name -> common / ambiguous / india_specific
    yolo_root = resolve(cfg["yolo"]["root"])      # where the converted 640 px dataset lives

    # ---- decide which classes are scored, and how IDD ids map onto the model's ids ----
    if args.coco_baseline:
        # Only an exact one-to-one COCO class can carry a baseline score. rider->person and
        # traffic sign->stop sign are deliberately excluded: they would score against the wrong GT.
        # "direct" = group common AND exactly one COCO id: person, car, truck, bus, motorcycle,
        # bicycle, traffic light, train. Scoring anything else would compare against the wrong truth.
        direct = [c for c in classes if c.group == "common" and len(c.coco_ids) == 1]
        if not direct:
            sys.exit("No direct-match classes in the mapping; --coco-baseline cannot be scored")
        id_map = {c.id: c.coco_ids[0] for c in direct}   # IDD id -> COCO id
        # Everything not scored gets a written reason, so the report can say WHY it is N/A.
        not_evaluable = {
            c.name: ("no COCO equivalent" if not c.coco_ids else c.note or f"not a direct match: {c.coco_names}")
            for c in classes if c not in direct
        }
        eval_names = None  # taken from the model once it is loaded
    else:
        id_map = {c.id: c.id for c in classes}  # identity: the fine-tuned model uses IDD ids
        not_evaluable = {}
        eval_names = [c.name for c in classes]

    from ultralytics import YOLO  # late import so --help works without torch installed
    model = YOLO(args.model)      # load the weights file (a .pt checkpoint) into a ready-to-use model

    if eval_names is None:
        # For the baseline, the class names are the model's own 80 COCO names.
        eval_names = [model.names[i] for i in range(len(model.names))]
        missing = [c.name for c in direct if c.coco_ids[0] >= len(eval_names)]
        if missing:
            sys.exit(f"coco_ids out of range for {args.model}: {missing}")

    # ---- build a temporary copy of the split whose labels use the model's class space ----
    out_dir = resolve(args.work_dir) / args.tag
    stems, kept, dropped, empty = build_eval_dir(yolo_root, args.split, id_map, out_dir, args.limit)

    data_yaml = out_dir / "eval.yaml"
    # check_det_dataset rejects a YAML without `train`, even though val() never reads it: point it
    # at the same images so the file is structurally complete.
    data_yaml.write_text(yaml.safe_dump(
        {"path": str(out_dir.resolve()), "train": f"images/{args.split}",
         args.split: f"images/{args.split}", "names": eval_names},
        sort_keys=False), encoding="utf-8")

    print(f"model     : {args.model}")
    print(f"mode      : {'coco_baseline' if args.coco_baseline else 'idd'}")
    print(f"images    : {len(stems)} ({args.split})")
    print(f"boxes kept: {kept}   dropped (no target class): {dropped}   images left empty: {empty}")
    print(f"classes   : {len(eval_names)} in the eval space")

    # ---- run Ultralytics' validator: it predicts on every image and computes the metrics ----
    # conf=0.001 keeps even very unsure detections, because AP integrates over the whole
    # precision-recall curve and needs them; iou=0.7 is the NMS (duplicate-removal) threshold.
    metrics = model.val(data=str(data_yaml), split=args.split, imgsz=args.imgsz, batch=args.batch,
                        device=args.device, conf=args.conf, iou=args.iou, plots=False, verbose=False)

    # ---- pull the per-class numbers out, pairing each with the right class id ----
    # metrics.box holds the detection metrics. Its important arrays:
    #   ap_class_index - ids of classes that have ground truth in the scored images
    #   maps           - AP50-95 per class id (LONG array, indexed by class id)
    #   p, r, ap50     - precision, recall and AP50 per present class (SHORT arrays, same order as
    #                    ap_class_index). p and r are read at the confidence that maximises F1.
    ap_by_id = ap_by_class_id(metrics.box.ap_class_index, metrics.box.maps)

    box = metrics.box
    p_by_id = compact_by_class_id(box.ap_class_index, box.p)       # precision per class
    r_by_id = compact_by_class_id(box.ap_class_index, box.r)       # recall per class
    ap50_by_id = compact_by_class_id(box.ap_class_index, box.ap50)  # AP at IoU 0.50

    # name_of converts a model class id back to the IDD class name for reporting.
    if args.coco_baseline:
        back = {c.coco_ids[0]: c.name for c in direct}  # COCO id -> IDD name
        name_of = lambda i: back.get(i)  # noqa: E731 - only the direct-match COCO ids are reported
    else:
        name_of = lambda i: eval_names[i] if i < len(eval_names) else None  # noqa: E731

    # per_class: {class name: AP50-95}. Ids with no IDD name (the 72 unused COCO ids) are skipped.
    per_class = {name_of(i): round(a, 4) for i, a in ap_by_id.items() if name_of(i)}
    # per_class_detail: the same classes with precision, recall, AP50 and AP50-95 side by side.
    per_class_detail = {
        name_of(i): {"precision": round(p_by_id[i], 4), "recall": round(r_by_id[i], 4),
                     "ap50": round(ap50_by_id[i], 4), "ap50_95": round(ap_by_id[i], 4)}
        for i in ap_by_id if name_of(i)
    }

    # ---- assemble the summary file: every setting that affects the numbers is recorded in it ----
    summary = {
        "tag": args.tag,
        "model": args.model,
        "mode": "coco_baseline" if args.coco_baseline else "idd",
        "split": args.split,
        "images": len(stems),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "boxes_kept": kept,
        "boxes_dropped_no_target_class": dropped,
        "images_left_empty": empty,
        # Overall precision, recall, mAP50 and mAP50-95 over ALL scored classes. For the baseline
        # this covers only 8 classes, so it must not be compared directly with the fine-tuned value.
        "overall": {k: round(float(v), 4) for k, v in metrics.results_dict.items()},
        # Sorted best-first so the JSON reads like a leaderboard.
        "per_class_ap50_95": dict(sorted(per_class.items(), key=lambda kv: -kv[1])),
        "per_class_detail": dict(sorted(per_class_detail.items(), key=lambda kv: -kv[1]["ap50_95"])),
        # {class name: reason} for classes the baseline could not be scored on (the "why N/A" list).
        "not_evaluable": not_evaluable,
        "per_group_ap50_95": group_means(per_class, groups),
        "run_dir": str(metrics.save_dir),
        "command": " ".join(sys.argv),  # the exact command line, so the run can be repeated
    }

    # Save as readable JSON (indent=2); compare_evals.py reads these files to build the final table.
    out = resolve("results") / f"eval_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Echo the headline numbers to the terminal as a quick sanity check.
    print(json.dumps(summary["per_group_ap50_95"], indent=2))
    print(json.dumps(summary["overall"], indent=2))
    print(f"written to {out}")


# Run main() only when executed as a script, not when imported (e.g. by the tests).
if __name__ == "__main__":
    main()
