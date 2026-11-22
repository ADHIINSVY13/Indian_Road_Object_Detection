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
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.config import load_classes, load_yaml, resolve


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--model", required=True, help="checkpoint to evaluate")
    ap.add_argument("--tag", required=True, help="name for results/eval_<tag>.json")
    ap.add_argument("--coco-baseline", action="store_true",
                    help="score a COCO-pretrained model by translating GT into COCO ids")
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, help="evaluate only the first N images of the split")
    ap.add_argument("--conf", type=float, default=0.001, help="AP needs low confidence predictions")
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--work-dir", default="data/_eval", help="scratch dataset built for the run")
    return ap.parse_args()


def build_eval_dir(yolo_root: Path, split: str, id_map: dict[int, int], out_dir: Path, limit: int | None):
    """Mirror the split with labels rewritten into the evaluation class space.

    Images are hardlinked (no disk cost), boxes whose class has no target id are dropped.
    """
    stems = sorted(p.stem for p in (yolo_root / "images" / split).glob("*.jpg"))
    if limit:
        stems = stems[:limit]
    if not stems:
        sys.exit(f"No images under {yolo_root / 'images' / split}")

    img_dst = out_dir / "images" / split
    lbl_dst = out_dir / "labels" / split
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    kept = dropped = empty = 0
    for stem in stems:
        src = yolo_root / "images" / split / f"{stem}.jpg"
        dst = img_dst / f"{stem}.jpg"
        if not dst.exists():
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)

        lines = []
        label_file = yolo_root / "labels" / split / f"{stem}.txt"
        for line in label_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cid, *rest = line.split()
            target = id_map.get(int(cid))
            if target is None:
                dropped += 1
                continue
            lines.append(" ".join([str(target), *rest]))
            kept += 1
        (lbl_dst / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if not lines:
            empty += 1

    return stems, kept, dropped, empty


def group_means(per_class: dict[str, float], groups: dict[str, str]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for name, ap in per_class.items():
        buckets.setdefault(groups.get(name, "ungrouped"), []).append(ap)
    return {g: round(sum(v) / len(v), 4) for g, v in sorted(buckets.items()) if v}


def main():
    args = parse_args()
    cfg = load_yaml(resolve(args.config))
    classes = load_classes(resolve(cfg["class_mapping"]))
    groups = {c.name: c.group for c in classes}
    yolo_root = resolve(cfg["yolo"]["root"])

    if args.coco_baseline:
        # Only an exact one-to-one COCO class can carry a baseline score. rider->person and
        # traffic sign->stop sign are deliberately excluded: they would score against the wrong GT.
        direct = [c for c in classes if c.group == "common" and len(c.coco_ids) == 1]
        if not direct:
            sys.exit("No direct-match classes in the mapping; --coco-baseline cannot be scored")
        id_map = {c.id: c.coco_ids[0] for c in direct}
        not_evaluable = {
            c.name: ("no COCO equivalent" if not c.coco_ids else c.note or f"not a direct match: {c.coco_names}")
            for c in classes if c not in direct
        }
        eval_names = None  # taken from the model once it is loaded
    else:
        id_map = {c.id: c.id for c in classes}
        not_evaluable = {}
        eval_names = [c.name for c in classes]

    from ultralytics import YOLO  # late import so --help works without torch installed
    model = YOLO(args.model)

    if eval_names is None:
        eval_names = [model.names[i] for i in range(len(model.names))]
        missing = [c.name for c in direct if c.coco_ids[0] >= len(eval_names)]
        if missing:
            sys.exit(f"coco_ids out of range for {args.model}: {missing}")

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

    metrics = model.val(data=str(data_yaml), split=args.split, imgsz=args.imgsz, batch=args.batch,
                        device=args.device, conf=args.conf, iou=args.iou, plots=False, verbose=False)

    box = metrics.box
    idx = [int(i) for i in box.ap_class_index]
    aps = [float(v) for v in box.maps]

    if args.coco_baseline:
        back = {c.coco_ids[0]: c.name for c in direct}
        per_class = {back[i]: round(a, 4) for i, a in zip(idx, aps) if i in back}
    else:
        per_class = {eval_names[i]: round(a, 4) for i, a in zip(idx, aps) if i < len(eval_names)}

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
        "overall": {k: round(float(v), 4) for k, v in metrics.results_dict.items()},
        "per_class_ap50_95": dict(sorted(per_class.items(), key=lambda kv: -kv[1])),
        "not_evaluable": not_evaluable,
        "per_group_ap50_95": group_means(per_class, groups),
        "run_dir": str(metrics.save_dir),
        "command": " ".join(sys.argv),
    }

    out = resolve("results") / f"eval_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary["per_group_ap50_95"], indent=2))
    print(json.dumps(summary["overall"], indent=2))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
