"""Step 7: qualitative comparison, ground truth | COCO baseline | IDD fine-tuned, on the same images.

    python scripts/qualitative.py --finetuned models/best.pt --baseline yolo11n.pt

Two galleries are written to `results/qualitative/`:

  random_*.jpg   N images drawn with a fixed seed from the val split. Nothing is hand-picked, so the
                 gallery is an honest sample and not a highlight reel.
  failure_*.jpg  the images where the fine-tuned model finds the smallest share of the ground truth
                 (class-aware recall at IoU 0.5), chosen by that rule from a seeded candidate pool.

`results/qualitative/index.json` records the seed, the stems, the thresholds and the measured stats
of every image (box count, median box size, recall), so each picture can be traced. Nothing about
*why* an image failed is asserted: only the measured size/crowding numbers are stored, and any
interpretation in the report is marked as a hypothesis.

The baseline is drawn with its raw 80 COCO class names: it has no IDD classes, which is the point.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.config import load_classes, load_yaml, resolve
from src.visualization.qualitative import (draw_boxes, median_box_area_fraction, read_yolo_labels,
                                           recall_at_iou, side_by_side)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--finetuned", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n-random", type=int, default=8)
    ap.add_argument("--n-failure", type=int, default=8)
    ap.add_argument("--pool", type=int, default=400, help="seeded candidate images scanned for failures")
    ap.add_argument("--min-gt", type=int, default=5, help="ignore near-empty images when ranking failures")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.25, help="display threshold (not the AP threshold)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/qualitative")
    return ap.parse_args()


def predict(model, path: Path, conf: float, device: str):
    r = model.predict(str(path), conf=conf, imgsz=640, device=device, verbose=False)[0]
    return [(int(c), tuple(map(float, xyxy)), float(s))
            for c, xyxy, s in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]


def main():
    args = parse_args()
    cfg = load_yaml(resolve(args.config))
    classes = load_classes(resolve(cfg["class_mapping"]))
    idd_names = [c.name for c in classes]
    yolo_root = resolve(cfg["yolo"]["root"])

    stems = sorted(p.stem for p in (yolo_root / "images" / args.split).glob("*.jpg"))
    if not stems:
        sys.exit(f"No images under {yolo_root / 'images' / args.split}")
    rng = random.Random(args.seed)
    random_stems = rng.sample(stems, args.n_random)
    pool = [s for s in rng.sample(stems, min(args.pool, len(stems))) if s not in random_stems]

    from ultralytics import YOLO  # late import so --help works without torch
    fine, base = YOLO(args.finetuned), YOLO(args.baseline)
    coco_names = [base.names[i] for i in range(len(base.names))]

    def load(stem):
        img_path = yolo_root / "images" / args.split / f"{stem}.jpg"
        img = Image.open(img_path)
        gt = read_yolo_labels(yolo_root / "labels" / args.split / f"{stem}.txt", img.width, img.height)
        return img_path, img, gt

    # ---- failure candidates: score the pool with the fine-tuned model only ----
    scored = []
    for stem in pool:
        img_path, img, gt = load(stem)
        if len(gt) < args.min_gt:
            continue
        pred = predict(fine, img_path, args.conf, args.device)
        rec = recall_at_iou(gt, [(c, b) for c, b, _ in pred])
        scored.append((rec, -len(gt), stem))
    scored.sort()
    failure_stems = [s for _, _, s in scored[:args.n_failure]]

    out_dir = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = {"seed": args.seed, "split": args.split, "conf_display": args.conf, "recall_iou": 0.5,
             "finetuned": args.finetuned, "baseline": args.baseline,
             "failure_rule": f"lowest class-aware recall@IoU0.5 among {len(scored)} pool images with >= {args.min_gt} GT boxes",
             "random": [], "failure": []}

    def render(stem, prefix, i):
        img_path, img, gt = load(stem)
        ft = predict(fine, img_path, args.conf, args.device)
        bs = predict(base, img_path, args.conf, args.device)
        rec = recall_at_iou(gt, [(c, b) for c, b, _ in ft])
        panels = [
            draw_boxes(img, [(c, b, None) for c, b in gt], idd_names, f"ground truth ({len(gt)} boxes)"),
            draw_boxes(img, bs, coco_names, f"COCO baseline ({len(bs)} detections)"),
            draw_boxes(img, ft, idd_names, f"IDD fine-tuned ({len(ft)} detections, recall@0.5 {rec:.2f})"),
        ]
        name = f"{prefix}_{i:02d}.jpg"
        side_by_side(panels).save(out_dir / name, quality=85)
        return {"file": name, "stem": stem, "gt_boxes": len(gt), "baseline_detections": len(bs),
                "finetuned_detections": len(ft), "finetuned_recall_iou50": round(rec, 3),
                "median_gt_box_area_fraction": round(median_box_area_fraction(gt, img.width, img.height) or 0, 5)}

    for i, stem in enumerate(random_stems):
        index["random"].append(render(stem, "random", i))
        print("random", i, stem)
    for i, stem in enumerate(failure_stems):
        index["failure"].append(render(stem, "failure", i))
        print("failure", i, stem)

    (out_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
