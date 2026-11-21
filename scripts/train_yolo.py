"""Step 4: fine-tune a COCO-pretrained YOLO detector on the converted IDD dataset.

This is the same entry point on a Colab GPU and on a CPU-only laptop, so the run that produced
the reported numbers can be reproduced. It trains, validates on `val`, and writes a summary whose
per-class and per-group APs are what the baseline-vs-finetuned comparison is built from.

    python scripts/train_yolo.py --dry-run                  # check the setup, train nothing
    python scripts/train_yolo.py --epochs 30 --batch 16     # Colab T4

Nothing here is timed or tuned automatically: every hyper-parameter the summary records is an
argument, so a number in the report can always be traced back to a command.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.config import load_classes, load_yaml, resolve
from src.preprocessing.convert import write_dataset_yaml


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--data-root", help="override yolo.root from the config, e.g. /content/idd_yolo on Colab")
    ap.add_argument("--model", default="yolo11n.pt", help="COCO-pretrained checkpoint to start from")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2, help="dataloader workers; Colab free has 2 vCPU")
    ap.add_argument("--device", default="", help="'' = auto, '0' = first GPU, 'cpu' to force CPU")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=10, help="early-stopping patience in epochs")
    ap.add_argument("--project", default="results/train", help="run directory; point at Drive on Colab")
    ap.add_argument("--name", default="idd_finetune")
    ap.add_argument("--resume", action="store_true", help="resume from <project>/<name>/weights/last.pt")
    ap.add_argument("--dry-run", action="store_true", help="validate the dataset and print the plan, then exit")
    return ap.parse_args()


def split_counts(yolo_root: Path) -> dict[str, int]:
    """Images actually present per split, so a missing split is caught before training."""
    counts = {}
    for d in sorted((yolo_root / "images").glob("*")):
        if d.is_dir():
            counts[d.name] = sum(1 for _ in d.glob("*.jpg"))
    return counts


def mean_ap_per_group(per_class_ap: dict[str, float], groups: dict[str, str]) -> dict[str, float]:
    """Mean AP50-95 per class group: the comparison the project is actually making."""
    buckets: dict[str, list[float]] = {}
    for name, ap in per_class_ap.items():
        buckets.setdefault(groups.get(name, "ungrouped"), []).append(ap)
    return {g: round(sum(v) / len(v), 4) for g, v in sorted(buckets.items()) if v}


def main():
    args = parse_args()
    cfg = load_yaml(resolve(args.config))
    classes = load_classes(resolve(cfg["class_mapping"]))
    groups = {c.name: c.group for c in classes}
    # The config points at <repo>/data/yolo, which is where the laptop keeps it. Colab unpacks to
    # /content/idd_yolo instead, so allow an explicit override rather than relying on symlinks.
    root_val = args.data_root or cfg["yolo"]["root"]
    yolo_root = Path(root_val) if Path(root_val).is_absolute() else resolve(root_val)

    counts = split_counts(yolo_root)
    if not counts.get("train"):
        sys.exit(f"No converted images under {yolo_root / 'images'}. Run scripts/prepare_dataset.py first.")

    # idd.yaml is committed with the absolute path of whoever converted the data, so rewrite it
    # here: the same checkout then trains on Colab, on the laptop, or anywhere else.
    data_yaml = write_dataset_yaml(yolo_root, classes, list(counts))

    print(f"dataset  : {data_yaml}")
    print(f"splits   : {counts}")
    print(f"classes  : {len(classes)}")
    print(f"model    : {args.model}")
    print(f"run dir  : {resolve(args.project) / args.name}")
    if args.dry_run:
        print("dry run: setup looks usable, nothing was trained")
        return

    from ultralytics import YOLO  # imported late so --dry-run works without torch installed

    run_dir = resolve(args.project) / args.name
    if args.resume:
        ckpt = run_dir / "weights" / "last.pt"
        if not ckpt.exists():
            sys.exit(f"--resume given but {ckpt} is missing")
        model = YOLO(str(ckpt))
        model.train(resume=True)
        save_dir = run_dir
    else:
        model = YOLO(args.model)
        model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            seed=args.seed,
            patience=args.patience,
            project=str(resolve(args.project)),
            name=args.name,
            exist_ok=True,
            pretrained=True,
            # Augmentation kept close to Ultralytics defaults on purpose: the claim under test is
            # "does IDD fine-tuning help", not "what is the best possible augmentation recipe".
            plots=True,
        )
        save_dir = run_dir

    metrics = model.val(data=str(data_yaml), split="val", imgsz=args.imgsz, batch=args.batch, device=args.device)

    per_class = {}
    box = getattr(metrics, "box", None)
    if box is not None and getattr(box, "maps", None) is not None:
        idx = list(getattr(box, "ap_class_index", range(len(box.maps))))
        per_class = {classes[i].name: round(float(m), 4) for i, m in zip(idx, box.maps)}

    summary = {
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "data": str(data_yaml),
        "images": counts,
        "metrics": {k: round(float(v), 4) for k, v in metrics.results_dict.items()},
        "per_class_ap50_95": per_class,
        "per_group_ap50_95": mean_ap_per_group(per_class, groups),
        "run_dir": str(save_dir),
        "results_csv": str(Path(save_dir) / "results.csv"),
        "weights_best": str(Path(save_dir) / "weights" / "best.pt"),
    }

    out = resolve("results") / f"{args.name}_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary["per_group_ap50_95"], indent=2))
    print(f"summary written to {out}")


if __name__ == "__main__":
    main()
