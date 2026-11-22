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

# argparse reads the command-line options, json writes the summary file, sys gives sys.exit/sys.path,
# and Path lets us build file paths that work on both Windows (laptop) and Linux (Colab).
import argparse
import json
import sys
from pathlib import Path

# Make `src` importable when this file is run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# load_yaml/load_classes read our config files, resolve() turns a repo-relative path into an absolute one,
# and write_dataset_yaml() produces the small YAML file that tells Ultralytics where the data is.
from src.data.config import load_classes, load_yaml, resolve
from src.preprocessing.convert import write_dataset_yaml


def parse_args():
    """Define the command-line options. Defaults reflect the settings used for the reported run."""
    # RawDescriptionHelpFormatter keeps the line breaks of the module docstring in `--help` output.
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Dataset configuration file (paths, image size, class-mapping location).
    ap.add_argument("--config", default="configs/dataset.yaml")
    # Where the converted YOLO images/labels live. On Colab they are unpacked to a fast local disk.
    ap.add_argument("--data-root", help="override yolo.root from the config, e.g. /content/idd_yolo on Colab")
    # Starting weights. yolo11n is the smallest (nano) YOLO11 model, small enough for a free Colab GPU.
    ap.add_argument("--model", default="yolo11n.pt", help="COCO-pretrained checkpoint to start from")
    # One epoch = one full pass over the training images.
    ap.add_argument("--epochs", type=int, default=30)
    # Images are resized to imgsz x imgsz pixels before entering the network (the data is already 640 px).
    ap.add_argument("--imgsz", type=int, default=640)
    # Images per optimisation step; larger needs more GPU memory (16 fits a 15 GB T4 at 640 px).
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2, help="dataloader workers; Colab free has 2 vCPU")
    ap.add_argument("--device", default="", help="'' = auto, '0' = first GPU, 'cpu' to force CPU")
    # Random seed so that data order and augmentation are repeatable (a fixed seed is part of honest reporting).
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=10, help="early-stopping patience in epochs")
    ap.add_argument("--cache", default="false", choices=["false", "ram", "disk"],
                    help="cache decoded images; the 2.5 GB dataset fits in RAM on Colab and Kaggle")
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="train on this fraction of the split, for a compute-limited run")
    # Ultralytics writes weights/, results.csv and plots to <project>/<name>/. On Colab point --project at
    # Google Drive, so checkpoints survive when the Colab session is disconnected.
    ap.add_argument("--project", default="results/train", help="run directory; point at Drive on Colab")
    ap.add_argument("--name", default="idd_finetune")
    ap.add_argument("--resume", action="store_true", help="resume from <project>/<name>/weights/last.pt")
    # A flag with no value: present = True. Lets you test the whole setup in seconds without training.
    ap.add_argument("--dry-run", action="store_true", help="validate the dataset and print the plan, then exit")
    return ap.parse_args()


def split_counts(yolo_root: Path) -> dict[str, int]:
    """Images actually present per split, so a missing split is caught before training."""
    counts = {}
    for d in sorted((yolo_root / "images").glob("*")):
        if d.is_dir():
            counts[d.name] = sum(1 for _ in d.glob("*.jpg"))  # count files without loading a list
    return counts


def mean_ap_per_group(per_class_ap: dict[str, float], groups: dict[str, str]) -> dict[str, float]:
    """Mean AP50-95 per class group: the comparison the project is actually making."""
    # Collect the AP values of every class under its group name (common / ambiguous / india_specific).
    buckets: dict[str, list[float]] = {}
    for name, ap in per_class_ap.items():
        buckets.setdefault(groups.get(name, "ungrouped"), []).append(ap)
    # Average each group (sum / count) and round for readable JSON; empty groups are skipped.
    return {g: round(sum(v) / len(v), 4) for g, v in sorted(buckets.items()) if v}


def main():
    args = parse_args()
    # Read the dataset config and the 15-class list (names, groups, COCO ids) from YAML.
    cfg = load_yaml(resolve(args.config))
    classes = load_classes(resolve(cfg["class_mapping"]))
    # class name -> group, used later to average AP per group.
    groups = {c.name: c.group for c in classes}
    # The config points at <repo>/data/yolo, which is where the laptop keeps it. Colab unpacks to
    # /content/idd_yolo instead, so allow an explicit override rather than relying on symlinks.
    root_val = args.data_root or cfg["yolo"]["root"]
    yolo_root = Path(root_val) if Path(root_val).is_absolute() else resolve(root_val)

    # ---- sanity check before spending hours of GPU time ----
    # If there are no training images the run would fail much later; stop here with a clear message.
    counts = split_counts(yolo_root)
    if not counts.get("train"):
        sys.exit(f"No converted images under {yolo_root / 'images'}. Run scripts/prepare_dataset.py first.")

    # idd.yaml is committed with the absolute path of whoever converted the data, so rewrite it
    # here: the same checkout then trains on Colab, on the laptop, or anywhere else.
    data_yaml = write_dataset_yaml(yolo_root, classes, list(counts))

    # Print the plan so the log of every run records exactly what was trained on.
    print(f"dataset  : {data_yaml}")
    print(f"splits   : {counts}")
    print(f"classes  : {len(classes)}")
    print(f"model    : {args.model}")
    print(f"run dir  : {resolve(args.project) / args.name}")
    # --dry-run stops here: everything above (paths, split counts, YAML) has been checked, nothing trained.
    if args.dry_run:
        print("dry run: setup looks usable, nothing was trained")
        return

    from ultralytics import YOLO  # imported late so --dry-run works without torch installed

    # <project>/<name>: the folder Ultralytics writes weights, plots and results.csv into.
    run_dir = resolve(args.project) / args.name
    if args.resume:
        # Continue an interrupted run (e.g. Colab disconnected) from its last saved checkpoint.
        ckpt = run_dir / "weights" / "last.pt"
        if not ckpt.exists():
            sys.exit(f"--resume given but {ckpt} is missing")
        # last.pt stores the weights AND the optimiser state, epoch number and original settings, so
        # model.train(resume=True) needs no other arguments and carries on from the next epoch.
        model = YOLO(str(ckpt))
        model.train(resume=True)
        save_dir = run_dir
    else:
        # Start from the COCO-pretrained weights; Ultralytics swaps the 80-class output layer for
        # a 15-class one automatically because the dataset YAML lists 15 names.
        model = YOLO(args.model)
        model.train(
            data=str(data_yaml),    # dataset YAML: image folders and the 15 class names
            epochs=args.epochs,     # number of passes over the training set
            imgsz=args.imgsz,       # network input size in pixels
            batch=args.batch,       # images per training step
            workers=args.workers,   # CPU processes that load and augment images in parallel
            device=args.device,     # '' lets Ultralytics pick the GPU if there is one, else the CPU
            seed=args.seed,         # fixed random seed for repeatability
            patience=args.patience,  # stop early if validation stops improving for this many epochs
            # Cache decoded images in RAM/disk so the slow JPEG decoding is not repeated every epoch.
            cache=False if args.cache == "false" else args.cache,
            fraction=args.fraction,  # use only this share of the training data (1.0 = all of it)
            project=str(resolve(args.project)),  # parent folder of the run
            name=args.name,                      # run folder name inside the project folder
            exist_ok=True,      # reuse the run folder instead of creating name2, name3, ...
            pretrained=True,    # initialise from the COCO weights (transfer learning), not from scratch
            # Augmentation kept close to Ultralytics defaults on purpose: the claim under test is
            # "does IDD fine-tuning help", not "what is the best possible augmentation recipe".
            plots=True,   # save confusion matrix, PR/F1 curves and training curves into the run folder
        )
        save_dir = run_dir

    # ---- final validation of the best checkpoint on the val split ----
    # model.val() runs inference on all val images and computes precision, recall and mAP.
    # After train(), the model object holds the best checkpoint's weights.
    metrics = model.val(data=str(data_yaml), split="val", imgsz=args.imgsz, batch=args.batch, device=args.device)

    # Per-class AP50-95 (strict localisation metric) is what the baseline comparison needs.
    per_class = {}
    box = getattr(metrics, "box", None)  # getattr with a default: tolerate a missing attribute
    if box is not None and getattr(box, "maps", None) is not None:
        # box.maps is indexed by class id (classes without ground truth hold the overall mAP), so
        # look each class up by id rather than zipping it against ap_class_index.
        idx = list(getattr(box, "ap_class_index", range(len(box.maps))))
        per_class = {classes[int(i)].name: round(float(box.maps[int(i)]), 4) for i in idx}

    # ---- summary file: settings + results, so the report can cite exactly what was run ----
    # Every setting that influenced the result is stored next to the result.
    summary = {
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "cache": args.cache,
        "fraction": args.fraction,
        "data": str(data_yaml),
        "images": counts,
        # Overall precision / recall / mAP50 / mAP50-95 as reported by Ultralytics.
        "metrics": {k: round(float(v), 4) for k, v in metrics.results_dict.items()},
        "per_class_ap50_95": per_class,
        "per_group_ap50_95": mean_ap_per_group(per_class, groups),
        # Where to find the raw training log and the best weights.
        "run_dir": str(save_dir),
        "results_csv": str(Path(save_dir) / "results.csv"),
        "weights_best": str(Path(save_dir) / "weights" / "best.pt"),
    }

    # Write the summary into results/, creating the folder if it does not exist yet.
    out = resolve("results") / f"{args.name}_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary["per_group_ap50_95"], indent=2))
    print(f"summary written to {out}")


# Run main() only when executed as a script, not when this file is imported (e.g. by the tests).
if __name__ == "__main__":
    main()
