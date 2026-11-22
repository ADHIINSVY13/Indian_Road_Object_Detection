"""Step 2: convert raw IDD (VOC xml) into the YOLO dataset and validate it.

    python scripts/prepare_dataset.py --limit 200      # small sample first
    python scripts/prepare_dataset.py                  # full conversion
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

# Make `src` importable when the file is run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from src.data.config import load_classes, load_yaml, resolve
from src.preprocessing.convert import SPLITS, convert_dataset
from src.preprocessing.validate import validate_yolo_dataset


def main():
    # ---- command-line options ----
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    ap.add_argument("--limit", type=int, help="convert only the first N samples per split")
    ap.add_argument("--check-images", action="store_true", help="also decode every converted image")
    ap.add_argument("--image-mode", choices=["hardlink", "copy", "symlink", "none"])
    ap.add_argument("--max-side", type=int, help="resize images so the longer side is at most this many px (e.g. 640); 0 keeps the originals")
    args = ap.parse_args()

    # ---- load configuration, then apply any command-line overrides on top of it ----
    cfg = load_yaml(resolve(args.config))
    if args.max_side is not None:
        # 0 means "keep the originals": without the `or None` the config's 640 would win.
        cfg["yolo"]["max_side"] = args.max_side or None
    if args.image_mode:
        cfg["yolo"]["image_mode"] = args.image_mode
    classes = load_classes(resolve(cfg["class_mapping"]))
    repo_root = resolve(".")

    # ---- step A: convert (tqdm draws a progress bar over the list of samples) ----
    report, issues = convert_dataset(cfg["raw"], cfg["yolo"], classes, repo_root, tuple(args.splits), args.limit, tqdm)
    for split, r in report["splits"].items():
        print(split, r)

    # ---- step B: validate the finished dataset independently of the converter ----
    yolo_root = resolve(cfg["yolo"]["root"])
    post = validate_yolo_dataset(yolo_root, len(classes), args.check_images)
    with open(yolo_root / "validation_issues.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "stem", "severity", "code", "detail"])
        w.writeheader()
        w.writerows(post)

    # ---- summary: how many of each kind of problem, and an overall pass/fail ----
    print("\nConversion issues by code:", dict(Counter(i["code"] for i in issues)))
    print("Post-conversion validation issues by code:", dict(Counter(i["code"] for i in post)))
    errors = [i for i in post if i["severity"] == "error"]
    print(f"Validation {'FAILED' if errors else 'passed'}: {len(errors)} errors, {len(post) - len(errors)} warnings")
    print(f"Reports in {yolo_root}")
    # A non-zero exit code lets other tools (CI, shell scripts) detect that validation failed.
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
