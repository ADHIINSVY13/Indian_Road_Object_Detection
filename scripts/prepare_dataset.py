"""Step 2: convert raw IDD (VOC xml) into the YOLO dataset and validate it.

    python scripts/prepare_dataset.py --limit 200      # small sample first
    python scripts/prepare_dataset.py                  # full conversion
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from src.data.config import load_classes, load_yaml, resolve
from src.preprocessing.convert import SPLITS, convert_dataset
from src.preprocessing.validate import validate_yolo_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    ap.add_argument("--limit", type=int, help="convert only the first N samples per split")
    ap.add_argument("--check-images", action="store_true", help="also decode every converted image")
    ap.add_argument("--image-mode", choices=["hardlink", "copy", "symlink", "none"])
    args = ap.parse_args()

    cfg = load_yaml(resolve(args.config))
    if args.image_mode:
        cfg["yolo"]["image_mode"] = args.image_mode
    classes = load_classes(resolve(cfg["class_mapping"]))
    repo_root = resolve(".")

    report, issues = convert_dataset(cfg["raw"], cfg["yolo"], classes, repo_root, tuple(args.splits), args.limit, tqdm)
    for split, r in report["splits"].items():
        print(split, r)

    yolo_root = resolve(cfg["yolo"]["root"])
    post = validate_yolo_dataset(yolo_root, len(classes), args.check_images)
    with open(yolo_root / "validation_issues.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "stem", "severity", "code", "detail"])
        w.writeheader()
        w.writerows(post)

    print("\nConversion issues by code:", dict(Counter(i["code"] for i in issues)))
    print("Post-conversion validation issues by code:", dict(Counter(i["code"] for i in post)))
    errors = [i for i in post if i["severity"] == "error"]
    print(f"Validation {'FAILED' if errors else 'passed'}: {len(errors)} errors, {len(post) - len(errors)} warnings")
    print(f"Reports in {yolo_root}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
