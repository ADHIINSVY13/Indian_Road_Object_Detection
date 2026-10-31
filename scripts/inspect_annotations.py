"""Step 1: inspect the real IDD files before any conversion.

    python scripts/inspect_annotations.py
    python scripts/inspect_annotations.py --root D:/data/IDD_Detection --n-show 10
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.config import REPO_ROOT, load_yaml, resolve
from src.preprocessing.inspect_annotations import inspect_dataset, to_markdown


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--root", help="override raw.root from the config")
    ap.add_argument("--n-show", type=int, default=5, help="annotation files to print in full")
    ap.add_argument("--n-scan", type=int, default=2000, help="annotation files sampled for class names")
    ap.add_argument("--out", default="results/eda")
    args = ap.parse_args()

    cfg = load_yaml(resolve(args.config))
    root = Path(args.root) if args.root else resolve(cfg["raw"]["root"])
    result = inspect_dataset(root, cfg["raw"].get("annotation_dir"), args.n_show, args.n_scan)

    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "annotation_inspection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "annotation_inspection.md").write_text(to_markdown(result), encoding="utf-8")
    print(to_markdown(result))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
