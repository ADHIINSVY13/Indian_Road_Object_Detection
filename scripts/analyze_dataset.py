"""Step 3: dataset statistics and EDA figures from the converted dataset.

    python scripts/analyze_dataset.py
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.config import load_classes, load_yaml, resolve
from src.visualization.eda import run_eda


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--out", default="results/eda")
    args = ap.parse_args()

    cfg = load_yaml(resolve(args.config))
    classes = load_classes(resolve(cfg["class_mapping"]))
    release = f"{cfg['release']['name']} {cfg['release'].get('downloaded_on', '')}".strip()
    stats = run_eda(resolve(cfg["yolo"]["root"]), [c.name for c in classes],
                    {c.name: c.group for c in classes}, resolve(args.out), release)
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
