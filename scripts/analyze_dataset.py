"""Step 3: dataset statistics and EDA figures from the converted dataset.

    python scripts/analyze_dataset.py
"""
# argparse = command-line options, json = pretty-print the statistics, sys/Path = import paths.
import argparse
import json
import sys
from pathlib import Path

# Make `src` importable when the file is run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# load_yaml/load_classes read the config files, resolve() makes repo-relative paths absolute, and
# run_eda() does the real work (statistics + figures); this script is only a thin command-line wrapper.
from src.data.config import load_classes, load_yaml, resolve
from src.visualization.eda import run_eda


def main():
    # Command-line options: which config to read and where to write the results.
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--out", default="results/eda")  # folder that receives figures, CSV and JSON
    args = ap.parse_args()

    # Read the dataset config, then the class list it points to (names in class-id order, with groups).
    cfg = load_yaml(resolve(args.config))
    classes = load_classes(resolve(cfg["class_mapping"]))
    # The release name and download date are stamped on every figure for traceability.
    release = f"{cfg['release']['name']} {cfg['release'].get('downloaded_on', '')}".strip()
    # run_eda needs: where the converted data is, the class names in id order, and each class's
    # group (common / ambiguous / india_specific) for the per-group statistics.
    stats = run_eda(resolve(cfg["yolo"]["root"]), [c.name for c in classes],
                    {c.name: c.group for c in classes}, resolve(args.out), release)
    # Print the statistics (they are also saved as results/eda/eda_stats.json). default=str makes
    # json.dumps print values it cannot serialise natively (e.g. numpy numbers) instead of crashing.
    print(json.dumps(stats, indent=2, default=str))


# Run main() only when executed as a script, not when this file is imported.
if __name__ == "__main__":
    main()
