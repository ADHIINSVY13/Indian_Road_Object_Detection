"""Regenerates notebooks/01_dataset_exploration.ipynb (no outputs stored).

Why generate a notebook from a script? A .ipynb file is a large JSON document that is awkward to
edit and to review in git. Keeping the cell text here, as plain Python strings, means the notebook
can be rebuilt at any time and always matches the library code in `src/`.

    python scripts/make_notebook.py
"""
import sys
from pathlib import Path

# nbformat is the official library for building notebook files from Python.
import nbformat as nbf

# Repository root: this file is <repo>/scripts/make_notebook.py, so go up one level from scripts/.
root = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()

# The notebook is described as a list of cells. Markdown cells hold explanations; code cells hold
# the Python that runs when the notebook is opened. The notebook reuses the same functions as the
# command-line scripts, so notebook and scripts can never disagree.
#
# Each code cell below is written as a triple-quoted string. The comments INSIDE those strings are
# part of the notebook: they are what a reader sees when they open it.
cells = [
    nbf.v4.new_markdown_cell(
        "# 01 Dataset exploration (IDD Detection)\n"
        "Run after `scripts/prepare_dataset.py`. All numbers come from the converted dataset; nothing is "
        "typed in by hand.\n"
        "\n"
        "This notebook answers: how many images and boxes are there, which classes are rare, how big are "
        "the objects, and did the conversion drop anything?"
    ),
    nbf.v4.new_markdown_cell(
        "## Setup\n"
        "Load the configuration, the class list and every converted label into one table. "
        "`sys.path.insert` lets the notebook import the repository's `src/` package, because the "
        "notebook lives in `notebooks/` (one folder below the repository root)."
    ),
    nbf.v4.new_code_cell(
        "import sys, json\n"
        "from pathlib import Path\n"
        "\n"
        "# Make `src` importable: the notebook runs inside notebooks/, so the repo root is one level up.\n"
        "sys.path.insert(0, str(Path.cwd().parent))\n"
        "import pandas as pd\n"
        "from IPython.display import Image, display\n"
        "from src.data.config import load_classes, load_yaml, resolve\n"
        "from src.visualization.eda import load_boxes, compute_stats, run_eda\n"
        "\n"
        "# Read the dataset configuration and the 15-class list (same files the scripts use).\n"
        "cfg = load_yaml(resolve('configs/dataset.yaml'))\n"
        "classes = load_classes(resolve(cfg['class_mapping']))\n"
        "names = [c.name for c in classes]               # class names in id order\n"
        "groups = {c.name: c.group for c in classes}     # class name -> common / ambiguous / india_specific\n"
        "yolo = resolve(cfg['yolo']['root'])             # folder with the converted YOLO dataset\n"
        "\n"
        "# manifest.csv has one row per converted image (split, name, size, number of boxes).\n"
        "manifest = pd.read_csv(yolo / 'manifest.csv')\n"
        "# load_boxes reads every label file into one table with one row per box.\n"
        "boxes = load_boxes(yolo, names)\n"
        "# The last expression of a cell is displayed: here, the number of images per split.\n"
        "manifest.groupby('split').size()"
    ),
    nbf.v4.new_markdown_cell(
        "## Conversion report and issues\n"
        "Every box or image the converter dropped, clipped or flagged is logged with a reason code. "
        "Reading the log here shows how clean the data was and that nothing disappeared silently."
    ),
    nbf.v4.new_code_cell(
        "# Per-split counts written by the converter (images converted, boxes kept/dropped, ...).\n"
        "print(json.dumps(json.load(open(yolo / 'conversion_report.json')), indent=2))\n"
        "\n"
        "# One row per logged problem; count them by severity (error/warning) and reason code.\n"
        "issues = pd.read_csv(yolo / 'conversion_issues.csv')\n"
        "issues.groupby(['severity', 'code']).size()"
    ),
    nbf.v4.new_markdown_cell(
        "## Class distribution\n"
        "Number of boxes per class and split. A very uneven table here is the class imbalance that "
        "explains why rare classes (caravan, trailer, train) are hard to learn and to measure."
    ),
    nbf.v4.new_code_cell(
        "# Count boxes per (class, split), turn splits into columns, keep all classes even if a count is 0.\n"
        "counts = boxes.groupby(['class', 'split']).size().unstack(fill_value=0).reindex(names, fill_value=0)\n"
        "# Add each class's group so common vs India-specific classes can be compared at a glance.\n"
        "counts['group'] = counts.index.map(groups)\n"
        "counts"
    ),
    nbf.v4.new_markdown_cell(
        "## Statistics and figures (written to results/eda/)\n"
        "`run_eda` is the same function `scripts/analyze_dataset.py` calls: it computes all statistics, "
        "saves five figures plus `eda_stats.json` and `class_counts.csv`, and returns the stats."
    ),
    nbf.v4.new_code_cell(
        "out = resolve('results/eda')\n"
        "# Compute the statistics and (re)draw every figure into results/eda/.\n"
        "stats = run_eda(yolo, names, groups, out, cfg['release']['name'])\n"
        "# Show each saved figure inside the notebook.\n"
        "for f in stats['figures']:\n"
        "    display(Image(filename=str(out / f)))"
    ),
    nbf.v4.new_markdown_cell(
        "## Observations\n"
        "Write these after looking at the real figures (imbalance, rare classes, small objects, "
        "train vs val differences)."
    ),
]
nb["cells"] = cells
out = root / "notebooks" / "01_dataset_exploration.ipynb"
nbf.write(nb, out)  # writes the .ipynb JSON file
print("wrote", out)
