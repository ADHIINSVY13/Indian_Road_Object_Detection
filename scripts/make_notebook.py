"""Regenerates notebooks/01_dataset_exploration.ipynb (no outputs stored)."""
import sys
from pathlib import Path

import nbformat as nbf

root = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells = [
    nbf.v4.new_markdown_cell("# 01 Dataset exploration (IDD Detection)\nRun after `scripts/prepare_dataset.py`. All numbers come from the converted dataset; nothing is typed in by hand."),
    nbf.v4.new_code_cell("import sys, json\nfrom pathlib import Path\nsys.path.insert(0, str(Path.cwd().parent))\nimport pandas as pd\nfrom IPython.display import Image, display\nfrom src.data.config import load_classes, load_yaml, resolve\nfrom src.visualization.eda import load_boxes, compute_stats, run_eda\n\ncfg = load_yaml(resolve('configs/dataset.yaml'))\nclasses = load_classes(resolve(cfg['class_mapping']))\nnames = [c.name for c in classes]\ngroups = {c.name: c.group for c in classes}\nyolo = resolve(cfg['yolo']['root'])\nmanifest = pd.read_csv(yolo / 'manifest.csv')\nboxes = load_boxes(yolo, names)\nmanifest.groupby('split').size()"),
    nbf.v4.new_markdown_cell("## Conversion report and issues"),
    nbf.v4.new_code_cell("print(json.dumps(json.load(open(yolo / 'conversion_report.json')), indent=2))\nissues = pd.read_csv(yolo / 'conversion_issues.csv')\nissues.groupby(['severity', 'code']).size()"),
    nbf.v4.new_markdown_cell("## Class distribution"),
    nbf.v4.new_code_cell("counts = boxes.groupby(['class', 'split']).size().unstack(fill_value=0).reindex(names, fill_value=0)\ncounts['group'] = counts.index.map(groups)\ncounts"),
    nbf.v4.new_markdown_cell("## Statistics and figures (written to results/eda/)"),
    nbf.v4.new_code_cell("out = resolve('results/eda')\nstats = run_eda(yolo, names, groups, out, cfg['release']['name'])\nfor f in stats['figures']:\n    display(Image(filename=str(out / f)))"),
    nbf.v4.new_markdown_cell("## Observations\nWrite these after looking at the real figures (imbalance, rare classes, small objects, train vs val differences)."),
]
nb["cells"] = cells
out = root / "notebooks" / "01_dataset_exploration.ipynb"
nbf.write(nb, out)
print("wrote", out)
