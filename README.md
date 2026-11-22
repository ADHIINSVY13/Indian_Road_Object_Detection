# P07: Object Detection on Unstructured Indian Roads

Deep learning project (team P07). Compares a COCO-pretrained YOLO detector with the same detector
fine-tuned on the India Driving Dataset (IDD) and asks how much region-specific fine-tuning helps on
India-specific classes compared with common classes.

Team: Aryan Saini, Divyanshi Sahu, Gaurav Rathore.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python -m pytest -q tests       # code self-test on tiny synthetic data
```

## Dataset pipeline
See [data/README.md](data/README.md). Short version:
```bash
python scripts/inspect_annotations.py
python scripts/prepare_dataset.py
python scripts/analyze_dataset.py
```
Class mapping: [docs/class_mapping.md](docs/class_mapping.md). Dataset findings: [docs/dataset_report.md](docs/dataset_report.md).

## Training
Training needs a CUDA GPU, and the project laptop has none, so it runs on a Colab T4.

```bash
python scripts/train_yolo.py --dry-run                       # validate the setup, train nothing
```

Full run: upload `data/idd_yolo_640.tar` to Drive, then open
`notebooks/02_train_colab.ipynb` (regenerate with `python scripts/make_train_notebook.py`). The same
entry point works anywhere:

```bash
python scripts/train_yolo.py --epochs 30 --batch 16 --name idd_yolo11n_e30
```

It writes `results/<name>_summary.json` with overall, per-class and per-group AP50-95. The per-group
numbers are the comparison this project is built on: common vs India-specific classes, COCO-pretrained
versus IDD-finetuned.

## Evaluation
Inference only, so this runs on CPU:

```bash
python scripts/evaluate.py --model models/best.pt --tag finetuned
python scripts/evaluate.py --model yolo11n.pt --tag coco_baseline --coco-baseline
python scripts/compare_evals.py
```

`--coco-baseline` is not the same task as the other run. A COCO model predicts 80 classes, so the IDD
val ground truth is translated into COCO ids first, and only for the classes the mapping calls a direct
match (`group: common` with exactly one `coco_id`). `rider->person` and `traffic sign->stop sign` are
excluded on purpose; `india_specific` classes are reported as N/A, never as 0.

Each run writes `results/eval_<tag>.json` in the same shape, and `compare_evals.py` folds them into
`results/comparison.json` plus `results/results_compare.png`. Full write-up: [docs/results_report.md](docs/results_report.md).

## Repository layout
`configs/` configuration, `src/` library code, `scripts/` command-line entry points, `notebooks/` analysis,
`results/` generated outputs, `docs/` and `report/` write-ups. The dataset and model weights are never committed.

## Integrity rule
Every number or figure must come from an executed run; nothing is ever invented.
