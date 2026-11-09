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

## Repository layout
`configs/` configuration, `src/` library code, `scripts/` command-line entry points, `notebooks/` analysis,
`results/` generated outputs, `docs/` and `report/` write-ups. The dataset and model weights are never committed.

## Integrity rule
Every number or figure must come from an executed run; nothing is always invented.
