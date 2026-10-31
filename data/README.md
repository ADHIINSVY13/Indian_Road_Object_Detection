# Data

The raw IDD data and the converted YOLO dataset are **not committed** (see `.gitignore`).

## Getting the data
1. Register / log in at https://idd.insaan.iiit.ac.in/ and accept the licence.
2. Download the **IDD Detection** release approved for the course (the project plan lists 31,569 train, 10,225 validation and 4,794 test images; confirm against what you actually download).
3. Extract so the folder is `data/raw/IDD_Detection/` (or set `raw.root` in `configs/dataset.yaml`). The expected layout is `JPEGImages/`, `Annotations/` and `train.txt` / `val.txt` / `test.txt`; this is verified, not assumed, by the next step.
4. Record the release name and download date in `configs/dataset.yaml` and `docs/dataset_report.md`.

## Pipeline
```bash
python scripts/inspect_annotations.py        # look at real files first (format, classes, coordinates)
# edit configs/class_mapping.yaml if the observed class names differ
python scripts/prepare_dataset.py --limit 200  # small sample, check the reports
python scripts/prepare_dataset.py              # full conversion + validation
python scripts/analyze_dataset.py              # EDA statistics and figures -> results/eda/
```

Outputs under `data/yolo/`: `images/`, `labels/`, `idd.yaml`, `manifest.csv`, `conversion_issues.csv`, `conversion_report.json`, `validation_issues.csv`.

Do not use the test split for quantitative results unless official test annotations are confirmed available (the converter reports a split without a split file as "not converted" instead of guessing).
