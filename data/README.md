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
python scripts/prepare_dataset.py --limit 1000 --splits train val   # sample gate: check the reports
python scripts/prepare_dataset.py                                   # full conversion + validation
python scripts/analyze_dataset.py                                   # EDA statistics and figures -> results/eda/
```

`yolo.max_side` is `640` in `configs/dataset.yaml`, so a plain `prepare_dataset.py` converts at
640 px on the long side. Override per run with `--max-side` (or `--max-side 0` for originals).

Outputs under `data/yolo/`: `images/`, `labels/`, `idd.yaml`, `manifest.csv`, `conversion_issues.csv`, `conversion_report.json`, `validation_issues.csv`.

Do not use the test split for quantitative results unless official test annotations are confirmed available (the converter reports a split without a split file as "not converted" instead of guessing).

## Training export
Training runs on a Colab T4 because the laptop has no CUDA GPU. Ship the converted dataset as **one**
tar file, never as 41,794 loose files — Drive throttles badly on many small files.

```bash
tar -cf data/idd_yolo_640.tar -C data/yolo .     # 2.56 GB, ~1 min
```

Then upload it to `MyDrive/idd_p07/` and run `notebooks/02_train_colab.ipynb`. That notebook unpacks to
`/content/idd_yolo`, clones this repo, and calls `scripts/train_yolo.py --data-root /content/idd_yolo`.

`results/` is tracked on purpose: the figures in `docs/dataset_report.md` and the training summaries
are cited by the report, so they must live in git rather than only on Drive.
