# Dataset report: IDD Detection

Fill every field from the real download and from the outputs of `scripts/inspect_annotations.py`,
`scripts/prepare_dataset.py` and `scripts/analyze_dataset.py`. Leave `TBD` until it is measured.

## 1. Release
| Item | Value |
|---|---|
| Release name / version | TBD |
| Download date | TBD |
| Source URL | https://idd.insaan.iiit.ac.in/dataset/details/ |
| Licence accepted | TBD |

## 2. Splits (exact counts from the files)
| Split | Images | Annotated | Notes |
|---|---|---|---|
| train | TBD | TBD | |
| val | TBD | TBD | used for model selection and the quantitative comparison |
| test | TBD | TBD | annotations available? TBD |

## 3. Annotation findings (checklist from the project plan, section 4.4)
- [ ] Official detection package downloaded
- [ ] Annotation directory and extensions inspected: TBD
- [ ] At least 5 annotation files opened (list them): TBD
- [ ] Class taxonomy and ids identified: TBD
- [ ] Bounding-box coordinate format: TBD
- [ ] Annotation file format (JSON / XML / txt): TBD
- [ ] Train and val annotations complete: TBD
- [ ] Test annotations available: TBD

## 4. Preprocessing and validation
Taken from `data/yolo/conversion_report.json` and `conversion_issues.csv`.

| Item | train | val |
|---|---|---|
| Images converted | TBD | TBD |
| Boxes kept | TBD | TBD |
| Boxes dropped (invalid / unknown class) | TBD | TBD |
| Boxes clipped to image | TBD | TBD |
| Images skipped (missing/corrupt) | TBD | TBD |

Unexplained issue codes remaining: TBD

## 5. Class distribution and statistics
Figures are in `results/eda/`. Summary numbers come from `results/eda/eda_stats.json`.
- Class imbalance (max/min non-zero class count): TBD
- Classes with zero boxes in a split: TBD
- Annotations per image (mean / median / max): TBD
- Small-object share: TBD
- Observations about train vs val consistency: TBD

## 6. Known limitations
TBD
