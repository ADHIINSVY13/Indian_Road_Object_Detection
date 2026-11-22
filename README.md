# P07: Object Detection on Unstructured Indian Roads

**Does region-specific fine-tuning help? COCO-pretrained versus IDD-fine-tuned YOLO11n.**

Deep learning project (Team P07, Sitare University). It compares a COCO-pretrained YOLO11n detector with
the same detector fine-tuned on the India Driving Dataset (IDD Detection), and asks how much
region-specific fine-tuning helps on India-specific classes compared with common ones. Both models share
one architecture and are scored on the identical labelled validation split with identical metrics; only
the weights differ.

## Contributors

| Name | GitHub |
|---|---|
| **Aryan Saini** | [aryan2006-saini](https://github.com/aryan2006-saini) |
| Divyanshi Sahu | |
| Gaurav Rathore | |

The submission report, [report/P07_Report_ELCVIA.pdf](report/P07_Report_ELCVIA.pdf), is authored by
Aryan Saini, Divyanshi Sahu and Gaurav Rathore.

## Headline results

Mean AP50-95 on the IDD validation split (10,225 images), from [docs/results_report.md](docs/results_report.md):

| Group | Baseline (COCO) | Fine-tuned (IDD) | Change |
|---|---|---|---|
| Common classes (7, excluding `train`) | 0.177 | 0.300 | +0.123 (+70%) |
| India-specific (`autorickshaw`, `vehicle fallback`) | N/A | 0.235 | not computable |
| Ambiguous (`rider`, `traffic sign`, `animal`) | N/A | 0.156 | not computable |

- Every common class the baseline can score improved and none got worse. The largest gains were truck
  (0.155 → 0.392), motorcycle (0.187 → 0.353), bus (0.346 → 0.487) and person (0.105 → 0.231).
- `autorickshaw`, which a COCO model cannot name at all, reaches AP50-95 0.439 after fine-tuning.
- 35% of validation boxes (44,154 of 126,059) belong to classes with no COCO equivalent.
- Small, distant objects are the main failure mode: recall is lowest for traffic light (0.080) and
  vehicle fallback (0.042).

N/A means the baseline has no output class for it. It is reported as N/A, never as zero.

## How it works

1. **Dataset.** IDD Detection (15 classes, Pascal-VOC XML) is converted to YOLO format at 640 px:
   41,794 images (31,569 train, 10,225 val), validated with 0 errors. The XML stores `<bndbox>` children
   in a non-standard order, so coordinates are read by tag name.
2. **Class mapping.** Each IDD class is mapped to COCO in [configs/class_mapping.yaml](configs/class_mapping.yaml)
   and put in a group: *common* (exactly one COCO match), *ambiguous* (rider, traffic sign, animal) or
   *india-specific* (autorickshaw, vehicle fallback, caravan, trailer).
3. **Training.** `yolo11n.pt` is fine-tuned with Ultralytics on a Colab Tesla T4: image size 640,
   batch 16, seed 0, default augmentation. Training stopped after 23 of 30 planned epochs, when the free
   GPU quota ran out.
4. **Evaluation.** Both models are scored at 640 px, confidence 0.001, NMS IoU 0.7. For the baseline the
   ground truth is translated into COCO ids, and only for the direct matches.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python -m pytest -q tests       # code self-test on tiny synthetic data (14 tests)
```

Training dependencies (`torch`, `torchvision`, `ultralytics`) are in `requirements-train.txt`. Install
them on a machine with a CUDA GPU or in Colab.

## Dataset pipeline

See [data/README.md](data/README.md). Short version:

```bash
python scripts/inspect_annotations.py
python scripts/prepare_dataset.py
python scripts/analyze_dataset.py
```

Class mapping: [docs/class_mapping.md](docs/class_mapping.md). Dataset findings: [docs/dataset_report.md](docs/dataset_report.md).

## Training

Training needs a CUDA GPU, so it runs on a Colab T4.

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

## Qualitative comparison and environment

```bash
python scripts/qualitative.py --finetuned models/best.pt --baseline yolo11n.pt   # results/qualitative/
python scripts/record_environment.py --tag local                                # results/environment_local.json
```

Side-by-side pictures (ground truth | COCO baseline | fine-tuned) from a seeded random sample, plus the
lowest-recall failure cases. Per-class precision and recall are in `results/eval_*.json` (`per_class_detail`).

## Limitations

Read these before quoting any number:

- **Incomplete training:** 23 of 30 epochs. The expected missing gain is about +0.003 mAP50-95.
- **Single seed:** small differences (traffic light, bicycle, car) are indicative only.
- **Selection on the evaluation split:** the public release has no test labels, so the checkpoint is
  chosen and scored on the same validation split. The fine-tuned numbers are slightly optimistic.
- **Small model, one dataset:** only YOLO11n and IDD's own validation split; cross-dataset
  generalisation is untested.
- **Rare classes:** caravan, trailer and train have 7 to 13 validation boxes, so their AP is noise.

## Reading the code

[docs/CODE_WALKTHROUGH.md](docs/CODE_WALKTHROUGH.md) explains every script and module, the design decisions
behind them, and likely questions about the code. Project status: [docs/PROJECT_STATUS_REPORT.md](docs/PROJECT_STATUS_REPORT.md).

## Repository layout

| Folder | Contents |
|---|---|
| `configs/` | dataset and class-mapping configuration |
| `src/` | library code: preprocessing, data config, visualisation |
| `scripts/` | command-line entry points |
| `notebooks/` | dataset exploration and the Colab training notebook |
| `results/` | generated outputs: EDA, evaluations, plots, qualitative images |
| `docs/`, `report/` | write-ups; the submission report is `report/P07_Report_ELCVIA.docx`, with a PDF copy |
| `models/` | `best.pt`, the IDD fine-tuned checkpoint |
| `tests/` | automated tests |

The dataset is never committed. The one exception among weights is `models/best.pt` (10.8 MB, the IDD
fine-tuned YOLO11n checkpoint, epoch 23 of 30, used for every reported fine-tuned number), kept in the
repo so the results can be reproduced. `yolo11n.pt` (COCO baseline) is downloaded automatically by
Ultralytics on first use.

## Integrity rule

Every number or figure must come from an executed run; nothing is ever invented.

## References

1. G. Varma et al., "IDD: A Dataset for Exploring Problems of Autonomous Navigation in Unconstrained Environments", WACV 2019.
2. T.-Y. Lin et al., "Microsoft COCO: Common Objects in Context", ECCV 2014.
3. G. Jocher, J. Qiu, "Ultralytics YOLO11", 2024. https://github.com/ultralytics/ultralytics
