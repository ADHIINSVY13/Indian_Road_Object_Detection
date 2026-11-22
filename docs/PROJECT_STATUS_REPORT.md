# Project status (P07: object detection on Indian roads)

Updated 2026-11-22. Short, so a teammate or another tool can see where the project stands. Details live in
the linked files; every number comes from an executed run.

## Question
How much does fine-tuning a COCO-pretrained YOLO on India Driving Dataset (IDD) help, and does it help
India-specific classes (autorickshaw, ...) differently from common ones (car, bus, ...)?

## Done
| Step | Output |
|---|---|
| Dataset inspected, converted to YOLO format (640 px), validated, EDA | `docs/dataset_report.md`, `results/eda/` (41,794 images, 507,574 boxes, 0 validation errors) |
| Class mapping IDD ↔ COCO | `configs/class_mapping.yaml`, `docs/class_mapping.md` |
| Fine-tuning `yolo11n` on Colab T4 (23 of 30 planned epochs; free quota ended) | `results/results.csv`, `results/training_curve.png`, `models/best.pt` (not committed) |
| COCO baseline vs fine-tuned, same val split | `results/eval_*.json`, `results/comparison.json`, `results/results_compare.png` |
| Per-class precision and recall | `per_class_detail` in `results/eval_*.json`, `docs/results_report.md` §8 |
| Qualitative side-by-side and failure cases | `results/qualitative/`, `docs/results_report.md` §9 |
| Environment versions (local) | `results/environment_local_cpu_eval.json` |
| Code explanation | `docs/CODE_WALKTHROUGH.md` |
| Automatic tests | `tests/test_pipeline.py` (14 pass) |

## Headline result
On the 7 classes the baseline can name, mean AP50-95 rose from 0.177 to 0.300; the largest gains were truck, motorcycle, bus and person,
and no supported class got worse. `autorickshaw`, which COCO cannot name, reached 0.4389 after fine-tuning. 35% of the
val boxes have no COCO class at all. Full analysis and limitations: `docs/results_report.md`.

## Known limitations
23/30 epochs; one seed; nano model; checkpoint chosen and scored on the same val split (the public release has
no test labels); very few val boxes for `caravan`, `trailer`, `train`. See `docs/results_report.md` §10.

## Not done
- Colab environment versions were not recorded during the run (run `scripts/record_environment.py --tag colab_t4` in a Colab cell next time).
- Confusion matrix and PR/F1 curves from the training run are on Drive and not yet copied into `results/`.
- Optional: resume the remaining 7 epochs (expected gain about +0.003 mAP50-95).

## A correction worth knowing
An earlier version of `docs/results_report.md` showed wrong per-class baseline numbers because AP values were paired
with the wrong class ids. It was fixed in `scripts/evaluate.py`, covered by a regression test, the baseline was re-run, and
the report was corrected.
