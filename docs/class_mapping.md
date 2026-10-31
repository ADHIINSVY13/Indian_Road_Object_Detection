# IDD to COCO class mapping

Source of truth: `configs/class_mapping.yaml` (it also defines the YOLO class ids by order).

> **Status: provisional.** The IDD class list is the expected taxonomy. Replace/confirm it with
> the classes observed by `scripts/inspect_annotations.py` (`results/eda/annotation_inspection.md`)
> before any training, then remove this note.

| IDD class | COCO equivalent | Group | Treatment |
|---|---|---|---|
| person | person | common | Direct comparison |
| rider | person (partial) | ambiguous | COCO has no rider class; do not claim a direct match |
| car | car | common | Direct comparison |
| truck | truck | common | Direct comparison |
| bus | bus | common | Direct comparison |
| motorcycle | motorcycle | common | Direct comparison |
| bicycle | bicycle | common | Direct comparison |
| autorickshaw | none | india_specific | Baseline reported as N/A, never 0 |
| animal | bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe | ambiguous | Many COCO species to one IDD class; decide and document before the baseline run |
| traffic light | traffic light | common | Direct comparison |
| traffic sign | stop sign (partial) | ambiguous | COCO only has stop sign |
| vehicle fallback | none | india_specific | Catch-all, N/A baseline |
| caravan | none | india_specific | N/A baseline, probably rare |
| trailer | none | india_specific | N/A baseline, probably rare |
| train | train | common | Direct comparison, probably rare |

## Rules for the baseline
- Only COCO ids listed under `coco_ids` for a class may be counted as predictions of that IDD class.
- A class with an empty `coco_ids` list is **not directly comparable**: report N/A, do not insert a zero.
- Ambiguous classes are reported separately and labelled as such in every table and figure.
