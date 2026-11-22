"""Step 6: turn the two eval summaries into the project's answer.

    python scripts/compare_evals.py

Reads `results/eval_coco_baseline.json` and `results/eval_finetuned.json`, writes
`results/comparison.json` and `results/results_compare.png`, and prints the per-group table that
`docs/results_report.md` quotes.

The comparison is deliberately group-first. A single overall mAP hides the thing being tested:
a COCO model cannot predict `india_specific` classes at all, so pooling every class into one
average mixes a real number with a structural zero. Classes the baseline cannot score stay `None`
and are excluded from its group mean, never counted as 0.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

# Draw straight to image files; no window needed (works on servers and Colab).
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Make `src` importable when the script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.config import load_classes, load_yaml, resolve

# Order in which class groups appear in tables and the figure.
GROUP_ORDER = ["common", "ambiguous", "india_specific"]


def parse_args():
    """Command-line options: where the two evaluation JSON files are, and where to save the figure."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", default="results/eval_coco_baseline.json")
    ap.add_argument("--finetuned", default="results/eval_finetuned.json")
    ap.add_argument("--out", default="results/results_compare.png")
    return ap.parse_args()


def load_eval(path: Path) -> dict:
    """Read an evaluation JSON, or stop with a helpful message if it has not been produced yet."""
    if not path.exists():
        sys.exit(f"{path} is missing; run scripts/evaluate.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float | None:
    """Average of a list rounded to 4 places, or None for an empty list.

    Returning None instead of 0 matters: "no classes to average" must not look like "score of zero".
    """
    return round(sum(values) / len(values), 4) if values else None


def main():
    args = parse_args()
    cfg = load_yaml(resolve("configs/dataset.yaml"))
    classes = load_classes(resolve(cfg["class_mapping"]))

    base = load_eval(resolve(args.baseline))
    fine = load_eval(resolve(args.finetuned))
    b_ap = base["per_class_ap50_95"]   # {class name: AP} for classes the baseline could score
    f_ap = fine["per_class_ap50_95"]   # {class name: AP} for every class

    # ---- one row per IDD class, with baseline, fine-tuned and the difference ----
    rows = []
    for c in classes:
        b = b_ap.get(c.name)  # None when the baseline has no output class for it
        f = f_ap.get(c.name)
        rows.append({
            "class": c.name,
            "group": c.group,
            "baseline_ap50_95": b,
            "finetuned_ap50_95": f,
            # delta only exists where both sides have a number; a missing baseline is the finding
            "delta": round(f - b, 4) if b is not None and f is not None else None,
            "baseline_status": "scored" if b is not None else base["not_evaluable"].get(c.name, "not scored"),
        })
    # Sort by group (in GROUP_ORDER), then by fine-tuned AP, best first (the minus flips the order).
    rows.sort(key=lambda r: (GROUP_ORDER.index(r["group"]) if r["group"] in GROUP_ORDER else 99,
                             -(r["finetuned_ap50_95"] or 0)))

    # ---- per-group averages; unscored classes are left out, not counted as zero ----
    groups = {}
    for g in GROUP_ORDER:
        b_vals = [r["baseline_ap50_95"] for r in rows if r["group"] == g and r["baseline_ap50_95"] is not None]
        f_vals = [r["finetuned_ap50_95"] for r in rows if r["group"] == g and r["finetuned_ap50_95"] is not None]
        d_vals = [r["delta"] for r in rows if r["group"] == g and r["delta"] is not None]
        groups[g] = {
            "classes": [r["class"] for r in rows if r["group"] == g],
            "scored_by_baseline": len(b_vals),
            "baseline_ap50_95": mean(b_vals),
            "finetuned_ap50_95": mean(f_vals),
            "delta": mean(d_vals),
        }

    out = {
        "baseline": {"tag": base["tag"], "model": base["model"], "images": base["images"],
                     "overall": base["overall"]},
        "finetuned": {"tag": fine["tag"], "model": fine["model"], "images": fine["images"],
                      "overall": fine["overall"]},
        "per_class": rows,
        "per_group": groups,
    }
    out_path = resolve("results") / "comparison.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ---- figure: one bar per model per class, N/A marked where the baseline has no class ----
    fig, ax = plt.subplots(figsize=(13, 6))
    xs = range(len(rows))
    # `or 0` only affects the DRAWING (a missing baseline is drawn as an empty bar); the saved JSON
    # above keeps the real None.
    base_vals = [r["baseline_ap50_95"] or 0 for r in rows]
    fine_vals = [r["finetuned_ap50_95"] or 0 for r in rows]
    # Offset the two bars by -/+0.2 so they sit side by side at each class position.
    ax.bar([x - 0.2 for x in xs], base_vals, width=0.4, label="COCO-pretrained (baseline)", color="#9aa0a6")
    ax.bar([x + 0.2 for x in xs], fine_vals, width=0.4, label="IDD fine-tuned", color="#1a73e8")

    for x, r in zip(xs, rows):
        if r["baseline_ap50_95"] is None:
            # Mark classes the baseline cannot name, so an empty bar is not mistaken for a zero score.
            ax.text(x - 0.2, 0.005, "N/A", rotation=90, ha="center", va="bottom", fontsize=7, color="#5f6368")
        if r["finetuned_ap50_95"] is not None:
            ax.text(x + 0.2, r["finetuned_ap50_95"], f'{r["finetuned_ap50_95"]:.2f}',
                    ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(list(xs))
    ax.set_xticklabels([f'{r["class"]}\n({r["group"]})' for r in rows], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("AP50-95 on IDD val")
    ax.set_title("COCO pretrained vs IDD fine-tuned, per class")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig_path = resolve(args.out)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)

    # ---- console table (the numbers quoted in docs/results_report.md) ----
    print(f"{'group':16s} {'n':>3s} {'baseline':>9s} {'finetuned':>10s} {'delta':>7s}")
    for g, v in groups.items():
        b = "-" if v["baseline_ap50_95"] is None else f'{v["baseline_ap50_95"]:.4f}'
        f = "-" if v["finetuned_ap50_95"] is None else f'{v["finetuned_ap50_95"]:.4f}'
        d = "-" if v["delta"] is None else f'{v["delta"]:+.4f}'
        print(f"{g:16s} {v['scored_by_baseline']:3d} {b:>9s} {f:>10s} {d:>7s}")

    print(f"\nwritten: {out_path}\nwritten: {fig_path}")


if __name__ == "__main__":
    main()
