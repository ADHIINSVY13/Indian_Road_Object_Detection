"""Step 4b: plot the training curve that Ultralytics wrote to results.csv.

    python scripts/plot_training_curve.py --csv results/results.csv

The curve is what justifies calling a short run "still improving" or "plateaued", so the figure and
the numbers printed next to it come from the same file the trainer produced -- nothing is read off
the console log by hand.

Reading the curve (for explaining it): mAP is the accuracy score on the validation images after
each epoch (one epoch = one pass over all training images). A rising line means the model is still
learning; a flat line means more epochs would add little. Training loss is the error on the training
images; a falling loss with a rising mAP means the model is genuinely improving, not just memorising.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

# Render to files only (no GUI window), so this runs anywhere.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Make `src` importable when run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.config import resolve


def parse_args():
    """Command-line options: input CSV written by the trainer, and output figure path."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="results/results.csv")
    ap.add_argument("--out", default="results/training_curve.png")
    return ap.parse_args()


def read_curve(path: Path) -> list[dict]:
    """Read results.csv into a list of row dicts (one per epoch).

    Input : path of the CSV the Ultralytics trainer writes after every epoch.
    Output: [{"epoch": "1", "metrics/mAP50(B)": "0.21", ...}, ...]; values are still text here and
            are converted to numbers in main().
    Why this exists: using the trainer's own file means the plot and the report share one source.
    """
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))  # DictReader maps each row to {column name: text value}
    # Ultralytics pads some column names with spaces, so strip every key once here.
    # Rows without an "epoch" value (blank trailing lines) are dropped.
    return [{k.strip(): v for k, v in r.items()} for r in rows if r.get("epoch")]


def main():
    """Read the training log, draw the two-panel curve, and print the numbers quoted in the report."""
    args = parse_args()
    path = resolve(args.csv)
    if not path.exists():
        sys.exit(f"{path} is missing; download it from the run's Drive folder")

    # ---- extract the columns we plot, converting text to numbers ----
    rows = read_curve(path)
    ep = [int(r["epoch"]) for r in rows]                       # epoch numbers 1..N
    m50 = [float(r["metrics/mAP50(B)"]) for r in rows]         # mAP at IoU 0.50
    m5095 = [float(r["metrics/mAP50-95(B)"]) for r in rows]    # mAP averaged over IoU 0.50..0.95
    box = [float(r["train/box_loss"]) for r in rows]           # localisation loss
    cls = [float(r["train/cls_loss"]) for r in rows]           # classification loss

    # Best epoch = the one with the highest mAP50-95. max() over the index uses the value as the key.
    best_i = max(range(len(m5095)), key=lambda i: m5095[i])
    # Slope over the last five epochs: average gain per epoch. A value near 0 means the curve has
    # flattened; a clearly positive value means the run was still improving when it stopped.
    last5 = m5095[-5:]
    slope = (last5[-1] - last5[0]) / max(len(last5) - 1, 1) if len(last5) > 1 else 0.0

    # ---- figure with two panels: validation mAP (left) and training loss (right) ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(ep, m50, marker="o", ms=3, label="mAP50")
    ax1.plot(ep, m5095, marker="o", ms=3, label="mAP50-95")
    # A red star marks the best epoch and its value is written in the legend.
    ax1.scatter([ep[best_i]], [m5095[best_i]], s=90, marker="*", color="#d93025", zorder=5,
                label=f"best: {m5095[best_i]:.4f} @ ep {ep[best_i]}")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("mAP on IDD val")
    ax1.set_title("Validation mAP")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    # Right panel: the two training losses. box_loss = how well boxes fit the objects (position and
    # size); cls_loss = how often the class is wrong. Both should go down as training progresses.
    ax2.plot(ep, box, marker="o", ms=3, label="box_loss")
    ax2.plot(ep, cls, marker="o", ms=3, label="cls_loss")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("training loss")
    ax2.set_title("Training loss")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()  # avoid overlapping titles and labels between the two panels
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)

    # ---- print the headline numbers quoted in the report ----
    secs = float(rows[-1]["time"]) if rows[-1].get("time") else 0.0  # cumulative seconds at the last epoch
    print(f"epochs      : {len(ep)} (1..{ep[-1]})")
    print(f"mAP50       : {m50[0]:.4f} -> {m50[-1]:.4f}")
    print(f"mAP50-95    : {m5095[0]:.4f} -> {m5095[-1]:.4f}")
    print(f"best        : {m5095[best_i]:.5f} @ epoch {ep[best_i]}")
    print(f"last-5 slope: {slope:+.5f} mAP50-95 per epoch")
    print(f"wall time   : {secs / 3600:.2f} h ({secs / max(len(ep), 1):.0f} s/epoch)")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
