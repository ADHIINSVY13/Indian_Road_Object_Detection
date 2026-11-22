"""EDA statistics and figures computed from the converted YOLO dataset.

Everything here is derived from the actual label files and manifest.csv;
nothing is hard-coded.

EDA = exploratory data analysis: before training we want to know how many objects each class has,
how crowded the images are, and how big the objects are, because those facts explain model behaviour
later (e.g. why rare classes score near zero).
"""
from __future__ import annotations

# json saves the statistics; Path builds file paths; matplotlib draws the figures; numpy does the
# log/clip maths; pandas holds the label data as tables (DataFrames).
import json
from pathlib import Path

import matplotlib

# "Agg" is a drawing backend that renders straight to image files and needs no window. It must be
# selected BEFORE pyplot is imported so the code also works on servers / Colab without a display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Object-size buckets, by box area as a fraction of the image area (width * height, both 0..1).
# 0.001 = 0.1% of the image, 0.01 = 1%; the last edge is just above 1.0 so a full-image box still fits.
SIZE_BINS = [0, 0.001, 0.01, 1.01]          # relative box area
SIZE_LABELS = ["small (<0.1% of image)", "medium (0.1-1%)", "large (>1%)"]


def load_boxes(yolo_root: Path, class_names: list[str]) -> pd.DataFrame:
    """Read every YOLO label file into ONE table with a row per box.

    Columns: split, stem (image name), cls (id), xc/yc/w/h (normalised box), plus derived
    class name, relative area and size bucket. Having one tidy table makes every later statistic
    a one-line groupby.
    """
    rows = []
    # One folder per split under labels/ (train/, val/); one .txt label file per image inside it.
    for split_dir in sorted((yolo_root / "labels").glob("*")):  # train/, val/
        for lf in split_dir.glob("*.txt"):
            # An image with no objects has an empty file: handle it instead of failing on split().
            text = lf.read_text(encoding="utf-8").strip()
            for line in text.splitlines() if text else []:
                # YOLO line format: class_id x_centre y_centre width height (all normalised 0..1).
                c, xc, yc, w, h = line.split()
                rows.append((split_dir.name, lf.stem, int(c), float(xc), float(yc), float(w), float(h)))
    # Turn the list of tuples into a table with named columns.
    df = pd.DataFrame(rows, columns=["split", "stem", "cls", "xc", "yc", "w", "h"])
    df["class"] = df["cls"].map(lambda i: class_names[i])   # id -> readable name
    df["area_rel"] = df["w"] * df["h"]                      # fraction of the image the box covers
    # pd.cut sorts each area into one of the three buckets; right=False makes intervals [low, high).
    df["size_bucket"] = pd.cut(df["area_rel"], SIZE_BINS, labels=SIZE_LABELS, right=False)
    return df


def compute_stats(boxes: pd.DataFrame, manifest: pd.DataFrame, class_names: list[str], groups: dict[str, str]) -> dict:
    """Compute all summary numbers and return them as a JSON-friendly dict (saved as eda_stats.json)."""
    # Start with the simplest facts: how many images and how many boxes each split has.
    stats: dict = {"images_per_split": manifest.groupby("split").size().to_dict(),
                   "boxes_per_split": boxes.groupby("split").size().to_dict()}

    # Boxes per (split, class). unstack() pivots classes into columns; reindex() forces ALL classes to
    # appear (with 0) even if one is absent from a split, so no class silently goes missing.
    per_class = (boxes.groupby(["split", "class"]).size().unstack(fill_value=0)
                 .reindex(columns=class_names, fill_value=0))
    stats["class_counts"] = {s: per_class.loc[s].to_dict() for s in per_class.index}

    # How many DISTINCT images contain each class (nunique on the image name).
    imgs_with = boxes.groupby(["split", "class"])["stem"].nunique().unstack(fill_value=0).reindex(columns=class_names, fill_value=0)
    stats["images_containing_class"] = {s: imgs_with.loc[s].to_dict() for s in imgs_with.index}

    # Distribution of "boxes per image": count, mean, std, min, median ("50%"), max.
    stats["annotations_per_image"] = (
        manifest.groupby("split")["n_boxes"].describe()[["count", "mean", "std", "min", "50%", "max"]].round(3).to_dict("index")
    )
    # Images with zero boxes (the converter should leave none, so this is a consistency check).
    stats["images_without_objects"] = manifest[manifest["n_boxes"] == 0].groupby("split").size().to_dict()

    # The five most common image resolutions per split (built as "WIDTHxHEIGHT" strings).
    stats["image_resolutions"] = (manifest.assign(res=manifest["width"].astype(str) + "x" + manifest["height"].astype(str))
                                  .groupby("split")["res"].value_counts().groupby(level=0).head(5)
                                  .rename("n").reset_index().to_dict("records"))
    # How many boxes fall in each size bucket (small / medium / large) per split.
    stats["size_bucket_counts"] = {s: g["size_bucket"].value_counts().to_dict() for s, g in boxes.groupby("split")}

    # Share of boxes per class GROUP (common / ambiguous / india_specific), which is the project's
    # central comparison.
    totals = boxes["class"].value_counts()
    stats["group_share_of_boxes"] = (boxes.assign(group=boxes["class"].map(groups)).groupby(["split", "group"]).size()
                                     .unstack(fill_value=0).to_dict("index"))
    if len(totals):
        # Imbalance ratio: biggest class divided by the smallest class that has at least one box.
        stats["imbalance_ratio_max_over_min_nonzero"] = float(totals.max() / totals[totals > 0].min())
    # A class with 0 boxes in a split could never be learned or validated, so list any such class.
    stats["classes_with_zero_boxes"] = {s: [c for c in class_names if per_class.loc[s, c] == 0] for s in per_class.index}
    return stats


def _save(fig, out: Path, name: str, source: str):
    """Finish and save a figure. Every figure gets a small grey footer saying where its data came
    from, as the project plan requires ("every figure has a source/run context")."""
    fig.text(0.01, 0.005, source, fontsize=7, color="gray", ha="left", va="bottom")
    # rect leaves a thin strip at the bottom free so the footer is not overlapped by the plot.
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    path = out / name
    fig.savefig(path, dpi=150)
    plt.close(fig)  # free the memory; matplotlib keeps figures alive otherwise
    return path


def plot_class_distribution(boxes, class_names, out: Path, source: str):
    """Bar chart of box counts per class, split by train/val, on a LOG scale.

    Log scale is needed because the biggest class has ~100,000 boxes and the smallest ~18; on a
    linear axis the rare classes would be invisible.
    """
    # Table: one row per class, one column per split, cell = number of boxes (0 if none).
    counts = boxes.groupby(["class", "split"]).size().unstack(fill_value=0).reindex(class_names, fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 5.5))  # figure size in inches
    counts.plot.bar(ax=ax, width=0.8)           # grouped bars: train and val side by side per class
    ax.set_yscale("log")
    ax.set_ylabel("Number of boxes (log scale)")
    ax.set_xlabel("IDD class")
    ax.set_title("IDD class distribution by split")
    ax.tick_params(axis="x", rotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")  # right-align the rotated labels so they sit under their bar
    return _save(fig, out, "class_distribution.png", source)


def plot_class_share(boxes, class_names, out: Path, source: str):
    """Bar chart of each class's PERCENTAGE of its split's boxes.

    Using percentages makes train (381k boxes) and val (126k boxes) directly comparable, which
    checks that the two splits have a similar class mix.
    """
    share = boxes.groupby(["class", "split"]).size().unstack(fill_value=0).reindex(class_names, fill_value=0)
    share = share / share.sum() * 100  # divide each column by its own total -> percent
    fig, ax = plt.subplots(figsize=(11, 5.5))
    share.plot.bar(ax=ax, width=0.8)
    ax.set_ylabel("Share of split's boxes (%)")
    ax.set_title("Class share per split (checks train/val consistency)")
    ax.tick_params(axis="x", rotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    return _save(fig, out, "class_share_per_split.png", source)


def plot_annotations_per_image(manifest, out: Path, source: str):
    """Histogram of how many boxes each image has (a measure of how crowded scenes are)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    # Clip the x-axis at the 99.5th percentile so a few extremely crowded images do not squash the plot.
    top = int(manifest["n_boxes"].quantile(0.995)) + 1
    for split, g in manifest.groupby("split"):
        # density=True normalises each split to fractions, so unequal split sizes do not matter.
        ax.hist(g["n_boxes"].clip(upper=top), bins=range(0, top + 2), alpha=0.55, label=f"{split} (n={len(g)})", density=True)
    ax.set_xlabel(f"Annotations per image (clipped at {top})")
    ax.set_ylabel("Fraction of images")
    ax.set_title("Number of annotations per image")
    ax.legend()
    return _save(fig, out, "annotations_per_image.png", source)


def plot_box_size(boxes, out: Path, source: str):
    """Two panels: overall box-size histogram per split, and a box plot of box size per class."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Box areas span several orders of magnitude, so plot log10 of the area. clip(lower=1e-7)
    # avoids log10(0) = -infinity for any degenerate box.
    log_area = np.log10(boxes["area_rel"].clip(lower=1e-7))
    for split, g in boxes.assign(la=log_area).groupby("split"):
        axes[0].hist(g["la"], bins=60, alpha=0.55, label=split, density=True)
    axes[0].set_xlabel("log10(box area / image area)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Relative box area")
    axes[0].legend()
    # Right panel: one box plot per class. Each `data` item holds the log10 areas of that class's boxes.
    order = list(boxes["class"].unique())
    data = [np.log10(boxes.loc[boxes["class"] == c, "area_rel"].clip(lower=1e-7)) for c in order]
    # showfliers=False hides extreme outlier dots so the boxes themselves stay readable.
    axes[1].boxplot(data, tick_labels=order, showfliers=False)
    axes[1].set_ylabel("log10(relative area)")
    axes[1].set_title("Relative box area per class")
    axes[1].tick_params(axis="x", rotation=60)
    return _save(fig, out, "box_size_distribution.png", source)


def plot_size_buckets(boxes, class_names, out: Path, source: str):
    """Stacked bars: for each class, what % of its boxes are small / medium / large."""
    t = boxes.groupby(["class", "size_bucket"], observed=False).size().unstack(fill_value=0).reindex(class_names, fill_value=0)
    # Divide each row by its total to get percentages; replace(0, nan) avoids dividing by zero
    # for a class with no boxes.
    t = t.div(t.sum(axis=1).replace(0, np.nan), axis=0) * 100
    fig, ax = plt.subplots(figsize=(11, 5.5))
    t.plot.bar(stacked=True, ax=ax, width=0.8)
    ax.set_ylabel("% of the class's boxes")
    ax.set_title("Object size mix per class (small objects are harder to detect)")
    ax.tick_params(axis="x", rotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    return _save(fig, out, "size_buckets_per_class.png", source)


def run_eda(yolo_root: Path, class_names: list[str], groups: dict[str, str], out_dir: Path, release: str = "") -> dict:
    """Run the whole EDA: load data, compute stats, draw all figures, save CSV + JSON.

    Returns the stats dict (plus the list of figure file names) for printing or further use.
    """
    out_dir.mkdir(parents=True, exist_ok=True)  # create the output folder if needed
    # manifest.csv (written by the converter) gives per-image facts: size and number of boxes.
    manifest = pd.read_csv(yolo_root / "manifest.csv")
    boxes = load_boxes(yolo_root, class_names)
    # No boxes usually means the dataset was not converted yet; fail with a clear message.
    if boxes.empty:
        raise RuntimeError("No boxes found in converted labels; run prepare_dataset.py first.")
    # Footer text stamped on every figure so it can always be traced back to its data.
    source = f"Source: {yolo_root.as_posix()} | release: {release or 'unspecified'} | generated by scripts/analyze_dataset.py"
    stats = compute_stats(boxes, manifest, class_names, groups)
    # Draw the five figures; each plot function saves a PNG and returns its path.
    figs = [
        plot_class_distribution(boxes, class_names, out_dir, source),
        plot_class_share(boxes, class_names, out_dir, source),
        plot_annotations_per_image(manifest, out_dir, source),
        plot_box_size(boxes, out_dir, source),
        plot_size_buckets(boxes, class_names, out_dir, source),
    ]
    # Also save the class-count table as a CSV, easy to open in a spreadsheet or cite in the report.
    counts = boxes.groupby(["class", "split"]).size().unstack(fill_value=0).reindex(class_names, fill_value=0)
    counts.to_csv(out_dir / "class_counts.csv")
    # default=str lets json.dumps cope with values it cannot serialise natively (e.g. numpy numbers).
    (out_dir / "eda_stats.json").write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    stats["figures"] = [f.name for f in figs]
    return stats
