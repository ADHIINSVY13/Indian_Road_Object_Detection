"""EDA statistics and figures computed from the converted YOLO dataset.

Everything here is derived from the actual label files and manifest.csv;
nothing is hard-coded.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SIZE_BINS = [0, 0.001, 0.01, 1.01]          # relative box area
SIZE_LABELS = ["small (<0.1% of image)", "medium (0.1-1%)", "large (>1%)"]


def load_boxes(yolo_root: Path, class_names: list[str]) -> pd.DataFrame:
    rows = []
    for split_dir in sorted((yolo_root / "labels").glob("*")):
        for lf in split_dir.glob("*.txt"):
            text = lf.read_text(encoding="utf-8").strip()
            for line in text.splitlines() if text else []:
                c, xc, yc, w, h = line.split()
                rows.append((split_dir.name, lf.stem, int(c), float(xc), float(yc), float(w), float(h)))
    df = pd.DataFrame(rows, columns=["split", "stem", "cls", "xc", "yc", "w", "h"])
    df["class"] = df["cls"].map(lambda i: class_names[i])
    df["area_rel"] = df["w"] * df["h"]
    df["size_bucket"] = pd.cut(df["area_rel"], SIZE_BINS, labels=SIZE_LABELS, right=False)
    return df


def compute_stats(boxes: pd.DataFrame, manifest: pd.DataFrame, class_names: list[str], groups: dict[str, str]) -> dict:
    stats: dict = {"images_per_split": manifest.groupby("split").size().to_dict(),
                   "boxes_per_split": boxes.groupby("split").size().to_dict()}
    per_class = (boxes.groupby(["split", "class"]).size().unstack(fill_value=0)
                 .reindex(columns=class_names, fill_value=0))
    stats["class_counts"] = {s: per_class.loc[s].to_dict() for s in per_class.index}
    imgs_with = boxes.groupby(["split", "class"])["stem"].nunique().unstack(fill_value=0).reindex(columns=class_names, fill_value=0)
    stats["images_containing_class"] = {s: imgs_with.loc[s].to_dict() for s in imgs_with.index}
    stats["annotations_per_image"] = (
        manifest.groupby("split")["n_boxes"].describe()[["count", "mean", "std", "min", "50%", "max"]].round(3).to_dict("index")
    )
    stats["images_without_objects"] = manifest[manifest["n_boxes"] == 0].groupby("split").size().to_dict()
    stats["image_resolutions"] = (manifest.assign(res=manifest["width"].astype(str) + "x" + manifest["height"].astype(str))
                                  .groupby("split")["res"].value_counts().groupby(level=0).head(5)
                                  .rename("n").reset_index().to_dict("records"))
    stats["size_bucket_counts"] = {s: g["size_bucket"].value_counts().to_dict() for s, g in boxes.groupby("split")}
    totals = boxes["class"].value_counts()
    stats["group_share_of_boxes"] = (boxes.assign(group=boxes["class"].map(groups)).groupby(["split", "group"]).size()
                                     .unstack(fill_value=0).to_dict("index"))
    if len(totals):
        stats["imbalance_ratio_max_over_min_nonzero"] = float(totals.max() / totals[totals > 0].min())
    stats["classes_with_zero_boxes"] = {s: [c for c in class_names if per_class.loc[s, c] == 0] for s in per_class.index}
    return stats


def _save(fig, out: Path, name: str, source: str):
    fig.text(0.01, 0.005, source, fontsize=7, color="gray", ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    path = out / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_class_distribution(boxes, class_names, out: Path, source: str):
    counts = boxes.groupby(["class", "split"]).size().unstack(fill_value=0).reindex(class_names, fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    counts.plot.bar(ax=ax, width=0.8)
    ax.set_yscale("log")
    ax.set_ylabel("Number of boxes (log scale)")
    ax.set_xlabel("IDD class")
    ax.set_title("IDD class distribution by split")
    ax.tick_params(axis="x", rotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    return _save(fig, out, "class_distribution.png", source)


def plot_class_share(boxes, class_names, out: Path, source: str):
    share = boxes.groupby(["class", "split"]).size().unstack(fill_value=0).reindex(class_names, fill_value=0)
    share = share / share.sum() * 100
    fig, ax = plt.subplots(figsize=(11, 5.5))
    share.plot.bar(ax=ax, width=0.8)
    ax.set_ylabel("Share of split's boxes (%)")
    ax.set_title("Class share per split (checks train/val consistency)")
    ax.tick_params(axis="x", rotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    return _save(fig, out, "class_share_per_split.png", source)


def plot_annotations_per_image(manifest, out: Path, source: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    top = int(manifest["n_boxes"].quantile(0.995)) + 1
    for split, g in manifest.groupby("split"):
        ax.hist(g["n_boxes"].clip(upper=top), bins=range(0, top + 2), alpha=0.55, label=f"{split} (n={len(g)})", density=True)
    ax.set_xlabel(f"Annotations per image (clipped at {top})")
    ax.set_ylabel("Fraction of images")
    ax.set_title("Number of annotations per image")
    ax.legend()
    return _save(fig, out, "annotations_per_image.png", source)


def plot_box_size(boxes, out: Path, source: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    log_area = np.log10(boxes["area_rel"].clip(lower=1e-7))
    for split, g in boxes.assign(la=log_area).groupby("split"):
        axes[0].hist(g["la"], bins=60, alpha=0.55, label=split, density=True)
    axes[0].set_xlabel("log10(box area / image area)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Relative box area")
    axes[0].legend()
    order = list(boxes["class"].unique())
    data = [np.log10(boxes.loc[boxes["class"] == c, "area_rel"].clip(lower=1e-7)) for c in order]
    axes[1].boxplot(data, tick_labels=order, showfliers=False)
    axes[1].set_ylabel("log10(relative area)")
    axes[1].set_title("Relative box area per class")
    axes[1].tick_params(axis="x", rotation=60)
    return _save(fig, out, "box_size_distribution.png", source)


def plot_size_buckets(boxes, class_names, out: Path, source: str):
    t = boxes.groupby(["class", "size_bucket"], observed=False).size().unstack(fill_value=0).reindex(class_names, fill_value=0)
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
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(yolo_root / "manifest.csv")
    boxes = load_boxes(yolo_root, class_names)
    if boxes.empty:
        raise RuntimeError("No boxes found in converted labels; run prepare_dataset.py first.")
    source = f"Source: {yolo_root.as_posix()} | release: {release or 'unspecified'} | generated by scripts/analyze_dataset.py"
    stats = compute_stats(boxes, manifest, class_names, groups)
    figs = [
        plot_class_distribution(boxes, class_names, out_dir, source),
        plot_class_share(boxes, class_names, out_dir, source),
        plot_annotations_per_image(manifest, out_dir, source),
        plot_box_size(boxes, out_dir, source),
        plot_size_buckets(boxes, class_names, out_dir, source),
    ]
    counts = boxes.groupby(["class", "split"]).size().unstack(fill_value=0).reindex(class_names, fill_value=0)
    counts.to_csv(out_dir / "class_counts.csv")
    (out_dir / "eda_stats.json").write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    stats["figures"] = [f.name for f in figs]
    return stats
