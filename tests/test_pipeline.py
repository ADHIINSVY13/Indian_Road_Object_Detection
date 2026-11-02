"""End-to-end test of inspect -> convert -> validate -> EDA on a tiny synthetic
VOC-style dataset. The synthetic data only exercises the code paths; it says
nothing about IDD itself."""
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.config import load_classes, resolve
from src.preprocessing.convert import convert_dataset
from src.preprocessing.inspect_annotations import inspect_dataset
from src.preprocessing.validate import box_issues, validate_yolo_dataset
from src.preprocessing.voc import parse_voc_xml
from src.visualization.eda import run_eda

XML = """<annotation><size><width>{w}</width><height>{h}</height></size>{objs}</annotation>"""
OBJ = "<object><name>{n}</name><bndbox><xmin>{a}</xmin><ymin>{b}</ymin><xmax>{c}</xmax><ymax>{d}</ymax></bndbox></object>"


def make_sample(root, stem, w, h, boxes, xml_size=None):
    (root / "JPEGImages" / Path(stem).parent).mkdir(parents=True, exist_ok=True)
    (root / "Annotations" / Path(stem).parent).mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h)).save(root / "JPEGImages" / f"{stem}.jpg")
    xw, xh = xml_size or (w, h)
    objs = "".join(OBJ.format(n=n, a=a, b=b, c=c, d=d) for n, a, b, c, d in boxes)
    (root / "Annotations" / f"{stem}.xml").write_text(XML.format(w=xw, h=xh, objs=objs), encoding="utf-8")


@pytest.fixture()
def dataset(tmp_path):
    raw = tmp_path / "raw"
    make_sample(raw, "seq1/a", 200, 100, [("car", 10, 10, 60, 50), ("autorickshaw", 100, 20, 150, 90)])
    make_sample(raw, "seq1/b", 200, 100, [("Person", -5, 0, 40, 60),       # clipped
                                          ("car", 50, 50, 50, 80),          # zero width -> dropped
                                          ("dragon", 1, 1, 20, 20)])       # unknown class -> dropped
    make_sample(raw, "seq2/c", 200, 100, [("animal", 0, 0, 100, 100)])
    make_sample(raw, "seq2/d", 200, 100, [("bus", 5, 5, 90, 60)], xml_size=(400, 100))  # aspect mismatch -> skipped
    (raw / "train.txt").write_text("seq1/a\nseq1/b\nseq2/c\nseq2/missing\n")
    (raw / "val.txt").write_text("seq2/d\n")
    raw_cfg = {"root": str(raw), "image_dir": "JPEGImages", "annotation_dir": "Annotations",
               "image_extensions": [".jpg"], "split_files": {"train": "train.txt", "val": "val.txt", "test": "test.txt"}}
    yolo_cfg = {"root": str(tmp_path / "yolo"), "image_mode": "copy", "min_box_side_px": 2}
    classes = load_classes(resolve("configs/class_mapping.yaml"))
    return raw, raw_cfg, yolo_cfg, classes, tmp_path


def test_box_issues():
    assert box_issues(0, 0, 10, 10, 100, 100) == []
    assert ("error", "non_positive_size") in box_issues(5, 5, 5, 9, 100, 100)
    assert ("warning", "out_of_bounds_clipped") in box_issues(-1, 0, 10, 10, 100, 100)
    assert ("warning", "tiny_box") in box_issues(0, 0, 1, 10, 100, 100)


def test_parse_malformed(tmp_path):
    p = tmp_path / "bad.xml"
    p.write_text("<annotation><oops")
    assert parse_voc_xml(p).problems[0].startswith("xml_parse_error")


def test_inspect(dataset):
    raw, *_ = dataset
    res = inspect_dataset(raw, "Annotations", n_show=2)
    assert res["detected_format"] == "voc_xml"
    assert res["scan"]["class_names_observed"]["car"] == 2
    assert res["split_files_line_counts"]["test.txt"] is None


def test_convert_validate_eda(dataset):
    raw, raw_cfg, yolo_cfg, classes, tmp = dataset
    report, issues = convert_dataset(raw_cfg, yolo_cfg, classes, Path("."))
    codes = {i["code"] for i in issues}
    assert {"missing_image", "unknown_class", "non_positive_size", "out_of_bounds_clipped",
            "aspect_mismatch_skipped"} <= codes
    train = report["splits"]["train"]
    assert train["converted"] == 3 and train["boxes_kept"] == 4
    assert report["splits"]["val"]["skipped_aspect_mismatch"] == 1
    assert "test" in report["splits"] and "not converted" in report["splits"]["test"]["status"]

    yolo = Path(yolo_cfg["root"])
    assert (yolo / "idd.yaml").exists() and (yolo / "manifest.csv").exists()
    lab = (yolo / "labels/train/seq1__b.txt").read_text().split()
    assert lab[0] == "0" and float(lab[1]) == pytest.approx(20 / 200)  # person, clipped x 0..40

    post = validate_yolo_dataset(yolo, len(classes), check_images=True)
    assert all(i["severity"] == "warning" for i in post)

    stats = run_eda(yolo, [c.name for c in classes], {c.name: c.group for c in classes}, tmp / "eda", "synthetic")
    assert stats["boxes_per_split"]["train"] == 4
    assert (tmp / "eda" / "class_distribution.png").exists()


def test_validator_catches_bad_labels(dataset):
    raw, raw_cfg, yolo_cfg, classes, tmp = dataset
    convert_dataset(raw_cfg, yolo_cfg, classes, Path("."))
    yolo = Path(yolo_cfg["root"])
    (yolo / "labels/train/seq1__a.txt").write_text("99 0.5 0.5 0.2 0.2\n0 1.5 0.5 0.2 0.2\nnonsense\n")
    codes = {i["code"] for i in validate_yolo_dataset(yolo, len(classes))}
    assert {"class_id_out_of_range", "coords_out_of_range", "bad_line_format"} <= codes


def test_resize_keeps_labels_valid(dataset):
    raw, raw_cfg, yolo_cfg, classes, tmp = dataset
    yolo_cfg = {**yolo_cfg, "max_side": 64}
    convert_dataset(raw_cfg, yolo_cfg, classes, Path("."))
    yolo = Path(yolo_cfg["root"])
    with Image.open(yolo / "images/train/seq1__a.jpg") as im:
        assert im.size == (64, 32)
    assert all(i["severity"] == "warning" for i in validate_yolo_dataset(yolo, len(classes), check_images=True))
