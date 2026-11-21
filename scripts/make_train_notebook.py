"""Regenerates notebooks/02_train_colab.ipynb (no outputs stored)."""
from pathlib import Path

import nbformat as nbf

root = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()

SETUP = '''# ---- configure these once, then run the notebook top to bottom ----
from pathlib import Path

DRIVE_ROOT = "/content/drive/MyDrive/idd_p07"          # your Drive folder
TAR        = f"{DRIVE_ROOT}/idd_yolo_640.tar"           # uploaded from the laptop
DATA_DIR   = "/content/idd_yolo"                        # ephemeral, wiped on disconnect
REPO_URL   = "https://github.com/ADHIINSVY13/P07_Indian_Road_Object_Detection.git"
BRANCH     = "feat/dataset-eda-pipeline"
REPO_DIR   = "/content/repo"

MODEL      = "yolo11n.pt"     # COCO-pretrained start point
EPOCHS     = 30
BATCH      = 16               # T4 16 GB handles 16 at 640 comfortably
IMGSZ      = 640
RUN_NAME   = "idd_yolo11n_e30"
PROJECT    = f"{DRIVE_ROOT}/runs"   # runs land on Drive, so a disconnect loses at most one epoch

print("tar   :", TAR, "exists:", Path(TAR).exists())
print("data  :", DATA_DIR)
print("run to:", f"{PROJECT}/{RUN_NAME}")'''

cells = [
    nbf.v4.new_markdown_cell(
        "# 02 Fine-tune YOLO on IDD (Colab GPU)\n"
        "\n"
        "The laptop has no NVIDIA GPU, so training happens here and only here. Everything upstream\n"
        "(extract, verify, convert, EDA) is already done locally and ships as one tar file.\n"
        "\n"
        "**Before running, from the laptop:**\n"
        "1. `python scripts/train_yolo.py --dry-run` must pass (it does).\n"
        "2. `git add -A && git commit -m \"add training entry point\" && git push` — the notebook clones\n"
        "   the repo, so an unpushed `scripts/train_yolo.py` will not be there.\n"
        "3. Upload `data/idd_yolo_640.tar` (2.4 GB) to `MyDrive/idd_p07/`. Upload the single tar, never\n"
        "   the 41,794 loose files — Drive throttles badly on many small files.\n"
        "\n"
        "**Session budget:** free Colab T4 gives ~4-12 h and disconnects after ~90 min idle. 30 epochs\n"
        "on 31,569 images is roughly 2-3 h, so one session is normally enough. If it dies, the resume\n"
        "cell picks up from the last epoch because checkpoints are written to Drive, not `/content`."
    ),
    nbf.v4.new_markdown_cell("## 0. Confirm we actually got a GPU\nTraining on CPU here would take days, so fail fast."),
    nbf.v4.new_code_cell("!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv\nimport torch\nprint('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"),
    nbf.v4.new_markdown_cell("## 1. Install Ultralytics\nColab images already carry a matching torch build, so only ultralytics is added."),
    nbf.v4.new_code_cell("!pip -q install ultralytics==8.3.* \nimport ultralytics\nprint('ultralytics', ultralytics.__version__)"),
    nbf.v4.new_markdown_cell("## 2. Mount Drive\nThe tar and every checkpoint live here, so they survive the disconnect."),
    nbf.v4.new_code_cell("from google.colab import drive\ndrive.mount('/content/drive')"),
    nbf.v4.new_markdown_cell("## 3. Paths and run parameters\nAll knobs in one place. `RUN_NAME` is what the summary file is called, so make it describe the run."),
    nbf.v4.new_code_cell(SETUP),
    nbf.v4.new_markdown_cell("## 4. Unpack the dataset into local disk\n`/content` is ephemeral and much faster than Drive. Re-running skips the work."),
    nbf.v4.new_code_cell("import os, subprocess, time\nif os.path.exists(f'{DATA_DIR}/images/train'):\n    print('already extracted, skipping')\nelse:\n    t = time.time()\n    os.makedirs(DATA_DIR, exist_ok=True)\n    subprocess.run(['tar', '-xf', TAR, '-C', DATA_DIR], check=True)\n    print(f'extracted in {time.time() - t:.0f}s')\n\nfor split in ('train', 'val'):\n    n = len(os.listdir(f'{DATA_DIR}/images/{split}'))\n    m = len(os.listdir(f'{DATA_DIR}/labels/{split}'))\n    print(f'{split}: {n} images, {m} labels', 'OK' if n == m else 'MISMATCH')\n    assert n == m, f'{split}: image/label count mismatch -- do not train on this'"),
    nbf.v4.new_markdown_cell("## 5. Get the code\n`train_yolo.py --data-root` points the trainer at `/content/idd_yolo` instead of the repo-relative\n`data/yolo`, and rewrites `idd.yaml` with this machine's paths, so the absolute path baked in on the\nlaptop is irrelevant."),
    nbf.v4.new_code_cell("import os\nif os.path.exists(REPO_DIR):\n    !git -C {REPO_DIR} fetch --all --prune\nelse:\n    !git clone {REPO_URL} {REPO_DIR}\n!git -C {REPO_DIR} checkout {BRANCH}\n!git -C {REPO_DIR} pull --ff-only\n!git -C {REPO_DIR} log -1 --format='%h %ad %s' --date=short"),
    nbf.v4.new_markdown_cell("## 6. Gate: the training entry point must exist and see the data\nIf this raises, the branch was not pushed from the laptop."),
    nbf.v4.new_code_cell("%cd {REPO_DIR}\nimport os\nassert os.path.exists('scripts/train_yolo.py'), 'scripts/train_yolo.py missing: commit and push it from the laptop first'\n!python scripts/train_yolo.py --data-root {DATA_DIR} --model {MODEL} --epochs {EPOCHS} --batch {BATCH} --imgsz {IMGSZ} --name {RUN_NAME} --project {PROJECT} --dry-run"),
    nbf.v4.new_markdown_cell(
        "## 7. Train\n"
        "`--project` points at Drive on purpose: `last.pt` is rewritten every epoch there, which is what\n"
        "makes the resume cell work after a disconnect. Expect ~4-6 min per epoch after the first one\n"
        "(the first also builds the label cache)."
    ),
    nbf.v4.new_code_cell("%cd {REPO_DIR}\n!python scripts/train_yolo.py --data-root {DATA_DIR} --model {MODEL} --epochs {EPOCHS} --batch {BATCH} --imgsz {IMGSZ} --workers 2 --name {RUN_NAME} --project {PROJECT}"),
    nbf.v4.new_markdown_cell("## 8. Resume after a disconnect\nOnly run this if cell 7 was cut short. Ultralytics continues from the stored epoch count."),
    nbf.v4.new_code_cell("%cd {REPO_DIR}\n!python scripts/train_yolo.py --data-root {DATA_DIR} --model {MODEL} --epochs {EPOCHS} --batch {BATCH} --imgsz {IMGSZ} --workers 2 --name {RUN_NAME} --project {PROJECT} --resume"),
    nbf.v4.new_markdown_cell(
        "## 9. Results\n"
        "`<RUN_NAME>_summary.json` holds per-class and per-group AP50-95. The group numbers are the\n"
        "project's actual question: does IDD fine-tuning move the India-specific classes more than the\n"
        "common ones? Ultralytics also writes `results.csv`, `confusion_matrix.png`, PR/F1 curves and\n"
        "`weights/best.pt` into the run directory on Drive."
    ),
    nbf.v4.new_code_cell("import json\np = f'/content/repo/results/{RUN_NAME}_summary.json'\ns = json.load(open(p))\nprint(json.dumps(s['metrics'], indent=2))\nprint(json.dumps(s['per_group_ap50_95'], indent=2))\nprint('weights:', s['weights_best'])"),
    nbf.v4.new_markdown_cell("## 10. Keep the numbers in the repo\nDownload this small bundle and commit it — the report cites it, so it must not live only on Drive."),
    nbf.v4.new_code_cell("import shutil, os\nsrc = f'{PROJECT}/{RUN_NAME}'\nbundle = f'{DRIVE_ROOT}/to_commit_{RUN_NAME}'\nos.makedirs(bundle, exist_ok=True)\nfor f in ('results.csv', 'confusion_matrix.png', 'confusion_matrix_normalized.png', 'BoxPR_curve.png', 'BoxF1_curve.png'):\n    if os.path.exists(f'{src}/{f}'):\n        shutil.copy(f'{src}/{f}', bundle)\nshutil.copy(f'/content/repo/results/{RUN_NAME}_summary.json', bundle)\nprint('collect these into the repo:', sorted(os.listdir(bundle)))"),
]

nb["cells"] = cells
out = root / "notebooks" / "02_train_colab.ipynb"
nbf.write(nb, out)
print("wrote", out)
