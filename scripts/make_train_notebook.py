"""Regenerates notebooks/02_train_colab.ipynb (no outputs stored).

The Colab notebook is generated from this script instead of being edited by hand, so the notebook
is always consistent with the training entry point (scripts/train_yolo.py) it calls. To change the
notebook, edit the cell text below and re-run `python scripts/make_train_notebook.py`.

Lines that start with `!` or `%` inside the code cells are Colab/IPython magics: `!cmd` runs a shell
command, `%cd` changes the notebook's working folder, and `{NAME}` inside them is replaced by the
Python variable of that name.
"""
from pathlib import Path

# nbformat is the official library for building notebook files from Python.
import nbformat as nbf

# Repository root: this file is <repo>/scripts/make_train_notebook.py.
root = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()

# SETUP is the text of ONE notebook code cell: the only place a user edits paths and settings
# (Drive folder, repo URL/branch, model, epochs, batch size, run name). Everything after it in the
# notebook reads these variables, so nothing else needs changing.
SETUP = '''# ---- configure these once, then run the notebook top to bottom ----
from pathlib import Path

# Where things live. Drive is permanent storage; /content is fast local disk that is wiped when the
# Colab session ends.
DRIVE_ROOT = "/content/drive/MyDrive/idd_p07"          # your Drive folder
TAR        = f"{DRIVE_ROOT}/idd_yolo_640.tar"           # uploaded from the laptop
DATA_DIR   = "/content/idd_yolo"                        # ephemeral, wiped on disconnect
# The code is cloned from GitHub so the notebook always runs the committed version of the scripts.
REPO_URL   = "https://github.com/ADHIINSVY13/P07_Indian_Road_Object_Detection.git"
BRANCH     = "feat/dataset-eda-pipeline"
REPO_DIR   = "/content/repo"

# Training settings (they are passed straight to scripts/train_yolo.py).
MODEL      = "yolo11n.pt"     # COCO-pretrained start point
EPOCHS     = 30               # passes over the training set
BATCH      = 16               # T4 16 GB handles 16 at 640 comfortably
IMGSZ      = 640              # network input size in pixels
RUN_NAME   = "idd_yolo11n_e30"
PROJECT    = f"{DRIVE_ROOT}/runs"   # runs land on Drive, so a disconnect loses at most one epoch

# Print the key paths so the run log shows what was used (and whether the dataset tar is found).
print("tar   :", TAR, "exists:", Path(TAR).exists())
print("data  :", DATA_DIR)
print("run to:", f"{PROJECT}/{RUN_NAME}")'''

# The notebook as an ordered list of cells: numbered markdown headings explain each step, and the
# code cells that follow do it. Order: check GPU -> install -> mount Drive -> set paths -> unpack the
# dataset -> get the code -> dry-run gate -> train -> (resume) -> read results -> save outputs.
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
        "cell picks up from the last epoch because checkpoints are written to Drive, not `/content`.\n"
        "\n"
        "**Big picture:** the notebook only prepares the machine (GPU, libraries, data, code) and then calls "
        "the same `scripts/train_yolo.py` that works on any computer. No training logic lives here."
    ),
    nbf.v4.new_markdown_cell(
        "## 0. Confirm we actually got a GPU\n"
        "Training on CPU here would take days, so fail fast. In Colab: Runtime > Change runtime type > T4 GPU."
    ),
    nbf.v4.new_code_cell(
        "# nvidia-smi lists the GPU. We ask only for its name, memory and driver version, as CSV.\n"
        "!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv\n"
        "import torch\n"
        "# cuda available: True means PyTorch can use the GPU; False means this run would fall back to the CPU.\n"
        "print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
    ),
    nbf.v4.new_markdown_cell(
        "## 1. Install Ultralytics\n"
        "Colab images already carry a matching torch build, so only ultralytics is added. The version is "
        "pinned to the 8.3 series so results are reproducible."
    ),
    nbf.v4.new_code_cell(
        "# -q = quiet. ultralytics==8.3.* accepts any 8.3 release (the series the code was tested with).\n"
        "!pip -q install ultralytics==8.3.* \n"
        "import ultralytics\n"
        "# Print the exact version, so it can be recorded next to the results.\n"
        "print('ultralytics', ultralytics.__version__)"
    ),
    nbf.v4.new_markdown_cell(
        "## 2. Mount Drive\n"
        "The tar and every checkpoint live here, so they survive the disconnect. Colab will ask you to "
        "grant access to your Google Drive."
    ),
    nbf.v4.new_code_cell(
        "# Makes your Google Drive visible as the folder /content/drive.\n"
        "from google.colab import drive\n"
        "drive.mount('/content/drive')"
    ),
    nbf.v4.new_markdown_cell(
        "## 3. Paths and run parameters\n"
        "All knobs in one place. `RUN_NAME` is what the summary file is called, so make it describe the run."
    ),
    nbf.v4.new_code_cell(SETUP),
    nbf.v4.new_markdown_cell(
        "## 4. Unpack the dataset into local disk\n"
        "`/content` is ephemeral and much faster than Drive, so training reads images from there. "
        "Re-running skips the work if the folder already exists."
    ),
    nbf.v4.new_code_cell(
        "import os, subprocess, time\n"
        "# If the train images are already there (cell was run before), do not extract a second time.\n"
        "if os.path.exists(f'{DATA_DIR}/images/train'):\n"
        "    print('already extracted, skipping')\n"
        "else:\n"
        "    t = time.time()\n"
        "    os.makedirs(DATA_DIR, exist_ok=True)\n"
        "    # tar -xf extracts the single archive from Drive into the fast local folder.\n"
        "    subprocess.run(['tar', '-xf', TAR, '-C', DATA_DIR], check=True)\n"
        "    print(f'extracted in {time.time() - t:.0f}s')\n"
        "\n"
        "# Safety check: every image must have exactly one label file. A mismatch means a damaged\n"
        "# download or tar, and training on it would silently give wrong results.\n"
        "for split in ('train', 'val'):\n"
        "    n = len(os.listdir(f'{DATA_DIR}/images/{split}'))\n"
        "    m = len(os.listdir(f'{DATA_DIR}/labels/{split}'))\n"
        "    print(f'{split}: {n} images, {m} labels', 'OK' if n == m else 'MISMATCH')\n"
        "    assert n == m, f'{split}: image/label count mismatch -- do not train on this'"
    ),
    nbf.v4.new_markdown_cell(
        "## 5. Get the code\n"
        "`train_yolo.py --data-root` points the trainer at `/content/idd_yolo` instead of the repo-relative\n"
        "`data/yolo`, and rewrites `idd.yaml` with this machine's paths, so the absolute path baked in on the\n"
        "laptop is irrelevant."
    ),
    nbf.v4.new_code_cell(
        "import os\n"
        "# First run: clone the repository. Later runs: just fetch the newest commits.\n"
        "if os.path.exists(REPO_DIR):\n"
        "    !git -C {REPO_DIR} fetch --all --prune\n"
        "else:\n"
        "    !git clone {REPO_URL} {REPO_DIR}\n"
        "# Switch to the project branch and fast-forward to its latest commit.\n"
        "!git -C {REPO_DIR} checkout {BRANCH}\n"
        "!git -C {REPO_DIR} pull --ff-only\n"
        "# Show which commit is checked out, so the run can be tied to an exact version of the code.\n"
        "!git -C {REPO_DIR} log -1 --format='%h %ad %s' --date=short"
    ),
    nbf.v4.new_markdown_cell(
        "## 6. Gate: the training entry point must exist and see the data\n"
        "If this raises, the branch was not pushed from the laptop. `--dry-run` checks the dataset and "
        "prints the plan but trains nothing, so a typo costs seconds instead of an hour."
    ),
    nbf.v4.new_code_cell(
        "# Move into the cloned repository so relative paths such as scripts/ and configs/ resolve.\n"
        "%cd {REPO_DIR}\n"
        "import os\n"
        "assert os.path.exists('scripts/train_yolo.py'), 'scripts/train_yolo.py missing: commit and push it from the laptop first'\n"
        "# Same command as the real training run below, plus --dry-run.\n"
        "!python scripts/train_yolo.py --data-root {DATA_DIR} --model {MODEL} --epochs {EPOCHS} --batch {BATCH} --imgsz {IMGSZ} --name {RUN_NAME} --project {PROJECT} --dry-run"
    ),
    nbf.v4.new_markdown_cell(
        "## 7. Train\n"
        "`--project` points at Drive on purpose: `last.pt` is rewritten every epoch there, which is what\n"
        "makes the resume cell work after a disconnect. Expect ~4-6 min per epoch after the first one\n"
        "(the first also builds the label cache)."
    ),
    nbf.v4.new_code_cell(
        "%cd {REPO_DIR}\n"
        "# --workers 2: Colab's free CPU has only two cores to load and augment images.\n"
        "!python scripts/train_yolo.py --data-root {DATA_DIR} --model {MODEL} --epochs {EPOCHS} --batch {BATCH} --imgsz {IMGSZ} --workers 2 --name {RUN_NAME} --project {PROJECT}"
    ),
    nbf.v4.new_markdown_cell(
        "## 8. Resume after a disconnect\n"
        "Only run this if cell 7 was cut short (disconnect, quota, crash). Ultralytics continues from the "
        "stored epoch count. After a disconnect, re-run cells 0-6 first so the GPU, libraries, Drive and "
        "dataset are back, then run this cell instead of cell 7."
    ),
    nbf.v4.new_code_cell(
        "%cd {REPO_DIR}\n"
        "# Same arguments as the training cell plus --resume, which loads <PROJECT>/<RUN_NAME>/weights/last.pt.\n"
        "!python scripts/train_yolo.py --data-root {DATA_DIR} --model {MODEL} --epochs {EPOCHS} --batch {BATCH} --imgsz {IMGSZ} --workers 2 --name {RUN_NAME} --project {PROJECT} --resume"
    ),
    nbf.v4.new_markdown_cell(
        "## 9. Results\n"
        "`<RUN_NAME>_summary.json` holds per-class and per-group AP50-95. The group numbers are the\n"
        "project's actual question: does IDD fine-tuning move the India-specific classes more than the\n"
        "common ones? Ultralytics also writes `results.csv`, `confusion_matrix.png`, PR/F1 curves and\n"
        "`weights/best.pt` into the run directory on Drive."
    ),
    nbf.v4.new_code_cell(
        "import json\n"
        "# The summary file written at the end of train_yolo.py.\n"
        "p = f'/content/repo/results/{RUN_NAME}_summary.json'\n"
        "s = json.load(open(p))\n"
        "# Overall precision / recall / mAP, then the mean AP per class group, then where the weights are.\n"
        "print(json.dumps(s['metrics'], indent=2))\n"
        "print(json.dumps(s['per_group_ap50_95'], indent=2))\n"
        "print('weights:', s['weights_best'])"
    ),
    nbf.v4.new_markdown_cell(
        "## 10. Keep the numbers in the repo\n"
        "Download this small bundle and commit it — the report cites it, so it must not live only on Drive."
    ),
    nbf.v4.new_code_cell(
        "import shutil, os\n"
        "src = f'{PROJECT}/{RUN_NAME}'\n"
        "bundle = f'{DRIVE_ROOT}/to_commit_{RUN_NAME}'\n"
        "os.makedirs(bundle, exist_ok=True)\n"
        "# Copy the small files worth committing: per-epoch metrics, confusion matrices and PR/F1 curves.\n"
        "# (The large weights stay on Drive; skip any file that was not produced.)\n"
        "for f in ('results.csv', 'confusion_matrix.png', 'confusion_matrix_normalized.png', 'BoxPR_curve.png', 'BoxF1_curve.png'):\n"
        "    if os.path.exists(f'{src}/{f}'):\n"
        "        shutil.copy(f'{src}/{f}', bundle)\n"
        "shutil.copy(f'/content/repo/results/{RUN_NAME}_summary.json', bundle)\n"
        "print('collect these into the repo:', sorted(os.listdir(bundle)))"
    ),
]

nb["cells"] = cells
out = root / "notebooks" / "02_train_colab.ipynb"
nbf.write(nb, out)  # writes the .ipynb JSON file, with no stored outputs
print("wrote", out)
