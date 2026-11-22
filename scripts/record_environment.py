"""Record the software environment a result was produced in.

    python scripts/record_environment.py                    # writes results/environment_<tag>.json
    python scripts/record_environment.py --tag colab_t4     # run this inside the Colab notebook too

`requirements-train.txt` only gives version *ranges*, so the exact versions behind a number would
otherwise be lost. Run it on every machine that produced a result and commit the JSON.

Why this matters (reproducibility): the same code can give slightly different numbers with a different
PyTorch or Ultralytics version, or on GPU instead of CPU. Writing the versions and the hardware down
lets anyone repeat the experiment under the same conditions, or explain a small difference.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

# Make `src` importable when run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.config import resolve


def version_of(module: str) -> str | None:
    """Return a package's version string, or None if it is not installed.

    __import__ loads the module by its name given as text; every well-behaved package exposes
    its version as __version__.
    """
    try:
        return __import__(module).__version__
    except Exception:  # noqa: BLE001 - a missing package is a fact worth recording, not an error
        return None


def gather() -> dict:
    """Collect Python, OS, key package versions and GPU details into one dict.

    Output keys: python, platform, packages {name: version or None}, cuda_available, cuda_version, gpu.
    "cuda" is NVIDIA's GPU computing platform; cuda_available=False means the code ran on the CPU.
    """
    # platform.platform() names the operating system, e.g. "Windows-10-..." or "Linux-6.6...".
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {m: version_of(m) for m in ("torch", "torchvision", "ultralytics", "numpy", "PIL", "yaml", "matplotlib")}}
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()   # is an NVIDIA GPU usable?
        info["cuda_version"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        # torch is not installed on this machine (e.g. the laptop): record "unknown".
        info["cuda_available"] = None
    return info


def main():
    """Write the gathered environment to results/environment_<tag>.json and print it."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="local")  # label for the output file, e.g. "colab_t4"
    args = ap.parse_args()
    out = resolve("results") / f"environment_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(gather(), indent=2), encoding="utf-8")
    print(out.read_text(encoding="utf-8"))  # echo the saved file so it can be pasted into a report


if __name__ == "__main__":
    main()
