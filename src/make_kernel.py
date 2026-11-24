"""Create a Kaggle kernel folder from kaggle/run_template.py and a job list.

Usage: .venv/bin/python src/make_kernel.py --name runA --account <your-kaggle-username> --jobs jobs/runA.json
"""

import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--account", required=True)
ap.add_argument("--jobs", required=True)
ap.add_argument("--resume_from", default=None, help="earlier kernel slug whose output is reused")
a = ap.parse_args()

jobs = json.load(open(a.jobs))
d = Path("kaggle/runs") / a.name
d.mkdir(parents=True, exist_ok=True)
src = Path("kaggle/run_template.py").read_text().replace("__JOBS__", json.dumps(jobs))
(d / "run.py").write_text(src)
meta = {
    "id": f"{a.account}/vlm-drive-{a.name.lower()}",
    "title": f"vlm-drive-{a.name.lower()}",
    "code_file": "run.py", "language": "python", "kernel_type": "script",
    "is_private": True, "enable_gpu": True, "enable_internet": True,
    "dataset_sources": [f"{a.account}/vlm-drive-data", "sahangunasekara92/nuscenes-v1-0-full-keyframes"],
    "competition_sources": [],
    "kernel_sources": [a.resume_from] if a.resume_from else [],
}
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
print(d, meta["id"])
