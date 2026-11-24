"""Full-run kernel: two GPU lanes run jobs in parallel, then jobs that need both GPUs.
Every job is resumable (scores are saved every 50 images) and has its own time limit, so a
session never exceeds Kaggle's 12-hour cap. JOBS is filled in by src/make_kernel.py."""

import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time

JOBS = json.loads(r'''__JOBS__''')
SESSION_HOURS = 11.3

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.1",
                "accelerate>=1.0", "pyarrow", "num2words"], check=True)

DATA = glob.glob("/kaggle/input/**/vlm_score.py", recursive=True)[0].rsplit("/", 1)[0]
OUT = "/kaggle/working/out"
os.makedirs(OUT, exist_ok=True)
# resume from earlier outputs attached as a kernel source, if any
for f in glob.glob("/kaggle/input/**/out/*.parquet", recursive=True):
    if not os.path.exists(f"{OUT}/{os.path.basename(f)}"):
        shutil.copy(f, OUT)
T0 = time.time()


def hours_left():
    return SESSION_HOURS - (time.time() - T0) / 3600


def run(job, gpus):
    left = hours_left() - 0.25
    if left < 0.3:
        print(f"skip {job['name']}: no time left", flush=True)
        return
    script = job.get("script") or ("baselines.py" if job.get("method") else "vlm_score.py")
    cmd = [sys.executable, f"{DATA}/{script}", "--model", job["model"], "--gpus", gpus,
           "--probes", f"{DATA}/probes.parquet", "--out", OUT,
           "--time_limit", f"{min(left, job.get('hours', 99)):.2f}"]
    for k in ("view", "split", "image_ids", "templates", "classes", "method", "full_scores", "part", "configs", "n"):
        if job.get(k):
            v = str(job[k]).replace("{OUT}", OUT).replace("{DATA}", DATA)
            cmd += [f"--{k}", v]
    if job.get("clamp"):
        cmd.append("--clamp")
    hf = f"/tmp/hf_{gpus.replace(',', '')}"
    t = time.time()
    with open(f"{OUT}/{job['name']}.log", "a") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "HF_HOME": hf})
    print(f"== {job['name']} gpus={gpus} rc={r.returncode} {(time.time() - t) / 60:.0f} min, "
          f"{hours_left():.1f} h left", flush=True)
    print(open(f"{OUT}/{job['name']}.log").read()[-600:], flush=True)


def lane(jobs, gpus):
    for j in jobs:
        run(j, gpus)
    shutil.rmtree(f"/tmp/hf_{gpus.replace(',', '')}", ignore_errors=True)


def lanes():
    th = [threading.Thread(target=lane, args=(JOBS.get("gpu0", []), "0")),
          threading.Thread(target=lane, args=(JOBS.get("gpu1", []), "1"))]
    [t.start() for t in th]
    [t.join() for t in th]


if JOBS.get("both_first"):
    lane(JOBS.get("both", []), "0,1")
    lanes()
else:
    lanes()
    lane(JOBS.get("both", []), "0,1")
print("RUN DONE", flush=True)
