"""Statistical fault injection (FI) into a VLM during image encoding, with optional Ranger.

Follows the fault models of the resilience literature:
  act     single transient bit flip in one output element of one Linear layer (LLTFI-style)
  weight  single bit flip in one weight element of one Linear layer for one inference (DLAFI-style)
Sites: 'vision' (vision encoder Linear layers) or 'llm' (language-model Linear layers; lm_head and
embeddings excluded). The fault fires during the prefix pass (image + prompt), so it reaches every
question of that image through the KV cache, as a transient fault during perception would.

Protection 'ranger' (Chen, Li & Pattabiraman, DSN 2021): clamp the output of every decoder layer
and vision block to the [min, max] observed on fault-free calibration images.

Outcome per probe, compared with the fault-free (golden) run of the same image in this process:
  benign, sdc_miss (yes -> no on a true object), sdc_false (no -> yes on an absent object),
  sdc_other (flip on a probe of the other kind), due (NaN/inf in the output).

Usage: python fi.py --model Qwen/Qwen3-VL-4B-Instruct --gpus 0 --probes probes.parquet \
         --image_ids baseline_test_ids.txt --configs vision:act:none,llm:act:none --n 1000 --out OUT
"""

import argparse
import glob
import json
import os
import random
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--gpus", default="0")
ap.add_argument("--probes", required=True)
ap.add_argument("--image_ids", required=True)
ap.add_argument("--configs", required=True, help="site:fault:protect,...")
ap.add_argument("--n", type=int, default=1000, help="injections per config")
ap.add_argument("--n_profile", type=int, default=100, help="images for Ranger bounds")
ap.add_argument("--out", required=True)
ap.add_argument("--time_limit", type=float, default=10.5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--clamp", action="store_true", help="unused; accepted for the run template")
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
os.environ.setdefault("HF_HOME", "/tmp/hf")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

T0 = time.time()
rng = random.Random(args.seed)


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def find_image_root():
    for pat in ("/kaggle/input/*/samples/CAM_FRONT", "/kaggle/input/*/*/samples/CAM_FRONT",
                "/kaggle/input/*/*/*/samples/CAM_FRONT"):
        h = glob.glob(pat)
        if h:
            return str(Path(h[0]).parent.parent)
    raise FileNotFoundError


if torch.cuda.get_device_capability(0)[0] < 7:
    sys.exit("need T4 (sm75+)")
proc = AutoProcessor.from_pretrained(args.model)
n_gpu = torch.cuda.device_count()
kw = dict(torch_dtype=torch.float16, attn_implementation="sdpa")
kw.update(dict(device_map="auto", max_memory={i: "14GiB" for i in range(n_gpu)})
          if n_gpu > 1 else dict(device_map={"": 0}))
model = AutoModelForImageTextToText.from_pretrained(args.model, **kw).eval()
DEV = model.get_input_embeddings().weight.device
TOK = proc.tokenizer


def ids_of(ws):
    s = set()
    for w in ws:
        t = TOK.encode(w, add_special_tokens=False)
        if len(t) == 1:
            s.add(t[0])
    return sorted(s)


YID, NID = ids_of(["yes", "Yes", "YES", " yes", " Yes"]), ids_of(["no", "No", "NO", " no", " No"])

# ------------------------------------------------------------------ fault sites
VIS_KEYS = ("visual", "vision_tower", "vision_model")
linear_vis, linear_llm, blocks = [], [], []
for name, mod in model.named_modules():
    if isinstance(mod, torch.nn.Linear):
        if any(k in name for k in VIS_KEYS):
            linear_vis.append((name, mod))
        elif "lm_head" not in name and ("language_model" in name or ".layers." in name):
            linear_llm.append((name, mod))
    if type(mod).__name__.endswith(("DecoderLayer", "VisionBlock", "VisionEncoderLayer")) or \
            (type(mod).__name__.endswith("Layer") and any(k in name for k in VIS_KEYS)
             and name.count(".") <= 4 and hasattr(mod, "mlp")):
        blocks.append((name, mod))
# weight each layer by its number of output elements / weights, so every element is equally likely
SITES = {"vision": linear_vis, "llm": linear_llm}
log(f"linear layers: vision {len(linear_vis)}, llm {len(linear_llm)}; protected blocks {len(blocks)}")


def flip_bit_fp16(t_flat, idx, bit):
    v = t_flat[idx:idx + 1].view(torch.int16)
    v ^= torch.tensor(np.int16(np.uint16(1 << bit).view(np.int16)), device=v.device)


def pick_layer(site, by="weight"):
    pool = SITES[site]
    sizes = np.array([m.weight.numel() for _, m in pool], dtype=float)
    i = np.searchsorted(np.cumsum(sizes) / sizes.sum(), rng.random())
    return pool[min(i, len(pool) - 1)]


# ------------------------------------------------------------------ Ranger
ranger_bounds = {}
ranger_handles = []
profiling = {"on": False}


def _profile_hook(name):
    def h(_m, _i, out):
        if not profiling["on"]:
            return
        x = out[0] if isinstance(out, tuple) else out
        lo, hi = float(x.min()), float(x.max())
        b = ranger_bounds.get(name)
        ranger_bounds[name] = (min(lo, b[0]), max(hi, b[1])) if b else (lo, hi)
    return h


def _ranger_hook(name):
    def h(_m, _i, out):
        lo, hi = ranger_bounds[name]
        x = out[0] if isinstance(out, tuple) else out
        x = torch.nan_to_num(x, nan=0.0, posinf=hi, neginf=lo).clamp(lo, hi)
        return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x
    return h


def ranger(on):
    for hd in ranger_handles:
        hd.remove()
    ranger_handles.clear()
    if on:
        for name, mod in blocks:
            if name in ranger_bounds:
                ranger_handles.append(mod.register_forward_hook(_ranger_hook(name)))


# ------------------------------------------------------------------ scoring (prefix + batched suffix)
def chat_text(q):
    m = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    return proc.apply_chat_template(m, add_generation_prompt=True, tokenize=False)


@torch.no_grad()
def score(img, questions, fault=None):
    """fault: callable(prefix=True) that arms the fault before the prefix pass and returns a disarm fn."""
    texts = [chat_text(q) for q in questions]
    cut = [t.index(q) for t, q in zip(texts, questions)]
    prefix = texts[0][:cut[0]]
    enc = proc(text=[prefix], images=[img], return_tensors="pt").to(DEV)
    if enc["pixel_values"].dtype == torch.float32:
        enc["pixel_values"] = enc["pixel_values"].half()
    disarm = fault() if fault else None
    try:
        out = model(**enc, use_cache=True)
    finally:
        if disarm:
            disarm()
    past, P = out.past_key_values, enc["input_ids"].shape[1]
    suf = [TOK(t[c:], add_special_tokens=False)["input_ids"] for t, c in zip(texts, cut)]
    B, L = len(suf), max(map(len, suf))
    ids = torch.full((B, L), TOK.pad_token_id or 0, dtype=torch.long)
    mk = torch.zeros((B, L), dtype=torch.long)
    for i, s in enumerate(suf):
        ids[i, :len(s)], mk[i, :len(s)] = torch.tensor(s), 1
    ids, mk = ids.to(DEV), mk.to(DEV)
    attn = torch.cat([torch.ones((B, P), dtype=torch.long, device=DEV), mk], 1)
    past.batch_repeat_interleave(B)
    extra = {}
    if getattr(model.config, "model_type", "") == "qwen2_5_vl":
        rd = model.model.rope_deltas.to(DEV).view(-1)[0]
        extra["position_ids"] = (torch.arange(P, P + L, device=DEV) + rd).view(1, 1, L).expand(3, B, L)
    o = model(input_ids=ids, attention_mask=attn, past_key_values=past,
              cache_position=torch.arange(P, P + L, device=DEV), use_cache=True, **extra)
    res = []
    for i, s in enumerate(suf):
        lg = o.logits[i, len(s) - 1].float()
        if not torch.isfinite(lg).all():
            res.append(float("nan"))
            continue
        lp = torch.log_softmax(lg, -1)
        res.append(float(torch.sigmoid(torch.logsumexp(lp[YID], 0) - torch.logsumexp(lp[NID], 0))))
    return res


def make_fault(site, kind, info, bits="all"):
    name, mod = pick_layer(site)
    # 'all': uniform over the 16 fp16 bits; 'hi': exponent and sign bits only (10..15), a
    # conditional campaign that over-samples the high-impact faults (reported separately)
    bit = rng.randrange(16) if bits == "all" else rng.randrange(10, 16)
    info.update(layer=name, bit=bit)
    if kind == "weight":
        idx = rng.randrange(mod.weight.numel())
        info["index"] = idx

        def arm():
            flat = mod.weight.data.view(-1)
            orig = flat[idx].clone()
            flip_bit_fp16(flat, idx, bit)
            info["value_before"], info["value_after"] = float(orig), float(flat[idx])

            def disarm():
                mod.weight.data.view(-1)[idx] = orig
            return disarm
        return arm

    def arm():  # transient activation fault, fires once on the next call of the layer
        state = {"done": False}

        def hook(_m, _i, out):
            if state["done"]:
                return
            state["done"] = True
            flat = out.view(-1)
            idx = rng.randrange(flat.numel())
            before = float(flat[idx])
            flip_bit_fp16(flat, idx, bit)
            info.update(index=idx, value_before=before, value_after=float(flat[idx]))
            return out
        hd = mod.register_forward_hook(hook)
        return hd.remove
    return arm


def main():
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.model.split("/")[-1]
    root = find_image_root()
    probes = pd.read_parquet(args.probes)
    keep = {int(x) for x in open(args.image_ids).read().split()}
    probes = probes[(probes.template == 0) & probes.image_id.isin(keep)]
    by_img = {k: g for k, g in probes.groupby("image_id")}
    img_ids = sorted(by_img)
    imgs = {}

    def load(i):
        if i not in imgs:
            g = by_img[i]
            imgs[i] = Image.open(os.path.join(root, g.filename.iloc[0])).convert("RGB")
        return imgs[i]

    # 1. golden runs (+ Ranger profiling on the first n_profile images)
    gold_path = out / f"{tag}__fi_golden.parquet"
    for name, mod in blocks:
        mod._fi_prof = mod.register_forward_hook(_profile_hook(name))
    gold = {}
    for k, i in enumerate(img_ids):
        profiling["on"] = k < args.n_profile
        g = by_img[i]
        gold[i] = score(load(i), list(g.question))
        if len(imgs) > 400:
            imgs.clear()
    profiling["on"] = False
    for name, mod in blocks:
        mod._fi_prof.remove()
    pd.DataFrame([dict(image_id=i, probe_id=int(p), p_yes=v) for i in img_ids
                  for p, v in zip(by_img[i].probe_id, gold[i])]).to_parquet(gold_path, index=False)
    json.dump(ranger_bounds, open(out / f"{tag}__ranger_bounds.json", "w"))
    log(f"golden done for {len(img_ids)} images; Ranger bounds for {len(ranger_bounds)} blocks")

    # 2. injections
    for cfg in args.configs.split(","):
        parts = cfg.split(":")
        site, kind, protect = parts[:3]
        bits = parts[3] if len(parts) > 3 else "all"
        path = out / f"{tag}__fi_{site}_{kind}_{protect}{'' if bits == 'all' else '_' + bits}.parquet"
        rows = pd.read_parquet(path).to_dict("records") if path.exists() else []
        done = len({r["inj"] for r in rows})
        ranger(protect == "ranger")
        # Ranger on the golden run: does clamping alone change any answer?
        if protect == "ranger" and done == 0:
            ch = 0
            for i in img_ids[:100]:
                r = score(load(i), list(by_img[i].question))
                ch += int(sum((a >= 0.5) != (b >= 0.5) for a, b in zip(r, gold[i])))
            log(f"Ranger fault-free decision changes on 100 images: {ch}")
            json.dump({"fault_free_changes_100img": ch},
                      open(out / f"{tag}__ranger_cost.json", "w"))
        for inj in range(done, args.n):
            if (time.time() - T0) / 3600 > args.time_limit:
                log("time limit")
                break
            i = rng.choice(img_ids)
            g = by_img[i]
            info = {}
            try:
                res = score(load(i), list(g.question), fault=make_fault(site, kind, info, bits))
                err = ""
            except Exception as e:  # a crash counts as a detected error
                res, err = [float("nan")] * len(g), repr(e)[:200]
            for (pid, lab, cls, gp), p in zip(zip(g.probe_id, g.label, g.cls, gold[i]), res):
                rows.append(dict(inj=inj, cfg=cfg, bits=bits, image_id=int(i), probe_id=int(pid), label=int(lab),
                                 cls=cls, p_gold=gp, p_fault=p, err=err, **info))
            if (inj + 1) % 100 == 0:
                pd.DataFrame(rows).to_parquet(path, index=False)
                log(f"{cfg}: {inj + 1}/{args.n}")
            if len(imgs) > 400:
                imgs.clear()
        pd.DataFrame(rows).to_parquet(path, index=False)
        ranger(False)
    log("FI DONE")


if __name__ == "__main__":
    main()
