"""Score yes/no presence probes with an open VLM: P(yes) from single-token logits.

The image and the chat prefix are encoded once per image (KV cache); all questions of
that image are then scored as short suffixes in one batch. --check N compares the cached
path with a full forward pass on N probes.

Usage (inside a Kaggle kernel):
  python vlm_score.py --model Qwen/Qwen2.5-VL-3B-Instruct --gpus 0 --probes probes.parquet \
      --split calib --max_images 150 --out /kaggle/working/out --check 24
Zoomed views (for RC-Zoom and baselines) are added later through --view.
"""

import argparse
import copy
import glob
import json
import math
import os
import sys
import time
import zlib
from pathlib import Path


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpus", default="0", help="'0', '1' or '0,1' (split a large model)")
    ap.add_argument("--probes", required=True)
    ap.add_argument("--split", default="all", help="calib, test or all")
    ap.add_argument("--image_ids", default=None, help="optional file with image_id per line")
    ap.add_argument("--max_images", type=int, default=0)
    ap.add_argument("--templates", default="0,1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", type=int, default=0)
    ap.add_argument("--time_limit", type=float, default=10.5, help="hours; stop cleanly before")
    ap.add_argument("--vision_fp32", action="store_true")
    ap.add_argument("--clamp", action="store_true",
                    help="clamp decoder-layer outputs to the fp16 range (T4 has no bf16)")
    ap.add_argument("--view", default="full",
                    help="full | hflip (mirror) | tile0 | tile1 | tile2 (horizon-band zoom) | vcd (noised image)")
    return ap.parse_args()


args = parse()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
os.environ.setdefault("HF_HOME", "/tmp/hf")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def find_image_root() -> str:
    for pat in ("/kaggle/input/*/samples/CAM_FRONT", "/kaggle/input/*/*/samples/CAM_FRONT",
                "/kaggle/input/*/*/*/samples/CAM_FRONT"):
        hits = glob.glob(pat)
        if hits:
            return str(Path(hits[0]).parent.parent)
    raise FileNotFoundError("samples/CAM_FRONT not found under /kaggle/input")


# Horizon band of the 1600x900 CAM_FRONT image: pedestrians beyond 20 m lie at
# y = 417..606 px (5th-95th percentile, nuScenes trainval). Three overlapping tiles,
# each upscaled 2.67x to 1600x800.
TILES = [(0, 360, 600, 660), (500, 360, 1100, 660), (1000, 360, 1600, 660)]
VCD_ALPHA_BAR = None


def make_view(img, view):
    if view == "full":
        return img
    if view == "hflip":  # metamorphic relation: presence is invariant to a mirror image
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    if view.startswith("tile"):
        return img.crop(TILES[int(view[4:])]).resize((1600, 800), Image.BICUBIC)
    if view == "vcd":
        # VCD (Leng et al., CVPR 2024): forward-diffusion noise, step 500 of 1000,
        # linear beta schedule 1e-4..0.02; fixed seed per image for reproducibility.
        global VCD_ALPHA_BAR
        if VCD_ALPHA_BAR is None:
            betas = np.linspace(1e-4, 0.02, 1000)
            VCD_ALPHA_BAR = float(np.prod(1 - betas[:500]))
        x = np.asarray(img, dtype=np.float32) / 127.5 - 1
        rng = np.random.default_rng(zlib.crc32(img.tobytes()[:65536]))
        x = math.sqrt(VCD_ALPHA_BAR) * x + math.sqrt(1 - VCD_ALPHA_BAR) * rng.standard_normal(x.shape)
        return Image.fromarray(((np.clip(x, -1, 1) + 1) * 127.5).astype(np.uint8))
    raise ValueError(view)


def yes_no_ids(tok):
    def ids(words):
        out = set()
        for w in words:
            t = tok.encode(w, add_special_tokens=False)
            if len(t) == 1:
                out.add(t[0])
        return sorted(out)
    y = ids(["yes", "Yes", "YES", " yes", " Yes"])
    n = ids(["no", "No", "NO", " no", " No"])
    assert y and n, "no single-token yes/no"
    return y, n


def load_model():
    cap = torch.cuda.get_device_capability(0)
    if cap[0] < 7:
        sys.exit(f"GPU capability {cap} < 7.0 (P100?). Set the accelerator to T4 x2.")
    n_gpu = torch.cuda.device_count()
    proc = AutoProcessor.from_pretrained(args.model)
    kw = dict(torch_dtype=torch.float16, attn_implementation="sdpa")
    if n_gpu > 1:
        kw["device_map"] = "auto"
        kw["max_memory"] = {i: "14GiB" for i in range(n_gpu)}
    else:
        kw["device_map"] = {"": 0}
    model = AutoModelForImageTextToText.from_pretrained(args.model, **kw).eval()
    if args.vision_fp32:
        for name in ("visual", "vision_tower", "vision_model"):
            sub = getattr(getattr(model, "model", model), name, None)
            if sub is not None:
                sub.float()
    if args.clamp:
        def _clamp(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = torch.nan_to_num(h, nan=0.0, posinf=6e4, neginf=-6e4).clamp_(-6e4, 6e4)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        n_hooked = 0
        for mod in model.modules():
            if type(mod).__name__.endswith(("DecoderLayer", "VisionBlock")):
                mod.register_forward_hook(_clamp)
                n_hooked += 1
        log(f"clamp hooks on {n_hooked} layers")
    log(f"loaded {args.model} on {n_gpu} GPU(s); mem",
        [round(torch.cuda.memory_allocated(i) / 2**30, 1) for i in range(n_gpu)], "GiB")
    return proc, model


def chat_text(proc, q):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    return proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def suffix_position_ids(model, P, L, B, dev):
    """Qwen2.5-VL in transformers 4.57 mis-broadcasts rope deltas for a batched suffix after
    a cached prefix; give the 3-D M-RoPE positions explicitly (text tokens: t = h = w)."""
    if getattr(model.config, "model_type", "") != "qwen2_5_vl":
        return {}
    rd = model.model.rope_deltas.to(dev).view(-1)[0]
    pos = torch.arange(P, P + L, device=dev) + rd
    return {"position_ids": pos.view(1, 1, L).expand(3, B, L)}


def p_yes_from_logits(logit_row, yid, nid):
    lp = torch.log_softmax(logit_row.float(), dim=-1)
    ly, ln = torch.logsumexp(lp[yid], 0), torch.logsumexp(lp[nid], 0)
    return (torch.sigmoid(ly - ln).item(), (ly.exp() + ln.exp()).item(), ly.item(), ln.item())


@torch.no_grad()
def score_cached(proc, model, img, questions, yid, nid):
    texts = [chat_text(proc, q) for q in questions]
    cut = [t.index(q) for t, q in zip(texts, questions)]
    prefix = texts[0][:cut[0]]
    assert all(t[:c] == prefix for t, c in zip(texts, cut)), "chat prefix differs between questions"
    dev = model.get_input_embeddings().weight.device
    enc = proc(text=[prefix], images=[img], return_tensors="pt").to(dev)
    if "pixel_values" in enc and enc["pixel_values"].dtype == torch.float32 and not args.vision_fp32:
        enc["pixel_values"] = enc["pixel_values"].half()
    n_img_tok = int(enc["input_ids"].shape[1])
    out = model(**enc, use_cache=True)
    past = out.past_key_values
    P = enc["input_ids"].shape[1]

    tok = proc.tokenizer
    suf = [tok(t[c:], add_special_tokens=False)["input_ids"] for t, c in zip(texts, cut)]
    B, L = len(suf), max(len(s) for s in suf)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.full((B, L), pad, dtype=torch.long)
    smask = torch.zeros((B, L), dtype=torch.long)
    for i, s in enumerate(suf):
        ids[i, :len(s)] = torch.tensor(s)
        smask[i, :len(s)] = 1
    ids, smask = ids.to(dev), smask.to(dev)
    attn = torch.cat([torch.ones((B, P), dtype=torch.long, device=dev), smask], dim=1)
    cache_pos = torch.arange(P, P + L, device=dev)

    res = []
    if hasattr(past, "batch_repeat_interleave"):
        past.batch_repeat_interleave(B)
        o = model(input_ids=ids, attention_mask=attn, past_key_values=past,
                  cache_position=cache_pos, use_cache=True,
                  **suffix_position_ids(model, P, L, B, dev))
        for i, s in enumerate(suf):
            res.append(p_yes_from_logits(o.logits[i, len(s) - 1], yid, nid))
    else:  # fallback: one suffix at a time on a copy of the cache
        for i, s in enumerate(suf):
            pc = copy.deepcopy(past)
            o = model(input_ids=ids[i:i + 1, :len(s)], attention_mask=attn[i:i + 1, :P + len(s)],
                      past_key_values=pc, cache_position=cache_pos[:len(s)], use_cache=True,
                      **suffix_position_ids(model, P, len(s), 1, dev))
            res.append(p_yes_from_logits(o.logits[0, -1], yid, nid))
    return res, n_img_tok


@torch.no_grad()
def score_full(proc, model, img, q, yid, nid):
    dev = model.get_input_embeddings().weight.device
    enc = proc(text=[chat_text(proc, q)], images=[img], return_tensors="pt").to(dev)
    if "pixel_values" in enc and enc["pixel_values"].dtype == torch.float32 and not args.vision_fp32:
        enc["pixel_values"] = enc["pixel_values"].half()
    o = model(**enc)
    return p_yes_from_logits(o.logits[0, -1], yid, nid)


def main():
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.model.split("/")[-1] + ("" if args.view == "full" else f"__{args.view}")
    root = find_image_root()
    probes = pd.read_parquet(args.probes)
    temps = [int(t) for t in args.templates.split(",")]
    probes = probes[probes.template.isin(temps)]
    if args.split != "all":
        probes = probes[probes.split == args.split]
    img_ids = sorted(probes.image_id.unique())
    if args.image_ids:
        keep = {int(x) for x in open(args.image_ids).read().split()}
        img_ids = [i for i in img_ids if i in keep]
    if args.max_images:
        img_ids = img_ids[:args.max_images]

    res_path = out_dir / f"{tag}.parquet"
    done = set()
    rows = []
    if res_path.exists():
        prev = pd.read_parquet(res_path)
        rows = prev.to_dict("records")
        done = set(prev.image_id.unique())
        log(f"resume: {len(done)} images already scored")

    proc, model = load_model()
    yid, nid = yes_no_ids(proc.tokenizer)
    log(f"yes ids {yid}, no ids {nid}; images to score {len(img_ids) - len(done)}; root {root}")

    by_img = {k: g for k, g in probes.groupby("image_id")}
    t_img, n_new = [], 0
    for k, iid in enumerate(img_ids):
        if iid in done:
            continue
        if (time.time() - T0) / 3600 > args.time_limit:
            log("time limit reached, stopping")
            break
        g = by_img[iid]
        img = make_view(Image.open(os.path.join(root, g.filename.iloc[0])).convert("RGB"), args.view)
        t1 = time.time()
        res, n_tok = score_cached(proc, model, img, list(g.question), yid, nid)
        torch.cuda.synchronize()
        t_img.append(time.time() - t1)
        for pid, (p, mass, ly, ln) in zip(g.probe_id, res):
            rows.append(dict(probe_id=int(pid), image_id=int(iid), model=tag, view=args.view, p_yes=p,
                             yn_mass=mass, logp_yes=ly, logp_no=ln, n_prefix_tok=n_tok))
        n_new += 1
        if n_new % 50 == 0:
            pd.DataFrame(rows).to_parquet(res_path, index=False)
            log(f"{n_new} images, {np.mean(t_img[-50:]):.2f} s/image")
    pd.DataFrame(rows).to_parquet(res_path, index=False)

    df = pd.DataFrame(rows)
    stats = dict(model=args.model, images=int(df.image_id.nunique()),
                 s_per_image=float(np.mean(t_img)) if t_img else None,
                 n_prefix_tok=float(df.n_prefix_tok.mean()),
                 nan_frac=float(df.p_yes.isna().mean()), yn_mass_mean=float(df.yn_mass.mean()),
                 yn_mass_p05=float(df.yn_mass.quantile(0.05)))
    lab = probes.set_index("probe_id").label
    d = df.join(lab, on="probe_id")
    stats["acc@0.5"] = float(((d.p_yes > 0.5) == (d.label == 1)).mean())

    if args.check:
        sample = probes[probes.image_id.isin(df.image_id.unique())].sample(
            n=min(args.check, len(df)), random_state=0)
        diffs = []
        for r in sample.itertuples():
            img = make_view(Image.open(os.path.join(root, r.filename)).convert("RGB"), args.view)
            p_full = score_full(proc, model, img, r.question, yid, nid)[0]
            p_cached = df.loc[df.probe_id == r.probe_id, "p_yes"].iloc[0]
            diffs.append(abs(p_full - p_cached))
        stats["check_max_abs_diff"] = float(np.max(diffs))
        stats["check_mean_abs_diff"] = float(np.mean(diffs))
    stats["peak_mem_gib"] = [round(torch.cuda.max_memory_allocated(i) / 2**30, 1)
                             for i in range(torch.cuda.device_count())]
    json.dump(stats, open(out_dir / f"{tag}.stats.json", "w"), indent=1)
    log("STATS", json.dumps(stats))


if __name__ == "__main__":
    main()
