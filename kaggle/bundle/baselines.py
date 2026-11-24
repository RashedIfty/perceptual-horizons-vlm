"""Training-free zoom/crop baselines for yes/no presence probes (Qwen2.5-VL / Qwen3-VL).

  vicrop  Zhang et al., ICLR 2025, rel-att (official Qwen2.5-VL path: layer 22 of 36, attention
          pass at max 256 visual tokens, general prompt "Write a general description of the
          image.", adaptive square box 224 px x ratio 1..2, final pass on [original, crop])
  ug      UG-Search, Kim et al. (arXiv 2510.00705; official code defaults): square windows of
          min(W,H)/2 with stride 1/2 (21 crops on 1600x900), each resized to the original size,
          scored on [original, crop, question] at max 768 visual tokens per image; crop with the
          lowest first-token entropy wins; its pass gives the answer (code variant)
  see     SEE, Popa et al. (arXiv 2611.23962): direct answer; if confidence < 0.9, K=3 sampled
          "where to look" boxes (T=0.8, top-p 0.95), each expanded 30 % per side, cropped at full
          resolution, short side upsampled to >= 448 px; confidence-weighted vote. The Read
          (transcription) view is off by default (--see_read to enable; it targets text images).

All methods output P(yes) per probe so that they can be thresholded under the same risk control.
Usage: python baselines.py --method ug --model Qwen/Qwen2.5-VL-3B-Instruct --gpus 0 \
          --probes probes.parquet --image_ids ids.txt --classes pedestrian,motorcycle --out OUT
"""

import argparse
import glob
import json
import math
import os
import re
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--method", required=True, choices=["vicrop", "ug", "see"])
ap.add_argument("--model", required=True)
ap.add_argument("--gpus", default="0")
ap.add_argument("--probes", required=True)
ap.add_argument("--image_ids", default=None)
ap.add_argument("--classes", default="pedestrian,motorcycle")
ap.add_argument("--template", type=int, default=0)
ap.add_argument("--full_scores", default=None, help="vanilla scores parquet (SEE direct answer)")
ap.add_argument("--out", required=True)
ap.add_argument("--time_limit", type=float, default=10.5)
ap.add_argument("--see_read", action="store_true")
ap.add_argument("--max_images", type=int, default=0)
ap.add_argument("--clamp", action="store_true", help="clamp layer outputs to the fp16 range")
ap.add_argument("--part", default="", help="suffix for the output file when a run is split")
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
os.environ.setdefault("HF_HOME", "/tmp/hf")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

T0 = time.time()
W0, H0 = 1600, 900


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def find_image_root():
    for pat in ("/kaggle/input/*/samples/CAM_FRONT", "/kaggle/input/*/*/samples/CAM_FRONT",
                "/kaggle/input/*/*/*/samples/CAM_FRONT"):
        hits = glob.glob(pat)
        if hits:
            return str(Path(hits[0]).parent.parent)
    raise FileNotFoundError("samples/CAM_FRONT")


# ---------------------------------------------------------------- model helpers
cap = torch.cuda.get_device_capability(0)
if cap[0] < 7:
    sys.exit(f"GPU capability {cap} < 7.0; use T4 x2")
n_gpu = torch.cuda.device_count()
proc = AutoProcessor.from_pretrained(args.model)
kw = dict(torch_dtype=torch.float16,
          attn_implementation="eager" if args.method == "vicrop" else "sdpa")
kw.update(dict(device_map="auto", max_memory={i: "14GiB" for i in range(n_gpu)})
          if n_gpu > 1 else dict(device_map={"": 0}))
model = AutoModelForImageTextToText.from_pretrained(args.model, **kw).eval()
if args.clamp:  # same fp16 safeguard as vlm_score.py --clamp
    def _clamp(_m, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = torch.nan_to_num(h, nan=0.0, posinf=6e4, neginf=-6e4).clamp_(-6e4, 6e4)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    for _mod in model.modules():
        if type(_mod).__name__.endswith(("DecoderLayer", "VisionBlock")):
            _mod.register_forward_hook(_clamp)
DEV =model.get_input_embeddings().weight.device
TOK = proc.tokenizer
IP = proc.image_processor
TOKEN_PX = getattr(IP, "patch_size", 14) * getattr(IP, "merge_size", 2)  # 28 (Qwen2.5) / 32 (Qwen3)
IMG_TOKEN_ID = model.config.image_token_id
N_LAYERS = model.config.text_config.num_hidden_layers if hasattr(model.config, "text_config") \
    else model.config.num_hidden_layers


def ids_of(words):
    out = set()
    for w in words:
        t = TOK.encode(w, add_special_tokens=False)
        if len(t) == 1:
            out.add(t[0])
    return sorted(out)


YID, NID = ids_of(["yes", "Yes", "YES", " yes", " Yes"]), ids_of(["no", "No", "NO", " no", " No"])


def resize_to_tokens(img, max_tokens):
    """Resize so that the image gives at most max_tokens visual tokens (multiples of TOKEN_PX)."""
    w, h = img.size
    s = math.sqrt(max_tokens * TOKEN_PX * TOKEN_PX / (w * h))
    if s >= 1:
        s = 1.0
    nw = max(TOKEN_PX, int(w * s // TOKEN_PX) * TOKEN_PX)
    nh = max(TOKEN_PX, int(h * s // TOKEN_PX) * TOKEN_PX)
    return img.resize((nw, nh), Image.BICUBIC)


def chat(n_images, text, system=None):
    content = [{"type": "image"} for _ in range(n_images)] + [{"type": "text", "text": text}]
    msgs = ([{"role": "system", "content": [{"type": "text", "text": system}]}] if system else []) + \
           [{"role": "user", "content": content}]
    return proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def encode(images, text):
    enc = proc(text=[text], images=images, return_tensors="pt").to(DEV)
    if enc["pixel_values"].dtype == torch.float32:
        enc["pixel_values"] = enc["pixel_values"].half()
    return enc


def yes_stats(logits_row):
    lp = torch.log_softmax(logits_row.float(), -1)
    ly, ln = torch.logsumexp(lp[YID], 0), torch.logsumexp(lp[NID], 0)
    p = torch.softmax(logits_row.float(), -1)
    ent = float(-(p * torch.log(p.clamp_min(1e-12))).sum() / math.log(p.numel()))
    return float(torch.sigmoid(ly - ln)), ent


@torch.no_grad()
def p_yes_batch(images, questions):
    """P(yes) and normalised first-token entropy for several questions on the same images."""
    texts = [chat(len(images), q) for q in questions]
    cut = [t.index(q) for t, q in zip(texts, questions)]
    prefix = texts[0][:cut[0]]
    enc = encode(images, prefix)
    out = model(**enc, use_cache=True)
    past, P = out.past_key_values, enc["input_ids"].shape[1]
    suf = [TOK(t[c:], add_special_tokens=False)["input_ids"] for t, c in zip(texts, cut)]
    B, L = len(suf), max(map(len, suf))
    ids = torch.full((B, L), TOK.pad_token_id or 0, dtype=torch.long)
    m = torch.zeros((B, L), dtype=torch.long)
    for i, s in enumerate(suf):
        ids[i, :len(s)], m[i, :len(s)] = torch.tensor(s), 1
    ids, m = ids.to(DEV), m.to(DEV)
    attn = torch.cat([torch.ones((B, P), dtype=torch.long, device=DEV), m], 1)
    past.batch_repeat_interleave(B)
    extra = {}
    if getattr(model.config, "model_type", "") == "qwen2_5_vl":  # see vlm_score.suffix_position_ids
        rd = model.model.rope_deltas.to(DEV).view(-1)[0]
        extra["position_ids"] = (torch.arange(P, P + L, device=DEV) + rd).view(1, 1, L).expand(3, B, L)
    o = model(input_ids=ids, attention_mask=attn, past_key_values=past,
              cache_position=torch.arange(P, P + L, device=DEV), use_cache=True, **extra)
    return [yes_stats(o.logits[i, len(s) - 1]) for i, s in enumerate(suf)]


# ---------------------------------------------------------------- ViCrop
VICROP_LAYER = round(22 / 36 * N_LAYERS)  # official: layer 22 for Qwen2.5-VL-3B (36 layers)
SUFFIX = " Answer the question using a single word or phrase."


@torch.no_grad()
def attention_map(img_small, question):
    enc = encode([img_small], chat(1, question + SUFFIX))
    out = model(**enc, output_attentions=True)
    att = out.attentions[VICROP_LAYER][0, :, -1, :].float().mean(0)
    vis = enc["input_ids"][0] == IMG_TOKEN_ID
    _, gh, gw = enc["image_grid_thw"][0].tolist()
    ms = getattr(IP, "merge_size", 2)
    return att[vis].reshape(gh // ms, gw // ms).cpu().numpy()


def vicrop_box(rel, W, H, bbox_size=224, ratios=(1, 1.2, 1.4, 1.6, 1.8, 2)):
    """Adaptive square box on the relative-attention map (port of bbox_from_att_image_adaptive)."""
    gh, gw = rel.shape
    cs = np.pad(rel.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    best = None
    for r in ratios:
        side = bbox_size * r
        bw, bh = max(1, round(side / W * gw)), max(1, round(side / H * gh))
        bw, bh = min(bw, gw), min(bh, gh)
        sums = cs[bh:, bw:] - cs[:-bh, bw:] - cs[bh:, :-bw] + cs[:-bh, :-bw]
        y, x = np.unravel_index(np.argmax(sums), sums.shape)
        mx = sums[y, x]
        neigh = [sums[min(max(y + dy, 0), sums.shape[0] - 1), min(max(x + dx, 0), sums.shape[1] - 1)]
                 for dy, dx in ((-bh, 0), (bh, 0), (0, -bw), (0, bw))]
        score = (mx - np.mean(neigh)) / (bw * bh)
        if best is None or score > best[0]:
            cx, cy = (x + bw / 2) / gw * W, (y + bh / 2) / gh * H
            best = (score, cx, cy, side)
    _, cx, cy, side = best
    x1 = int(min(max(cx - side / 2, 0), W - side))
    y1 = int(min(max(cy - side / 2, 0), H - side))
    return x1, y1, int(x1 + side), int(y1 + side)


def run_vicrop(img, questions):
    small = resize_to_tokens(img, 256)  # official max_pixels = 256*28*28
    a_gen = attention_map(small, "Write a general description of the image.")
    res = []
    for q in questions:
        rel = attention_map(small, q) / (a_gen + 1e-8)
        box = vicrop_box(rel, *img.size)
        crop = img.crop(box)
        (p, _), = p_yes_batch([img, crop], [q])
        res.append(dict(p_yes=p, n_pass=3, box=json.dumps(box)))
    return res


# ---------------------------------------------------------------- UG-Search
def ug_windows(W, H, ratio=2):
    s = int(min(W, H) / ratio)
    st = s // 2
    xs = list(range(0, W - s + 1, st))
    ys = list(range(0, H - s + 1, st))
    if xs[-1] != W - s:
        xs.append(W - s)
    if ys[-1] != H - s:
        ys.append(H - s)
    return [(x, y, x + s, y + s) for y in ys for x in xs]


def run_ug(img, questions):
    orig = resize_to_tokens(img, 768)  # official max_pixels = 768*28*28
    per_q = [[] for _ in questions]
    wins = ug_windows(*img.size)
    for box in wins:
        crop = resize_to_tokens(img.crop(box).resize(img.size, Image.LANCZOS), 768)
        for i, (p, ent) in enumerate(p_yes_batch([orig, crop], questions)):
            per_q[i].append((ent, p, box))
    res = []
    for lst in per_q:
        ent, p, box = min(lst, key=lambda t: t[0])
        res.append(dict(p_yes=p, n_pass=len(wins), box=json.dumps(box), ug_entropy=ent))
    return res


# ---------------------------------------------------------------- SEE
WHERE = ("To answer the question \"{q}\", which regions of the image should be examined closely? "
         "Output the bounding boxes in JSON format as a list of {{\"bbox_2d\": [x1, y1, x2, y2]}}. "
         "If the whole image is needed, output [].")
READ = "Transcribe all text visible in the image."


def parse_boxes(text, in_w, in_h, W, H):
    nums = re.findall(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
                      r"\s*(-?\d+(?:\.\d+)?)\s*\]", text)
    if not nums:
        return None
    b = np.array([[float(v) for v in n] for n in nums])
    if "Qwen3" in args.model:  # Qwen3-VL: relative 0..1000 grid
        sx, sy = W / 1000, H / 1000
    else:  # Qwen2.5-VL: absolute pixels of the resized model input
        sx, sy = W / in_w, H / in_h
    x1, y1 = b[:, 0].min() * sx, b[:, 1].min() * sy
    x2, y2 = b[:, 2].max() * sx, b[:, 3].max() * sy
    if x2 <= x1 or y2 <= y1:
        return None
    mw, mh = 0.3 * (x2 - x1), 0.3 * (y2 - y1)
    box = (int(max(0, x1 - mw)), int(max(0, y1 - mh)), int(min(W, x2 + mw)), int(min(H, y2 + mh)))
    if box[2] - box[0] < 8 or box[3] - box[1] < 8:  # box (almost) outside the image: treat as no box
        return None
    return box


@torch.no_grad()
def run_see(img, questions, direct):
    res = []
    for q, p0 in zip(questions, direct):
        y0, c0 = (p0 >= 0.5), max(p0, 1 - p0)
        if c0 >= 0.9:
            res.append(dict(p_yes=p0, n_pass=1, see_gated=True))
            continue
        enc = encode([img], chat(1, WHERE.format(q=q)))
        _, gh, gw = enc["image_grid_thw"][0].tolist()
        in_w, in_h = gw * getattr(IP, "patch_size", 14), gh * getattr(IP, "patch_size", 14)
        gen = model.generate(**enc, do_sample=True, temperature=0.8, top_p=0.95, max_new_tokens=256,
                             num_return_sequences=3)
        texts = TOK.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        votes = {True: c0 if y0 else 0.0, False: 0.0 if y0 else c0}
        n_pass = 2
        for t in texts:
            box = parse_boxes(t, in_w, in_h, *img.size)
            if box is None:
                continue
            crop = img.crop(box)
            if min(crop.size) < 448:
                s = 448 / min(crop.size)
                crop = crop.resize((round(crop.size[0] * s), round(crop.size[1] * s)), Image.LANCZOS)
            (p, _), = p_yes_batch([crop], [q])
            votes[p >= 0.5] += max(p, 1 - p)
            n_pass += 1
        if args.see_read:
            enc_r = encode([img], chat(1, READ))
            g = model.generate(**enc_r, do_sample=False, max_new_tokens=512)
            tr = TOK.decode(g[0, enc_r["input_ids"].shape[1]:], skip_special_tokens=True)
            (p, _), = p_yes_batch([img], [f"OCR of the image: {tr}\n{q}"])
            votes[p >= 0.5] += max(p, 1 - p)
            n_pass += 2
        tot = votes[True] + votes[False]
        res.append(dict(p_yes=votes[True] / tot if tot > 0 else p0, n_pass=n_pass, see_gated=False))
    return res


# ---------------------------------------------------------------- main loop
def main():
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model.split('/')[-1]}__{args.method}" + (f"_{args.part}" if args.part else "")
    root = find_image_root()
    probes = pd.read_parquet(args.probes)
    probes = probes[(probes.template == args.template) & probes.cls.isin(args.classes.split(","))]
    img_ids = sorted(probes.image_id.unique())
    if args.image_ids:
        keep = {int(x) for x in open(args.image_ids).read().split()}
        img_ids = [i for i in img_ids if i in keep]
    if args.max_images:
        img_ids = img_ids[:args.max_images]
    direct = None
    if args.method == "see" and args.full_scores and os.path.exists(args.full_scores):
        direct = pd.read_parquet(args.full_scores).set_index("probe_id").p_yes
    path = out / f"{tag}.parquet"
    rows, done = [], set()
    if path.exists():
        prev = pd.read_parquet(path)
        rows, done = prev.to_dict("records"), set(prev.image_id)
    log(f"{tag}: {len(img_ids) - len(done)} images, layer {VICROP_LAYER}/{N_LAYERS}, token {TOKEN_PX}px")
    by_img = {k: g for k, g in probes.groupby("image_id")}
    times = []
    for iid in img_ids:
        if iid in done:
            continue
        if (time.time() - T0) / 3600 > args.time_limit:
            log("time limit")
            break
        g = by_img[iid]
        img = Image.open(os.path.join(root, g.filename.iloc[0])).convert("RGB")
        qs = list(g.question)
        t1 = time.time()
        if args.method == "vicrop":
            res = run_vicrop(img, qs)
        elif args.method == "ug":
            res = run_ug(img, qs)
        else:  # SEE step 1, the direct answer: reuse vanilla scores or compute them here
            if direct is not None and all(p in direct.index for p in g.probe_id):
                d0 = [float(direct[p]) for p in g.probe_id]
            else:
                d0 = [p for p, _ in p_yes_batch([img], qs)]
            res = run_see(img, qs, d0)
        times.append(time.time() - t1)
        for pid, r in zip(g.probe_id, res):
            rows.append(dict(probe_id=int(pid), image_id=int(iid), model=tag, view=args.method, **r))
        if len(times) % 25 == 0:
            pd.DataFrame(rows).to_parquet(path, index=False)
            log(f"{len(times)} images, {np.mean(times[-25:]):.1f} s/image")
    pd.DataFrame(rows).to_parquet(path, index=False)
    stats = dict(model=args.model, method=args.method, images=len(set(r["image_id"] for r in rows)),
                 s_per_image=float(np.mean(times)) if times else None,
                 peak_mem_gib=[round(torch.cuda.max_memory_allocated(i) / 2**30, 1) for i in range(n_gpu)])
    json.dump(stats, open(out / f"{tag}.stats.json", "w"), indent=1)
    log("STATS", json.dumps(stats))


if __name__ == "__main__":
    main()
