"""Safeguard comparison under a common guarantee on the VRU miss rate.

Methods (all reduced to one score; 'yes' if score >= t):
  vanilla  p_f                                 1 pass
  vcd      sigmoid(2 logit p_f - logit p_noisy)  2 passes
  zoom_or  max(p_f, p_t)                       4 passes (p_t = max over the 3 horizon tiles)
  fused    sigmoid(logit p_f + w logit p_t)    4 passes; w chosen on CALIBRATION scenes only
  fused_g  fused, tiles only when 0.01 <= p_f <= 0.99 (else p_f)   1 + 3*share passes
  vicrop / see / ug (baseline subset only)     cost from the baseline outputs

Outputs per model and probe set (full benchmark / baseline subset):
  curves.csv   test false-alarm rate at fixed test miss rates (threshold set on test: trade-off curve)
  crc.csv      threshold by conformal risk control on calibration scenes at alpha in {0.05,0.1,0.2,0.3};
               test miss, false alarm, passes; scene-bootstrap 95% CIs; paired difference vs vanilla
Usage: .venv/bin/python src/safeguard.py --dirs results/runA/out results/runB/out results/runC/out results/runD/out results/runF/out
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

VRU = ["pedestrian", "motorcycle"]
ALPHAS = [0.05, 0.10, 0.20, 0.30]
MISS_GRID = np.round(np.arange(0.02, 0.41, 0.01), 2)


def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sg(z):
    return 1 / (1 + np.exp(-z))


def read(dirs, tag):
    fs = [f for d in dirs for f in glob.glob(f"{d}/{tag}.parquet")]
    if not fs:
        return None
    d = pd.concat([pd.read_parquet(f) for f in fs]).drop_duplicates("probe_id", keep="last")
    keep = ["probe_id", "p_yes"] + (["n_pass"] if "n_pass" in d else [])
    return d[keep]


def build(dirs, model, probes):
    base = probes[(probes.template == 0) & probes.cls.isin(VRU)][
        ["probe_id", "image_id", "scene_name", "split", "label", "stratum", "easy_dist"]]
    full = read(dirs, model)
    if full is None:
        return None, None
    d = base.merge(full.rename(columns={"p_yes": "full"}), on="probe_id")
    for k in ("tile0", "tile1", "tile2", "vcd"):
        x = read(dirs, f"{model}__{k}")
        if x is not None:
            d = d.merge(x[["probe_id", "p_yes"]].rename(columns={"p_yes": k}), on="probe_id", how="left")
    sub = None
    parts = {}
    for k in ("vicrop", "see"):
        x = read(dirs, f"{model}__{k}")
        if x is not None:
            parts[k] = x
    ug = [read(dirs, f"{model}__ug_{p}") for p in ("p1", "p2")]
    ug = [u for u in ug if u is not None]
    if ug:
        parts["ug"] = pd.concat(ug)
    if parts:
        sub = d.copy()
        for k, x in parts.items():
            x = x.rename(columns={"p_yes": k, "n_pass": f"cost_{k}"}) if "n_pass" in x else x.rename(columns={"p_yes": k})
            sub = sub.merge(x, on="probe_id", how="inner")
    return d, sub


def scores(d, w):
    s = {"vanilla": d.full.values}
    c = {"vanilla": np.ones(len(d))}
    if "vcd" in d and d.vcd.notna().all():
        s["vcd"] = sg(2 * lg(d.full.values) - lg(d.vcd.values))
        c["vcd"] = np.full(len(d), 2.0)
    if all(k in d for k in ("tile0", "tile1", "tile2")) and d[["tile0", "tile1", "tile2"]].notna().all().all():
        pt = d[["tile0", "tile1", "tile2"]].max(axis=1).values
        s["zoom_or"] = np.maximum(d.full.values, pt)
        c["zoom_or"] = np.full(len(d), 4.0)
        s["fused"] = sg(lg(d.full.values) + w * lg(pt))
        c["fused"] = np.full(len(d), 4.0)
        g = (d.full.values >= 0.01) & (d.full.values <= 0.99)
        s["fused_g"] = np.where(g, s["fused"], d.full.values)
        c["fused_g"] = 1 + 3 * g
    for k in ("vicrop", "see", "ug"):
        if k in d:
            s[k] = d[k].values
            c[k] = d[f"cost_{k}"].values if f"cost_{k}" in d else np.full(len(d), np.nan)
    return s, c


def fa_at_miss(score, label, miss):
    pos, neg = score[label == 1], score[label == 0]
    t = np.quantile(pos, miss)
    return float((neg >= t).mean())


def choose_w(cal):
    """Tile weight for the fused score, chosen on calibration scenes: mean false-alarm rate over miss 5-30%."""
    best = None
    for w in (0.25, 0.5, 0.75, 1.0):
        s, _ = scores(cal, w)
        if "fused" not in s:
            return 0.5
        v = np.mean([fa_at_miss(s["fused"], cal.label.values, m) for m in (0.05, 0.1, 0.2, 0.3)])
        if best is None or v < best[1]:
            best = (w, v)
    return best[0]


def crc_threshold(score, label, alpha):
    pos = np.sort(score[label == 1])
    n = len(pos)
    # largest t with (k(t)+1)/(n+1) <= alpha, k(t) = #pos with score < t
    kmax = int(np.floor(alpha * (n + 1) - 1))
    if kmax < 0:
        return -np.inf
    return pos[kmax] if kmax < n else np.inf  # scores < pos[kmax] are exactly kmax misses


def boot(test, fn, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    scen = test.scene_name.values
    u, inv = np.unique(scen, return_inverse=True)
    idx = [np.where(inv == i)[0] for i in range(len(u))]
    out = []
    for _ in range(n):
        pick = np.concatenate([idx[i] for i in rng.integers(0, len(u), len(u))])
        out.append(fn(pick))
    out = np.array(out)
    return np.nanpercentile(out, 2.5, axis=0), np.nanpercentile(out, 97.5, axis=0)


def evaluate(d, model, setname, boot_n):
    cal, test = d[d.split == "calib"].reset_index(drop=True), d[d.split == "test"].reset_index(drop=True)
    w = choose_w(cal)
    sc, cc = scores(cal, w)
    st, ct = scores(test, w)
    yl = test.label.values
    curves, crc = [], []
    for m in st:
        for miss in MISS_GRID:
            curves.append(dict(model=model, set=setname, method=m, miss=miss, fa=fa_at_miss(st[m], yl, miss)))
        for a in ALPHAS:
            t = crc_threshold(sc[m], cal.label.values, a)
            yes = st[m] >= t
            yes_v = st["vanilla"] >= crc_threshold(sc["vanilla"], cal.label.values, a)

            def stat(ix, yes=yes, yes_v=yes_v):
                p, n_ = yl[ix] == 1, yl[ix] == 0
                return np.array([1 - yes[ix][p].mean(), yes[ix][n_].mean(),
                                 yes[ix][n_].mean() - yes_v[ix][n_].mean()])
            lo, hi = boot(test, stat, boot_n)
            v = stat(np.arange(len(test)))
            night = test.stratum.str.contains("night").values & (yl == 1)
            far = (test.easy_dist.values > 35) & (yl == 1)
            crc.append(dict(model=model, set=setname, method=m, alpha=a, w=w, t=t,
                            miss=v[0], miss_lo=lo[0], miss_hi=hi[0], fa=v[1], fa_lo=lo[1], fa_hi=hi[1],
                            dfa_vs_vanilla=v[2], dfa_lo=lo[2], dfa_hi=hi[2],
                            miss_night=1 - yes[night].mean(), miss_far=1 - yes[far].mean(),
                            passes=float(np.nanmean(ct[m])), n_pos=int((yl == 1).sum()), n_neg=int((yl == 0).sum())))
    return pd.DataFrame(curves), pd.DataFrame(crc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--models", default="Qwen3-VL-4B-Instruct,Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--probes", default="results/probes/probes.parquet")
    ap.add_argument("--out", default="results/analysis/safeguard")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    probes = pd.read_parquet(a.probes)
    C, R = [], []
    for m in a.models.split(","):
        full, sub = build(a.dirs, m, probes)
        if full is None:
            print("skip", m, "(no scores)")
            continue
        for name, d in (("benchmark", full), ("baseline_subset", sub)):
            if d is None or not len(d):
                continue
            c, r = evaluate(d, m, name, a.boot)
            C.append(c)
            R.append(r)
            print(f"\n== {m} [{name}] probes={len(d)} images={d.image_id.nunique()} w={r.w.iloc[0]}")
            print(r[r.alpha.isin([0.1, 0.2])][["method", "alpha", "miss", "miss_lo", "miss_hi", "fa", "fa_lo", "fa_hi",
                                               "dfa_vs_vanilla", "dfa_lo", "dfa_hi", "miss_night", "miss_far",
                                               "passes"]].round(3).to_string(index=False))
    pd.concat(C).to_csv(out / "curves.csv", index=False)
    pd.concat(R).to_csv(out / "crc.csv", index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
