"""Label-free metamorphic testing, validated against the nuScenes oracle.

Metamorphic relations (MRs), each checked per (model, image, class):
  paraphrase : answer(template 0) == answer(template 1)            (full view)
  mirror     : answer(full) == answer(hflip)                        (template 0)
  zoom       : yes on any horizon tile  =>  yes on the full image   (template 0)
               (a tile shows a subset of the scene, so a tile-yes / full-no pair means
               at least one of the two answers is wrong)

For each MR we report
  * violation rate overall, per stratum and per distance bin (pedestrian positives);
  * how well a violation flags an oracle error: P(err | viol), P(err | no viol), lift,
    precision / recall for errors (and, for zoom, for misses);
  * whether the label-free violation rate tracks the oracle error rate across
    operational cells (stratum x distance bin): Spearman correlation.

Usage: .venv/bin/python src/metamorphic.py --scores results/runA/out results/runB/out results/runC/out results/runD/out
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

VIEWS = {"full", "hflip", "tile0", "tile1", "tile2"}
DIST_BINS = [0, 15, 25, 35, 50, 1e9]
DIST_LAB = ["<15", "15-25", "25-35", "35-50", ">50"]


def load(dirs):
    frames = []
    for d in dirs:
        for f in sorted(glob.glob(f"{d}/*.parquet")):
            if "__fi_" in f:
                continue
            x = pd.read_parquet(f)
            if "p_yes" not in x:
                continue
            if "view" not in x:
                x["view"] = "full"
            x = x[x.view.isin(VIEWS)]
            if len(x):
                x = x.assign(base=x.model.str.split("__").str[0])
                frames.append(x[["probe_id", "image_id", "base", "view", "p_yes"]])
    s = pd.concat(frames, ignore_index=True).drop_duplicates(["probe_id", "base", "view"], keep="last")
    return s.dropna(subset=["p_yes"])


def scene_boot_ci(df, fn, n=500, seed=0):
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby("scene_name")]
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        v = fn(pd.concat([groups[i] for i in pick]))
        if np.isfinite(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [np.nan, np.nan]


def detector_stats(df, flag, target):
    f, t = df[flag].astype(bool), df[target].astype(bool)
    pe_v = t[f].mean() if f.any() else np.nan
    pe_n = t[~f].mean() if (~f).any() else np.nan
    return dict(n=int(len(df)), viol_rate=float(f.mean()), err_rate=float(t.mean()),
                p_err_given_viol=float(pe_v), p_err_given_ok=float(pe_n),
                lift=float(pe_v / t.mean()) if t.mean() > 0 else np.nan,
                precision=float(pe_v), recall=float((f & t).sum() / t.sum()) if t.sum() else np.nan)


def build_pairs(s, probes):
    """One row per (model, image, class) with the decisions needed for each MR."""
    p = probes.set_index("probe_id")
    s = s.join(p[["image_id_p", "cls", "template", "label"]] if "image_id_p" in p else
               p[["cls", "template", "label"]], on="probe_id")
    s["yes"] = s.p_yes >= 0.5
    key = ["base", "image_id", "cls"]
    wide = s.pivot_table(index=key, columns=["view", "template"], values="yes", aggfunc="first")
    wide.columns = [f"{v}_t{t}" for v, t in wide.columns]
    wide = wide.reset_index()
    meta = probes[probes.template == 0][["image_id", "cls", "label", "scene_name", "stratum", "split",
                                        "easy_dist", "vru"]].drop_duplicates(["image_id", "cls"])
    return wide.merge(meta, on=["image_id", "cls"], how="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--probes", default="results/probes/probes.parquet")
    ap.add_argument("--out", default="results/analysis/metamorphic")
    ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    probes = pd.read_parquet(a.probes)
    w = build_pairs(load(a.scores), probes)
    w["dist_bin"] = pd.cut(w.easy_dist, DIST_BINS, labels=DIST_LAB, right=False)
    res, cells = {}, []

    for base, g in w.groupby("base"):
        r = {}
        # oracle error of the reference answer (full view, template 0)
        if "full_t0" not in g:
            continue
        g = g[g.full_t0.notna()].copy()
        g["full_yes"] = g.full_t0.astype(bool)
        g["err"] = g.full_yes != g.label.astype(bool)
        g["miss"] = g.label.astype(bool) & ~g.full_yes

        mrs = {}
        if "full_t1" in g:
            m = g.full_t1.notna()
            g.loc[m, "viol_paraphrase"] = g.loc[m, "full_t0"].astype(bool) != g.loc[m, "full_t1"].astype(bool)
            mrs["paraphrase"] = m
        if "hflip_t0" in g:
            m = g.hflip_t0.notna()
            g.loc[m, "viol_mirror"] = g.loc[m, "full_t0"].astype(bool) != g.loc[m, "hflip_t0"].astype(bool)
            mrs["mirror"] = m
        tiles = [c for c in ("tile0_t0", "tile1_t0", "tile2_t0") if c in g]
        if len(tiles) == 3:
            m = g[tiles].notna().all(axis=1)
            any_tile = g.loc[m, tiles].astype(bool).any(axis=1)
            g.loc[m, "viol_zoom"] = any_tile & ~g.loc[m, "full_yes"]
            mrs["zoom"] = m

        for name, m in mrs.items():
            if m.sum() == 0:
                continue  # this model was not run with the transformed view
            d = g[m].copy()
            d["viol"] = d[f"viol_{name}"].astype(bool)
            st = detector_stats(d, "viol", "err")
            st["ci_viol_rate"] = scene_boot_ci(d, lambda x: x.viol.mean(), a.boot)
            st["ci_lift"] = scene_boot_ci(
                d, lambda x: x.err[x.viol].mean() / x.err.mean() if x.viol.any() and x.err.mean() > 0 else np.nan,
                a.boot)
            if name == "zoom":
                st["as_miss_detector"] = detector_stats(d, "viol", "miss")
                vru = d[d.vru.astype(bool)]
                st["as_vru_miss_detector"] = detector_stats(vru, "viol", "miss")
            st["by_stratum"] = {k: dict(viol=float(x.viol.mean()), err=float(x.err.mean()), n=int(len(x)))
                                for k, x in d.groupby("stratum")}
            ped = d[(d.cls == "pedestrian") & (d.label == 1)]
            st["ped_by_dist"] = {str(k): dict(viol=float(x.viol.mean()), miss=float(x.miss.mean()), n=int(len(x)))
                                 for k, x in ped.groupby("dist_bin", observed=True)}
            # operational cells: stratum x distance bin, pedestrian positives (boundary recovery)
            cc = ped.groupby(["stratum", "dist_bin"], observed=True).agg(
                viol=("viol", "mean"), miss=("miss", "mean"), n=("viol", "size")).reset_index()
            cc = cc[cc.n >= 15]
            if len(cc) >= 5:
                rho, pv = spearmanr(cc.viol, cc.miss)
                st["cells_spearman"] = dict(rho=float(rho), p=float(pv), n_cells=int(len(cc)))
            cells.append(cc.assign(base=base, mr=name))
            r[name] = st
        res[base] = r

        print(f"\n== {base}")
        for name, st in r.items():
            print(f"  {name:10s} viol {st['viol_rate']:.3f} {np.round(st['ci_viol_rate'], 3)}  "
                  f"P(err|viol) {st['p_err_given_viol']:.3f}  P(err|ok) {st['p_err_given_ok']:.3f}  "
                  f"lift {st['lift']:.2f}  recall {st['recall']:.3f}"
                  + (f"  cells rho {st['cells_spearman']['rho']:.2f} (p={st['cells_spearman']['p']:.3g})"
                     if "cells_spearman" in st else ""))
            if name == "zoom":
                z = st["as_vru_miss_detector"]
                print(f"  {'':10s} as VRU-miss detector: precision {z['precision']:.3f} recall {z['recall']:.3f}")

    json.dump(res, open(out / "metamorphic.json", "w"), indent=1, default=float)
    if cells:
        pd.concat(cells).to_csv(out / "cells.csv", index=False)
    print(f"\nwrote {out}/metamorphic.json")


if __name__ == "__main__":
    main()
