"""Benchmark analysis: accuracy, omission/hallucination, psychometric distance curves, GEE.

Usage: .venv/bin/python src/analyze.py --scores results/pilot/out --probes results/probes/probes.parquet
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

STRATA = ["boston_day", "boston_rain", "singapore_day", "singapore_night", "singapore_night_rain"]


def load_scores(score_dir: str, view: str = "full") -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(f"{score_dir}/*.parquet")):
        d = pd.read_parquet(f)
        if "view" not in d:
            d["view"] = "full"
        frames.append(d[d.view == view])
    return pd.concat(frames, ignore_index=True)


def auroc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def scene_bootstrap(df, stat, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    scenes = df.scene_name.unique()
    groups = {s: g for s, g in df.groupby("scene_name")}
    vals = []
    for _ in range(n):
        pick = rng.choice(scenes, size=len(scenes), replace=True)
        vals.append(stat(pd.concat([groups[s] for s in pick])))
    vals = np.array([v for v in vals if np.isfinite(v)])
    return np.percentile(vals, [2.5, 97.5]) if len(vals) else (np.nan, np.nan)


def d50(df):
    """Distance at which P(yes | pedestrian present) = 0.5, logistic in log distance."""
    if df.label.nunique() == 0 or len(df) < 20 or df.det.nunique() < 2:
        return np.nan
    X = sm.add_constant(np.log(df.easy_dist.values))
    try:
        b = sm.Logit(df.det.values, X).fit(disp=0).params
    except Exception:
        return np.nan
    if b[1] >= 0:
        return np.inf  # detection does not fall with distance
    return float(np.exp(-b[0] / b[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--probes", default="results/probes/probes.parquet")
    ap.add_argument("--out", default=None)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    out = Path(args.out or Path(args.scores).parent / "analysis")
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)

    probes = pd.read_parquet(args.probes)
    sc = load_scores(args.scores).merge(probes, on=["probe_id", "image_id"])
    sc["det"] = (sc.p_yes > 0.5).astype(int)
    summary = {}

    # 1. Overall per model
    rows = []
    for m, g in sc.groupby("model"):
        vru_pos = g[(g.label == 1) & g.vru]
        neg = g[g.label == 0]
        rows.append(dict(model=m, n=len(g), acc=((g.det == g.label).mean()),
                         auroc=auroc(g.label, g.p_yes),
                         vru_omission=1 - vru_pos.det.mean(), halluc=neg.det.mean(),
                         yes_rate=g.det.mean(), yn_mass=g.yn_mass.mean(),
                         tmpl_agree=(g.pivot_table(index=["image_id", "cls"], columns="template",
                                                   values="det").dropna().nunique(axis=1) == 1).mean()))
    overall = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    print("\n== Overall (threshold 0.5)\n", overall.round(3).to_string(index=False))
    overall.to_csv(out / "overall.csv", index=False)

    # 2. Per stratum: pedestrian omission and hallucination
    ped = sc[sc.cls == "pedestrian"]
    tab = ped.groupby(["model", "stratum"]).apply(
        lambda g: pd.Series(dict(omission=1 - g[g.label == 1].det.mean(),
                                 halluc=g[g.label == 0].det.mean(),
                                 n_pos=int((g.label == 1).sum()), n_neg=int((g.label == 0).sum()))),
        include_groups=False)
    print("\n== Pedestrian omission / hallucination by stratum\n", tab.round(3).unstack("stratum").to_string())
    tab.to_csv(out / "pedestrian_by_stratum.csv")

    # 3. Psychometric d50 (pedestrian positives, template 0) with scene bootstrap CI
    pp = ped[(ped.label == 1) & (ped.template == 0)]
    rows = []
    for (m, st), g in pp.groupby(["model", "stratum"]):
        lo, hi = scene_bootstrap(g, d50, n=args.boot) if args.boot else (np.nan, np.nan)
        rows.append(dict(model=m, stratum=st, n=len(g), d50=d50(g), ci_lo=lo, ci_hi=hi))
    psy = pd.DataFrame(rows)
    print("\n== d50 (m) for pedestrians\n", psy.round(1).to_string(index=False))
    psy.to_csv(out / "d50.csv", index=False)

    # Binned detection curves for plotting
    bins = [0, 15, 25, 35, 50, 200]
    pp = pp.assign(dbin=pd.cut(pp.easy_dist, bins))
    curve = pp.groupby(["model", "stratum", "dbin"], observed=False).det.agg(["mean", "size"])
    curve.to_csv(out / "curves.csv")

    # 4. GEE logistic, scene clusters: detection of pedestrians
    rows = []
    for m, g in pp.groupby("model"):
        g = g.assign(log_dist=np.log(g.easy_dist), log_n=np.log(g.n_inst),
                     vis=g.easy_vis.astype(int).astype(str))
        if g.det.nunique() < 2:
            continue
        try:
            fit = smf.gee("det ~ log_dist + C(vis, Treatment('4')) + log_n + "
                          "C(stratum, Treatment('singapore_day'))", groups="scene_name", data=g,
                          family=sm.families.Binomial(),
                          cov_struct=sm.cov_struct.Exchangeable()).fit()
            ci = fit.conf_int()
            for k in fit.params.index:
                rows.append(dict(model=m, term=k, OR=np.exp(fit.params[k]),
                                 lo=np.exp(ci.loc[k, 0]), hi=np.exp(ci.loc[k, 1]), p=fit.pvalues[k]))
        except Exception as e:  # small pilot strata can fail to converge
            print("GEE failed for", m, e)
    gee = pd.DataFrame(rows)
    if len(gee):
        print("\n== GEE odds ratios (pedestrian detection)\n", gee.round(3).to_string(index=False))
        gee.to_csv(out / "gee.csv", index=False)

    summary["models"] = overall.model.tolist()
    json.dump(summary, open(out / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
