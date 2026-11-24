"""RC-Zoom and baselines under the same statistical guarantee on VRU omission.

Every method is a one-parameter family of yes/no rules, monotone in the threshold t
(a larger t never adds a "yes"), so the omission rate is monotone in t.

Guarantees, calibrated on the calibration scenes and evaluated once on the test scenes:
  crc  conformal risk control (Angelopoulos et al., ICLR 2024): the largest t with
       (misses(t) + 1) / (n + 1) <= alpha  ->  E[omission on new data] <= alpha
  ltt  Learn-then-Test (Angelopoulos et al., 2021): exact binomial p-value, Bonferroni over
       the grid  ->  P(omission > alpha) <= delta  (stricter; needs a large calibration set)

Methods (score for P(yes) per probe; cost = forward passes per probe)
  vanilla        p_full                                             cost 1
  vcd            sigmoid(2 z_full - z_noised), Leng et al. CVPR 2024  cost 2
  zoom_all       max(p_full, p_tile0..2): always zoom                 cost 4
  rc_zoom@s      yes if p_full >= t, or (s <= p_full < t and max p_tile >= t)
                 zoom only when the full view is unsure but not hopeless
                 cost 1 + 3 * P(s <= p_full < t)
  vicrop, ug, see   baseline P(yes); cost from the recorded number of passes

Usage: .venv/bin/python src/rc_zoom.py --scores DIR [DIR ...] --classes pedestrian,motorcycle --common
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

SKIPS = (0.0, 0.001, 0.01, 0.05)
BASELINES = ("vicrop", "ug", "see")


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load(dirs, probes, template=0):
    frames = []
    for d in dirs:
        for f in glob.glob(f"{d}/*.parquet"):
            x = pd.read_parquet(f)
            if "view" not in x:
                x["view"] = "full"
            if "n_pass" not in x:
                x["n_pass"] = np.nan
            frames.append(x[["probe_id", "model", "view", "p_yes", "n_pass"]])
    s = pd.concat(frames, ignore_index=True)
    s["base"] = s.model.str.split("__").str[0]
    s = s.drop_duplicates(["base", "view", "probe_id"], keep="last")
    wide = s.pivot_table(index=["base", "probe_id"], columns="view", values="p_yes").reset_index()
    cost = s[s.view.isin(BASELINES)].pivot_table(index=["base", "probe_id"], columns="view",
                                                 values="n_pass")
    cost.columns = [f"cost_{c}" for c in cost.columns]
    wide = wide.merge(cost.reset_index(), on=["base", "probe_id"], how="left")
    p = probes[probes.template == template]
    return wide.merge(p, on="probe_id")


def decide(df, method, t):
    """Return (yes, passes per probe) for threshold t."""
    pf = df["full"].values
    if method == "vanilla":
        return pf >= t, np.ones(len(df))
    if method == "vcd":
        s = 1 / (1 + np.exp(-(2 * logit(pf) - logit(df["vcd"].values))))
        return s >= t, np.full(len(df), 2.0)
    pt = np.nanmax(df[["tile0", "tile1", "tile2"]].values, axis=1) if "tile0" in df else None
    if method == "zoom_all":
        return np.maximum(pf, pt) >= t, np.full(len(df), 4.0)
    if method.startswith("rc_zoom@"):
        s = float(method.split("@")[1])
        zoom = (pf < t) & (pf >= s)
        return (pf >= t) | (zoom & (pt >= t)), 1 + 3 * zoom
    if method in BASELINES:
        c = df.get(f"cost_{method}")
        return df[method].values >= t, (c.values if c is not None else np.full(len(df), np.nan))
    raise ValueError(method)


def calibrate(cal, method, alpha, delta, grid, guarantee):
    pos = ((cal.label == 1) & cal.vru).values
    n = int(pos.sum())
    best = None
    for t in grid:  # increasing t -> omission non-decreasing
        yes, _ = decide(cal, method, t)
        k = int((~yes[pos]).sum())
        ok = ((k + 1) / (n + 1) <= alpha) if guarantee == "crc" \
            else (binom.cdf(k, n, alpha) <= delta / len(grid))
        if ok:
            best = t
    return best, n


def metrics(df, yes, passes):
    pos = ((df.label == 1) & df.vru).values
    neg = (df.label == 0).values
    return dict(omission=1 - yes[pos].mean(), halluc=yes[neg].mean(),
                passes=float(np.nanmean(passes)), n_pos=int(pos.sum()), n_neg=int(neg.sum()))


def boot_diff(test, ya, yb, n=2000, seed=0):
    """Paired scene bootstrap of the hallucination difference a - b (95 % CI)."""
    rng = np.random.default_rng(seed)
    neg = (test.label == 0).values
    d = pd.DataFrame(dict(scene=test.scene_name.values[neg], a=ya[neg], b=yb[neg]))
    agg = d.groupby("scene").agg(a=("a", "sum"), b=("b", "sum"), n=("a", "size")).values
    idx = rng.integers(0, len(agg), size=(n, len(agg)))
    s = agg[idx].sum(1)
    diff = (s[:, 0] - s[:, 1]) / s[:, 2]
    return np.percentile(diff, [2.5, 97.5]), float(np.mean(diff > 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--probes", default="results/probes/probes.parquet")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--guarantee", choices=["crc", "ltt"], default="crc")
    ap.add_argument("--classes", default="pedestrian,motorcycle")
    ap.add_argument("--common", action="store_true", help="only probes scored by every method")
    ap.add_argument("--out", default="results/analysis")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)

    grid = np.unique(np.concatenate([np.geomspace(1e-4, 0.05, 40), np.linspace(0.05, 0.95, 46)]))
    data = load(a.scores, pd.read_parquet(a.probes))
    data = data[data.cls.isin(a.classes.split(","))]
    rows, diffs = [], []
    for base, df in data.groupby("base"):
        cols = [c for c in ("full", "vcd", "tile0", "tile1", "tile2") + BASELINES if c in df]
        if a.common:
            df = df.dropna(subset=cols)
        if "full" not in df or len(df) == 0:
            continue
        methods = ["vanilla"] + (["vcd"] if "vcd" in df else [])
        if {"tile0", "tile1", "tile2"} <= set(df):
            methods += ["zoom_all"] + [f"rc_zoom@{s}" for s in SKIPS]
        methods += [m for m in BASELINES if m in df]
        cal, test = df[df.split == "calib"], df[df.split == "test"]
        chosen = {}
        for m in methods:
            t, n_cal = calibrate(cal, m, a.alpha, a.delta, grid, a.guarantee)
            if t is None:
                rows.append(dict(model=base, method=m, t=None, n_cal_pos=n_cal))
                continue
            yes, passes = decide(test, m, t)
            chosen[m] = yes
            e = metrics(test, yes, passes)
            nt = test.condition.str.startswith("night").values
            en = metrics(test[nt], yes[nt], passes[nt])
            far = (test.easy_dist > 35).values
            rows.append(dict(model=base, method=m, t=round(t, 5), n_cal_pos=n_cal,
                             omission=e["omission"], halluc=e["halluc"], passes=e["passes"],
                             night_omission=en["omission"], night_halluc=en["halluc"],
                             far_omission=1 - yes[far & (test.label == 1).values].mean(),
                             n_test_pos=e["n_pos"], n_test_neg=e["n_neg"]))
        ref = [m for m in chosen if m.startswith("rc_zoom@")]
        for r in ref:
            for m in chosen:
                if m != r and not m.startswith("rc_zoom@"):
                    ci, pb = boot_diff(test, chosen[r].astype(float), chosen[m].astype(float))
                    diffs.append(dict(model=base, a=r, b=m, halluc_diff_lo=ci[0], halluc_diff_hi=ci[1],
                                      p_boot_a_worse=pb))
    res, dif = pd.DataFrame(rows), pd.DataFrame(diffs)
    tag = f"{a.guarantee}_a{a.alpha}{'_common' if a.common else ''}"
    print(f"\n== {a.guarantee.upper()} alpha={a.alpha}: test omission (VRU), hallucination, passes/probe\n")
    print(res.round(4).to_string(index=False))
    if len(dif):
        print("\n== RC-Zoom minus method: hallucination difference, paired scene bootstrap 95% CI\n")
        print(dif.round(4).to_string(index=False))
    res.to_csv(out / f"risk_{tag}.csv", index=False)
    dif.to_csv(out / f"risk_diff_{tag}.csv", index=False)


if __name__ == "__main__":
    main()
