"""Fault-injection analysis (statistical FI, Leveugle et al. DATE 2009).

Outcome of one injection on one probe (decision threshold 0.5 on P(yes)):
  DUE        : crash or non-finite output (detectable)
  SDC-miss   : golden says yes on an object that is present, faulty run says no
  SDC-FA     : golden says no on an absent class, faulty run says yes (false alarm)
  SDC-other  : any other decision flip (golden answer was already wrong)
  benign     : same decision
An injection is an SDC if any of its probes is an SDC (image-level outcome, as in DNN FI work).

Reports per config (site x fault x protection x bits): rates with Wilson 95% CIs; SDC by bit
position, component (attention / MLP / other) and depth; the boundary hypothesis (flip
probability vs golden margin |logit p_gold| and vs pedestrian distance); Ranger vs no
protection (Fisher exact); and, for transient activation faults, how many fault-induced VRU
misses RC-Zoom's fault-free second look recovers (vanilla@t vs RC-Zoom@t).

Usage: .venv/bin/python src/fi_analyze.py --fi results/runD/out results/runE/out \
           --tiles results/runA/out --t 0.5
"""

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import fisher_exact
from statsmodels.stats.proportion import proportion_confint


def wilson(k, n):
    lo, hi = proportion_confint(k, n, method="wilson") if n else (np.nan, np.nan)
    return [float(lo), float(hi)]


def component(layer):
    l = layer.lower()
    if re.search(r"(q_proj|k_proj|v_proj|o_proj|qkv|attn|attention)", l):
        return "attention"
    if re.search(r"(mlp|gate_proj|up_proj|down_proj|fc1|fc2|linear_fc)", l):
        return "mlp"
    return "other"


def depth(layer):
    m = re.search(r"\.(?:layers|blocks|layer)\.(\d+)\.", layer)
    return int(m.group(1)) if m else -1


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def classify(df):
    due = df.p_fault.isna() | (df.err.fillna("") != "") | ~np.isfinite(df.p_fault.fillna(np.nan))
    g, f = df.p_gold >= 0.5, df.p_fault >= 0.5
    flip = (g != f) & ~due
    out = np.select([due, flip & (df.label == 1) & g, flip & (df.label == 0) & ~g, flip],
                    ["DUE", "SDC-miss", "SDC-FA", "SDC-other"], "benign")
    return pd.Series(out, index=df.index)


def load(dirs):
    frames = []
    for d in dirs:
        for f in sorted(glob.glob(f"{d}/*__fi_*.parquet")):
            if f.endswith("__fi_golden.parquet"):
                continue
            x = pd.read_parquet(f)
            x["model"] = Path(f).name.split("__fi_")[0]
            if "bits" not in x:
                x["bits"] = "all"
            frames.append(x)
    df = pd.concat(frames, ignore_index=True)
    df["cfg_full"] = df.cfg.str.replace(":hi", "", regex=False) + ":" + df.bits
    df["outcome"] = classify(df)
    df["component"] = df.layer.map(component)
    df["depth"] = df.layer.map(depth)
    return df


def injection_level(df):
    keys = ["model", "cfg_full", "inj"]
    rank = {"benign": 0, "SDC-other": 1, "SDC-FA": 2, "SDC-miss": 3, "DUE": 4}
    d = df.assign(r=df.outcome.map(rank))
    inj = d.groupby(keys).agg(r=("r", "max"), bit=("bit", "first"), layer=("layer", "first"),
                              component=("component", "first"), depth=("depth", "first")).reset_index()
    inv = {v: k for k, v in rank.items()}
    inj["outcome"] = inj.r.map(inv)
    inj["sdc"] = inj.outcome.str.startswith("SDC")
    return inj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fi", nargs="+", required=True, help="dirs with *__fi_*.parquet")
    ap.add_argument("--tiles", nargs="*", default=[], help="dirs with <model>__tile{0,1,2}.parquet")
    ap.add_argument("--probes", default="results/probes/probes.parquet")
    ap.add_argument("--t", type=float, default=0.5, help="deployed decision threshold (calibrated)")
    ap.add_argument("--s", type=float, default=0.0, help="RC-Zoom skip gate")
    ap.add_argument("--out", default="results/analysis/fi")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    probes = pd.read_parquet(a.probes).set_index("probe_id")
    df = load(a.fi)
    df = df.join(probes[["easy_dist", "stratum", "vru", "template"]], on="probe_id")
    inj = injection_level(df)
    res = {}

    # ---- rates per config
    rows = []
    for (m, c), g in inj.groupby(["model", "cfg_full"]):
        n = len(g)
        r = dict(model=m, cfg=c, n_inj=n)
        for o in ["benign", "DUE", "SDC-miss", "SDC-FA", "SDC-other"]:
            k = int((g.outcome == o).sum())
            r[o] = k / n
            r[f"{o}_ci"] = wilson(k, n)
        k = int(g.sdc.sum())
        r["SDC"], r["SDC_ci"] = k / n, wilson(k, n)
        pr = df[(df.model == m) & (df.cfg_full == c)]
        r["probe_flip_rate"] = float(pr.outcome.str.startswith("SDC").mean())
        r["mean_abs_dp"] = float((pr.p_fault - pr.p_gold).abs().mean())
        rows.append(r)
    rates = pd.DataFrame(rows)
    rates.to_csv(out / "rates.csv", index=False)
    print(rates[["model", "cfg", "n_inj", "benign", "DUE", "SDC-miss", "SDC-FA", "SDC"]].to_string(index=False,
          float_format=lambda v: f"{v:.3f}"))

    # ---- by bit, component, depth (unprotected only)
    un = inj[inj.cfg_full.str.contains(":none:")]
    by_bit = un.groupby(["model", "cfg_full", "bit"]).agg(n=("sdc", "size"), sdc=("sdc", "mean"),
                                                         due=("outcome", lambda o: (o == "DUE").mean()))
    by_bit.to_csv(out / "by_bit.csv")
    by_comp = un.groupby(["model", "cfg_full", "component"]).agg(n=("sdc", "size"), sdc=("sdc", "mean"))
    by_comp.to_csv(out / "by_component.csv")
    un = un.assign(rel_depth=un.groupby(["model", "cfg_full"]).depth.transform(
        lambda d: pd.cut(d / max(d.max(), 1), [-.01, .33, .67, 1.01], labels=["early", "mid", "late"])))
    un.groupby(["model", "cfg_full", "rel_depth"], observed=True).agg(
        n=("sdc", "size"), sdc=("sdc", "mean")).to_csv(out / "by_depth.csv")

    # ---- boundary hypothesis: P(flip) vs golden margin and pedestrian distance (probe level)
    pv = df[(df.outcome != "DUE") & df.cfg_full.str.contains(":none:")].copy()
    pv["flip"] = pv.outcome.str.startswith("SDC").astype(int)
    pv["margin"] = np.abs(logit(pv.p_gold.values))
    bnd = {}
    for (m, c), g in pv.groupby(["model", "cfg_full"]):
        if g.flip.sum() < 5:
            continue
        X = sm.add_constant(np.log1p(g.margin.values))
        try:
            fit = sm.Logit(g.flip.values, X).fit(disp=0)
            b = dict(coef_log_margin=float(fit.params[1]), p=float(fit.pvalues[1]))
        except Exception:
            b = {}
        g = g.assign(mbin=pd.cut(g.margin, [0, 1, 2, 4, 8, 1e9], right=False))
        b["flip_by_margin"] = {str(k): [float(x.flip.mean()), int(len(x))] for k, x in g.groupby("mbin", observed=True)}
        ped = g[(g.label == 1) & (g.cls == "pedestrian")]
        ped = ped.assign(dbin=pd.cut(ped.easy_dist, [0, 15, 25, 35, 50, 1e9], right=False,
                                     labels=["<15", "15-25", "25-35", "35-50", ">50"]))
        b["ped_miss_by_dist"] = {str(k): [float((x.outcome == "SDC-miss").mean()), int(len(x))]
                                 for k, x in ped.groupby("dbin", observed=True)}
        bnd[f"{m}|{c}"] = b
    res["boundary"] = bnd

    # ---- Ranger effect
    rg = {}
    for (m, c), g in inj.groupby(["model", "cfg_full"]):
        if ":none:" not in c:
            continue
        c2 = c.replace(":none:", ":ranger:")
        h = inj[(inj.model == m) & (inj.cfg_full == c2)]
        if not len(h):
            continue
        tab = [[int(g.sdc.sum()), int((~g.sdc).sum())], [int(h.sdc.sum()), int((~h.sdc).sum())]]
        due = [float((g.outcome == "DUE").mean()), float((h.outcome == "DUE").mean())]
        odds, p = fisher_exact(tab)
        rg[f"{m}|{c}"] = dict(sdc_none=g.sdc.mean(), sdc_ranger=h.sdc.mean(), due_none=due[0], due_ranger=due[1],
                              reduction=1 - h.sdc.mean() / g.sdc.mean() if g.sdc.mean() > 0 else np.nan,
                              fisher_p=float(p))
        print(f"Ranger {m} {c}: SDC {g.sdc.mean():.3f} -> {h.sdc.mean():.3f}, DUE {due[0]:.3f} -> {due[1]:.3f}, p={p:.3g}")
    res["ranger"] = rg

    # ---- RC-Zoom as a detector of transient (activation) fault-induced VRU misses
    det = {}
    for m in df.model.unique():
        tf = []
        for d in a.tiles:
            for k in range(3):
                f = Path(d) / f"{m}__tile{k}.parquet"
                if f.exists():
                    tf.append(pd.read_parquet(f)[["probe_id", "p_yes"]].rename(columns={"p_yes": f"tile{k}"}))
        if len(tf) < 3:
            continue
        tiles = tf[0].merge(tf[1], on="probe_id").merge(tf[2], on="probe_id").set_index("probe_id")
        pt_all = tiles.max(axis=1)
        for c in df[(df.model == m)].cfg_full.unique():
            if ":act:" not in c:
                continue  # weight faults persist across passes; a second look is not independent
            g = df[(df.model == m) & (df.cfg_full == c) & (df.label == 1) & df.vru.astype(bool)
                   & (df.outcome != "DUE")].copy()
            g = g[g.probe_id.isin(pt_all.index)]
            if not len(g):
                continue
            pt = pt_all.reindex(g.probe_id).values
            t, s = a.t, a.s

            def rc(p):
                return (p >= t) | ((p < t) & (p >= s) & (pt >= t))

            van_g, van_f = g.p_gold.values >= t, g.p_fault.values >= t
            rc_g, rc_f = rc(g.p_gold.values), rc(g.p_fault.values)
            fm_van = int((van_g & ~van_f).sum())   # fault-induced misses, vanilla
            fm_rc = int((rc_g & ~rc_f).sum())      # fault-induced misses that survive RC-Zoom
            n_inj = df[(df.model == m) & (df.cfg_full == c)].inj.nunique()
            det[f"{m}|{c}"] = dict(n_inj=int(n_inj), fault_misses_vanilla=fm_van, fault_misses_rczoom=fm_rc,
                                   recovered=float(1 - fm_rc / fm_van) if fm_van else np.nan)
            print(f"RC-Zoom {m} {c}: fault-induced VRU misses {fm_van} -> {fm_rc}")
    res["rczoom_detection"] = det

    json.dump(res, open(out / "fi.json", "w"), indent=1, default=float)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
