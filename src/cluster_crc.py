"""Scene-level (cluster) conformal risk control, compared with question-level CRC.
Scene-level CRC treats scenes as the exchangeable units: for each calibration scene s with VRU positives,
L_s(t) = share of its VRU positives with score < t; choose the largest t with (sum_s L_s(t) + 1)/(S + 1) <= alpha,
which bounds the expected per-scene miss rate of a new scene by alpha."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
import safeguard as SG

def scene_crc(cal, s, alpha):
    pos = cal.label.values == 1
    scen = cal.scene_name.values[pos]; sc = s[pos]
    df = pd.DataFrame({"scene": scen, "s": sc})
    S = df.scene.nunique()
    best = -np.inf
    for t in np.unique(sc):
        L = df.assign(m=df.s < t).groupby("scene").m.mean().sum()
        if (L + 1) / (S + 1) <= alpha:
            best = max(best, t)
    return best

def main():
    dirs = ["results/runA/out", "results/runB/out", "results/runB2/out", "results/runC/out", "results/runD/out", "results/runF/out", "results/runF2/out"]
    probes = pd.read_parquet("results/probes/probes.parquet")
    rows = []
    for m in ("Qwen3-VL-4B-Instruct", "Qwen2.5-VL-3B-Instruct"):
        full, _ = SG.build(dirs, m, probes)
        cal, test = full[full.split == "calib"].reset_index(drop=True), full[full.split == "test"].reset_index(drop=True)
        w = SG.choose_w(cal)
        sc, _ = SG.scores(cal, w); st, _ = SG.scores(test, w)
        yl = test.label.values
        for meth in ("vanilla", "fused"):
            for a in (0.1, 0.2):
                for kind in ("question", "scene"):
                    t = SG.crc_threshold(sc[meth], cal.label.values, a) if kind == "question" else scene_crc(cal, sc[meth], a)
                    yes = st[meth] >= t
                    pos = yl == 1
                    pooled = 1 - yes[pos].mean()
                    per_scene = pd.DataFrame({"s": test.scene_name.values[pos], "m": ~yes[pos]}).groupby("s").m.mean()
                    def stat(ix):
                        p = yl[ix] == 1
                        return 1 - yes[ix][p].mean()
                    lo, hi = SG.boot(test, lambda ix: np.array([stat(ix)]), 1000)
                    rows.append(dict(model=m, method=meth, alpha=a, calibration=kind, miss_pooled=pooled, miss_lo=lo[0], miss_hi=hi[0],
                                     miss_per_scene=per_scene.mean(), fa=(st[meth][yl == 0] >= t).mean()))
    out = pd.DataFrame(rows); Path("results/analysis/safeguard").mkdir(parents=True, exist_ok=True)
    out.to_csv("results/analysis/safeguard/cluster_crc.csv", index=False)
    print(out.round(3).to_string(index=False))

if __name__ == "__main__":
    main()


def mondrian(dirs=("results/runA/out", "results/runB/out", "results/runB2/out", "results/runC/out", "results/runD/out", "results/runF/out", "results/runF2/out")):
    """Stratum-conditional (Mondrian) CRC: one threshold per stratum, each calibrated by Eq. 14 on that stratum's
    calibration positives; bounds the expected miss rate within every stratum, hence also pooled."""
    probes = pd.read_parquet("results/probes/probes.parquet")
    rows = []
    for m in ("Qwen3-VL-4B-Instruct", "Qwen2.5-VL-3B-Instruct"):
        full, _ = SG.build(list(dirs), m, probes)
        cal, test = full[full.split == "calib"].reset_index(drop=True), full[full.split == "test"].reset_index(drop=True)
        w = SG.choose_w(cal); sc, _ = SG.scores(cal, w); st, _ = SG.scores(test, w)
        for meth in ("vanilla", "fused"):
            for a in (0.1, 0.2):
                yes = np.zeros(len(test), bool); per = {}
                for k in test.stratum.unique():
                    cm, tm = (cal.stratum == k).values, (test.stratum == k).values
                    t = SG.crc_threshold(sc[meth][cm], cal.label.values[cm], a)
                    yes[tm] = st[meth][tm] >= t
                    p = tm & (test.label.values == 1)
                    per[k] = 1 - yes[p].mean() if p.any() else np.nan
                yl = test.label.values
                lo, hi = SG.boot(test, lambda ix: np.array([1 - yes[ix][yl[ix] == 1].mean()]), 1000)
                rows.append(dict(model=m, method=meth, alpha=a, miss=1 - yes[yl == 1].mean(), miss_lo=lo[0], miss_hi=hi[0],
                                 fa=yes[yl == 0].mean(), worst_stratum_miss=np.nanmax(list(per.values())),
                                 night_miss=per.get("singapore_night")))
    out = pd.DataFrame(rows); out.to_csv("results/analysis/safeguard/mondrian_crc.csv", index=False)
    print(out.round(3).to_string(index=False))
