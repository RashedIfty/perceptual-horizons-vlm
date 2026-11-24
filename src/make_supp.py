"""Build the LaTeX tables of the Supplementary Information from the analysis outputs.
Writes paper/supp/supp_tables.tex. Placeholders (\\tbd{}) mark results of runs that are still running."""

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("paper/supp")
NICE = {"Qwen3-VL-8B-Instruct": "Qwen3-VL-8B", "Qwen3-VL-4B-Instruct": "Qwen3-VL-4B",
        "InternVL3-8B-hf": "InternVL3-8B", "InternVL3-2B-hf": "InternVL3-2B",
        "Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B", "Qwen2.5-VL-3B-Instruct": "Qwen2.5-VL-3B",
        "SmolVLM2-2.2B-Instruct": "SmolVLM2-2.2B"}
HF = {"Qwen3-VL-8B-Instruct": "Qwen/Qwen3-VL-8B-Instruct", "Qwen3-VL-4B-Instruct": "Qwen/Qwen3-VL-4B-Instruct",
      "InternVL3-8B-hf": "OpenGVLab/InternVL3-8B-hf", "InternVL3-2B-hf": "OpenGVLab/InternVL3-2B-hf",
      "Qwen2.5-VL-7B-Instruct": "Qwen/Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-3B-Instruct": "Qwen/Qwen2.5-VL-3B-Instruct",
      "SmolVLM2-2.2B-Instruct": "HuggingFaceTB/SmolVLM2-2.2B-Instruct"}
SIZE = {"Qwen3-VL-8B-Instruct": "8", "Qwen3-VL-4B-Instruct": "4", "InternVL3-8B-hf": "8", "InternVL3-2B-hf": "2",
        "Qwen2.5-VL-7B-Instruct": "7", "Qwen2.5-VL-3B-Instruct": "3", "SmolVLM2-2.2B-Instruct": "2.2"}
STRATA = [("boston_day", "BOS day"), ("boston_rain", "BOS rain"), ("singapore_day", "SG day"),
          ("singapore_night", "SG night"), ("singapore_night_rain", "SG night+rain")]


def table(caption, label, colspec, header, rows, size="scriptsize", sep="3pt", note=None):
    t = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
         rf"\{size}\setlength{{\tabcolsep}}{{{sep}}}", rf"\begin{{tabular}}{{{colspec}}}\toprule", header + r"\\\midrule"]
    t += [r + r"\\" if not r.startswith(r"\midrule") else r for r in rows]
    t += [r"\bottomrule", r"\end{tabular}"]
    if note:
        t.append(rf"\par\smallskip\parbox{{0.95\linewidth}}{{\scriptsize {note}}}")
    t.append(r"\end{table}")
    return "\n".join(t) + "\n\n"


def num(x, f):
    t = f.format(x)
    return "$-$" + t[1:] if t.startswith("-") else t


def ci(v, lo, hi, f="{:.2f}"):
    sep = " to " if (lo < 0 or hi < 0) else "--"
    return f"{num(v, f)} ({num(lo, f)}{sep}{num(hi, f)})"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tex = ""

    # ---------------- S1: models
    stats = {}
    for f in glob.glob("results/run[ABD]/out/*.stats.json"):
        d = json.load(open(f))
        name = Path(f).name.replace(".stats.json", "")
        if "__" not in name and "method" not in d:
            stats[name] = d
    rows = []
    for m in NICE:
        d = stats[m]
        mem = "/".join(f"{x:.1f}" for x in d["peak_mem_gib"])
        gpus = "2" if len(d["peak_mem_gib"]) > 1 else "1"
        note = "block outputs clamped to fp16 range" if m == "Qwen2.5-VL-3B-Instruct" else (
            "3,349 of 3,359 frames (time limit)" if m == "Qwen2.5-VL-7B-Instruct" else "")
        mark = "$^{a}$" if m == "Qwen2.5-VL-3B-Instruct" else ("$^{b}$" if m == "Qwen2.5-VL-7B-Instruct" else "")
        rows.append(f"{NICE[m]}{mark} & \\texttt{{{HF[m]}}} & {SIZE[m]} & {int(d['n_prefix_tok']):,} & {d['s_per_image']:.2f} & {gpus} & {mem}")
    tex += table(r"\textbf{Models and inference settings:} parameters, image-plus-prompt tokens per frame, time per frame and peak GPU memory (fp16, NVIDIA T4).",
                 "tab:s_models", "llrrrrr",
                 r"Model & Hugging Face identifier & Size (B) & Prefix tokens & s/frame & GPUs & Memory (GiB)",
                 rows, note="Time per frame covers encoding the image and prompt once and answering all questions of the frame (up to 12) from the key-value cache. Transformers 4.57.1; default image resolution of each model. $^{a}$Every block output clamped to the fp16 range to prevent overflow. $^{b}$3,349 of 3,359 frames completed within the time limit.")

    # ---------------- S2: fault-injection configurations and results
    fi = pd.read_csv("results/analysis/fi_D/rates.csv")
    rows = []
    lab = {"vision": "vision encoder", "llm": "language model"}
    for _, r in fi.iterrows():
        site, kind, prot, bits = r.cfg.split(":")
        rows.append(f"Qwen3-VL-4B & {lab[site]} & {'activation' if kind == 'act' else kind} & {prot if prot != 'none' else '--'} & {'0--15' if bits == 'all' else '10--15'} & {int(r.n_inj)} & "
                    f"{100 * r.benign:.1f} & {100 * r.DUE:.1f} & {100 * r['SDC-miss']:.1f} & {100 * r['SDC-FA']:.1f} & {100 * r.SDC:.1f}")
    for site in ("vision", "llm"):
        for prot in ("--", "ranger"):
            rows.append(f"Qwen3-VL-4B & {lab[site]} & weight & {prot} & 0--15 & 1000 & \\tbd{{E}} & \\tbd{{E}} & \\tbd{{E}} & \\tbd{{E}} & \\tbd{{E}}")
    for site in ("vision", "llm"):
        for prot in ("--", "ranger"):
            rows.append(f"Qwen3-VL-4B & {lab[site]} & activation & {prot} & 10--15 & 1000 & \\tbd{{E}} & \\tbd{{E}} & \\tbd{{E}} & \\tbd{{E}} & \\tbd{{E}}")
    rows.append(r"\midrule")
    for kind, bits in (("weight", "0--15"), ("activation", "0--15")):
        for site in ("vision", "llm"):
            for prot in ("--", "ranger"):
                rows.append(f"InternVL3-2B & {lab[site]} & {kind} & {prot} & {bits} & 1000 & \\tbd{{E}} & \\tbd{{E}} & \\tbd{{E}} & \\tbd{{E}} & \\tbd{{E}}")
    for site in ("vision", "llm"):
        rows.append(f"InternVL3-2B & {lab[site]} & activation & -- & 10--15 & 1000 & \\tbd{{E}} & \\tbd{{E}} & \\tbd{{E}} & \\tbd{{E}} & \\tbd{{E}}")
    tex += table(r"\textbf{Fault-injection configurations and outcomes:} share of injections (\%) per outcome class of Eq.~11.",
                 "tab:s_fi", "lllllrrrrrr",
                 r"Model & Site & Fault & Protection & Bits & $n_{\mathrm{inj}}$ & Benign & DUE & SDC-miss & SDC-FA & SDC (any)",
                 rows, note="All configurations use 350 test frames (70 per stratum) and the six classes in the first wording. Protection: range guard of Eq.~12 (Ranger). SDC (any) includes SDC-other. Wilson 95\\% CIs are within $\\pm$3.1 percentage points for every rate (Eq.~10). \\tbd{E}: results of the running campaign.")

    # ---------------- S3: safeguard results
    crc = pd.read_csv("results/analysis/safeguard/crc.csv")
    names = {"vanilla": "Unaided", "vcd": "VCD", "zoom_or": "Zoom-OR", "fused": "FZ", "fused_g": "GFZ",
             "vicrop": "ViCrop", "see": "SEE", "ug": "UG-Search"}
    rows = []
    for setname, setlab in (("benchmark", "all frames"), ("baseline_subset", "700-frame subset")):
        g = crc[(crc.model == "Qwen3-VL-4B-Instruct") & (crc.set == setname)]
        for a in (0.1, 0.2):
            for m in ["vanilla", "vcd", "zoom_or", "fused", "fused_g", "vicrop", "see", "ug"]:
                h = g[(g.method == m) & (np.isclose(g.alpha, a))]
                if h.empty:
                    continue
                r = h.iloc[0]
                d = "--" if m == "vanilla" else ci(100 * r.dfa_vs_vanilla, 100 * r.dfa_lo, 100 * r.dfa_hi, "{:.1f}")
                rows.append(f"{setlab} & {a:.1f} & {names[m]} & {ci(100 * r['miss'], 100 * r.miss_lo, 100 * r.miss_hi, '{:.1f}')} & "
                            f"{ci(100 * r.fa, 100 * r.fa_lo, 100 * r.fa_hi, '{:.1f}')} & {d} & {100 * r.miss_night:.1f} & {100 * r.miss_far:.1f} & {r.passes:.1f}")
            rows.append(r"\midrule")
    rows.append(r"Qwen2.5-VL-3B & 0.1, 0.2 & all methods & \tbd{F} & \tbd{F} & \tbd{F} & \tbd{F} & \tbd{F} & \tbd{F}")
    tex += table(r"\textbf{Safeguards calibrated by conformal risk control:} realised test miss and false-alarm rates (\%) with scene-bootstrap 95\% CIs.",
                 "tab:s_safe", "llllllrrr",
                 r"Frames & $\alpha$ & Method & $\mathrm{MR}_{\mathrm{VRU}}$ & $\mathrm{FAR}_{\varnothing}$ & $\Delta\mathrm{FAR}$ vs unaided & MR night & MR $>$35\,m & Passes",
                 rows, size="tiny", sep="2.5pt",
                 note="Model: Qwen3-VL-4B unless stated. Thresholds $\\hat\\tau_\\alpha$ from Eq.~14 on calibration scenes; rates on test scenes. $\\Delta\\mathrm{FAR}$: paired difference to the unaided model (percentage points). MR night and MR $>$35\\,m: miss rate on night-time and on distant VRU positives. Tile weight $\\beta$ chosen on calibration scenes: 1.0 (all frames), 0.5 (subset). \\tbd{F}: results of the running SEE job for Qwen2.5-VL-3B.")

    # ---------------- S4: GEE odds ratios
    gee = pd.read_csv("results/analysis/all/gee.csv")
    terms = [("singapore_night]", "Night", lambda x: x), ("singapore_night_rain]", "Night + rain", lambda x: x),
             ("boston_rain]", "Rain", lambda x: x), ("log_dist", "Distance $\\times2$", lambda x: x ** np.log(2)),
             ("[T.1]", "Visibility 0--40\\%", lambda x: x), ("[T.2]", "Visibility 40--60\\%", lambda x: x),
             ("[T.3]", "Visibility 60--80\\%", lambda x: x)]
    rows = []
    for m in NICE:
        g = gee[gee.model == m]
        cells = []
        for key, _, tf in terms:
            h = g[g.term.str.contains(key, regex=False)]
            if key == "singapore_night]":
                h = h[~h.term.str.contains("night_rain")]
            r = h.iloc[0]
            cells.append(f"{tf(r.OR):.2f} ({tf(r.lo):.2f}--{tf(r.hi):.2f})")
        rows.append(f"{NICE[m]} & " + " & ".join(cells))
    tex += table(r"\textbf{Odds ratios of pedestrian detection from the GEE model of Eq.~5} with 95\% CIs.",
                 "tab:s_gee", "l" + "c" * len(terms), "Model & " + " & ".join(t[1] for t in terms), rows, size="tiny", sep="2.5pt",
                 note="Night and night with rain relative to Singapore daytime; rain relative to Boston daytime; visibility levels relative to 80--100\\%. Distance $\\times2$: odds ratio per doubling of distance. All models are adjusted for distance, visibility, number of pedestrians and stratum, with scenes as clusters.")

    # ---------------- S5: d50 for all strata
    d50 = pd.read_csv("results/analysis/all/d50.csv")
    rows = []
    for m in NICE:
        cells = []
        for st, _ in STRATA:
            r = d50[(d50.model == m) & (d50.stratum == st)].iloc[0]
            cells.append(f"{r.d50:.0f} ({r.ci_lo:.0f}--{r.ci_hi:.0f})")
        rows.append(f"{NICE[m]} & " + " & ".join(cells))
    tex += table(r"\textbf{Distance of 50\% pedestrian detection $d_{50}$ (m) for all strata} with scene-bootstrap 95\% CIs.",
                 "tab:s_d50", "lccccc", "Model & " + " & ".join(s[1] for s in STRATA), rows,
                 note="Night with rain has 49 frames with pedestrians, so its intervals are wide. BOS, Boston; SG, Singapore.")

    # ---------------- S6: metamorphic relations
    meta = json.load(open("results/analysis/metamorphic_all/metamorphic.json"))
    rows = []
    for m in NICE:
        for r in ("paraphrase", "mirror", "zoom"):
            if r not in meta.get(m, {}):
                continue
            s = meta[m][r]
            k = s.get("cells_spearman", {})
            kap = f"{k['rho']:.2f} ({k['p']:.1g})" if k else "--"
            rows.append(f"{NICE[m]} & {r} & {ci(100 * s['viol_rate'], 100 * s['ci_viol_rate'][0], 100 * s['ci_viol_rate'][1], '{:.1f}')} & "
                        f"{100 * s['p_err_given_viol']:.1f} & {100 * s['p_err_given_ok']:.1f} & {ci(s['lift'], s['ci_lift'][0], s['ci_lift'][1])} & "
                        f"{100 * s['recall']:.1f} & {kap}")
    tex += table(r"\textbf{Metamorphic relations validated against the reference labels:} violation rate, error rates, error lift $\Gamma_r$ and horizon agreement $\kappa_r$.",
                 "tab:s_meta", "llcrrcrc",
                 r"Model & Relation & Violations (\%) & Error\,$|$\,viol. (\%) & Error\,$|$\,no viol. (\%) & $\Gamma_r$ & Recall (\%) & $\kappa_r$ ($p$)",
                 rows, size="tiny", sep="2.5pt",
                 note="Mirror and zoom relations were evaluated for the models with mirrored and tiled views (Qwen3-VL-4B; Qwen2.5-VL-3B). Violation rate and $\\Gamma_r$ with scene-bootstrap 95\\% CIs; $\\kappa_r$ is the Spearman correlation across operational cells with its $p$ value.")
    (OUT / "supp_tables.tex").write_text(tex)
    print("wrote", OUT / "supp_tables.tex")


if __name__ == "__main__":
    main()
