# Perceptual horizons and silent failures of vision-language models in autonomous driving

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24507275.svg)](https://doi.org/10.5281/zenodo.24507275)

Code and data for the manuscript *Perceptual horizons and silent failures of vision-language models in autonomous driving* (Rashedul Arefin Ifty and Nasif Shahriar Ifaz; submitted to *Scientific Reports*).

The study tests seven open vision-language models (VLMs) as pedestrian detectors on 3,359 nuScenes front-camera frames:

- **T1 perceptual horizon:** 39,096 yes/no questions scored against reference labels derived from the nuScenes 3D annotations; detection curves, the distance of 50% detection (d50) and GEE odds ratios.
- **T2 label-free metamorphic tests:** paraphrase, mirror and zoom consistency, validated against the reference labels.
- **T3 statistical fault injection:** fp16 bit flips in activations and weights of the vision encoder and the language model, with and without range restriction (Ranger).
- **T4 safeguards:** unaided model, VCD, Zoom-OR, a fused zoom (FZ/GFZ), and re-implementations of ViCrop, UG-Search and SEE, all calibrated by conformal risk control to the same miss rate.

## Repository layout

| Path | Content |
|---|---|
| `src/build_index.py` | projects nuScenes 3D boxes into CAM_FRONT images (per-image and per-object tables) |
| `src/make_probes.py` | samples frames per stratum, splits scenes into calibration/test, builds questions and reference labels |
| `src/analyze.py` | T1: accuracy, miss and false-alarm rates, d50 with scene bootstrap, GEE |
| `src/metamorphic.py` | T2: violation rates, error lift, horizon agreement |
| `src/fi_analyze.py` | T3: outcome rates, bit/component/depth breakdown, decision-margin analysis, Ranger effect |
| `src/safeguard.py` | T4: operating curves, conformal calibration, paired scene-bootstrap differences |
| `src/rc_zoom.py` | earlier safeguard analysis (conformal risk control and Learn-then-Test) |
| `src/cluster_crc.py` | question-level, scene-level and stratum-conditional (Mondrian) conformal calibration |
| `src/figures.py`, `src/fig1_formal.py` | figures of the manuscript |
| `src/make_supp.py` | tables of the Supplementary Information |
| `kaggle/bundle/` | scripts run on GPUs: `vlm_score.py` (scoring with KV-cached prefix), `baselines.py` (ViCrop, UG-Search, SEE), `fi.py` (fault injection), frame-id lists |
| `kaggle/run_template.py`, `src/make_kernel.py`, `jobs/` | job definitions and the runner used on Kaggle (2× NVIDIA T4 per session) |
| `data/questions/` | `images.parquet` (sampled frames, strata, splits) and `probes.parquet` (questions with reference labels, distance and visibility of the most visible instance) |
| `data/model_scores/` | yes-propensity ψ of every model, view and baseline for every question (`*.parquet`) and per-run statistics (`*.stats.json`) |
| `results/` | analysis outputs used for the figures and tables |

Fault-injection outcomes are stored per configuration as `<model>__fi_<site>_<fault>_<protection>[_hi].parquet` (one row per question and injection, with layer, bit and values before and after the flip), together with the fault-free reference runs (`__fi_golden.parquet`) and the profiled Ranger bounds (`runD`: activation faults on Qwen3-VL-4B; `runE`: weight and exponent-bit faults on Qwen3-VL-4B and ten configurations on InternVL3-2B; `runG`: exponent-bit activation faults with Ranger on InternVL3-2B; 24 configurations and 24,000 injections in total). `runF`/`runF2` hold the SEE outputs for Qwen2.5-VL-3B, and `runB2/Qwen2.5-VL-7B-Instruct_merged.parquet` the complete Qwen2.5-VL-7B scores (all 3,359 frames).

## Reproducing

1. Obtain nuScenes v1.0-trainval (metadata and CAM_FRONT keyframes) from <https://www.nuscenes.org> under its terms of use.
2. `python src/build_index.py` and `python src/make_probes.py` to rebuild `data/questions/` (fixed seed).
3. Score the models: upload `kaggle/bundle/` as a Kaggle dataset (set your username in `dataset-metadata.json`), build a kernel with `python src/make_kernel.py --name runA --account <your-kaggle-username> --jobs jobs/runA.json`, and push it with a T4 accelerator. The scripts can also be run directly on any GPU with `python kaggle/bundle/vlm_score.py --model <hf-id> --probes data/questions/probes.parquet --out <dir>`.
4. Run the analyses in `src/` on the score files, for example `python src/analyze.py --scores <dir-with-full-view-scores>`.

Software: Python 3.11+, PyTorch, Hugging Face Transformers 4.57.1 (GPU runs), pandas, numpy, scipy, statsmodels, matplotlib, pyarrow, Pillow (see `requirements.txt`).

## Licences

- **Code:** MIT licence (`LICENSE`).
- **Data in `data/` and `results/`:** derived from the nuScenes annotations and released under CC BY-NC-SA 4.0, consistent with the nuScenes terms of use (`DATA_LICENSE.md`). No nuScenes images or original metadata are redistributed.

## Citation

Please cite the article once published; citation details will be added here. To cite this code and data, use the Zenodo archive: https://doi.org/10.5281/zenodo.24507275 (all versions).
