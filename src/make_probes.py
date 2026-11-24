"""Select images and generate yes/no presence probes with ground truth from nuScenes labels.

Strata (condition is confounded with city in nuScenes, so each condition is compared
with daytime in the same city):
  boston_day, boston_rain, singapore_day, singapore_night, singapore_night_rain

Every probe asks whether a class is present. Ground truth:
  positive  >= 1 instance of the class with a sensor return is visible in the image
  negative  no annotation of the class, or of any easily confused class, projects into the image
  dropped   the class (or a confusable class) is present only as boxes without sensor return

Output (results/probes/): images.parquet, probes.parquet, split by scene (calib / test).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# probe class -> (singular phrase, plural phrase, positive classes, blocking classes)
CLASSES = {
    "pedestrian": ("pedestrian", "pedestrians", {"pedestrian"},
                   {"pedestrian", "pedestrian_other", "cyclist", "motorcyclist"}),
    "motorcycle": ("motorcycle", "motorcycles", {"motorcycle", "motorcyclist"},
                   {"motorcycle", "motorcyclist"}),
    "car": ("car", "cars", {"car"}, {"car", "emergency_vehicle"}),
    "truck": ("truck", "trucks", {"truck"}, {"truck", "construction_vehicle", "emergency_vehicle"}),
    "bus": ("bus", "buses", {"bus"}, {"bus"}),
    "traffic_cone": ("traffic cone", "traffic cones", {"traffic_cone"}, {"traffic_cone"}),
}
VRU = {"pedestrian", "motorcycle"}
TEMPLATES = [
    "Is there a {sing} in this image? Answer with yes or no.",
    "Look at this driving scene. Can you see any {plur}? Answer yes or no.",
]


def stratum(row) -> str:
    city = "boston" if row.location == "boston-seaport" else "singapore"
    return f"{city}_{row.condition}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("results/index"))
    ap.add_argument("--out", type=Path, default=Path("results/probes"))
    ap.add_argument("--per_stratum", type=int, default=800)
    ap.add_argument("--stride", type=int, default=4, help="keep every k-th keyframe of a scene")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    images = pd.read_parquet(args.index / "images.parquet")
    objects = pd.read_parquet(args.index / "objects.parquet")
    images["stratum"] = images.apply(stratum, axis=1)

    # 1. Thin each scene to every k-th keyframe (2 s apart) to limit near-duplicates.
    images = images.sort_values(["scene_name", "timestamp"])
    images["kf_idx"] = images.groupby("scene_name").cumcount()
    offset = images.scene_name.map(lambda s: int(rng.integers(args.stride)))
    images = images[(images.kf_idx - offset) % args.stride == 0]

    # 2. Scene-level split: half of the scenes of each stratum for calibration, half for test.
    split = {}
    for st, g in images.groupby("stratum"):
        scenes = np.array(sorted(g.scene_name.unique()), dtype=object)
        rng.shuffle(scenes)
        for i, s in enumerate(scenes):
            split[s] = "calib" if i < len(scenes) / 2 else "test"
    images["split"] = images.scene_name.map(split)

    # 3. Sample up to per_stratum images per stratum, equal share per split.
    sel = []
    for (st, sp), g in images.groupby(["stratum", "split"]):
        n = min(len(g), args.per_stratum // 2)
        sel.append(g.sample(n=n, random_state=args.seed))
    images = pd.concat(sel).reset_index(drop=True)
    images["image_id"] = np.arange(len(images))

    # 4. Probes
    obj = objects[objects.sample_token.isin(images.sample_token)]
    by_sample = {k: g for k, g in obj.groupby("sample_token")}
    rows = []
    for im in images.itertuples():
        o = by_sample.get(im.sample_token, objects.iloc[:0])
        for cname, (sing, plur, pos, block) in CLASSES.items():
            pos_obj = o[o.cls.isin(pos) & o.has_pts]
            blk_obj = o[o.cls.isin(block)]
            if len(pos_obj):
                label = 1
            elif len(blk_obj):
                continue  # only unconfirmed or confusable instances: drop
            else:
                label = 0
            easiest = pos_obj.loc[pos_obj.box_h.idxmax()] if label else None
            base = dict(
                image_id=im.image_id, sample_token=im.sample_token, filename=im.filename,
                scene_name=im.scene_name, stratum=im.stratum, location=im.location,
                condition=im.condition, split=im.split, cls=cname, vru=cname in VRU,
                label=label, n_inst=len(pos_obj),
                min_dist=pos_obj.dist.min() if label else np.nan,
                max_box_h=pos_obj.box_h.max() if label else np.nan,
                max_vis=pos_obj.visibility.max() if label else np.nan,
                easy_dist=easiest.dist if label else np.nan,
                easy_vis=easiest.visibility if label else np.nan,
                easy_trunc=easiest.truncation if label else np.nan,
            )
            for t, tpl in enumerate(TEMPLATES):
                rows.append({**base, "template": t, "question": tpl.format(sing=sing, plur=plur)})

    probes = pd.DataFrame(rows)
    probes.insert(0, "probe_id", np.arange(len(probes)))
    images.to_parquet(args.out / "images.parquet", index=False)
    probes.to_parquet(args.out / "probes.parquet", index=False)
    probes.to_csv(args.out / "probes.csv", index=False)

    p0 = probes[probes.template == 0]
    print(f"images {len(images)}, probes {len(probes)} ({len(p0)} per template)")
    print(images.groupby(["stratum", "split"]).size().unstack())
    print("\npositives / negatives per class and stratum (template 0)")
    print(p0.pivot_table(index="cls", columns="stratum", values="label",
                         aggfunc=lambda s: f"{int(s.sum())}/{int((1 - s).sum())}"))
    vp = p0[(p0.label == 1) & p0.vru]
    print("\nVRU positives by distance of the easiest instance")
    print(vp.groupby([pd.cut(vp.easy_dist, [0, 15, 25, 35, 50, 200]), "stratum"],
                     observed=False).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
