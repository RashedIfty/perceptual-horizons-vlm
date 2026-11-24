"""Build per-image and per-object tables for nuScenes CAM_FRONT keyframes.

Output (results/index/):
  images.parquet   one row per CAM_FRONT keyframe: scene, location, condition, file name
  objects.parquet  one row per annotated object visible in that image: class, distance,
                   visibility bin, projected 2D box, truncation

Geometry follows the nuScenes devkit conventions (quaternions w, x, y, z; box size w, l, h),
re-implemented here so that the devkit is not needed.

Usage: .venv/bin/python src/build_index.py --meta data/nuscenes_meta/v1.0-trainval_meta/v1.0-trainval
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

IMG_W, IMG_H = 1600, 900
CAMERA = "CAM_FRONT"


def load(meta: Path, name: str):
    with open(meta / f"{name}.json") as f:
        return json.load(f)


def quat_to_rot(q):
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def box_corners(center, size, rot):
    """8 corners (3 x 8) of a box in the frame of `center`; size = (w, l, h)."""
    w, l, h = size
    x = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    return rot @ np.vstack([x, y, z]) + np.asarray(center).reshape(3, 1)


def coarse_class(category: str, attributes: list[str]) -> str:
    if category.startswith("human.pedestrian"):
        if category.endswith(("adult", "child", "police_officer", "construction_worker")):
            return "pedestrian"
        return "pedestrian_other"  # wheelchair, stroller, personal mobility
    if category in ("vehicle.bicycle", "vehicle.motorcycle"):
        base = category.split(".")[1]
        if "cycle.with_rider" in attributes:
            return "cyclist" if base == "bicycle" else "motorcyclist"
        return base  # parked bicycle / motorcycle
    if category == "vehicle.car":
        return "car"
    if category.startswith("vehicle.bus"):
        return "bus"
    if category in ("vehicle.truck", "vehicle.trailer"):
        return "truck"
    if category == "vehicle.construction":
        return "construction_vehicle"
    if category.startswith("vehicle.emergency"):
        return "emergency_vehicle"
    if category == "movable_object.trafficcone":
        return "traffic_cone"
    if category == "movable_object.barrier":
        return "barrier"
    if category == "animal":
        return "animal"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/index"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    meta = args.meta

    # Small tables
    scenes = {s["token"]: s for s in load(meta, "scene")}
    logs = {l["token"]: l for l in load(meta, "log")}
    samples = {s["token"]: s for s in load(meta, "sample")}
    cam_sensor = next(s["token"] for s in load(meta, "sensor") if s["channel"] == CAMERA)
    calib = {c["token"]: c for c in load(meta, "calibrated_sensor") if c["sensor_token"] == cam_sensor}
    cats = {c["token"]: c["name"] for c in load(meta, "category")}
    attrs = {a["token"]: a["name"] for a in load(meta, "attribute")}
    inst_cat = {i["token"]: cats[i["category_token"]] for i in load(meta, "instance")}
    print(f"scenes {len(scenes)}, samples {len(samples)}, front-camera calibrations {len(calib)}")

    # sample_data: keep CAM_FRONT keyframes only (large file, freed right after)
    sd = load(meta, "sample_data")
    cam_sd = [d for d in sd if d["is_key_frame"] and d["calibrated_sensor_token"] in calib]
    del sd
    gc.collect()
    print(f"CAM_FRONT keyframes {len(cam_sd)}")
    needed_pose = {d["ego_pose_token"] for d in cam_sd}

    poses = {p["token"]: p for p in load(meta, "ego_pose") if p["token"] in needed_pose}
    gc.collect()

    anns_by_sample: dict[str, list] = {}
    for a in load(meta, "sample_annotation"):
        anns_by_sample.setdefault(a["sample_token"], []).append(a)
    gc.collect()

    img_rows, obj_rows = [], []
    for d in cam_sd:
        smp = samples[d["sample_token"]]
        scene = scenes[smp["scene_token"]]
        log = logs[scene["log_token"]]
        desc = scene["description"].lower()
        night, rain = "night" in desc, "rain" in desc
        condition = ("night_rain" if night and rain else "night" if night
                     else "rain" if rain else "day")
        img_rows.append(dict(
            sample_token=d["sample_token"], sample_data_token=d["token"],
            filename=d["filename"], timestamp=d["timestamp"],
            scene_name=scene["name"], scene_description=scene["description"],
            location=log["location"], date=log["date_captured"],
            night=night, rain=rain, condition=condition,
        ))

        pose, cs = poses[d["ego_pose_token"]], calib[d["calibrated_sensor_token"]]
        R_ego, t_ego = quat_to_rot(pose["rotation"]), np.array(pose["translation"])
        R_cs, t_cs = quat_to_rot(cs["rotation"]), np.array(cs["translation"])
        K = np.array(cs["camera_intrinsic"])

        for a in anns_by_sample.get(d["sample_token"], []):
            # Boxes without any sensor return are kept with has_pts=False: they are not
            # used as positives, but they block a probe from counting as a true negative.
            corners_g = box_corners(a["translation"], a["size"], quat_to_rot(a["rotation"]))
            center_g = np.array(a["translation"])
            # global -> ego -> camera
            to_cam = lambda p: R_cs.T @ (R_ego.T @ (p - t_ego.reshape(3, -1)) - t_cs.reshape(3, -1))
            c_cam = to_cam(corners_g)
            ctr_ego = R_ego.T @ (center_g - t_ego)
            ctr_cam = to_cam(center_g.reshape(3, 1)).ravel()
            if ctr_cam[2] <= 0.5:
                continue  # centre behind or at the camera
            front = c_cam[2] > 0.1
            if front.sum() == 0:
                continue
            pts = K @ c_cam[:, front]
            u, v = pts[0] / pts[2], pts[1] / pts[2]
            x1, y1, x2, y2 = u.min(), v.min(), u.max(), v.max()
            cx1, cy1 = max(x1, 0.0), max(y1, 0.0)
            cx2, cy2 = min(x2, IMG_W), min(y2, IMG_H)
            if cx2 <= cx1 or cy2 <= cy1:
                continue  # outside the image
            full_area = (x2 - x1) * (y2 - y1)
            area = (cx2 - cx1) * (cy2 - cy1)
            a_names = [attrs[t] for t in a["attribute_tokens"]]
            cat = inst_cat[a["instance_token"]]
            obj_rows.append(dict(
                sample_token=d["sample_token"], ann_token=a["token"],
                instance_token=a["instance_token"], category=cat,
                cls=coarse_class(cat, a_names), attributes="|".join(a_names),
                visibility=int(a["visibility_token"]) if a["visibility_token"] else 0,
                dist=float(np.hypot(ctr_ego[0], ctr_ego[1])),
                depth=float(ctr_cam[2]), lateral=float(ctr_cam[0]),
                x1=cx1, y1=cy1, x2=cx2, y2=cy2,
                box_h=cy2 - cy1, box_area=area,
                truncation=float(1 - area / full_area) if full_area > 0 else 0.0,
                center_u=(cx1 + cx2) / 2,
                num_lidar_pts=a["num_lidar_pts"], num_radar_pts=a["num_radar_pts"],
                has_pts=(a["num_lidar_pts"] + a["num_radar_pts"]) > 0,
            ))

    images = pd.DataFrame(img_rows)
    objects = pd.DataFrame(obj_rows)
    images.to_parquet(args.out / "images.parquet", index=False)
    objects.to_parquet(args.out / "objects.parquet", index=False)

    print(f"\nimages {len(images)}, objects {len(objects)}")
    print(images.groupby(["condition", "location"]).size().unstack(fill_value=0))
    vru = objects[objects.has_pts & objects.cls.isin(["pedestrian", "cyclist", "motorcyclist"])]
    vru = vru.merge(images[["sample_token", "condition"]], on="sample_token")
    bins = pd.cut(vru.dist, [0, 10, 20, 30, 40, 50, 75, 200])
    print("\nVRU objects by distance and condition")
    print(vru.groupby([bins, "condition"], observed=False).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
