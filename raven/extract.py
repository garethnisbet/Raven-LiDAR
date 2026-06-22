"""Prepare working data: decode camera frames and stage the fused point cloud.

Examples::

    python -m raven.extract --data /path/to/Scan --images --cloud
    python -m raven.extract --data /path/to/Scan --images --stride 3 --long-edge 2000
    python -m raven.extract --data /path/to/Scan --images --rotate   # sensor portrait
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
from tqdm import tqdm

from . import bag_io
from .calib import load_calib
from .paths import Capture, add_data_arg


def extract_images(cap: Capture, stride: int, long_edge: int | None, rotate: bool, quality: int) -> None:
    out_dir = cap.ensure_work() / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    calib = load_calib(cap.calib())

    bag = cap.image_bag()
    total = bag_io.count_images(bag)
    kept = 0
    index: list[dict] = []
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

    for i, (stamp_ns, _fmt, data) in enumerate(
        tqdm(bag_io.iter_images(bag), total=total, desc="images")
    ):
        if i % stride != 0:
            continue
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        if rotate:
            img = calib.rotate_to_sensor(img)
        if long_edge:
            h, w = img.shape[:2]
            scale = long_edge / max(h, w)
            if scale < 1.0:
                img = cv2.resize(
                    img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
                )
        name = f"frame_{i:05d}.jpg"
        cv2.imwrite(str(out_dir / name), img, enc)
        index.append({"name": name, "stamp_ns": stamp_ns, "src_index": i})
        kept += 1

    # Record intrinsics hint at the written resolution for COLMAP.
    if index:
        sample = cv2.imread(str(out_dir / index[0]["name"]))
        h, w = sample.shape[:2]
        meta = {
            "count": kept,
            "size": [w, h],
            "rotated_to_sensor": rotate,
            "colmap_camera_model": "OPENCV_FISHEYE",
            "colmap_params": calib.colmap_params((w, h)),
            "frames": index,
        }
        (out_dir / "frames.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {kept} images to {out_dir}")


def stage_cloud(cap: Capture) -> None:
    src = cap.find_cloud()
    dst = cap.staged_cloud()
    print(f"staged cloud: {src} -> {dst}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_data_arg(ap)
    ap.add_argument("--images", action="store_true", help="decode camera frames")
    ap.add_argument("--cloud", action="store_true", help="stage fused thumbnail PLY")
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth frame")
    ap.add_argument("--long-edge", type=int, default=None, help="downscale long edge to N px")
    ap.add_argument("--rotate", action="store_true", help="rotate frames to sensor portrait")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality")
    args = ap.parse_args()

    cap = Capture.from_args(args)
    if not (args.images or args.cloud):
        ap.error("nothing to do: pass --images and/or --cloud")
    if args.images:
        extract_images(cap, args.stride, args.long_edge, args.rotate, args.quality)
    if args.cloud:
        stage_cloud(cap)


if __name__ == "__main__":
    main()
