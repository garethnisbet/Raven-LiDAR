"""Structure-from-Motion over the extracted camera frames, via pycolmap.

The Raven frames are a video-like handheld sweep, so sequential matching is
much faster than exhaustive. The camera is modelled as ``OPENCV_FISHEYE`` and
seeded with the intrinsics from ``calibration/calib.json`` (written into
``work/images/frames.json`` by :mod:`raven.extract`); COLMAP refines them.

Output (under ``work/colmap``):
    database.db
    sparse/0/                COLMAP model (cameras/images/points3D)
    undistorted/             PINHOLE images + sparse model for splat training

Usage::

    python -m raven.colmap_pipeline                 # full run
    python -m raven.colmap_pipeline --no-undistort  # stop after sparse
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pycolmap

from .paths import Capture, add_data_arg


def run(
    image_dir: Path,
    out_dir: Path,
    overlap: int = 10,
    loop_detection: bool = False,
    device: str = "cpu",
    undistort: bool = True,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "database.db"
    sparse_dir = out_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)

    dev = {"auto": pycolmap.Device.auto, "cpu": pycolmap.Device.cpu, "cuda": pycolmap.Device.cuda}[device]

    # Seed intrinsics from frames.json if present.
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "OPENCV_FISHEYE"
    frames_json = image_dir / "frames.json"
    if frames_json.exists():
        meta = json.loads(frames_json.read_text())
        reader.camera_params = ",".join(str(p) for p in meta["colmap_params"])
        print(f"seeded OPENCV_FISHEYE params from {frames_json.name}")

    if db_path.exists():
        db_path.unlink()

    print("[1/4] feature extraction")
    pycolmap.extract_features(
        database_path=db_path,
        image_path=image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,  # all frames share one camera
        reader_options=reader,
        device=dev,
    )

    print("[2/4] sequential matching")
    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = overlap
    pairing.quadratic_overlap = True
    pairing.loop_detection = loop_detection  # needs a vocab tree if enabled
    pycolmap.match_sequential(database_path=db_path, pairing_options=pairing, device=dev)

    print("[3/4] incremental mapping")
    maps = pycolmap.incremental_mapping(
        database_path=db_path, image_path=image_dir, output_path=sparse_dir
    )
    if not maps:
        raise RuntimeError("COLMAP failed to reconstruct any model")
    # Pick the reconstruction that registered the most images.
    best_id = max(maps, key=lambda k: maps[k].num_reg_images())
    best = maps[best_id]
    print(
        f"reconstructed {len(maps)} model(s); "
        f"best #{best_id}: {best.num_reg_images()} images, {best.num_points3D()} points"
    )
    model0 = sparse_dir / "0"
    if best_id != 0:
        # Ensure the best model lives at sparse/0 for downstream steps.
        model0.mkdir(exist_ok=True)
        for f in (sparse_dir / str(best_id)).glob("*"):
            shutil.copy2(f, model0 / f.name)

    if undistort:
        print("[4/4] undistorting to PINHOLE")
        undist = out_dir / "undistorted"
        pycolmap.undistort_images(
            output_path=undist, input_path=model0, image_path=image_dir
        )
        print(f"undistorted dataset: {undist}")

    return model0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_data_arg(ap)
    ap.add_argument("--images", default=None, help="frames dir (default <data>/work/images)")
    ap.add_argument("--out", default=None, help="COLMAP workspace (default <data>/work/colmap)")
    ap.add_argument("--overlap", type=int, default=10, help="sequential match window")
    ap.add_argument("--loop-detection", action="store_true", help="enable loop closure (needs vocab tree)")
    ap.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="cpu",
        help="cpu is required unless pycolmap/COLMAP was built with CUDA SIFT",
    )
    ap.add_argument("--no-undistort", action="store_true", help="stop after sparse model")
    args = ap.parse_args()

    cap = Capture.from_args(args)
    run(
        image_dir=Path(args.images) if args.images else cap.p("images"),
        out_dir=Path(args.out) if args.out else cap.p("colmap"),
        overlap=args.overlap,
        loop_detection=args.loop_detection,
        device=args.device,
        undistort=not args.no_undistort,
    )


if __name__ == "__main__":
    main()
