"""Align the edited lidar cloud into the COLMAP coordinate frame.

3DGS trains in COLMAP's frame; COLMAP's own sparse points are always a valid
initialization. This step is optional and *upgrades* that init: it fits a
similarity transform (scale + rotation + translation) bringing the dense, colored
**lidar** cloud into the COLMAP frame so the splats start from real geometry.

Strategy: coarse global registration (FPFH + scale-aware RANSAC) followed by
ICP refinement. If the resulting fit is poor, we report it and leave the caller
to fall back to COLMAP-sparse init.

Output: ``work/aligned/cloud_in_colmap.ply`` (+ ``transform.json``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import pycolmap

from . import cloud_ops as ops
from .paths import Capture, add_data_arg


def colmap_sparse_to_o3d(model_path: Path) -> o3d.geometry.PointCloud:
    rec = pycolmap.Reconstruction(str(model_path))
    xyz, rgb = [], []
    for _pid, p in rec.points3D.items():
        xyz.append(p.xyz)
        rgb.append(p.color[:3])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz, float))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(rgb, float) / 255.0)
    return pcd


def _prep(pcd: o3d.geometry.PointCloud, voxel: float):
    down = pcd.voxel_down_sample(voxel)
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100)
    )
    return down, fpfh


def align(
    source: o3d.geometry.PointCloud,  # lidar
    target: o3d.geometry.PointCloud,  # colmap sparse
    voxel: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Return (4x4 similarity source->target, fitness, inlier_rmse)."""
    # Choose a voxel size from the target (COLMAP) scale.
    if voxel is None:
        ext = target.get_axis_aligned_bounding_box().get_extent()
        voxel = float(np.linalg.norm(ext)) / 120.0

    src_d, src_f = _prep(source, voxel)
    tgt_d, tgt_f = _prep(target, voxel)

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_d, tgt_d, src_f, tgt_f, True, voxel * 1.5,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(True),  # with scaling
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel * 1.5),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(400000, 0.999),
    )

    # ICP refine (point-to-point with scaling) on the coarse result.
    icp = o3d.pipelines.registration.registration_icp(
        src_d, tgt_d, voxel * 2, result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
    )
    return icp.transformation, icp.fitness, icp.inlier_rmse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_data_arg(ap)
    ap.add_argument("--cloud", default=None,
                    help="edited lidar cloud (default <data>/work/cloud_edited.ply)")
    ap.add_argument("--model", default=None,
                    help="COLMAP model dir (default <data>/work/colmap/undistorted/sparse)")
    ap.add_argument("--out", default=None, help="default <data>/work/aligned")
    ap.add_argument("--min-fitness", type=float, default=0.3,
                    help="below this, alignment is considered unreliable")
    args = ap.parse_args()

    cap = Capture.from_args(args)
    cloud_path = Path(args.cloud) if args.cloud else cap.p("cloud_edited.ply")
    if not cloud_path.exists():
        cloud_path = cap.p("cloud_raw.ply")
    model_path = Path(args.model) if args.model else cap.p("colmap", "undistorted", "sparse")
    if not model_path.exists():
        model_path = cap.p("colmap", "sparse", "0")
    out = Path(args.out) if args.out else cap.p("aligned")

    lidar = ops.load(str(cloud_path))
    sparse = colmap_sparse_to_o3d(model_path)
    print(f"lidar:  {ops.info(lidar)}")
    print(f"colmap: {ops.info(sparse)}")

    T, fitness, rmse = align(lidar, sparse)
    print(f"alignment fitness={fitness:.3f} rmse={rmse:.4f}")
    scale = float(np.cbrt(np.linalg.det(T[:3, :3])))
    print(f"recovered scale (lidar->colmap) ~ {scale:.4f}")

    out.mkdir(parents=True, exist_ok=True)
    aligned = ops.clone(lidar).transform(T)
    ops.save(aligned, str(out / "cloud_in_colmap.ply"))
    (out / "transform.json").write_text(json.dumps({
        "transform_lidar_to_colmap": T.tolist(),
        "fitness": fitness,
        "inlier_rmse": rmse,
        "scale": scale,
        "reliable": fitness >= args.min_fitness,
    }, indent=2))

    if fitness < args.min_fitness:
        print(f"WARNING: fitness < {args.min_fitness}; alignment may be unreliable.")
        print("         Train with --init colmap (default) instead of --init lidar.")
    else:
        print(f"wrote {out/'cloud_in_colmap.ply'} (use --init lidar for training)")


if __name__ == "__main__":
    main()
