"""Headless point-cloud operations shared by the GUI editor and the pipeline.

Every function takes and returns an ``open3d.geometry.PointCloud`` (non-mutating)
so they compose and are unit-testable without a display.
"""

from __future__ import annotations

import copy

import numpy as np
import open3d as o3d


def load(path: str) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        raise ValueError(f"empty / unreadable point cloud: {path}")
    return pcd


def save(pcd: o3d.geometry.PointCloud, path: str) -> None:
    o3d.io.write_point_cloud(path, pcd, write_ascii=False, compressed=False)


def statistical_outlier(
    pcd: o3d.geometry.PointCloud, nb_neighbors: int = 20, std_ratio: float = 2.0
) -> o3d.geometry.PointCloud:
    out, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return out


def radius_outlier(
    pcd: o3d.geometry.PointCloud, nb_points: int = 16, radius: float = 0.05
) -> o3d.geometry.PointCloud:
    out, _ = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    return out


def voxel_downsample(pcd: o3d.geometry.PointCloud, voxel: float = 0.01) -> o3d.geometry.PointCloud:
    return pcd.voxel_down_sample(voxel_size=voxel)


def decimate(pcd: o3d.geometry.PointCloud, factor: int = 2) -> o3d.geometry.PointCloud:
    """Keep every Nth point (uniform decimation)."""
    return pcd.uniform_down_sample(max(2, int(factor)))


def crop_aabb(
    pcd: o3d.geometry.PointCloud, min_xyz, max_xyz, invert: bool = False
) -> o3d.geometry.PointCloud:
    """Keep (or with ``invert`` remove) points inside an axis-aligned box."""
    pts = np.asarray(pcd.points)
    lo = np.asarray(min_xyz, float)
    hi = np.asarray(max_xyz, float)
    inside = np.all((pts >= lo) & (pts <= hi), axis=1)
    keep = ~inside if invert else inside
    return select(pcd, np.nonzero(keep)[0])


def z_filter(
    pcd: o3d.geometry.PointCloud, zmin: float | None, zmax: float | None
) -> o3d.geometry.PointCloud:
    z = np.asarray(pcd.points)[:, 2]
    keep = np.ones(len(z), bool)
    if zmin is not None:
        keep &= z >= zmin
    if zmax is not None:
        keep &= z <= zmax
    return select(pcd, np.nonzero(keep)[0])


def percentile_crop(
    pcd: o3d.geometry.PointCloud, low: float = 1.0, high: float = 99.0
) -> o3d.geometry.PointCloud:
    """Crop to the per-axis [low, high] percentile box, trimming sparse tails."""
    pts = np.asarray(pcd.points)
    lo = np.percentile(pts, low, axis=0)
    hi = np.percentile(pts, high, axis=0)
    return crop_aabb(pcd, lo, hi)


def auto_clean(
    pcd: o3d.geometry.PointCloud,
    sor_neighbors: int = 20,
    sor_std: float = 2.0,
    crop_low: float = 1.0,
    crop_high: float = 99.0,
    radius_nb: int = 5,
    radius: float = 0.25,
) -> o3d.geometry.PointCloud:
    """Headless 'remove unnecessary points': SOR + percentile auto-crop + radius.

    Mirrors the manual editor workflow so it is reproducible without the GUI.
    """
    out = statistical_outlier(pcd, sor_neighbors, sor_std)
    out = percentile_crop(out, crop_low, crop_high)
    out = radius_outlier(out, radius_nb, radius)
    return out


# --------------------------------------------------------------------------- #
# Index / keep-mask variants -- so the GUI can show exactly what gets removed
# --------------------------------------------------------------------------- #
def statistical_outlier_keep(pcd, nb_neighbors: int = 20, std_ratio: float = 2.0) -> np.ndarray:
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return np.asarray(ind, dtype=np.int64)


def radius_outlier_keep(pcd, nb_points: int = 16, radius: float = 0.05) -> np.ndarray:
    _, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    return np.asarray(ind, dtype=np.int64)


def aabb_keep(pcd, min_xyz, max_xyz, invert: bool = False) -> np.ndarray:
    pts = np.asarray(pcd.points)
    inside = np.all((pts >= np.asarray(min_xyz, float)) & (pts <= np.asarray(max_xyz, float)), axis=1)
    keep = ~inside if invert else inside
    return np.nonzero(keep)[0]


def auto_clean_mask(
    pcd: o3d.geometry.PointCloud,
    sor_neighbors: int = 20,
    sor_std: float = 2.0,
    crop_low: float = 1.0,
    crop_high: float = 99.0,
    radius_nb: int = 5,
    radius: float = 0.25,
) -> np.ndarray:
    """Boolean keep-mask over the *original* points for :func:`auto_clean`.

    Tracks original indices through SOR -> percentile crop -> radius removal so
    the editor can preview removed points without re-deriving the steps.
    """
    n = len(pcd.points)
    orig = np.arange(n)
    cur = pcd

    ind = statistical_outlier_keep(cur, sor_neighbors, sor_std)
    orig, cur = orig[ind], cur.select_by_index(ind)

    pts = np.asarray(cur.points)
    lo = np.percentile(pts, crop_low, axis=0)
    hi = np.percentile(pts, crop_high, axis=0)
    ind = aabb_keep(cur, lo, hi)
    orig, cur = orig[ind], cur.select_by_index(ind)

    ind = radius_outlier_keep(cur, radius_nb, radius)
    orig = orig[ind]

    mask = np.zeros(n, dtype=bool)
    mask[orig] = True
    return mask


def select(pcd: o3d.geometry.PointCloud, indices) -> o3d.geometry.PointCloud:
    return pcd.select_by_index(list(indices))


def clone(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    return copy.deepcopy(pcd)


def info(pcd: o3d.geometry.PointCloud) -> str:
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return "empty"
    mn = pts.min(0)
    mx = pts.max(0)
    return (
        f"{len(pts):,} pts  "
        f"bbox=[{mn[0]:.2f},{mn[1]:.2f},{mn[2]:.2f}]..[{mx[0]:.2f},{mx[1]:.2f},{mx[2]:.2f}]  "
        f"colors={'yes' if pcd.has_colors() else 'no'}"
    )


def main() -> None:
    """Headless auto-clean: SOR + percentile crop + radius outlier removal.

        python -m raven.cloud_ops work/cloud_raw.ply work/cloud_edited.ply
    """
    import argparse

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--sor-neighbors", type=int, default=20)
    ap.add_argument("--sor-std", type=float, default=2.0)
    ap.add_argument("--crop-low", type=float, default=1.0)
    ap.add_argument("--crop-high", type=float, default=99.0)
    ap.add_argument("--radius-nb", type=int, default=5)
    ap.add_argument("--radius", type=float, default=0.25)
    args = ap.parse_args()

    pcd = load(args.input)
    print("input: ", info(pcd))
    out = auto_clean(pcd, args.sor_neighbors, args.sor_std, args.crop_low,
                     args.crop_high, args.radius_nb, args.radius)
    print("output:", info(out))
    save(out, args.output)
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
