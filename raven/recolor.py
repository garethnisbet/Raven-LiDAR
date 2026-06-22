"""Recolour the lidar point cloud with true RGB sampled from the camera photos.

The fused thumbnail cloud carries only a false-colour height ramp (red channel
is constant 0). Real colour lives in the 922 photos. Given COLMAP poses
(undistorted PINHOLE images + sparse model) and the lidar->COLMAP similarity
transform from :mod:`raven.align`, we transform the cloud into the COLMAP frame,
project every point into each photo, and average the sampled colours.

A per-image z-buffer rejects occluded points (so a point behind a wall is not
coloured by a photo that sees the wall). Points never seen keep a neutral grey.

    python -m raven.recolor --cloud work/cloud_edited.ply --out work/cloud_colored.ply
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pycolmap
from tqdm import tqdm

from .paths import Capture, add_data_arg


def load_transform(path: Path) -> np.ndarray:
    if path and path.exists():
        T = np.asarray(json.loads(path.read_text())["transform_lidar_to_colmap"], float)
        return T.reshape(4, 4)
    return np.eye(4)


def recolor(
    cloud_path: Path,
    model_path: Path,
    image_dir: Path,
    transform_path: Path,
    out_path: Path,
    occ_downsample: int = 4,
    occ_tol: float = 0.02,
    drop_unseen: bool = False,
    occ_cloud_path: Path | None = None,
) -> None:
    pcd = o3d.io.read_point_cloud(str(cloud_path))
    T = load_transform(transform_path)
    # If a dense reference cloud is given, build occlusion from it so a
    # decimated/trimmed ``cloud_path`` is still coloured correctly.
    occ_points = None
    if occ_cloud_path and Path(occ_cloud_path).exists():
        occ_points = np.asarray(o3d.io.read_point_cloud(str(occ_cloud_path)).points)
    colors, seen, n_imgs = sample_colors(
        np.asarray(pcd.points), T, model_path, image_dir, occ_downsample, occ_tol,
        occ_points=occ_points,
    )
    pcd.colors = o3d.utility.Vector3dVector(colors)
    if drop_unseen and (~seen).any():
        pcd = pcd.select_by_index(np.nonzero(seen)[0])
    o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False)
    print(f"recoloured {int(seen.sum()):,}/{len(seen):,} points from {n_imgs} photos -> {out_path}")
    print(f"  unseen: {int((~seen).sum()):,}  ({'dropped' if drop_unseen else 'kept grey'})")


def _quat_to_R(qw, qx, qy, qz):
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def _load_colmap_text(model_path: Path):
    """Parse a COLMAP cameras.txt + images.txt model (e.g. a JMStudio export)."""
    intr = {}
    for line in (model_path / "cameras.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        t = line.split()
        cid, model, w, h = int(t[0]), t[1], int(t[2]), int(t[3])
        p = list(map(float, t[4:]))
        if model in ("PINHOLE", "OPENCV"):
            fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            fx = fy = p[0]; cx, cy = p[1], p[2]
        else:
            fx = fy = p[0]; cx, cy = w / 2.0, h / 2.0
        intr[cid] = (np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), w, h)
    lines = [l for l in (model_path / "images.txt").read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    cams = []
    for i in range(0, len(lines), 2):           # two lines per image
        t = lines[i].split()
        qw, qx, qy, qz = map(float, t[1:5])
        tx, ty, tz = map(float, t[5:8])
        K, w, h = intr[int(t[8])]
        cams.append({"name": t[9], "K": K, "R": _quat_to_R(qw, qx, qy, qz),
                     "t": np.array([tx, ty, tz]), "w": w, "h": h})
    return cams


def load_cameras(model_path: Path):
    """Posed cameras from a COLMAP model: pycolmap first, text fallback.

    Returns a list of dicts {name, K(3x3), R(3x3 world->cam), t(3), w, h}.
    """
    mp = Path(model_path)
    # Only a full model (with points3D) loads via pycolmap; otherwise (e.g. a
    # JMStudio text export with just cameras/images) parse the text directly.
    if any((mp / f).exists() for f in ("points3D.bin", "points3D.txt")):
        try:
            rec = pycolmap.Reconstruction(str(mp))
            cams = []
            for im in rec.images.values():
                if not im.has_pose:
                    continue
                cam = rec.cameras[im.camera_id]
                m = np.asarray(im.cam_from_world().matrix(), float)
                cams.append({"name": im.name, "K": np.asarray(cam.calibration_matrix(), float),
                             "R": m[:, :3], "t": m[:, 3], "w": cam.width, "h": cam.height})
            if cams:
                return cams
        except Exception:
            pass
    return _load_colmap_text(mp)


def sample_colors(
    pts_lidar: np.ndarray,
    T: np.ndarray,
    model_path: Path,
    image_dir: Path,
    occ_downsample: int = 4,
    occ_tol: float = 0.02,
    progress=None,
    occ_points: np.ndarray | None = None,
):
    """Project photos onto points; return (colors Nx3 [0,1], seen mask, n_images).

    Unseen points get neutral grey. Reusable from the GUI and the CLI.
    ``progress``, if given, is called ``progress(done, total)`` after each image
    so a GUI can drive a progress bar (the CLI relies on tqdm instead).

    The per-image z-buffer that rejects occluded points is built from
    ``occ_points`` (lidar frame, same ``T``) when given, otherwise from the
    points being coloured. Pass the dense, *unedited* cloud here so occlusion
    keeps working after the coloured subset is decimated or trimmed — a sparse
    subset leaves the depth buffer full of holes and lets photos colour points
    they can't actually see.
    """
    n = len(pts_lidar)
    # Bring the (lidar-frame) cloud into the COLMAP frame the poses live in.
    pts = (T @ np.hstack([pts_lidar, np.ones((n, 1))]).T).T[:, :3]
    if occ_points is not None and len(occ_points):
        opts = (T @ np.hstack([occ_points, np.ones((len(occ_points), 1))]).T).T[:, :3]
    else:
        opts = pts

    cams = load_cameras(Path(model_path))
    accum = np.zeros((n, 3), np.float64)   # weighted colour sum
    wsum = np.zeros(n, np.float64)         # weight sum

    for i, cam in enumerate(tqdm(cams, desc="recolor", unit="img")):
        if progress is not None:
            progress(i + 1, len(cams))
        K = cam["K"]
        W, H = cam["w"], cam["h"]
        cam_pts = (cam["R"] @ pts.T + cam["t"][:, None]).T    # Nx3 in camera frame
        z = cam_pts[:, 2]

        infront = z > 1e-6
        uv = (K @ cam_pts.T).T
        u = uv[:, 0] / np.where(infront, z, 1.0)
        v = uv[:, 1] / np.where(infront, z, 1.0)
        inb = infront & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not inb.any():
            continue

        idx = np.nonzero(inb)[0]
        ui, vi, zi = u[idx], v[idx], z[idx]

        # Per-image z-buffer occlusion at reduced resolution. Build the depth
        # map from the dense occlusion cloud (``opts``) so it stays watertight
        # even when the coloured subset has been decimated/trimmed.
        dw, dh = W // occ_downsample + 1, H // occ_downsample + 1
        depth = np.full(dw * dh, np.inf)
        if opts is pts:
            o_idx, oui, ovi, ozi = idx, ui, vi, zi
        else:
            ocam = (cam["R"] @ opts.T + cam["t"][:, None]).T
            oz = ocam[:, 2]
            ofront = oz > 1e-6
            ouv = (K @ ocam.T).T
            ou = ouv[:, 0] / np.where(ofront, oz, 1.0)
            ov = ouv[:, 1] / np.where(ofront, oz, 1.0)
            oin = ofront & (ou >= 0) & (ou < W) & (ov >= 0) & (ov < H)
            oui, ovi, ozi = ou[oin], ov[oin], oz[oin]
        opu = (oui / occ_downsample).astype(np.int32)
        opv = (ovi / occ_downsample).astype(np.int32)
        np.minimum.at(depth, opv * dw + opu, ozi)

        pu = (ui / occ_downsample).astype(np.int32)
        pv = (vi / occ_downsample).astype(np.int32)
        flat = pv * dw + pu
        visible = zi <= depth[flat] * (1.0 + occ_tol)
        if not visible.any():
            continue

        img = cv2.imread(str(image_dir / cam["name"]))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        idx_v = idx[visible]
        su = np.clip(np.round(ui[visible]).astype(np.int32), 0, W - 1)
        sv = np.clip(np.round(vi[visible]).astype(np.int32), 0, H - 1)
        colors = rgb[sv, su].astype(np.float64) / 255.0
        # Closer + more frontal views weigh more.
        wt = 1.0 / np.maximum(zi[visible], 0.1)
        accum[idx_v] += colors * wt[:, None]
        wsum[idx_v] += wt

    seen = wsum > 0
    out = np.full((n, 3), 0.5)  # neutral grey for unseen points
    out[seen] = accum[seen] / wsum[seen, None]
    return out, seen, len(cams)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_data_arg(ap)
    ap.add_argument("--cloud", default=None, help="default <data>/work/cloud_edited.ply")
    ap.add_argument("--model", default=None)
    ap.add_argument("--images", default=None)
    ap.add_argument("--transform", default=None)
    ap.add_argument("--out", default=None, help="default <data>/work/cloud_colored.ply")
    ap.add_argument("--occ-downsample", type=int, default=4)
    ap.add_argument("--occ-tol", type=float, default=0.02)
    ap.add_argument("--drop-unseen", action="store_true", help="discard points no photo saw")
    ap.add_argument("--occ-cloud", default=None,
                    help="dense cloud for occlusion (default: the source cloud, so a "
                         "decimated/trimmed --cloud is still coloured correctly)")
    args = ap.parse_args()

    cap = Capture.from_args(args)
    cloud = Path(args.cloud) if args.cloud else cap.p("cloud_edited.ply")
    if not cloud.exists():
        cloud = cap.staged_cloud()
    detected = cap.colmap()   # our pipeline output OR a JMStudio project's poses
    if args.model and args.images:
        model, images = Path(args.model), Path(args.images)
    elif detected is not None:
        model, images = detected[0], detected[1]
    else:
        raise SystemExit("no COLMAP poses found; run colmap_pipeline + align first")
    transform = Path(args.transform) if args.transform else cap.p("aligned", "transform.json")
    # Default the occlusion reference to the dense source cloud so a decimated
    # or trimmed --cloud is still coloured with watertight occlusion.
    if args.occ_cloud:
        occ_cloud = Path(args.occ_cloud)
    else:
        try:
            occ_cloud = cap.staged_cloud()
        except Exception:
            occ_cloud = None
        if occ_cloud and Path(occ_cloud).resolve() == Path(cloud).resolve():
            occ_cloud = None   # source == cloud: no denser reference to add
    recolor(
        cloud_path=cloud,
        model_path=model,
        image_dir=images,
        transform_path=transform,
        out_path=Path(args.out) if args.out else cap.p("cloud_colored.ply"),
        occ_downsample=args.occ_downsample,
        occ_tol=args.occ_tol,
        drop_unseen=args.drop_unseen,
        occ_cloud_path=occ_cloud,
    )


if __name__ == "__main__":
    main()
