"""Train 3D Gaussian Splats from the posed COLMAP frames, via gsplat.

Initialization comes from either COLMAP's sparse points (default, always valid)
or the lidar cloud aligned into the COLMAP frame (``--init lidar``, requires
:mod:`raven.align`). Training uses gsplat's rasterizer with an L1 + SSIM loss and
the default adaptive-density strategy, and exports a standard 3DGS ``.ply``.

    python -m raven.train_splat --iters 30000
    python -m raven.train_splat --init lidar --iters 30000

Requires ``torch`` (CUDA) and ``gsplat``::

    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install gsplat
"""

from __future__ import annotations

import argparse
import os
import threading
from pathlib import Path

import cv2
import numpy as np

from .paths import Capture, add_data_arg


def ensure_cuda_home() -> None:
    """Point CUDA_HOME at a usable nvcc so gsplat can JIT-compile its kernels.

    gsplat builds CUDA extensions on first use and needs ``nvcc``. If the system
    has none, fall back to a micromamba ``cuda121`` env (see README) when present.
    """
    home = os.environ.get("CUDA_HOME")
    if home and (Path(home) / "bin" / "nvcc").exists():
        return
    candidates = [
        Path.home() / ".local/micromamba/envs/cuda121",
        Path("/usr/local/cuda"),
    ]
    for c in candidates:
        if (c / "bin" / "nvcc").exists():
            os.environ["CUDA_HOME"] = str(c)
            os.environ["PATH"] = f"{c / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
            return


# --------------------------------------------------------------------------- #
# Data loading (COLMAP model -> posed images + intrinsics)
# --------------------------------------------------------------------------- #
def load_colmap_dataset(model_path: Path, image_dir: Path):
    """Return list of dicts: {image HxWx3 float[0,1], K 3x3, viewmat 4x4 (w2c)}.

    Uses the shared loader, so it reads our pycolmap models *and* JMStudio's
    text-only ``cameras.txt``/``images.txt`` exports.
    """
    from raven.recolor import load_cameras

    frames = []
    for c in load_cameras(Path(model_path)):
        bgr = cv2.imread(str(image_dir / c["name"]))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        w2c = np.eye(4)
        w2c[:3, :3] = c["R"]
        w2c[:3, 3] = c["t"]
        frames.append({"image": rgb, "K": np.asarray(c["K"], np.float64), "viewmat": w2c, "name": c["name"]})
    if not frames:
        raise RuntimeError(f"no posed images loaded from {model_path} / {image_dir}")
    return frames


def _colmap_sparse_points(model_path: Path):
    """(xyz, rgb[0,1]) from a COLMAP model's points3D, or None if unavailable."""
    try:
        import pycolmap

        rec = pycolmap.Reconstruction(str(model_path))
        pts = np.asarray([p.xyz for p in rec.points3D.values()], np.float32)
        rgb = np.asarray([p.color[:3] for p in rec.points3D.values()], np.float32) / 255.0
        return (pts, rgb) if len(pts) else None
    except Exception:
        return None


def init_points(init: str, lidar_ply: Path, max_pts: int, model_path: Path):
    """Return (means Nx3, rgb Nx3 [0,1])."""
    if init == "lidar" and lidar_ply.exists():
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(str(lidar_ply))
        pts = np.asarray(pcd.points, np.float32)
        rgb = (
            np.asarray(pcd.colors, np.float32)
            if pcd.has_colors()
            else np.full((len(pts), 3), 0.5, np.float32)
        )
        print(f"init from lidar cloud: {len(pts):,} points")
    else:
        sp = _colmap_sparse_points(model_path)
        if sp is None:
            raise RuntimeError(
                "model has no sparse points (text-only export); use --init lidar "
                "with a point cloud"
            )
        pts, rgb = sp
        print(f"init from COLMAP sparse: {len(pts):,} points")
    if len(pts) > max_pts:
        idx = np.random.choice(len(pts), max_pts, replace=False)
        pts, rgb = pts[idx], rgb[idx]
    return pts, rgb


# --------------------------------------------------------------------------- #
# Gaussian model helpers
# --------------------------------------------------------------------------- #
def rgb_to_sh(rgb):
    return (rgb - 0.5) / 0.28209479177387814


def se3_exp(xi):
    """Differentiable SE3 exponential of a 6-vector (w, v) -> 4x4 matrix.

    Used for per-camera pose refinement: a small learnable ``xi`` left-composed
    with the COLMAP world->camera matrix lets bundle-style pose error (which
    blurs multi-view training) be corrected jointly with the Gaussians.
    """
    import torch

    w, v = xi[:3], xi[3:]
    z = torch.zeros((), device=xi.device, dtype=xi.dtype)
    wx = torch.stack([
        torch.stack([z, -w[2], w[1]]),
        torch.stack([w[2], z, -w[0]]),
        torch.stack([-w[1], w[0], z]),
    ])
    g = torch.zeros(4, 4, device=xi.device, dtype=xi.dtype)
    g[:3, :3] = wx
    g[:3, 3] = v
    return torch.matrix_exp(g)


_SSIM_WINDOW = {}


def windowed_ssim(pred, gt, window_size: int = 11, sigma: float = 1.5):
    """Standard Gaussian-windowed SSIM on two HxWx3 tensors in [0, 1]."""
    import torch
    import torch.nn.functional as F

    device = pred.device
    key = (window_size, sigma, device)
    win = _SSIM_WINDOW.get(key)
    if win is None:
        coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = (g / g.sum())
        win2d = (g[:, None] * g[None, :])[None, None]
        win = win2d.expand(3, 1, window_size, window_size).contiguous()
        _SSIM_WINDOW[key] = win

    a = pred.permute(2, 0, 1)[None]  # 1,3,H,W
    b = gt.permute(2, 0, 1)[None]
    pad = window_size // 2
    mu_a = F.conv2d(a, win, padding=pad, groups=3)
    mu_b = F.conv2d(b, win, padding=pad, groups=3)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    va = F.conv2d(a * a, win, padding=pad, groups=3) - mu_a2
    vb = F.conv2d(b * b, win, padding=pad, groups=3) - mu_b2
    cov = F.conv2d(a * b, win, padding=pad, groups=3) - mu_ab
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    smap = ((2 * mu_ab + c1) * (2 * cov + c2)) / ((mu_a2 + mu_b2 + c1) * (va + vb + c2))
    return smap.mean()


def knn_scale(points, k: int = 4):
    """Initial isotropic scale = mean distance to k nearest neighbours."""
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    d, _ = tree.query(points, k=k + 1)
    return np.clip(d[:, 1:].mean(axis=1), 1e-4, None)


def build_gaussians(points, rgb, sh_degree, device):
    import torch

    n = len(points)
    means = torch.tensor(points, dtype=torch.float32, device=device)
    scales = torch.tensor(np.log(knn_scale(points)), dtype=torch.float32, device=device)
    scales = scales[:, None].repeat(1, 3)
    quats = torch.zeros(n, 4, device=device)
    quats[:, 0] = 1.0
    opacities = torch.logit(torch.full((n,), 0.1, device=device))

    num_sh = (sh_degree + 1) ** 2
    colors = torch.zeros(n, num_sh, 3, device=device)
    colors[:, 0, :] = torch.tensor(rgb_to_sh(rgb), dtype=torch.float32, device=device)

    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities),
        "sh0": torch.nn.Parameter(colors[:, :1, :]),
        "shN": torch.nn.Parameter(colors[:, 1:, :]),
    }).to(device)
    return params


def make_optimizers(params, lr_scale: float):
    import torch

    specs = {
        "means": 1.6e-4 * lr_scale,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 2.5e-3 / 20,
    }
    return {
        name: torch.optim.Adam([params[name]], lr=lr, eps=1e-15)
        for name, lr in specs.items()
    }


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(args):
    import torch
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy

    device = "cuda"
    model_path = Path(args.model)
    image_dir = Path(args.images)
    frames = load_colmap_dataset(model_path, image_dir)
    print(f"loaded {len(frames)} posed frames")

    pts, rgb = init_points(args.init, Path(args.lidar), args.max_init_points, model_path)
    scene_scale = float(np.linalg.norm(pts.std(axis=0))) + 1e-6

    # Confine the final splat to the (reduced) init cloud's extent. The cameras
    # see beyond the kept points, so the densifier creates Gaussians outside the
    # region of interest; cropping the export keeps only what the edited cloud
    # covers, so "edit the cloud -> reduced splat" holds.
    crop_aabb = None
    if args.crop_to_init:
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        margin = (hi - lo) * args.crop_margin
        crop_aabb = (lo - margin, hi + margin)
        print(f"crop-to-init AABB: {np.round(crop_aabb[0],2)} .. {np.round(crop_aabb[1],2)}")
    params = build_gaussians(pts, rgb, args.sh_degree, device)
    optimizers = make_optimizers(params, lr_scale=scene_scale)

    # gsplat's DefaultStrategy constants are tuned for the reference 30k-iter
    # schedule and do NOT scale with --iters: at e.g. 7k, densification (meant to
    # run to 15k) is cut off mid-cycle and training ends ~1k iters after an
    # opacity reset (every 3k), so gaussians stay under-densified and washed out.
    # Scale the schedule to --iters so shorter runs still bake properly; >=30k
    # reproduces the reference exactly.
    REF_ITERS = 30_000
    sc = args.iters / REF_ITERS
    strategy = DefaultStrategy(
        verbose=True,
        refine_start_iter=max(1, round(500 * sc)),
        refine_stop_iter=max(1, round(15_000 * sc)),
        reset_every=max(1, round(3_000 * sc)),
        # refine_every is a cadence, not a milestone — keep it dense for short runs.
        refine_every=max(1, min(100, round(100 * sc))),
        # Lower => densify more aggressively (more, finer Gaussians on detail).
        grow_grad2d=args.grow_grad2d,
    )
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)
    # Reach full SH at the same ~10%-of-training point as the reference schedule.
    sh_increase_every = max(1, round(args.sh_increase_every * sc))

    # Keep images on CPU (pinned) and move per-step to scale to many frames;
    # the tiny K/viewmat tensors live on the GPU.
    for f in frames:
        f["image_t"] = torch.tensor(f["image"]).pin_memory()
        f["K_t"] = torch.tensor(f["K"], dtype=torch.float32, device=device)
        f["viewmat_t"] = torch.tensor(f["viewmat"], dtype=torch.float32, device=device)

    # Optional joint pose refinement: a learnable SE3 delta per camera, left-
    # composed with its COLMAP pose, optimised at a low LR after a short warmup
    # (so Gaussians form first). Corrects residual multi-view pose error that a
    # fixed-pose run can only resolve by blurring.
    pose_opt = None
    if args.refine_poses:
        pose_adjust = torch.zeros(len(frames), 6, device=device, requires_grad=True)
        pose_opt = torch.optim.Adam([pose_adjust], lr=args.pose_lr)
        print(f"joint pose refinement ON (lr={args.pose_lr}, warmup={args.pose_warmup})")

    rng = np.random.default_rng(0)
    order = rng.permutation(len(frames))
    cur = 0
    for step in range(args.iters):
        if cur >= len(order):
            order = rng.permutation(len(frames))
            cur = 0
        fi = int(order[cur])
        f = frames[fi]
        cur += 1

        H, W = f["image"].shape[:2]
        sh_deg = min(args.sh_degree, step // sh_increase_every)
        colors = torch.cat([params["sh0"], params["shN"]], dim=1)

        pose_active = pose_opt is not None and step >= args.pose_warmup
        viewmat = se3_exp(pose_adjust[fi]) @ f["viewmat_t"] if pose_active else f["viewmat_t"]

        renders, alphas, info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=colors,
            viewmats=viewmat[None],
            Ks=f["K_t"][None],
            width=W,
            height=H,
            sh_degree=sh_deg,
            packed=False,
        )
        strategy.step_pre_backward(params, optimizers, strategy_state, step, info)

        pred = renders[0].clamp(0, 1)
        gt = f["image_t"].to(device, non_blocking=True)
        l1 = (pred - gt).abs().mean()
        loss = (1 - args.ssim_lambda) * l1 + args.ssim_lambda * (1 - windowed_ssim(pred, gt))

        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        if pose_active:
            pose_opt.step()
            pose_opt.zero_grad(set_to_none=True)

        strategy.step_post_backward(params, optimizers, strategy_state, step, info)

        if step % 100 == 0:
            print(f"step {step:6d}  loss {loss.item():.4f}  gaussians {params['means'].shape[0]:,}")
            # Machine-readable line the GUI scrapes to drive a progress bar.
            print(f"PROGRESS {step + 1}/{args.iters}", flush=True)

    print(f"PROGRESS {args.iters}/{args.iters}", flush=True)

    # Train-view PSNR readout, evaluated with each frame's (possibly refined)
    # pose so pose-refinement runs are scored fairly.
    with torch.no_grad():
        colors = torch.cat([params["sh0"], params["shN"]], dim=1)
        tot, n = 0.0, 0
        for fi in range(0, len(frames), max(1, len(frames) // 30)):
            f = frames[fi]
            vm = se3_exp(pose_adjust[fi]) @ f["viewmat_t"] if pose_opt is not None else f["viewmat_t"]
            H, W = f["image"].shape[:2]
            out, _, _ = rasterization(
                means=params["means"], quats=params["quats"], scales=torch.exp(params["scales"]),
                opacities=torch.sigmoid(params["opacities"]), colors=colors,
                viewmats=vm[None], Ks=f["K_t"][None], width=W, height=H,
                sh_degree=args.sh_degree, packed=False)
            mse = (out[0].clamp(0, 1) - f["image_t"].to(device)).pow(2).mean()
            tot += float(10 * torch.log10(1.0 / mse)); n += 1
        print(f"train-view PSNR (mean over {n} frames): {tot / n:.2f} dB")

    export_ply(params, Path(args.out), crop_aabb)
    print(f"exported splat: {args.out}")


def export_ply(params, path: Path, crop_aabb=None):
    """Write a standard 3DGS .ply (means, sh dc+rest, opacity, scale, rot).

    If ``crop_aabb`` (lo, hi) is given, only Gaussians whose centres fall inside
    the box are written, confining the splat to the edited cloud's region.
    """
    import torch
    from plyfile import PlyData, PlyElement

    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        means = params["means"].cpu().numpy()
        sh0 = params["sh0"].cpu().numpy().reshape(len(means), -1)
        shN = params["shN"].cpu().numpy().reshape(len(means), -1)
        opac = params["opacities"].cpu().numpy().reshape(-1, 1)
        scales = params["scales"].cpu().numpy()
        quats = params["quats"].cpu().numpy()

    if crop_aabb is not None:
        lo, hi = crop_aabb
        keep = np.all((means >= lo) & (means <= hi), axis=1)
        means, sh0, shN = means[keep], sh0[keep], shN[keep]
        opac, scales, quats = opac[keep], scales[keep], quats[keep]
        print(f"crop-to-init: kept {keep.sum():,} / {len(keep):,} gaussians")

    fields = ["x", "y", "z", "nx", "ny", "nz"]
    fields += [f"f_dc_{i}" for i in range(sh0.shape[1])]
    fields += [f"f_rest_{i}" for i in range(shN.shape[1])]
    fields += ["opacity"]
    fields += [f"scale_{i}" for i in range(scales.shape[1])]
    fields += [f"rot_{i}" for i in range(quats.shape[1])]

    normals = np.zeros_like(means)
    data = np.concatenate([means, normals, sh0, shN, opac, scales, quats], axis=1)
    dtype = [(f, "f4") for f in fields]
    arr = np.empty(len(means), dtype=dtype)
    for i, f in enumerate(fields):
        arr[f] = data[:, i]
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def export_surfel(cloud_path: Path, out_path: Path, radius_mult: float = 1.5,
                  thin_frac: float = 0.15, max_points: int | None = None,
                  opacity: float = 0.99, adaptive: bool = True) -> None:
    """Convert a coloured point cloud straight to a 3DGS .ply — no training.

    Each point becomes a flat, normal-oriented disk Gaussian sized to the local
    point spacing and coloured from the point. Because it skips multi-view
    photometric optimisation it stays as crisp as the cloud (good for distant /
    orbit viewing) and runs on CPU in seconds. The output uses the same 62-field
    SH-degree-3 layout as the trained splat (with zero view-dependent terms), so
    viewers treat the two identically.

    With ``adaptive`` (default), each disk is sized to its *own* local spacing
    (dense regions get small, sharp disks; sparse regions get larger ones to fill
    gaps), clamped so isolated outliers don't become huge disks. Otherwise every
    disk uses the global median spacing.
    """
    import torch
    import open3d as o3d
    from scipy.spatial import cKDTree

    pcd = o3d.io.read_point_cloud(str(cloud_path))
    P = np.asarray(pcd.points, np.float32)
    if len(P) == 0:
        raise SystemExit(f"empty cloud: {cloud_path}")
    if pcd.has_colors():
        C = np.asarray(pcd.colors, np.float32)
    else:
        print("warning: cloud has no colours; surfel will be flat grey")
        C = np.full((len(P), 3), 0.5, np.float32)
    if max_points and len(P) > max_points:
        idx = np.random.default_rng(0).choice(len(P), max_points, replace=False)
        P, C = P[idx], C[idx]
    n = len(P)

    # Normals (disk orientation) and local spacing (disk radius).
    pn = o3d.geometry.PointCloud()
    pn.points = o3d.utility.Vector3dVector(P)
    pn.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=16))
    N = np.asarray(pn.normals, np.float32)
    if adaptive:
        # Per-point local spacing -> per-point disk size.
        dist, _ = cKDTree(P).query(P, k=4, workers=-1)
        sp = dist[:, 1:].mean(1).astype(np.float32)
        med = float(np.median(sp))
        sp = np.clip(sp, 0.3 * med, 3.0 * med)         # tame isolated outliers
        print(f"surfel: {n:,} pts, median spacing={med * 1000:.1f}mm, "
              f"adaptive disk r≈{med * radius_mult * 1000:.0f}mm")
    else:
        q = P[:: max(1, n // 200_000)]                 # global median from a subsample
        dist, _ = cKDTree(P).query(q, k=4)
        med = float(np.median(dist[:, 1:].mean(1)))
        sp = np.full(n, med, np.float32)
        print(f"surfel: {n:,} pts, spacing={med * 1000:.1f}mm, "
              f"uniform disk r={med * radius_mult * 1000:.0f}mm")
    radius_pp = sp * radius_mult
    thin_pp = sp * thin_frac

    # Orthonormal tangents -> rotation [t1|t2|n]; thin (3rd) axis along normal.
    a = np.tile(np.array([1, 0, 0], np.float32), (n, 1))
    a[np.abs(N[:, 0]) > 0.9] = (0, 1, 0)
    t1 = np.cross(a, N); t1 /= np.linalg.norm(t1, axis=1, keepdims=True) + 1e-9
    t2 = np.cross(N, t1)
    R = np.stack([t1, t2, N], axis=2); rr = lambda i, j: R[:, i, j]
    qw = 0.5 * np.sqrt(np.maximum(0, 1 + rr(0, 0) + rr(1, 1) + rr(2, 2)))
    qx = 0.5 * np.sqrt(np.maximum(0, 1 + rr(0, 0) - rr(1, 1) - rr(2, 2))) * np.sign(rr(2, 1) - rr(1, 2))
    qy = 0.5 * np.sqrt(np.maximum(0, 1 - rr(0, 0) + rr(1, 1) - rr(2, 2))) * np.sign(rr(0, 2) - rr(2, 0))
    qz = 0.5 * np.sqrt(np.maximum(0, 1 - rr(0, 0) - rr(1, 1) + rr(2, 2))) * np.sign(rr(1, 0) - rr(0, 1))
    quats = np.stack([qw, qx, qy, qz], 1).astype(np.float32)

    logit_opac = float(np.log(opacity / (1 - opacity)))
    log_scale = np.log(np.stack([radius_pp, radius_pp, thin_pp], 1)).astype(np.float32)
    tt = lambda x: torch.from_numpy(np.ascontiguousarray(x.astype(np.float32)))
    params = {
        "means": tt(P),
        "sh0": tt(rgb_to_sh(C).reshape(n, 1, 3)),       # diffuse colour
        "shN": torch.zeros(n, 15, 3),                   # deg-3 rest = 0 (no view-dependence)
        "opacities": tt(np.full(n, logit_opac)),
        "scales": tt(log_scale),                        # export_ply writes these as-is (log)
        "quats": tt(quats),
    }
    export_ply(params, Path(out_path))
    print(f"exported surfel splat: {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_data_arg(ap)
    ap.add_argument("--model", default=None)
    ap.add_argument("--images", default=None)
    ap.add_argument("--init", choices=["colmap", "lidar"], default="colmap")
    ap.add_argument("--lidar", default=None)
    ap.add_argument("--out", default=None, help="default <data>/work/splat.ply")
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--sh-increase-every", type=int, default=1000)
    ap.add_argument("--ssim-lambda", type=float, default=0.2)
    ap.add_argument("--max-init-points", type=int, default=1_000_000)
    ap.add_argument("--crop-to-init", dest="crop_to_init", action="store_true", default=None,
                    help="confine exported splat to init-cloud AABB (default: on for --init lidar)")
    ap.add_argument("--no-crop-to-init", dest="crop_to_init", action="store_false",
                    help="keep all gaussians, even outside the edited region")
    ap.add_argument("--crop-margin", type=float, default=0.03,
                    help="expand crop AABB by this fraction of its extent")
    ap.add_argument("--surfel", action="store_true",
                    help="fast no-training export: turn the (coloured) --lidar cloud directly "
                         "into normal-oriented disk gaussians (CPU, seconds)")
    ap.add_argument("--surfel-radius", type=float, default=1.5,
                    help="surfel disk radius as a multiple of local point spacing")
    ap.add_argument("--surfel-thin", type=float, default=0.15,
                    help="surfel disk thickness as a fraction of local point spacing")
    ap.add_argument("--surfel-uniform", action="store_true",
                    help="size every surfel disk by the global median spacing "
                         "(default: adaptive per-point sizing)")
    ap.add_argument("--grow-grad2d", type=float, default=0.0002,
                    help="densification gradient threshold; lower => more/finer gaussians")
    ap.add_argument("--refine-poses", action="store_true",
                    help="jointly optimise per-camera poses (fixes residual multi-view error)")
    ap.add_argument("--pose-lr", type=float, default=1e-3, help="pose-refinement learning rate")
    ap.add_argument("--pose-warmup", type=int, default=500,
                    help="iters before pose refinement starts (let gaussians form first)")
    ap.add_argument("--view", action="store_true",
                    help="after export, open the splat in the SuperSplat web editor")
    args = ap.parse_args()

    cap = Capture.from_args(args)
    args.lidar = args.lidar or str(cap.p("aligned", "cloud_in_colmap.ply"))
    args.out = args.out or str(cap.p("splat.ply"))

    # Fast path: no-training surfel export straight from the coloured cloud.
    # Needs only the cloud (no poses/COLMAP/CUDA).
    if args.surfel:
        export_surfel(Path(args.lidar), Path(args.out),
                      radius_mult=args.surfel_radius, thin_frac=args.surfel_thin,
                      adaptive=not args.surfel_uniform)
        _maybe_view(args)
        return

    args.model = args.model or str(cap.p("colmap", "undistorted", "sparse"))
    args.images = args.images or str(cap.p("colmap", "undistorted", "images"))

    # Default: crop to the edited cloud when initialising from lidar.
    if args.crop_to_init is None:
        args.crop_to_init = args.init == "lidar"

    # Fall back to the non-undistorted sparse model if needed.
    if not Path(args.model).exists():
        args.model = str(cap.p("colmap", "sparse", "0"))
        args.images = str(cap.p("images"))
    ensure_cuda_home()
    train(args)
    _maybe_view(args)


def _maybe_view(args) -> None:
    """If ``--view``, serve the exported splat and open it in SuperSplat."""
    if not getattr(args, "view", False):
        return
    from raven.view_splat import open_in_supersplat

    httpd, url = open_in_supersplat(Path(args.out))
    print(f"opened in SuperSplat: {url}\nCtrl-C to stop serving.")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()


if __name__ == "__main__":
    main()
