"""Load, display, and save 3D Gaussian-Splat PLYs, *format-preserving*.

Lets the cloud editor trim/clip a **finished** splat (e.g. 3DMakerpro's export)
rather than the point cloud: every gaussian attribute (scale, rotation, opacity,
spherical-harmonic colour) is carried through the edit untouched and written back
in the original field layout, so the result still opens in the same viewer.

Editing only *selects subsets* of gaussians (crop / trim / clean / decimate), so
the structured PLY record is simply masked — no attribute is recomputed.

    python -m raven.splat_io in.ply out.ply --crop xmin ymin zmin xmax ymax zmax
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

# DC term of the SH basis: rgb = 0.5 + C0 * f_dc.
SH_C0 = 0.28209479177387814


def is_splat_ply(path: str | Path) -> bool:
    """True if the PLY looks like a 3DGS splat (SH dc / scale / rot fields).

    Sniffs only the header so it's cheap even on multi-GB files.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(4096).decode("latin-1", "ignore")
    except Exception:
        return False
    if "element vertex" not in head:
        return False
    return "f_dc_0" in head or ("scale_0" in head and "rot_0" in head)


def load_splat(path: str | Path):
    """Return ``(data, field_names)`` for a splat PLY (full structured record)."""
    el = PlyData.read(str(path))["vertex"]
    return el.data, [p.name for p in el.properties]


def xyz(data) -> np.ndarray:
    return np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)


def display_colours(data) -> np.ndarray:
    """RGB in [0,1] for display: from SH DC (``f_dc_*``) if present, else stored
    ``red/green/blue``, else neutral grey."""
    names = data.dtype.names
    if "f_dc_0" in names:
        dc = np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], 1).astype(np.float64)
        return np.clip(0.5 + SH_C0 * dc, 0.0, 1.0)
    if "red" in names:
        return np.stack([data["red"], data["green"], data["blue"]], 1).astype(np.float64) / 255.0
    return np.full((len(data), 3), 0.6)


def save_splat(path: str | Path, data, indices=None) -> int:
    """Write ``data`` (or only its ``indices`` rows) as a binary splat PLY,
    preserving every field. Returns the gaussian count written."""
    out = data if indices is None else data[np.asarray(indices, np.int64)]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(out, "vertex")], text=False).write(str(path))
    return len(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--crop", type=float, nargs=6, metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
                    help="keep only gaussians whose centres fall inside this AABB")
    ap.add_argument("--invert", action="store_true", help="with --crop, remove inside instead")
    args = ap.parse_args()

    if not is_splat_ply(args.input):
        raise SystemExit(f"{args.input} doesn't look like a gaussian-splat PLY")
    data, fields = load_splat(args.input)
    idx = None
    if args.crop:
        lo = np.array(args.crop[:3]); hi = np.array(args.crop[3:])
        p = xyz(data)
        inside = np.all((p >= lo) & (p <= hi), axis=1)
        keep = ~inside if args.invert else inside
        idx = np.nonzero(keep)[0]
    n = save_splat(args.output, data, idx)
    print(f"{args.input}: {len(data):,} gaussians ({len(fields)} fields) "
          f"-> {args.output}: {n:,} kept")


if __name__ == "__main__":
    main()
