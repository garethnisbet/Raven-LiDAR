"""Capture-folder abstraction.

A *capture* is a 3DMakerpro Raven export folder, e.g.::

    /FastDrive/Dropbox/LIDAR/data/I16BeamlineScan/
        IMAGE_*.bag
        LIDAR_*.bag
        project_parameters.json
        calibration/calib.json
        thumbnail/LIDAR_*.ply
        camera/...

The raven package lives elsewhere; every command points at a capture with
``--data <folder>`` (default: current directory). All derived artefacts are
written to ``<capture>/work`` so each scan keeps its own outputs.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# Directory that contains the ``raven`` package (used as cwd for subprocesses).
CODE_ROOT = Path(__file__).resolve().parent.parent


class Capture:
    """A scan's inputs (``root``) and its output location (``work``).

    By default outputs live in ``<root>/work``. Pass ``project`` to keep the
    scan folder clean and write to ``<project>/<scan-name>/`` instead (so one
    project can hold many scans without collisions); ``work`` overrides both
    with an explicit directory. Inputs are always read from ``root``.
    """

    def __init__(self, root: str | Path, project: str | Path | None = None,
                 work: str | Path | None = None):
        self.root = Path(root).expanduser().resolve()
        self.project = Path(project).expanduser().resolve() if project else None
        if work is not None:
            self.work = Path(work).expanduser().resolve()
        elif self.project is not None:
            self.work = self.project / self.root.name
        else:
            self.work = self.root / "work"

    @classmethod
    def from_args(cls, args) -> "Capture":
        """Build from a parser that used :func:`add_data_arg`."""
        return cls(args.data, project=getattr(args, "project", None),
                   work=getattr(args, "work", None))

    # ---- inputs -------------------------------------------------------------
    def _glob_one(self, pattern: str, where: Path) -> Path:
        matches = sorted(where.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"no {pattern} found in {where}")
        return matches[0]

    def image_bag(self) -> Path:
        return self._glob_one("IMAGE_*.bag", self.root)

    def lidar_bag(self) -> Path:
        return self._glob_one("LIDAR_*.bag", self.root)

    def calib(self) -> Path:
        return self.root / "calibration" / "calib.json"

    # Gaussian-splat exports are point-cloud-shaped but must not be edited as one.
    _SPLAT_HINTS = ("splat", "gaussian")

    def find_cloud(self) -> Path:
        """Locate the *original* editable point cloud for the capture.

        Always prefers the untouched source so "Load folder" shows the original,
        not a previous edit: the staged original copy (``work/cloud_raw.ply``),
        then the Raven ``thumbnail`` fused cloud, then any ``.ply``/``.pcd`` in the
        capture root (JMStudio exports), skipping Gaussian-splat ``.ply`` files.
        Derived edits (``cloud_edited.ply``/``cloud_colored.ply``) are never
        auto-loaded -- open them explicitly with "Load file…".
        """
        raw = self.work / "cloud_raw.ply"
        if raw.exists():
            return raw
        for pat in ("thumbnail/LIDAR_*.ply", "thumbnail/*.ply"):
            m = sorted(self.root.glob(pat))
            if m:
                return m[0]
        plys = [p for p in sorted(self.root.glob("*.ply"))
                if not any(h in p.name.lower() for h in self._SPLAT_HINTS)]
        if plys:
            return plys[0]
        pcds = sorted(self.root.glob("*.pcd"))
        if pcds:
            return pcds[0]
        raise FileNotFoundError(
            f"no point cloud (.ply/.pcd) found in {self.root} "
            "(looked in work/, thumbnail/, and the folder root)"
        )

    def colmap(self):
        """Locate an existing COLMAP result (poses + undistorted images).

        Returns ``(model_dir, images_dir, integrated)`` or ``None``. ``integrated``
        is True when the poses already share the point-cloud frame (a JMStudio
        project export), so no lidar->COLMAP alignment is needed. Detects our own
        pipeline output and JMStudio's ``shading/Colmap`` layout.
        """
        ours_undist = self.p("colmap", "undistorted", "sparse")
        if ours_undist.exists():
            return ours_undist, self.p("colmap", "undistorted", "images"), False
        ours = self.p("colmap", "sparse", "0")
        if ours.exists():
            return ours, self.p("images"), False
        jm = self.root / "shading" / "Colmap"
        if (jm / "sparse" / "0").exists() and (jm / "images").is_dir():
            return jm / "sparse" / "0", jm / "images", True
        return None

    def looks_like_capture(self) -> bool:
        if not self.root.is_dir():
            return False
        try:
            self.find_cloud()
            return True
        except FileNotFoundError:
            return any(self.root.glob("LIDAR_*.bag"))

    # ---- outputs ------------------------------------------------------------
    def ensure_work(self) -> Path:
        self.work.mkdir(parents=True, exist_ok=True)
        return self.work

    def p(self, *parts) -> Path:
        """Path inside the work dir, e.g. cap.p('colmap', 'undistorted')."""
        return self.work.joinpath(*parts)

    def staged_cloud(self) -> Path:
        """Return an editable cloud in ``work/``, staging it from the folder if needed."""
        src = self.find_cloud()
        if src.parent == self.work:          # already a work output: edit in place
            return src
        self.ensure_work()
        dst = self.work / "cloud_raw.ply"
        if src.suffix.lower() == ".ply":
            shutil.copy2(src, dst)
        else:                                # e.g. .pcd -> normalise to .ply
            import open3d as o3d
            o3d.io.write_point_cloud(str(dst), o3d.io.read_point_cloud(str(src)))
        return dst


def add_data_arg(ap: argparse.ArgumentParser) -> None:
    """Add the shared ``--data`` capture-folder argument (+ output location)."""
    ap.add_argument(
        "--data", default=".",
        help="capture folder (contains the .bag files, calibration/, thumbnail/)",
    )
    ap.add_argument(
        "--project", default=None,
        help="output project folder; artefacts go to <project>/<scan-name>/ "
             "instead of <scan>/work, keeping the scan folder clean",
    )
    ap.add_argument(
        "--work", default=None,
        help="explicit output dir (overrides --project and the default <scan>/work)",
    )
