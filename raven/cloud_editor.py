"""Editor-first Open3D GUI for reducing the fused point cloud, then building splats.

    python -m raven.cloud_editor              # launch, then use the Load buttons
    python -m raven.cloud_editor <folder>     # optionally open a capture at startup

Features
  * Load: "Load folder…" opens a capture (auto-stages its fused cloud);
    "Load file…" opens any .ply directly.
  * In-scene selection: toggle *Box select*, left-drag a rectangle in the view;
    points inside highlight red live as you drag. "selection removes inside"
    flips between cropping to the box and deleting its contents.
  * Orthographic toggle (wheel-zoom rescales the ortho frustum), great with the
    Top / Front / Side presets for clean axis-aligned box selection.
  * 3D box crop: sliders define an axis-aligned box (drawn live); Keep / Remove.
  * Delete-preview: SOR / radius / auto-clean highlight the points they would
    remove in red with a count, then Apply or Cancel.
  * One-click Auto-clean (SOR + percentile crop + radius), all reusing
    :mod:`raven.cloud_ops`.
  * View presets + point size.
  * Save + Build Splat: writes cloud_edited.ply and runs align (+ SfM if needed)
    and gsplat training in the background, streaming progress into the panel.

All editing maps to a single keep-mask preview model, so every operation is
reviewed before it mutates the cloud.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

if __package__ in (None, ""):
    # Allow `python3 cloud_editor.py` from inside raven/ as well as `-m raven.cloud_editor`.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from raven import cloud_ops as ops
    from raven import splat_io
    from raven.paths import CODE_ROOT, Capture, add_data_arg
else:
    from . import cloud_ops as ops
    from . import splat_io
    from .paths import CODE_ROOT, Capture, add_data_arg

CLOUD = "cloud"
REMOVED = "removed"
CROPBOX = "cropbox"
RECT = "selrect"

# Remembers the chosen project (output) folder between sessions.
CONFIG_PATH = Path.home() / ".config" / "raven" / "editor.json"


def _load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text())
    except Exception:
        return {}


def _save_config(cfg: dict) -> None:
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    except Exception:
        pass


class EditorApp:
    def __init__(self, cap: Capture | None = None, path: str | None = None,
                 project: str | Path | None = None):
        # Output project folder: when set, every scan's artefacts go to
        # <project>/<scan-name>/ instead of <scan>/work, keeping scan folders
        # clean. Threaded into Captures (in-process) and subprocess steps.
        self.project = str(Path(project).expanduser().resolve()) if project else None
        self.cap = cap
        # Capture that actually holds the photos/poses for the current cloud,
        # which may differ from ``cap`` when the cloud and the camera data live
        # in separate folders (set when Recolour resolves one). Build Splat reuses
        # it so it looks in the same place Recolour did.
        self.photo_cap: Capture | None = None
        self.path = path
        # When editing a finished gaussian splat (not a point cloud): the full
        # per-gaussian record + field order. The displayed pcd carries each
        # gaussian's original index in its normals, so subset edits (crop/trim/
        # decimate/denoise — all select_by_index) keep the splat attributes in
        # lockstep; Save writes back the surviving rows in the original layout.
        self.is_splat = False
        self.splat = None            # structured ndarray of all gaussians
        self.splat_fields = None
        self.pcd: o3d.geometry.PointCloud | None = None
        self.original: o3d.geometry.PointCloud | None = None
        self.undo_stack: list[o3d.geometry.PointCloud] = []

        self.point_size = 2.0
        self.select_mode = False
        self.sel_start = None
        self.rect_remove = False          # selection removes inside vs keeps inside
        self._rect_shown = False          # selection rectangle currently drawn
        self._last_sel_draw = 0.0         # throttle timestamp for the rectangle
        self.preview_keep: np.ndarray | None = None  # True = keep
        self.show_box = False
        self.build_proc: subprocess.Popen | None = None
        # Local HTTP server backing "Open in SuperSplat"; kept alive for the
        # process lifetime so the editor can (re)fetch the served splat.
        self._supersplat_httpd = None

        self.fov = 60.0
        self.ortho = False
        self.ortho_half_h = None          # ortho frustum half-height (wheel zoom)
        self.diag = 1.0

        self.app = gui.Application.instance
        self.window = self.app.create_window("Raven Cloud Editor", 1360, 860)
        w = self.window
        self.em = em = w.theme.font_size

        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(w.renderer)
        self.scene.scene.set_background([0.05, 0.05, 0.07, 1.0])
        self.scene.set_on_mouse(self._on_mouse)
        w.add_child(self.scene)

        self.panel = gui.Vert(0.25 * em, gui.Margins(em, 0.5 * em, em, 0.5 * em))
        self._build_panel()
        w.add_child(self.panel)
        w.set_on_layout(self._on_layout)

        if path:
            if splat_io.is_splat_ply(path):
                self._set_splat(path, cap)
            else:
                self._set_cloud(path, cap)
        else:
            self._update_status()

    # ---- layout -------------------------------------------------------------
    def _on_layout(self, ctx):
        r = self.window.content_rect
        pw = 24 * self.em
        self.scene.frame = gui.Rect(r.x, r.y, r.width - pw, r.height)
        self.panel.frame = gui.Rect(r.get_right() - pw, r.y, pw, r.height)
        if self.ortho:
            self._apply_ortho()

    def _row(self, label, widget):
        h = gui.Horiz(0.25 * self.em)
        h.add_child(gui.Label(label))
        h.add_stretch()
        h.add_child(widget)
        return h

    def _num(self, value, kind=gui.NumberEdit.DOUBLE):
        n = gui.NumberEdit(kind)
        if kind == gui.NumberEdit.INT:
            n.int_value = int(value)
        else:
            n.double_value = float(value)
        return n

    def _build_panel(self):
        em = self.em
        p = self.panel

        # ---- load ----
        load = gui.Horiz(0.25 * em)
        bf = gui.Button("Load folder…")
        bf.set_on_clicked(self._load_folder_dialog)
        bfile = gui.Button("Load file…")
        bfile.set_on_clicked(self._load_file_dialog)
        load.add_child(bf)
        load.add_child(bfile)
        p.add_child(load)

        self.project_btn = gui.Button(self._project_label())
        self.project_btn.set_on_clicked(self._project_dialog)
        p.add_child(self.project_btn)

        self.status = gui.Label("")
        p.add_child(self.status)

        # ---- display ----
        p.add_child(gui.Label("— Display —"))
        ps = gui.Slider(gui.Slider.DOUBLE)
        ps.set_limits(1.0, 8.0)
        ps.double_value = self.point_size
        ps.set_on_value_changed(self._on_point_size)
        p.add_child(self._row("point size", ps))
        views = gui.Horiz(0.25 * em)
        for name in ("Top", "Front", "Side", "Iso"):
            b = gui.Button(name)
            b.set_on_clicked(lambda n=name: self._view_preset(n))
            views.add_child(b)
        p.add_child(views)
        self.ortho_chk = gui.Checkbox("Orthographic")
        self.ortho_chk.set_on_checked(self._toggle_ortho)
        p.add_child(self.ortho_chk)
        self.recolor_btn = gui.Button("Recolour from photos")
        self.recolor_btn.set_on_clicked(self._recolor)
        p.add_child(self.recolor_btn)

        # ---- selection / crop ----
        p.add_child(gui.Label("— Select & Crop —"))
        self.sel_btn = gui.Button("Box select: OFF")
        self.sel_btn.set_on_clicked(self._toggle_select)
        p.add_child(self.sel_btn)
        self.rect_rm_chk = gui.Checkbox("selection removes inside")
        self.rect_rm_chk.set_on_checked(lambda v: setattr(self, "rect_remove", v))
        p.add_child(self.rect_rm_chk)

        self.box_chk = gui.Checkbox("Show crop box")
        self.box_chk.set_on_checked(self._on_show_box)
        p.add_child(self.box_chk)
        self.box_sliders = {}
        lo = self.original.get_min_bound() if self.original is not None else np.zeros(3)
        hi = self.original.get_max_bound() if self.original is not None else np.ones(3)
        for i, ax in enumerate("xyz"):
            smin = self._num(lo[i])
            smax = self._num(hi[i])
            smin.set_on_value_changed(lambda v, a=i, k="min": self._on_box_edit())
            smax.set_on_value_changed(lambda v, a=i, k="max": self._on_box_edit())
            self.box_sliders[(ax, "min")] = smin
            self.box_sliders[(ax, "max")] = smax
            row = gui.Horiz(0.25 * em)
            row.add_child(gui.Label(f"{ax}"))
            row.add_child(smin)
            row.add_child(smax)
            p.add_child(row)
        crop_btns = gui.Horiz(0.25 * em)
        bk = gui.Button("Keep box")
        bk.set_on_clicked(lambda: self._box_apply(keep_inside=True))
        br = gui.Button("Remove box")
        br.set_on_clicked(lambda: self._box_apply(keep_inside=False))
        crop_btns.add_child(bk)
        crop_btns.add_child(br)
        p.add_child(crop_btns)

        # ---- denoise (preview) ----
        p.add_child(gui.Label("— Denoise (preview) —"))
        self.so_nb = self._num(20, gui.NumberEdit.INT)
        self.so_std = self._num(2.0)
        p.add_child(self._row("SOR neighbors", self.so_nb))
        p.add_child(self._row("SOR std", self.so_std))
        b = gui.Button("Preview statistical outliers")
        b.set_on_clicked(self._preview_sor)
        p.add_child(b)
        self.ro_nb = self._num(5, gui.NumberEdit.INT)
        self.ro_r = self._num(0.25)
        p.add_child(self._row("radius min-pts", self.ro_nb))
        p.add_child(self._row("radius (m)", self.ro_r))
        b = gui.Button("Preview radius outliers")
        b.set_on_clicked(self._preview_radius)
        p.add_child(b)

        # ---- decimate ----
        p.add_child(gui.Label("— Decimate —"))
        self.dec_factor = self._num(2, gui.NumberEdit.INT)
        p.add_child(self._row("keep 1 in N", self.dec_factor))
        b = gui.Button("Preview decimate")
        b.set_on_clicked(self._preview_decimate)
        p.add_child(b)

        # ---- auto-clean ----
        p.add_child(gui.Label("— Auto-clean —"))
        self.ac_std = self._num(2.0)
        self.ac_pct = self._num(1.0)
        self.ac_r = self._num(0.25)
        p.add_child(self._row("SOR std", self.ac_std))
        p.add_child(self._row("crop tail %", self.ac_pct))
        p.add_child(self._row("radius (m)", self.ac_r))
        b = gui.Button("Preview auto-clean")
        b.set_on_clicked(self._preview_autoclean)
        p.add_child(b)

        # ---- preview apply/cancel ----
        self.preview_row = gui.Horiz(0.25 * em)
        self.apply_btn = gui.Button("Apply")
        self.apply_btn.set_on_clicked(self._apply_preview)
        self.cancel_btn = gui.Button("Cancel")
        self.cancel_btn.set_on_clicked(self._cancel_preview)
        self.preview_row.add_child(self.apply_btn)
        self.preview_row.add_child(self.cancel_btn)
        p.add_child(self.preview_row)
        self.preview_row.visible = False

        # ---- edit / output ----
        p.add_child(gui.Label("— Edit —"))
        edit = gui.Horiz(0.25 * em)
        bu = gui.Button("Undo")
        bu.set_on_clicked(self._undo)
        bre = gui.Button("Reset")
        bre.set_on_clicked(self._reset)
        edit.add_child(bu)
        edit.add_child(bre)
        p.add_child(edit)

        p.add_child(gui.Label("— Output —"))
        b = gui.Button("Save As...")
        b.set_on_clicked(self._save_dialog)
        p.add_child(b)
        self.build_iters = self._num(30000, gui.NumberEdit.INT)
        p.add_child(self._row("train iters", self.build_iters))
        self.surfel_chk = gui.Checkbox("Surfel (fast, no training)")
        p.add_child(self.surfel_chk)
        self.build_btn = gui.Button("Save + Build Splat")
        self.build_btn.set_on_clicked(self._build_splat)
        p.add_child(self.build_btn)
        self.view_btn = gui.Button("Open in SuperSplat")
        self.view_btn.set_on_clicked(self._open_in_supersplat)
        p.add_child(self.view_btn)
        self.build_status = gui.Label("")
        p.add_child(self.build_status)
        self.build_progress = gui.ProgressBar()
        self.build_progress.value = 0.0
        p.add_child(self.build_progress)

    # ---- project (output) folder -------------------------------------------
    def _project_label(self) -> str:
        if self.project:
            return f"Project: {Path(self.project).name}  ▸"
        return "Project: per-scan work/  — set…"

    def _project_dialog(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN_DIR,
                             "Choose project (output) folder", self.window.theme)
        dlg.set_on_cancel(self.window.close_dialog)

        def done(path):
            self.window.close_dialog()
            self._set_project(path)

        dlg.set_on_done(done)
        self.window.show_dialog(dlg)

    def _set_project(self, path):
        self.project = str(Path(path).expanduser().resolve()) if path else None
        cfg = _load_config()
        cfg["project"] = self.project
        _save_config(cfg)
        # Re-point existing captures at the new output location (future saves
        # and builds land there; the loaded cloud in memory is unaffected).
        if self.cap is not None:
            self.cap = Capture(self.cap.root, project=self.project)
        if self.photo_cap is not None:
            self.photo_cap = Capture(self.photo_cap.root, project=self.project)
        self.project_btn.text = self._project_label()
        self.window.set_needs_layout()
        if self.project:
            self._message(f"Project folder set:\n{self.project}\n\n"
                          "Each scan's outputs go to <project>/<scan-name>/.")

    # ---- loading ------------------------------------------------------------
    def _load_folder_dialog(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN_DIR, "Open capture folder", self.window.theme)
        dlg.set_on_cancel(self.window.close_dialog)

        def done(path):
            self.window.close_dialog()
            cap = Capture(path, project=self.project)
            try:
                cloud = str(cap.staged_cloud())
            except Exception as exc:  # noqa: BLE001
                self._message(f"No fused cloud found in:\n{path}\n{exc}")
                return
            self._set_cloud(cloud, cap)

        dlg.set_on_done(done)
        self.window.show_dialog(dlg)

    def _load_file_dialog(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Open point cloud or splat", self.window.theme)
        dlg.add_filter(".ply", "Point cloud or gaussian splat (.ply)")
        dlg.add_filter("", "All files")
        dlg.set_on_cancel(self.window.close_dialog)

        def done(path):
            self.window.close_dialog()
            p = Path(path)
            base = p.parent.parent if p.parent.name == "work" else p.parent
            cap = Capture(base, project=self.project)
            if splat_io.is_splat_ply(path):
                self._set_splat(path, cap)
            else:
                self._set_cloud(path, cap)

        dlg.set_on_done(done)
        self.window.show_dialog(dlg)

    def _set_splat(self, path, cap=None):
        """Load a finished gaussian splat for editing (carries all attributes)."""
        try:
            data, fields = splat_io.load_splat(path)
        except Exception as exc:  # noqa: BLE001
            self._message(f"Could not load splat:\n{path}\n{exc}")
            return
        n = len(data)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(splat_io.xyz(data))
        pcd.colors = o3d.utility.Vector3dVector(splat_io.display_colours(data))
        idx = np.zeros((n, 3))
        idx[:, 0] = np.arange(n, dtype=np.float64)   # gaussian index rides in normals
        pcd.normals = o3d.utility.Vector3dVector(idx)
        self.is_splat = True
        self.splat = data
        self.splat_fields = fields
        self.build_btn.text = "Save Edited Splat"
        self._finish_load(pcd, path, cap)
        self.build_status.text = f"splat: {n:,} gaussians, {len(fields)} fields — trim & Save"

    def _set_cloud(self, path, cap=None):
        try:
            pcd = ops.load(path)
        except Exception as exc:  # noqa: BLE001
            self._message(f"Could not load:\n{path}\n{exc}")
            return
        self.is_splat = False
        self.splat = None
        self.splat_fields = None
        self.build_btn.text = "Save + Build Splat"
        self._finish_load(pcd, path, cap)

    def _finish_load(self, pcd, path, cap):
        self.cap = cap
        self.photo_cap = None   # re-resolved per cloud (see _photo_capture)
        self.path = path
        self.pcd = pcd
        self.original = ops.clone(pcd)
        self.undo_stack = []
        self.preview_keep = None
        self.preview_row.visible = False
        self.select_mode = False
        self.sel_btn.text = "Box select: OFF"
        self.diag = float(np.linalg.norm(self.original.get_max_bound() - self.original.get_min_bound())) or 1.0
        lo = self.original.get_min_bound()
        hi = self.original.get_max_bound()
        for i, ax in enumerate("xyz"):
            self.box_sliders[(ax, "min")].double_value = float(lo[i])
            self.box_sliders[(ax, "max")].double_value = float(hi[i])
        self._refresh(reset_camera=True)
        self.window.set_needs_layout()

    # ---- rendering ----------------------------------------------------------
    def _material(self):
        m = rendering.MaterialRecord()
        m.shader = "defaultUnlit"
        m.point_size = self.point_size
        return m

    def _refresh(self, reset_camera=False):
        s = self.scene.scene
        s.clear_geometry()
        self._rect_shown = False
        if self.pcd is None:
            self._update_status()
            return
        if self.preview_keep is not None:
            keep_idx = np.nonzero(self.preview_keep)[0]
            rem_idx = np.nonzero(~self.preview_keep)[0]
            s.add_geometry(CLOUD, ops.select(self.pcd, keep_idx), self._material())
            if len(rem_idx):
                rem = ops.select(self.pcd, rem_idx)
                rem.paint_uniform_color([1.0, 0.15, 0.15])
                s.add_geometry(REMOVED, rem, self._material())
        else:
            s.add_geometry(CLOUD, self.pcd, self._material())
        if self.show_box:
            self._draw_box()
        if reset_camera:
            b = self.pcd.get_axis_aligned_bounding_box()
            self.scene.setup_camera(self.fov, b, b.get_center())
            if self.ortho:
                self._apply_ortho()
        self._update_status()

    def _update_status(self):
        if self.pcd is None:
            self.status.text = "No cloud loaded.\nUse “Load folder…” (a capture)\nor “Load file…” (a .ply)."
            return
        txt = ops.info(self.pcd)
        if self.preview_keep is not None:
            txt = f"PREVIEW: would remove {int((~self.preview_keep).sum()):,}\n" + txt
        elif self.select_mode:
            txt = "BOX SELECT: drag a rectangle\n" + txt
        self.status.text = txt

    # ---- preview model ------------------------------------------------------
    def _set_preview(self, keep_mask: np.ndarray):
        self.preview_keep = keep_mask.astype(bool)
        self.preview_row.visible = True
        self._refresh()
        self.window.set_needs_layout()

    def _apply_preview(self):
        if self.preview_keep is None:
            return
        keep = np.nonzero(self.preview_keep)[0]
        if len(keep) == 0:
            self._message("That would remove all points; cancelled.")
            self._cancel_preview()
            return
        self.undo_stack.append(self.pcd)
        self.pcd = ops.select(self.pcd, keep)
        self.preview_keep = None
        self.preview_row.visible = False
        self._refresh()
        self.window.set_needs_layout()

    def _cancel_preview(self):
        self.preview_keep = None
        self.preview_row.visible = False
        self._refresh()
        self.window.set_needs_layout()

    # ---- denoise / auto-clean ----------------------------------------------
    def _keep_from_indices(self, kept_idx):
        mask = np.zeros(len(self.pcd.points), bool)
        mask[np.asarray(kept_idx, int)] = True
        return mask

    def _preview_sor(self):
        if self.pcd is None:
            return
        kept = ops.statistical_outlier_keep(self.pcd, self.so_nb.int_value, self.so_std.double_value)
        self._set_preview(self._keep_from_indices(kept))

    def _preview_radius(self):
        if self.pcd is None:
            return
        kept = ops.radius_outlier_keep(self.pcd, self.ro_nb.int_value, self.ro_r.double_value)
        self._set_preview(self._keep_from_indices(kept))

    def _preview_decimate(self):
        if self.pcd is None:
            return
        factor = max(2, self.dec_factor.int_value)
        keep = np.zeros(len(self.pcd.points), bool)
        keep[::factor] = True   # keep every Nth point
        self._set_preview(keep)

    def _preview_autoclean(self):
        if self.pcd is None:
            return
        pct = self.ac_pct.double_value
        mask = ops.auto_clean_mask(
            self.pcd, sor_std=self.ac_std.double_value,
            crop_low=pct, crop_high=100.0 - pct, radius=self.ac_r.double_value,
        )
        self._set_preview(mask)

    # ---- box crop -----------------------------------------------------------
    def _box_bounds(self):
        lo = np.array([self.box_sliders[(ax, "min")].double_value for ax in "xyz"])
        hi = np.array([self.box_sliders[(ax, "max")].double_value for ax in "xyz"])
        return lo, hi

    def _on_show_box(self, checked):
        self.show_box = checked
        self._refresh()

    def _on_box_edit(self):
        if self.show_box:
            self._refresh()

    def _draw_box(self):
        lo, hi = self._box_bounds()
        aabb = o3d.geometry.AxisAlignedBoundingBox(lo, hi)
        ls = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb)
        ls.paint_uniform_color([1.0, 0.85, 0.1])
        m = rendering.MaterialRecord()
        m.shader = "unlitLine"
        m.line_width = 2.0
        self.scene.scene.add_geometry(CROPBOX, ls, m)

    def _box_apply(self, keep_inside: bool):
        if self.pcd is None:
            return
        lo, hi = self._box_bounds()
        inside = ops.aabb_keep(self.pcd, lo, hi)
        mask = self._keep_from_indices(inside)
        self._set_preview(mask if keep_inside else ~mask)

    # ---- in-scene rectangle selection --------------------------------------
    def _toggle_select(self):
        if self.pcd is None:
            return
        self.select_mode = not self.select_mode
        self.sel_btn.text = f"Box select: {'ON' if self.select_mode else 'OFF'}"
        if not self.select_mode:
            self._hide_rect()
        self._update_status()

    def _on_mouse(self, e):
        CR = gui.Widget.EventCallbackResult
        # Orthographic wheel zoom: dolly does nothing in ortho, so rescale frustum.
        if self.ortho and e.type == gui.MouseEvent.Type.WHEEL:
            step = 0.9 if e.wheel_dy > 0 else 1.0 / 0.9
            self.ortho_half_h = max(self.diag * 1e-3, (self.ortho_half_h or self.diag) * step)
            self._apply_ortho()
            return CR.CONSUMED

        if not self.select_mode:
            return CR.IGNORED
        if e.type == gui.MouseEvent.Type.BUTTON_DOWN:
            self.sel_start = (e.x, e.y)
            return CR.CONSUMED
        if e.type == gui.MouseEvent.Type.DRAG and self.sel_start:
            self._draw_rect(self.sel_start, (e.x, e.y))    # cheap: only an outline
            return CR.CONSUMED
        if e.type == gui.MouseEvent.Type.BUTTON_UP and self.sel_start:
            start = self.sel_start
            self.sel_start = None
            self._finish_select(start, (e.x, e.y))         # heavy work once, here
            return CR.CONSUMED
        return CR.IGNORED

    def _project(self, pts):
        cam = self.scene.scene.camera
        V = np.asarray(cam.get_view_matrix(), float)
        P = np.asarray(cam.get_projection_matrix(), float)
        fr = self.scene.frame
        homog = np.hstack([pts, np.ones((len(pts), 1))])
        clip = (P @ V @ homog.T).T
        w = clip[:, 3]
        valid = np.abs(w) > 1e-9
        ndc = clip[:, :3] / np.where(valid, w, 1.0)[:, None]
        sx = (ndc[:, 0] * 0.5 + 0.5) * fr.width + fr.x
        sy = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * fr.height + fr.y
        return sx, sy, valid & (ndc[:, 2] > -1.0) & (ndc[:, 2] < 1.0)

    def _draw_rect(self, p0, p1):
        """Draw just the selection rectangle outline (no cloud work) during drag.

        The four screen corners are unprojected to a plane very close to the
        camera, so the loop renders in front of the cloud and tracks the drag.
        Only 4 line segments are uploaded, so it stays responsive on big clouds.
        """
        now = time.monotonic()
        if now - self._last_sel_draw < 0.016:   # cap at ~60 fps
            return
        self._last_sel_draw = now
        (x0, y0), (x1, y1) = p0, p1
        if abs(x1 - x0) < 2 or abs(y1 - y0) < 2:   # skip degenerate rectangles
            return
        fr = self.scene.frame
        W, H = max(float(fr.width), 1.0), max(float(fr.height), 1.0)
        cam = self.scene.scene.camera
        # Depth must sit just past the near plane and in front of the cloud.
        # Perspective depth is non-linear (z=0.5 is still very near the camera);
        # ortho is linear, so a small z keeps the outline in front.
        z = 0.05 if self.ortho else 0.5

        def corner(x, y):
            vx = min(max(x - fr.x, 0), W)
            vy = min(max(y - fr.y, 0), H)
            return np.asarray(cam.unproject(float(vx), float(vy), z, W, H), float)

        pts = [corner(x0, y0), corner(x1, y0), corner(x1, y1), corner(x0, y1)]
        if not all(np.all(np.isfinite(c)) for c in pts):
            return
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(np.asarray(pts))
        ls.lines = o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 3], [3, 0]])
        ls.paint_uniform_color([1.0, 0.9, 0.2])
        m = rendering.MaterialRecord()
        m.shader = "unlitLine"
        m.line_width = 2.0
        s = self.scene.scene
        if s.has_geometry(RECT):
            s.remove_geometry(RECT)
        s.add_geometry(RECT, ls, m)
        self._rect_shown = True
        self.status.text = f"BOX SELECT: drag…  ({'remove' if self.rect_remove else 'keep'} inside)"

    def _hide_rect(self):
        if self._rect_shown and self.scene.scene.has_geometry(RECT):
            self.scene.scene.remove_geometry(RECT)
        self._rect_shown = False

    def _finish_select(self, p0, p1):
        """On mouse-up: project the cloud once and build the keep/remove preview.

        A click (no real drag) clears any active selection/preview.
        """
        self._hide_rect()
        (x0, y0), (x1, y1) = p0, p1
        xmin, xmax = sorted((x0, x1))
        ymin, ymax = sorted((y0, y1))
        if xmax - xmin < 3 or ymax - ymin < 3:
            self._cancel_preview()   # click away to unselect
            return
        sx, sy, valid = self._project(np.asarray(self.pcd.points))
        inside = valid & (sx >= xmin) & (sx <= xmax) & (sy >= ymin) & (sy <= ymax)
        if inside.sum() == 0:
            self._message("No points in selection.")
            return
        self.preview_keep = ~inside if self.rect_remove else inside
        self.preview_row.visible = True
        self._refresh()
        self.window.set_needs_layout()

    # ---- display helpers ----------------------------------------------------
    def _on_point_size(self, v):
        self.point_size = float(v)
        self._refresh()

    def _view_preset(self, name):
        if self.pcd is None:
            return
        b = self.pcd.get_axis_aligned_bounding_box()
        c = b.get_center()
        ext = np.linalg.norm(b.get_extent()) + 1e-6
        d = ext * 1.2
        eye = {
            "Top": c + [0, 0, d],
            "Front": c + [0, -d, 0],
            "Side": c + [d, 0, 0],
            "Iso": c + [d, -d, d] / np.sqrt(3),
        }[name]
        up = [0, 1, 0] if name == "Top" else [0, 0, 1]
        self.scene.scene.camera.look_at(c, eye, up)
        if self.ortho:
            self._apply_ortho()

    # ---- projection ---------------------------------------------------------
    def _cam_distance(self):
        cam = self.scene.scene.camera
        eye = np.asarray(cam.get_model_matrix(), float)[:3, 3]
        center = self.pcd.get_axis_aligned_bounding_box().get_center()
        return float(np.linalg.norm(eye - center))

    def _toggle_ortho(self, on):
        if self.pcd is None:
            self.ortho_chk.checked = False
            return
        self.ortho = bool(on)
        if self.ortho:
            # Match the current perspective framing at the scene-centre plane.
            d = max(self._cam_distance(), 1e-3)
            self.ortho_half_h = d * np.tan(np.radians(self.fov / 2))
            self._apply_ortho()
        else:
            cam = self.scene.scene.camera
            fr = self.scene.frame
            aspect = max(fr.width, 1) / max(fr.height, 1)
            d = max(self._cam_distance(), 1e-3)
            near = max(0.01, d - self.diag)
            far = d + 2 * self.diag
            cam.set_projection(self.fov, aspect, near, far, rendering.Camera.FovType.Vertical)
        self.scene.force_redraw()

    def _apply_ortho(self):
        cam = self.scene.scene.camera
        fr = self.scene.frame
        aspect = max(fr.width, 1) / max(fr.height, 1)
        hh = self.ortho_half_h or self.diag * 0.6
        hw = hh * aspect
        d = max(self._cam_distance(), 1e-3)
        near = max(0.001, d - 2 * self.diag)
        far = d + 2 * self.diag
        cam.set_projection(rendering.Camera.Projection.Ortho, -hw, hw, -hh, hh, near, far)

    # ---- recolour from photos (independent of splatting) -------------------
    @staticmethod
    def _has_photos(cap) -> bool:
        try:
            cap.image_bag()
        except FileNotFoundError:
            return False
        return cap.calib().exists()

    def _photo_capture(self):
        """Capture holding the photos/poses for the current cloud, or None.

        Prefers the folder the cloud came from; falls back to the one Recolour
        resolved (``self.photo_cap``) so Build Splat finds the camera data in
        the same place Recolour did, even across folders.
        """
        for c in (self.cap, self.photo_cap):
            if c is not None and (c.colmap() is not None or self._has_photos(c)):
                return c
        return None

    def _recolor(self):
        if self.build_proc and self.build_proc.poll() is None:
            self._stop_build()
            return
        if self.pcd is None:
            return
        if self.is_splat:
            self._message("Recolour isn't available for a splat —\nits colour is baked into each gaussian.")
            return
        if self.cap is None:
            self._message("Recolour needs a folder.\nUse “Load folder…”.")
            return
        self._start_recolor(self.cap)

    def _pick_capture(self, cont):
        dlg = gui.FileDialog(gui.FileDialog.OPEN_DIR,
                             "Capture folder with photos (IMAGE_*.bag)", self.window.theme)
        dlg.set_on_cancel(self.window.close_dialog)

        def done(path):
            self.window.close_dialog()
            cont(Capture(path, project=self.project))

        dlg.set_on_done(done)
        self.window.show_dialog(dlg)

    def _start_recolor(self, pcap):
        """Colour the current cloud using ``pcap``'s photos.

        If a COLMAP result already exists (our pipeline *or* a JMStudio project's
        ``shading/Colmap``), colour straight away. Otherwise, if the folder has
        the raw photos, compute poses first; if it has neither, ask for the
        capture folder that does.
        """
        if pcap.colmap() is not None:
            self.photo_cap = pcap          # remember for Build Splat
            self.recolor_btn.text = "Recolouring…"
            threading.Thread(target=self._recolor_worker, args=(pcap,), daemon=True).start()
            return
        if not self._has_photos(pcap):
            self._message(
                "This folder has no camera poses and no photos\n"
                "(IMAGE_*.bag). Pick the capture folder that has\n"
                "the photos for this scan."
            )
            self._pick_capture(self._start_recolor)
            return
        self.photo_cap = pcap              # remember for Build Splat

        # Compute poses from the photos, aligning *this* cloud, then colour.
        data = str(pcap.root)
        steps = []
        if not (pcap.p("images").exists() and any(pcap.p("images").glob("*.jpg"))):
            steps.append([sys.executable, "-m", "raven.extract", "--data", data,
                          "--images", "--rotate", "--long-edge", "1600", "--stride", "2"])
        steps.append([sys.executable, "-m", "raven.colmap_pipeline", "--data", data, "--device", "cpu"])
        cloud_for_align = pcap.ensure_work() / "recolor_cloud.ply"
        ops.save(self.pcd, str(cloud_for_align))
        steps.append([sys.executable, "-m", "raven.align", "--data", data, "--cloud", str(cloud_for_align)])

        self.recolor_btn.text = "Stop"
        self.build_status.text = "colourising: SfM + align, then recolour…"
        threading.Thread(
            target=self._run_steps, args=(steps, dict(os.environ)),
            kwargs={"on_success": lambda: self._recolor_worker(pcap)}, daemon=True,
        ).start()

    def _recolor_worker(self, pcap):
        """Sample ``pcap``'s photo colours onto the current cloud (off the UI thread)."""
        import json as _json

        from raven import recolor as rc

        model, images, integrated = pcap.colmap()
        tform = pcap.p("aligned", "transform.json")
        if tform.exists():
            T = np.asarray(_json.loads(tform.read_text())["transform_lidar_to_colmap"], float).reshape(4, 4)
        else:
            T = np.eye(4)   # JMStudio: cloud already shares the pose frame
        self._post_status("recolouring points…")
        self._post_progress(0.0)
        # Occlusion uses the full, unedited cloud so decimating/trimming the
        # coloured subset doesn't poke holes in the z-buffer (which would let
        # photos colour points hidden behind surfaces).
        occ = np.asarray(self.original.points) if self.original is not None else None
        colors, seen, n_imgs = rc.sample_colors(
            np.asarray(self.pcd.points), T, model, images, occ_points=occ,
            progress=lambda done, total: self._post_progress(done / total),
        )

        def apply():
            self.undo_stack.append(ops.clone(self.pcd))
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
            self._refresh()
            self._reset_buttons()
            self.build_status.text = ""
            self._message(f"Recoloured {int(seen.sum()):,}/{len(seen):,} points\nfrom {n_imgs} photos.")

        self.app.post_to_main_thread(self.window, apply)

    # ---- edit / output ------------------------------------------------------
    def _undo(self):
        if self.undo_stack:
            self.pcd = self.undo_stack.pop()
            self._cancel_preview()

    def _reset(self):
        if self.original is None:
            return
        self.undo_stack.append(self.pcd)
        self.pcd = ops.clone(self.original)
        self._cancel_preview()
        self._refresh(reset_camera=True)

    def _message(self, text):
        dlg = gui.Dialog("Notice")
        em = self.em
        v = gui.Vert(em, gui.Margins(em, em, em, em))
        v.add_child(gui.Label(text))
        ok = gui.Button("OK")
        ok.set_on_clicked(self.window.close_dialog)
        v.add_child(ok)
        dlg.add_child(v)
        self.window.show_dialog(dlg)

    def _confirm(self, text, on_yes, yes_label="OK"):
        """Two-button modal: run ``on_yes`` if the user accepts, else dismiss."""
        dlg = gui.Dialog("Confirm")
        em = self.em
        v = gui.Vert(em, gui.Margins(em, em, em, em))
        v.add_child(gui.Label(text))
        row = gui.Horiz(0.5 * em)
        yes = gui.Button(yes_label)

        def accept():
            self.window.close_dialog()
            on_yes()

        yes.set_on_clicked(accept)
        no = gui.Button("Cancel")
        no.set_on_clicked(self.window.close_dialog)
        row.add_child(yes)
        row.add_child(no)
        v.add_child(row)
        dlg.add_child(v)
        self.window.show_dialog(dlg)

    @staticmethod
    def _looks_uncoloured(pcd) -> bool:
        """True if the cloud carries no real photo colour — either no colours at
        all, a single flat colour, or the fused thumbnail's false-colour height
        ramp (red channel constant 0). Such a cloud makes a poor splat init."""
        if pcd is None or not pcd.has_colors():
            return True
        c = np.asarray(pcd.colors)
        if len(c) == 0:
            return True
        return float(c[:, 0].max()) < 0.02 or float(c.std()) < 1e-3

    def _surviving_indices(self):
        """Original gaussian indices still present, read from the carried normals."""
        return np.asarray(self.pcd.normals)[:, 0].astype(np.int64)

    def _save_dialog(self):
        if self.pcd is None:
            return
        title = "Save edited splat" if self.is_splat else "Save point cloud"
        dlg = gui.FileDialog(gui.FileDialog.SAVE, title, self.window.theme)
        dlg.add_filter(".ply", "Gaussian splat (.ply)" if self.is_splat else "Point cloud (.ply)")
        start = str(self.cap.ensure_work()) if self.cap else str(Path(self.path).parent)
        dlg.set_path(start)
        dlg.set_on_cancel(self.window.close_dialog)

        def on_done(path):
            self.window.close_dialog()
            if not path.endswith(".ply"):
                path += ".ply"
            if self.is_splat:
                n = splat_io.save_splat(path, self.splat, self._surviving_indices())
                self._message(f"Saved edited splat: {n:,} gaussians\n(of {len(self.splat):,}) to\n{path}")
            else:
                ops.save(self.pcd, path)
                self._message(f"Saved {len(self.pcd.points):,} points to\n{path}")

        dlg.set_on_done(on_done)
        self.window.show_dialog(dlg)

    # ---- build handoff ------------------------------------------------------
    def _build_splat(self, skip_colour_check=False):
        if self.build_proc and self.build_proc.poll() is None:
            self._stop_build()
            return
        if self.pcd is None:
            return
        # Editing a finished splat: "build" just writes the trimmed splat back out.
        if self.is_splat:
            out = (self.cap.ensure_work() / "splat_edited.ply") if self.cap \
                else Path(self.path).with_name("splat_edited.ply")
            n = splat_io.save_splat(out, self.splat, self._surviving_indices())
            self._message(f"Saved edited splat: {n:,} gaussians\n(of {len(self.splat):,}) to\n{out}")
            return
        if self.cap is None:
            self._message("Building splats needs a capture folder\n(the photos + calibration).\nUse “Load folder…”.")
            return
        # A flat / false-colour init makes gsplat start every gaussian the wrong
        # colour; rarely-seen gaussians never recover, so the splat looks washed
        # out. Steer the user to Recolour first rather than build silently.
        if not skip_colour_check and self._looks_uncoloured(self.pcd):
            self._confirm(
                "This cloud isn't colourised — the splat will init from\n"
                "flat colour and look poor (rarely-seen points stay wrong).\n"
                "Recolour from photos first for best results.\n\nBuild anyway?",
                lambda: self._build_splat(skip_colour_check=True),
                yes_label="Build anyway",
            )
            return
        # Surfel mode converts the coloured cloud straight to disk gaussians —
        # no poses/COLMAP/training needed, so skip the photo-capture resolution.
        if self.surfel_chk.checked:
            self._build_surfel()
            return
        pcap = self._photo_capture()
        if pcap is None:
            self._message(
                "This cloud has no camera poses and no photos (IMAGE_*.bag)\n"
                "in its folder. Pick the capture folder that has the photos\n"
                "(or COLMAP poses) for this scan."
            )
            self._pick_capture(self._build_splat_with)
            return
        self._build_splat_with(pcap)

    def _build_surfel(self):
        """Fast no-training export: the coloured cloud -> disk gaussians (CPU)."""
        cap = self.cap
        edited = cap.ensure_work() / "cloud_edited.ply"
        ops.save(self.pcd, str(edited))
        out = str(cap.p("splat.ply"))
        steps = [[sys.executable, "-m", "raven.train_splat", "--surfel",
                  "--data", str(cap.root), "--lidar", str(edited), "--out", out]]
        self.build_btn.text = "Stop Build"
        self.build_status.text = "surfel export (no training)…"
        self.build_progress.value = 0.0
        threading.Thread(
            target=self._run_steps, args=(steps, dict(os.environ)),
            kwargs={"done_msg": f"surfel → {out}"}, daemon=True,
        ).start()

    def _build_splat_with(self, pcap):
        """Build splats for the current cloud using ``pcap``'s photos/poses.

        ``pcap`` may be a different folder from the cloud's (``self.cap``); the
        edited cloud and the output splat stay with the cloud, while photos,
        COLMAP model and alignment come from ``pcap``.
        """
        if not (pcap.colmap() is not None or self._has_photos(pcap)):
            self._message("That folder has no COLMAP poses and no photos\n(IMAGE_*.bag); can't build splats from it.")
            return
        self.photo_cap = pcap
        cap = self.cap
        pdata = str(pcap.root)
        iters = self.build_iters.int_value
        edited = cap.ensure_work() / "cloud_edited.ply"
        ops.save(self.pcd, str(edited))
        out = str(cap.p("splat.ply"))
        aligned_cloud = str(pcap.p("aligned", "cloud_in_colmap.ply"))

        existing = pcap.colmap()   # our pipeline output OR a JMStudio project's poses
        steps = []
        if existing is not None:
            model, images, integrated = existing
            train = [sys.executable, "-m", "raven.train_splat", "--data", pdata,
                     "--model", str(model), "--images", str(images),
                     "--init", "lidar", "--iters", str(iters), "--out", out]
            # "integrated" (JMStudio) poses already share the cloud's frame, so
            # init straight from the edited cloud — and align *can't* run on
            # those text models anyway (no points3D for pycolmap). Only our own
            # reconstructions (which carry points3D) get a lidar->COLMAP align.
            if integrated:
                train += ["--lidar", str(edited)]
                note = "train (existing poses)"
            else:
                steps.append([sys.executable, "-m", "raven.align", "--data", pdata,
                              "--cloud", str(edited), "--model", str(model),
                              "--out", str(pcap.p("aligned"))])
                train += ["--lidar", aligned_cloud]
                note = "align + train"
            steps.append(train)
        else:
            # No poses yet — need the raw capture to compute them.
            if not pcap.calib().exists():
                self._message("Missing calibration/calib.json in the photo capture.")
                return
            if not (pcap.p("images").exists() and any(pcap.p("images").glob("*.jpg"))):
                steps.append([sys.executable, "-m", "raven.extract", "--data", pdata,
                              "--images", "--rotate", "--long-edge", "1600", "--stride", "2"])
            steps.append([sys.executable, "-m", "raven.colmap_pipeline", "--data", pdata, "--device", "cpu"])
            steps.append([sys.executable, "-m", "raven.align", "--data", pdata, "--cloud", str(edited)])
            steps.append([sys.executable, "-m", "raven.train_splat", "--data", pdata,
                          "--init", "lidar", "--iters", str(iters),
                          "--lidar", aligned_cloud, "--out", out])
            note = "SfM (slow) + align + train"

        env = dict(os.environ)
        cuda = os.path.expanduser("~/.local/micromamba/envs/cuda121")
        if os.path.exists(os.path.join(cuda, "bin", "nvcc")):
            env["CUDA_HOME"] = cuda
            env["PATH"] = os.path.join(cuda, "bin") + os.pathsep + env.get("PATH", "")

        self.build_btn.text = "Stop Build"
        self.build_status.text = f"building: {note} ({iters} it)…"
        self.build_progress.value = 0.0
        threading.Thread(
            target=self._run_steps, args=(steps, env),
            kwargs={"done_msg": f"done → {self.cap.p('splat.ply')}"}, daemon=True,
        ).start()

    def _open_in_supersplat(self):
        """Serve the current/just-built splat and open it in the SuperSplat editor."""
        # Prefer a splat loaded for editing; otherwise the capture's built splat.
        if self.is_splat and self.path:
            ply = Path(self.path)
        elif self.cap is not None:
            ply = self.cap.p("splat.ply")
        else:
            self._message("Load a capture or splat first.")
            return
        if not ply.exists():
            self._message("No splat to view yet — build one first\n(Save + Build Splat).")
            return

        from raven.view_splat import open_in_supersplat

        # Replace any previous server so opens don't leak ports.
        if self._supersplat_httpd is not None:
            self._supersplat_httpd.shutdown()
            self._supersplat_httpd = None
        try:
            self._supersplat_httpd, _url = open_in_supersplat(ply)
        except Exception as exc:  # noqa: BLE001
            self._message(f"Couldn't open SuperSplat:\n{exc}")
            return
        self._post_status(f"opened {ply.name} in SuperSplat (browser)")

    def _run_steps(self, steps, env, done_msg=None, on_success=None):
        log = open(self.cap.ensure_work() / "build.log", "w")
        try:
            for cmd in steps:
                # Every step is a raven.* command, so route its outputs to the
                # project folder too (no-op when no project is set).
                if self.project and "--project" not in cmd:
                    cmd = cmd + ["--project", self.project]
                self._post_status(f"▶ {cmd[2].split('.')[-1]}")
                self.build_proc = subprocess.Popen(
                    cmd, cwd=str(CODE_ROOT), env=env, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1,
                )
                for line in self.build_proc.stdout:
                    log.write(line)
                    log.flush()
                    s = line.strip()
                    if not s:
                        continue
                    m = re.match(r"PROGRESS (\d+)/(\d+)", s)
                    if m:
                        done, total = int(m.group(1)), int(m.group(2))
                        if total:
                            self._post_progress(done / total)
                    else:
                        self._post_status(s[:60])
                rc = self.build_proc.wait()
                if rc != 0:
                    self._post_status(f"step failed (rc={rc}); see work/build.log")
                    self._reset_buttons()
                    return
            self.build_proc = None
            if on_success is not None:
                on_success()              # e.g. in-process recolour
            else:
                self._post_status(done_msg or "done")
                self._reset_buttons()
        finally:
            log.close()
            self.build_proc = None

    def _stop_build(self):
        if self.build_proc and self.build_proc.poll() is None:
            self.build_proc.terminate()
        self._post_status("stopped")
        self._reset_buttons()

    def _post_status(self, text):
        self.app.post_to_main_thread(self.window, lambda: setattr(self.build_status, "text", text or ""))

    def _post_progress(self, frac):
        frac = max(0.0, min(1.0, float(frac)))
        self.app.post_to_main_thread(self.window, lambda: setattr(self.build_progress, "value", frac))

    def _reset_buttons(self):
        def fn():
            self.build_btn.text = "Save + Build Splat"
            self.recolor_btn.text = "Recolour from photos"
            self.build_progress.value = 0.0
        self.app.post_to_main_thread(self.window, fn)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data", nargs="?", default=None,
                    help="optional capture folder or .ply to open at startup "
                         "(otherwise use the Load buttons)")
    ap.add_argument("--cloud", default=None, help="explicit .ply to edit")
    ap.add_argument("--project", default=None,
                    help="output project folder; artefacts go to <project>/<scan-name>/ "
                         "instead of <scan>/work (remembered between sessions)")
    args = ap.parse_args()

    # CLI --project wins; otherwise reuse the last folder chosen in the GUI.
    project = args.project or _load_config().get("project")
    project = str(Path(project).expanduser()) if project else None

    cap = None
    cloud = None
    if args.data:
        # `data` may be a capture folder or, for convenience, a .ply file.
        p = Path(args.data)
        if p.is_file() and p.suffix == ".ply":
            base = p.parent.parent if p.parent.name == "work" else p.parent
            cap = Capture(base, project=project)
            cloud = str(p)
        else:
            cap = Capture(args.data, project=project)
            cloud = args.cloud or str(cap.staged_cloud())
    elif args.cloud:
        cloud = args.cloud
        cap = Capture(Path(cloud).parent, project=project)

    gui.Application.instance.initialize()
    EditorApp(cap, cloud, project=project)
    gui.Application.instance.run()


if __name__ == "__main__":
    sys.exit(main())
