"""Headless smoke: drive the real EditorApp splat load + edit + save path."""
import sys, numpy as np
import open3d.visualization.gui as gui
from raven.cloud_editor import EditorApp
from raven import splat_io, cloud_ops as ops

SPLAT = "/FastDrive/Dropbox/LIDAR/data/20260527212832/I16_With_Gaussian_Splat.ply"
OUT = "/tmp/splat_gui_smoke_out.ply"

gui.Application.instance.initialize()
app = EditorApp(path=SPLAT)
assert app.is_splat, "splat not detected"
n0 = len(app.splat)
print(f"loaded splat: {n0:,} gaussians, {len(app.splat_fields)} fields")
print(f"pcd points={len(app.pcd.points):,}  index-stash present={app.pcd.has_normals()}")

# Apply a real edit through the same machinery the GUI uses: keep 1-in-4.
kept = app.pcd.select_by_index(list(range(0, len(app.pcd.points), 4)))
app.pcd = kept
idx = app._surviving_indices()
print(f"after decimate: {len(idx):,} surviving indices  (max={idx.max()}, <n0={idx.max() < n0})")

n = splat_io.save_splat(OUT, app.splat, idx)
data, fields = splat_io.load_splat(OUT)
assert fields == app.splat_fields, "field layout changed!"
assert len(data) == n == len(idx)
print(f"saved {n:,} gaussians, fields preserved exactly -> {OUT}")
print("SMOKE OK")
