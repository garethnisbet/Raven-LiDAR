# Raven LiDAR → Point-Cloud Editor + Gaussian Splats

Tools to turn a **3DMakerpro Raven (JMK7)** handheld capture into an editable
point cloud and a trained **3D Gaussian Splat**.

The capture (`LIDAR_*.bag`, `IMAGE_*.bag`, `calibration/`, `thumbnail/`) holds
LiDAR scans + IMU, 922 12 MP fisheye photos, calibration, and JMStudio's already
SLAM-fused colored cloud. The bags contain **no camera poses**, so poses are
recovered with COLMAP and the splats are trained from the posed photos.

## Pipeline

```
extract ──▶ cloud_editor ──▶ colmap_pipeline ──▶ (align) ──▶ train_splat
 images        edit PLY          SfM poses        lidar init     splat.ply
 + PLY
```

## Install

**No-root install to `/scratch` (HPC / full home disk).** One self-contained
script provisions Python, the whole pip stack, torch+gsplat, and an nvcc toolkit
under a single prefix — nothing touches `$HOME` or the system:

```bash
scripts/install_scratch.sh                       # -> /scratch/$USER/raven-env
# or:  scripts/install_scratch.sh /scratch/team/raven-env
source /scratch/$USER/raven-env/activate.sh      # each new shell
```

pip/conda caches and `TMPDIR` are redirected into the prefix too; `rm -rf` the
prefix to uninstall. The manual steps below do the same thing piecemeal:

```bash
pip install -r requirements.txt          # numpy, open3d, opencv, rosbags, pycolmap, ...

# Gaussian splatting (CUDA GPU required):
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install gsplat
```

`pycolmap` (pip) ships a **CPU-only** SIFT build — feature extraction/matching
run on CPU (fine for a few hundred frames). For GPU SfM install a CUDA COLMAP
(`sudo apt install colmap` on a CUDA box, or a CUDA-enabled `pycolmap`).

**`nvcc` for gsplat.** gsplat JIT-compiles its CUDA kernels on first use and
needs a CUDA toolkit (`nvcc`) matching torch's CUDA (12.1). If you lack one and
can't `apt install nvidia-cuda-toolkit`, install it without root via micromamba —
`raven.train_splat` auto-detects this env:

```bash
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba
MAMBA_ROOT_PREFIX=~/.local/micromamba ./bin/micromamba create -y \
  -p ~/.local/micromamba/envs/cuda121 -c nvidia/label/cuda-12.1.1 -c conda-forge \
  cuda-nvcc cuda-cudart-dev cuda-cccl libcusparse-dev libcublas-dev
# train_splat finds ~/.local/micromamba/envs/cuda121 automatically;
# otherwise export CUDA_HOME=/path/to/cuda before training.
```

## Usage

Run the commands from the **code root** (where the `raven/` package lives) and
point each at a **capture folder** with `--data`. A capture is a Raven export
holding the `.bag` files, `calibration/`, and `thumbnail/`. Every output is
written to `<capture>/work/`, so each scan keeps its own results.

```bash
DATA=/FastDrive/Dropbox/LIDAR/data/I16BeamlineScan

# 1. Extract frames (rotate to the calibrated portrait orientation) + stage cloud.
python -m raven.extract --data "$DATA" --images --cloud --rotate --long-edge 1600 --stride 2

# 2. Edit the point cloud (Open3D GUI). Launch with no args and use the Load
#    buttons, or pass a capture folder to open it immediately.
python -m raven.cloud_editor                  # then "Load folder…" / "Load file…"
python -m raven.cloud_editor "$DATA"          # or open the capture at startup
#    - Load folder: opens a capture (auto-stages the fused cloud); Load file: any .ply.
#    - Box select: toggle on, left-drag a rectangle (selection highlights red
#      live); "selection removes inside" flips crop-to-box vs delete-contents.
#    - Orthographic toggle + Top/Front/Side presets for clean axis-aligned crops.
#    - Crop box: sliders define a 3D box (drawn live); Keep box / Remove box.
#    - Denoise: Preview SOR/radius outliers (removed shown red) then Apply/Cancel.
#    - Auto-clean: one click = SOR + percentile crop + radius (previewed).
#    - "Save + Build Splat": writes cloud_edited.ply and runs align (+ SfM if
#      needed) and gsplat training in the background, streaming progress.
# Or headless auto-clean (no GUI):
python -m raven.cloud_ops "$DATA"/work/cloud_raw.ply "$DATA"/work/cloud_edited.ply

# 3. Structure-from-Motion (sequential matcher, fisheye model, CPU SIFT).
python -m raven.colmap_pipeline --data "$DATA" --device cpu

# 4. (optional) Align the lidar cloud into the COLMAP frame for a denser init.
python -m raven.align --data "$DATA"

# 4b. (optional) Recolour the cloud with TRUE RGB from the photos. The fused
#     cloud ships only a false-colour height ramp (red channel is constant 0);
#     this projects the undistorted photos onto the points (z-buffer occlusion).
#     INDEPENDENT of splatting: it needs camera poses (steps 3 + 4) but never
#     trains splats. The editor's "Recolour from photos" button runs steps
#     3 + 4 automatically if needed. If the cloud's folder has no photos (e.g. a
#     JMStudio project folder), the button asks you to pick the capture folder
#     that does (the .bag files) — same scan, different folder — and colours the
#     current cloud with those photos.
python -m raven.recolor --data "$DATA"

# 5. Train splats and export a standard 3DGS .ply.
python -m raven.train_splat --data "$DATA" --iters 30000               # COLMAP init
python -m raven.train_splat --data "$DATA" --iters 30000 --init lidar  # lidar init
python -m raven.train_splat --data "$DATA" --iters 30000 --view        # + open in SuperSplat

# Or the whole thing at once:
scripts/run_pipeline.sh "$DATA"
```

Outputs land in `<capture>/work/`: `images/`, `colmap/`, `aligned/`,
`cloud_edited.ply`, `cloud_colored.ply`, `splat.ply`.

**Existing poses are reused automatically.** If a folder already has camera poses
— our own `work/colmap/…` output, or a **JMStudio project** with
`shading/Colmap/` (text `cameras.txt`/`images.txt` + undistorted images) — recolour
and Build Splat detect them and **skip extraction/COLMAP/align entirely**,
colouring or training straight from the existing photos in seconds. A JMStudio
project's cloud already shares the pose frame, so no lidar→COLMAP alignment is
needed.

### Tuning notes

- **Frame count vs quality:** `--stride` trades reconstruction completeness for
  speed. Start dense (`--stride 2`, ~460 frames); drop to `--stride 1` for a final
  run. Too sparse and COLMAP fails to register frames.
- **Fisheye orientation:** the calibrated intrinsics are portrait while the JPEGs
  are landscape; always extract with `--rotate` so COLMAP's seeded fisheye model
  is valid. COLMAP also refines intrinsics.
- **Alignment reliability:** `raven.align` reports a `fitness`; below ~0.3 it is
  unreliable — train with the default `--init colmap` instead of `--init lidar`.
- **Viewing splats:** `work/splat.ply` loads in standard 3DGS viewers
  (e.g. SuperSplat, antimatter15/splat, the Nerfstudio viewer). For a one-click
  open, use `python -m raven.view_splat --data "$DATA"`, the trainer's `--view`
  flag, or the cloud editor's **Open in SuperSplat** button — each serves the
  splat over a short-lived localhost server and launches the
  [SuperSplat](https://superspl.at/editor) web editor pointed at it (nothing is
  uploaded; the browser fetches it from your machine).

## Module map

| Module | Role |
| --- | --- |
| `raven/bag_io.py` | Read ROS1 bags (clouds, IMU, images) with the ROS1 typestore |
| `raven/calib.py` | Fisheye intrinsics, distortion, extrinsic, image rotation |
| `raven/extract.py` | Decode frames + stage the fused cloud |
| `raven/cloud_ops.py` | Headless point-cloud ops + auto-clean (crop / denoise / mask) |
| `raven/cloud_editor.py` | Open3D GUI: in-scene select/crop, delete-preview, auto-clean, build handoff |
| `raven/colmap_pipeline.py` | pycolmap SfM → sparse + undistorted PINHOLE model |
| `raven/align.py` | Fit similarity transform: lidar cloud → COLMAP frame |
| `raven/recolor.py` | Project photos onto the cloud for true RGB (z-buffer occlusion) |
| `raven/train_splat.py` | gsplat training → 3DGS `.ply` |
| `raven/view_splat.py` | Serve `splat.ply` locally + open it in the SuperSplat web editor |
```
