#!/usr/bin/env bash
# End-to-end Raven -> Gaussian splat pipeline for one capture folder.
#
#   scripts/run_pipeline.sh /path/to/capture
#
# Edit the point cloud manually (step 2) between extract and SfM if you like.
set -euo pipefail
cd "$(dirname "$0")/.."          # run from the code root (where raven/ lives)

DATA="${1:-.}"
STRIDE="${STRIDE:-2}"
LONG_EDGE="${LONG_EDGE:-1600}"
ITERS="${ITERS:-30000}"
INIT="${INIT:-colmap}"           # colmap | lidar

if [ -z "${CUDA_HOME:-}" ] && [ -x "$HOME/.local/micromamba/envs/cuda121/bin/nvcc" ]; then
  export CUDA_HOME="$HOME/.local/micromamba/envs/cuda121"
  export PATH="$CUDA_HOME/bin:$PATH"
fi

echo "== capture: $DATA =="

echo "== 1. extract =="
python3 -m raven.extract --data "$DATA" --images --cloud --rotate --long-edge "$LONG_EDGE" --stride "$STRIDE"

echo "== 2. edit cloud (manual, optional) =="
echo "   python3 -m raven.cloud_editor $DATA   (Save As work/cloud_edited.ply)"

echo "== 3. colmap SfM =="
python3 -m raven.colmap_pipeline --data "$DATA" --device "${COLMAP_DEVICE:-cpu}"

if [ "$INIT" = "lidar" ]; then
  echo "== 4. align lidar -> colmap =="
  python3 -m raven.align --data "$DATA"
  echo "== 4b. recolour (optional) =="
  echo "   python3 -m raven.recolor --data $DATA"
fi

echo "== 5. train splats =="
python3 -m raven.train_splat --data "$DATA" --iters "$ITERS" --init "$INIT"

echo "done -> $DATA/work/splat.ply"
