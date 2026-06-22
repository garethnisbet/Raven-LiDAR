#!/usr/bin/env bash
# Install all Raven dependencies under /scratch with NO root.
#
# Fully self-contained: a private micromamba provides Python, the pip stack
# (open3d, pycolmap, rosbags, ...), torch (CUDA 12.1) + gsplat, and a separate
# CUDA toolkit (nvcc) for gsplat's JIT kernel build. Nothing is installed to the
# system or to $HOME -- everything (incl. pip/conda caches and TMPDIR) lives in
# the prefix, so a small or full home disk doesn't matter.
#
# Usage:
#   scripts/install_scratch.sh [PREFIX]
#   PREFIX=/scratch/$USER/raven-env  scripts/install_scratch.sh
#
# Default PREFIX: /scratch/$USER/raven-env
#
# After it finishes, activate the environment with:
#   source <PREFIX>/activate.sh
# then run the pipeline normally, e.g.  python -m raven.train_splat --help
set -euo pipefail

# --------------------------------------------------------------------------- #
PREFIX="${1:-${PREFIX:-/scratch/${USER}/raven-env}}"
PY_VERSION="${PY_VERSION:-3.11}"
TORCH_CUDA="${TORCH_CUDA:-cu121}"          # matches the cuda-12.1 nvcc env below
CUDA_LABEL="${CUDA_LABEL:-12.1.1}"
MAMBA_URL="https://micro.mamba.pm/api/micromamba/linux-64/latest"

# Repo root = parent of this script's dir (so requirements.txt is found).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
REQ="$REPO_DIR/requirements.txt"

PY_ENV="$PREFIX/envs/raven"            # python + pip deps + torch + gsplat
CUDA_ENV="$PREFIX/envs/cuda121"        # cuda-nvcc toolkit (CUDA_HOME for gsplat)
MAMBA_BIN="$PREFIX/bin/micromamba"

export MAMBA_ROOT_PREFIX="$PREFIX/mamba"
export PIP_CACHE_DIR="$PREFIX/cache/pip"
export TMPDIR="$PREFIX/tmp"            # gsplat/torch builds need scratch tmp space

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# --------------------------------------------------------------------------- #
log "Target prefix: $PREFIX"
parent="$(dirname "$PREFIX")"
if [ ! -d "$parent" ]; then
    echo "ERROR: $parent does not exist. Is /scratch mounted on this node?" >&2
    exit 1
fi
mkdir -p "$PREFIX/bin" "$MAMBA_ROOT_PREFIX" "$PIP_CACHE_DIR" "$TMPDIR"

if [ ! -f "$REQ" ]; then
    echo "ERROR: requirements.txt not found at $REQ" >&2
    exit 1
fi

# --------------------------------------------------------------------------- #
log "1/4  Bootstrapping micromamba (no root, static binary)"
if [ ! -x "$MAMBA_BIN" ]; then
    curl -Ls "$MAMBA_URL" | tar -xj -C "$PREFIX" bin/micromamba
    chmod +x "$MAMBA_BIN"
fi
"$MAMBA_BIN" --version

mm() { "$MAMBA_BIN" "$@"; }

# --------------------------------------------------------------------------- #
log "2/4  Creating Python $PY_VERSION env + CUDA $CUDA_LABEL nvcc toolkit"
if [ ! -x "$PY_ENV/bin/python" ]; then
    mm create -y -p "$PY_ENV" -c conda-forge "python=$PY_VERSION" pip
fi
# Separate nvcc toolkit kept out of the python env so torch's bundled runtime
# libs are never shadowed (gsplat only needs nvcc + headers at build time).
if [ ! -x "$CUDA_ENV/bin/nvcc" ]; then
    mm create -y -p "$CUDA_ENV" \
        -c "nvidia/label/cuda-$CUDA_LABEL" -c conda-forge \
        cuda-nvcc cuda-cudart-dev cuda-cccl libcusparse-dev libcublas-dev
fi

PYBIN="$PY_ENV/bin/python"
"$PYBIN" -m pip install --upgrade pip wheel setuptools

# --------------------------------------------------------------------------- #
log "3/4  Installing pip dependencies (open3d, pycolmap, rosbags, ...)"
"$PYBIN" -m pip install -r "$REQ"

log "3b   Installing torch ($TORCH_CUDA) + gsplat"
"$PYBIN" -m pip install torch --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
# gsplat builds its CUDA extension against the nvcc env; expose it for the build.
CUDA_HOME="$CUDA_ENV" PATH="$CUDA_ENV/bin:$PATH" "$PYBIN" -m pip install gsplat

# --------------------------------------------------------------------------- #
log "4/4  Writing activation script"
cat > "$PREFIX/activate.sh" <<EOF
# Source this to use the Raven scratch environment:  source "$PREFIX/activate.sh"
export MAMBA_ROOT_PREFIX="$MAMBA_ROOT_PREFIX"
export PIP_CACHE_DIR="$PIP_CACHE_DIR"
export TMPDIR="$TMPDIR"
# CUDA toolkit (nvcc) for gsplat's JIT kernels; train_splat honours CUDA_HOME.
export CUDA_HOME="$CUDA_ENV"
export PATH="$PY_ENV/bin:$CUDA_ENV/bin:\$PATH"
EOF
echo "wrote $PREFIX/activate.sh"

# --------------------------------------------------------------------------- #
log "Verifying the install"
# shellcheck disable=SC1090
source "$PREFIX/activate.sh"
python - <<'PY'
import importlib, sys
mods = ["numpy", "scipy", "cv2", "open3d", "rosbags", "pycolmap", "plyfile", "tqdm", "torch", "gsplat"]
ok = True
for m in mods:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "?")
        print(f"  {m:12s} {v}")
    except Exception as exc:                       # noqa: BLE001
        ok = False
        print(f"  {m:12s} FAILED: {exc}")
print(f"  torch.cuda.is_available() = {__import__('torch').cuda.is_available()}")
sys.exit(0 if ok else 1)
PY
nvcc --version | tail -2 || echo "WARNING: nvcc not on PATH"

cat <<EOF

================================================================================
Raven dependencies installed under: $PREFIX

To use it (each new shell):
    source "$PREFIX/activate.sh"
    cd "$REPO_DIR"
    python -m raven.train_splat --help

Notes:
  * Nothing was installed to \$HOME or system dirs; delete $PREFIX to uninstall.
  * pip/conda caches and TMPDIR live under the prefix, not your home disk.
  * gsplat compiles its CUDA kernels on first run using CUDA_HOME=$CUDA_ENV.
================================================================================
EOF
