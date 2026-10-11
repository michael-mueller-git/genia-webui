#!/usr/bin/env bash
# Install GenIA into the PyTorch CUDA base image.
#
# Mirrors GenIA's install.sh, but uses prebuilt CUDA wheels for the extensions
# that are otherwise slow to compile (pytorch3d, nvdiffrast, fused-ssim) and
# falls back to a source build only when a wheel is unavailable. torch itself is
# provided by the base image, so it is pinned on the requirements line (as
# install.sh does) to stop the resolver from replacing it.
set -euo pipefail

GENIA_DIR="${GENIA_DIR:-/opt/GenIA}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-2}"
export TORCH_CUDA_ARCH_LIST="${GENIA_TORCH_CUDA_ARCH_LIST:-8.0}"

MIRROR="https://miropsota.github.io/torch_packages_builder"
TORCH_TAG="pt2.7.1"
CUDA_TAG="cu128"
PY_TAG="cp311"

cd "$GENIA_DIR"

echo "=== submodules ==="
git submodule update --init submodules/sam-3d-objects submodules/map-anything
if git -C submodules/sam-3d-objects apply --reverse --check ../sam-3d-objects.patch 2>/dev/null; then
    echo "submodules/sam-3d-objects: patch already applied"
else
    git -C submodules/sam-3d-objects apply ../sam-3d-objects.patch
fi
# ActionMesh pins an SSH submodule URL; rewrite it to HTTPS for anonymous fetches.
git -c url."https://github.com/".insteadOf="git@github.com:" \
    submodule update --init --recursive submodules/actionmesh

echo "=== python dependencies ==="
TORCH_PINS=(torch==2.7.1+cu128 torchvision==0.22.1+cu128
            --extra-index-url https://download.pytorch.org/whl/cu128)
python -m pip install --upgrade pip wheel "setuptools==75.8.2"
# torch pinned on the same line: a requirement wanting another torch fails here
# instead of silently replacing the build the extensions are compiled against.
python -m pip install -r requirements.txt "${TORCH_PINS[@]}"

# Resolve the URL of a prebuilt wheel from the simple index by filename pattern.
wheel_url() {  # $1=pkg directory, $2=version grep (optional)
    curl -fsSL "${MIRROR}/$1/" \
        | grep -oE 'href="[^"]+\.whl"' \
        | sed 's/^href="//;s/"$//' \
        | grep -F "${TORCH_TAG}${CUDA_TAG}-${PY_TAG}-${PY_TAG}-linux_x86_64.whl" \
        | { if [ -n "${2:-}" ]; then grep -E "$2"; else cat; fi; } \
        | sort | tail -n1
}

install_prebuilt() {  # $1=pkg dir, $2=version grep, $3=source fallback
    local url=""
    url="$(wheel_url "$1" "${2:-}")" || url=""
    if [ -n "$url" ]; then
        echo "=== prebuilt $1: $url ==="
        python -m pip install "$url"
    else
        echo "=== no prebuilt wheel for $1; building from source ==="
        python -m pip install --no-build-isolation "$3"
    fi
}

echo "=== CUDA extensions ==="
# diso ships only an sdist; small CUDA source build pinned to the target arch.
# Its setup.py only selects the CUDA extension when a GPU is visible; the Docker
# build has no GPU, so torch.cuda.is_available() is False and it would otherwise
# build a CPU-only extension that still includes <cuda_runtime.h>. Force it.
FORCE_CUDA=1 python -m pip install --no-build-isolation diso==0.1.4

install_prebuilt pytorch3d '0\.7\.9' \
    "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9"
install_prebuilt nvdiffrast '0\.4\.0' \
    "git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0"
install_prebuilt fused-ssim '' \
    "git+https://github.com/rahul-goel/fused-ssim.git"

# gsplat has no prebuilt CUDA wheel; build it eagerly for the target arch.
python -m pip install --no-build-isolation \
    "git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3"

echo "=== prebuilt runtime wheels ==="
python -m pip install --no-deps xformers==0.0.31
python -m pip install --no-deps spconv-cu126==2.3.8 cumm-cu126==0.7.11
python -m pip install --no-deps \
    "git+https://github.com/microsoft/MoGe.git@07444410f1e33f402353b99d6ccd26bd31e469e8"
python -m pip install --no-deps \
    "git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183"

echo "=== editable packages ==="
python -m pip install --no-deps -e submodules/map-anything
python -m pip install --no-deps uniception==0.1.6
python -m pip install --no-deps -e submodules/actionmesh
python -m pip install --no-deps -e .

echo "=== done ==="
python -c "import torch, pytorch3d, nvdiffrast; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
