# syntax=docker/dockerfile:1.7
#
# GenIA container: a Gradio Web UI plus the GenIA pipeline, built on the official
# PyTorch CUDA image (torch 2.7.1 + CUDA 12.8 + cuDNN 9 + Python 3.11), which
# matches GenIA's pins. CUDA extensions use prebuilt wheels where available, so
# the image builds in minutes instead of hours. Model weights are NOT baked in:
# they download into the /data/weights volume at first start (SAM 3D Objects is
# gated, so the licence must be accepted per user).
#
# Built for the CMP 170HX (GA100, compute capability 8.0) by default.

ARG CUDA_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel
FROM ${CUDA_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
# Compile parallelism for the few source-built extensions (diso, gsplat).
ARG MAX_JOBS=2
# Target GPU arch. CMP 170HX / A100 = 8.0; L40S/4090 = 8.9; H100 = 9.0.
ARG GENIA_TORCH_CUDA_ARCH_LIST=8.0
# Pin GenIA so builds are reproducible (release branch tip).
ARG GENIA_REF=6fbd691ac90248eedb16e1099f2c7bef6831c69c

ENV GENIA_HOME=/opt/GenIA \
    GENIA_DIR=/opt/GenIA \
    HF_HUB_CACHE=/data/weights/huggingface/hub \
    TORCH_HOME=/data/weights/torch \
    U2NET_HOME=/data/weights/rembg \
    GENIA_RESULTS=/data/results \
    CUDA_HOME=/usr/local/cuda \
    MAX_JOBS=${MAX_JOBS} \
    GENIA_TORCH_CUDA_ARCH_LIST=${GENIA_TORCH_CUDA_ARCH_LIST} \
    TORCH_CUDA_ARCH_LIST=${GENIA_TORCH_CUDA_ARCH_LIST} \
    PYTHONUNBUFFERED=1

# Headless runtime libs for open3d/pyvista/opencv/ffmpeg.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
 && apt-get install -y --no-install-recommends \
        git ffmpeg curl ca-certificates \
        libgl1 libglib2.0-0 libgomp1 libsm6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/facebookresearch/GenIA "${GENIA_DIR}" \
 && git -C "${GENIA_DIR}" checkout "${GENIA_REF}"

COPY scripts/install_genia_docker.sh /opt/genia-bootstrap/install_genia_docker.sh
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    bash /opt/genia-bootstrap/install_genia_docker.sh

COPY webui/requirements-webui.txt /opt/genia-bootstrap/requirements-webui.txt
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install -r /opt/genia-bootstrap/requirements-webui.txt

COPY webui/ "${GENIA_DIR}/webui/"
COPY scripts/entrypoint.sh /usr/local/bin/genia-entrypoint
RUN chmod +x /usr/local/bin/genia-entrypoint

WORKDIR ${GENIA_DIR}
EXPOSE 7860
VOLUME ["/data/weights", "/data/results"]

ENTRYPOINT ["genia-entrypoint"]
CMD ["webui"]
