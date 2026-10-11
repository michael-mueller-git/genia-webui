# GenIA Web UI (Docker)

A Gradio Web UI for [GenIA](https://github.com/facebookresearch/GenIA) — generative
3D reconstruction from a single image, a few views, or a monocular video. Runs the
GenIA pipeline on GPU, with model weights downloaded into a Docker volume on first
start (they are not baked into the image; `facebook/sam-3d-objects` is gated and
must be licensed per user).

## Target hardware

Built for an **NVIDIA CMP 170HX** (GA100, compute capability **8.0**, 64 GB HBM2e)
and tested arch `sm_80`. Because GA100 is the same silicon as A100, the build pins
`TORCH_CUDA_ARCH_LIST=8.0` (also GenIA's default). For other GPUs override the
build arg: L40S/4090 → `8.9`, H100 → `9.0`.

Host requirements: an NVIDIA driver new enough for CUDA 12.8 (R525+ works via CUDA
minor-version compatibility; a recent driver, ≥570, is recommended) and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Verify the card:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv   # expect 8.0
```

## Quick start

```bash
# 1. Accept the licence at https://huggingface.co/facebook/sam-3d-objects
#    and create a read token: https://huggingface.co/settings/tokens
export HF_TOKEN=hf_...

# 2. Build and run (downloads weights into the genia-weights volume on first start)
docker compose up --build
# open http://localhost:7860
```

The first start downloads the model weights (SAM 3D Objects, map-anything, MoGe,
ActionMesh + TripoSG/DINOv2/RMBG — tens of GB) into the `genia-weights` volume.
Subsequent starts are instant. Static scenes only (skips the ~17 GB ActionMesh
weights): set `GENIA_SKIP_ACTIONMESH=1`.

### Without compose

```bash
docker build -t genia-webui:local --build-arg GENIA_TORCH_CUDA_ARCH_LIST=8.0 .
docker run --gpus all -p 7860:7860 -e HF_TOKEN=hf_... \
  -v genia-weights:/data/weights -v genia-results:/data/results \
  genia-webui:local webui
```

## UI

Three tabs matching GenIA's demos:

- **Single image** — one RGB (+ optional mask) → canonical object.
- **Multiview** — several views of a static scene.
- **Video** — a monocular frame sequence.

Masks are optional: with none provided, the foreground is segmented automatically
with **RMBG-1.4** (weights already fetched for ActionMesh). Extra Hydra overrides
can be passed as text (e.g. `appearance_init_2.inference_steps=10`). Results show
the turntable orbit video, the input-view renders, and downloads (final `.zip`,
mesh `.glb`, `poses.json`).

The Setup tab shows GPU/weights status and can trigger the weight download.

## Container modes

```bash
docker run ... genia-webui:local download      # fetch weights and exit
docker run ... genia-webui:local run image --scene sloth   # demo/run_demo.py passthrough
docker run ... genia-webui:local status        # GPU + volume status
docker run ... genia-webui:local shell
```

Volumes: `/data/weights` (HF + torch hub caches) and `/data/results` (runs).

If the deployment mounts a cluster CA at `/etc/ssl/certs/internal-ca.pem`
(e.g. for an in-cluster HuggingFace mirror), the entrypoint builds a combined
trust bundle = the image's public roots + that CA and points
`SSL_CERT_FILE`/`REQUESTS_CA_BUNDLE`/`CURL_CA_BUNDLE` at it. Override the path
with `GENIA_CA_BUNDLE`. This keeps TLS to public hosts working (torch.hub fetches
DINOv2 from GitHub) while trusting the cluster CA.

## Build notes

- Base image: `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel` — exactly GenIA's
  torch 2.7.1 + CUDA 12.8 + cuDNN 9 + Python 3.11 pins, with the CUDA toolkit.
- CUDA extensions use prebuilt wheels where possible (pytorch3d, nvdiffrast,
  fused-ssim via the MiroPsota `torch_packages_builder` index; spconv/cumm/xformers
  from PyPI). Only `diso` and `gsplat` build from source, so the image builds in
  minutes rather than hours. See `scripts/install_genia_docker.sh`.
- GenIA is cloned at a pinned commit (`GENIA_REF` build arg).

## CI

`.github/workflows/build.yml` builds and pushes `ghcr.io/<owner>/genia-webui` and
uses a **registry-backed BuildKit cache** in GHCR (`:buildcache`, `mode=max`),
because the image exceeds the 10 GB `actions/cache` limit.

## Licensing

GenIA and its third-party components ship their own licences (see `LICENSE` and
`THIRD_PARTY_LICENSES.md` in the GenIA checkout, and `facebook/sam-3d-objects` on
Hugging Face). Review them before use.
