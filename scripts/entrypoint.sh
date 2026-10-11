#!/usr/bin/env bash
# Container entrypoint: bootstrap model weights into the volume, then run.
#
#   webui     start the Gradio Web UI (default)
#   download  fetch all model weights into /data/weights and exit
#   run ...   run demo/run_demo.py with the given arguments
#   shell     interactive shell
#   status    print GPU + volume status
set -euo pipefail

GENIA_DIR="${GENIA_DIR:-/opt/GenIA}"
WEIGHTS_DIR="${WEIGHTS_DIR:-/data/weights}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${WEIGHTS_DIR}/huggingface/hub}"
export TORCH_HOME="${TORCH_HOME:-${WEIGHTS_DIR}/torch}"
export U2NET_HOME="${U2NET_HOME:-${WEIGHTS_DIR}/rembg}"
export GENIA_RESULTS="${GENIA_RESULTS:-/data/results}"
export GENIA_HOME="$GENIA_DIR"
MARKER="${WEIGHTS_DIR}/.genia_weights_ok"

mkdir -p "$HF_HUB_CACHE" "$TORCH_HOME" "$U2NET_HOME" "$GENIA_RESULTS"

# Trust BOTH the image's public roots and the cluster CA.
#
# Deployments may point SSL_CERT_FILE/REQUESTS_CA_BUNDLE/CURL_CA_BUNDLE at a
# file holding only the cluster CA (e.g. /etc/ssl/certs/internal-ca.pem). That
# drops the public roots, so TLS to public hosts fails — torch.hub.load(...,
# source="github") for DINOv2 then dies with CERTIFICATE_VERIFY_FAILED. Build a
# combined bundle (image roots + cluster CA) and point the CA vars at it, so
# in-cluster (the HF mirror) and public endpoints both verify. No-op when the
# cluster CA is not mounted (e.g. local docker-compose).
CA_BUNDLE="${GENIA_CA_BUNDLE:-/opt/genia-bootstrap/ca-certificates.crt}"
INTERNAL_CA="/etc/ssl/certs/internal-ca.pem"
if [ -f "$INTERNAL_CA" ]; then
    mkdir -p "$(dirname "$CA_BUNDLE")"
    : > "$CA_BUNDLE"
    SYSTEM_CA="/etc/ssl/certs/ca-certificates.crt"
    [ -f "$SYSTEM_CA" ] && cat "$SYSTEM_CA" >> "$CA_BUNDLE" || true
    cat "$INTERNAL_CA" >> "$CA_BUNDLE"
    export SSL_CERT_FILE="$CA_BUNDLE"
    export REQUESTS_CA_BUNDLE="$CA_BUNDLE"
    export CURL_CA_BUNDLE="$CA_BUNDLE"
    export PIP_CERT="$CA_BUNDLE"
    export GIT_SSL_CAINFO="$CA_BUNDLE"
    echo "[entrypoint] CA bundle: ${CA_BUNDLE} (public roots + cluster CA)"
fi

# huggingface_hub reads HF_TOKEN directly; mirror it for older clients too.
if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

ensure_weights() {
    if [ "${GENIA_FORCE_DOWNLOAD:-0}" = "1" ] || [ ! -f "$MARKER" ]; then
        echo "[entrypoint] downloading GenIA model weights into ${WEIGHTS_DIR} ..."
        echo "[entrypoint] (facebook/sam-3d-objects is gated: accept its licence and set HF_TOKEN)"
        local args=()
        [ "${GENIA_SKIP_ACTIONMESH:-0}" = "1" ] && args+=(--no-actionmesh)
        cd "$GENIA_DIR"
        python -m genia.core.download_weights "${args[@]}"
        touch "$MARKER"
    else
        echo "[entrypoint] model weights already present (${MARKER})"
    fi
}

case "${1:-webui}" in
    webui)
        ensure_weights
        exec python "${GENIA_DIR}/webui/app.py" \
            --server-name 0.0.0.0 --server-port "${WEBUI_PORT:-7860}"
        ;;
    download)
        ensure_weights
        ;;
    run)
        shift
        ensure_weights
        cd "$GENIA_DIR"
        exec python demo/run_demo.py "$@"
        ;;
    shell)
        exec bash
        ;;
    status)
        nvidia-smi || echo "no GPU visible (run with --gpus all)"
        echo "weights: ${WEIGHTS_DIR} ($(du -sh "$WEIGHTS_DIR" 2>/dev/null | cut -f1))"
        echo "marker:  $([ -f "$MARKER" ] && echo present || echo missing)"
        ;;
    *)
        exec "$@"
        ;;
esac
