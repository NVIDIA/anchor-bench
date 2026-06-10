#!/usr/bin/env bash
# Rebuild and run commands in the Anchor Bench full container.
#
# Usage:
#   ./docker/run-gui.sh
#   ./docker/run-gui.sh anchor-bench benchmark --robot so101 --motion inputs/datasets/so101/custom_figure8_with_camera
#   HEADLESS=1 ./docker/run-gui.sh anchor-bench benchmark --robot so101 --motion inputs/datasets/so101/custom_figure8_with_camera --headless
#
# Env knobs:
#   HEADLESS=1                 skip X11 forwarding
#   ANCHOR_BENCH_DATA_DIR=...  mount host read-only data at /data/anchor-bench
#   ANCHOR_BENCH_RUN_AS_ROOT=1 run container as root (default: host UID/GID)

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,12p' "$0"
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${REPO_ROOT}/docker-compose.yaml"
IMAGE="${ANCHOR_BENCH_FULL_IMAGE:-anchor-bench:full}"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

mkdir -p "${REPO_ROOT}/assets" "${REPO_ROOT}/outputs" "${REPO_ROOT}/third_party/artifacts/h1"

echo "[INFO] Rebuilding Docker image..."
ANCHOR_BENCH_IMAGE="${IMAGE}" ANCHOR_BENCH_DOCKER_TARGET=full \
  docker compose --file "${COMPOSE_FILE}" build anchor-bench

X11_OVERLAY=()
TMP_DIR=""
if [[ -n "${HEADLESS:-}" ]]; then
  echo "[INFO] Running in headless mode (no GUI)"
else
  echo "[INFO] Setting up X11 forwarding..."
  TMP_DIR="$(mktemp -d)"
  TMP_XAUTH="$(mktemp --suffix=.xauth --tmpdir="${TMP_DIR}")"
  XAUTH_COOKIE="$(xauth nlist "${DISPLAY:-}" 2>/dev/null || true)"
  if [[ -n "${XAUTH_COOKIE}" ]]; then
    echo "${XAUTH_COOKIE}" | sed 's/ffff//' | xauth -f "${TMP_XAUTH}" nmerge -
    export __ANCHOR_BENCH_TMP_XAUTH="${TMP_XAUTH}"
    export __ANCHOR_BENCH_TMP_DIR="${TMP_DIR}"
    X11_OVERLAY=(--file "${SCRIPT_DIR}/x11.yaml")
    echo "[INFO] X11 forwarding enabled"
  else
    echo "[WARN] No X11 display cookie found; running without GUI forwarding"
  fi
fi

DATA_ARGS=()
if [[ -n "${ANCHOR_BENCH_DATA_DIR:-}" ]]; then
  DATA_ARGS=(
    --volume "${ANCHOR_BENCH_DATA_DIR}:/data/anchor-bench:ro"
    --env ANCHOR_BENCH_DATA_DIR=/data/anchor-bench
  )
fi

USER_ARGS=(
  --user "${HOST_UID}:${HOST_GID}"
  -e HOME=/tmp/anchor-bench-home
  -e XDG_CACHE_HOME=/tmp/anchor-bench-cache
  -e MPLCONFIGDIR=/tmp/anchor-bench-cache/matplotlib
)
if [[ "${ANCHOR_BENCH_RUN_AS_ROOT:-0}" == "1" ]]; then
  USER_ARGS=(
    --user 0:0
    -e HOME=/tmp/anchor-bench-home
    -e XDG_CACHE_HOME=/tmp/anchor-bench-cache
    -e MPLCONFIGDIR=/tmp/anchor-bench-cache/matplotlib
  )
  echo "[INFO] ANCHOR_BENCH_RUN_AS_ROOT=1; container will run as root"
else
  echo "[INFO] Container user mapped to host UID:GID ${HOST_UID}:${HOST_GID}"
fi

cleanup() {
  if [[ -n "${TMP_DIR}" ]]; then
    rm -rf "${TMP_DIR}"
  fi
}
trap cleanup EXIT

if [[ "$#" -eq 0 ]]; then
  echo "[INFO] Running interactive shell..."
  ANCHOR_BENCH_IMAGE="${IMAGE}" ANCHOR_BENCH_DOCKER_TARGET=full \
    docker compose --file "${COMPOSE_FILE}" "${X11_OVERLAY[@]}" run --rm "${USER_ARGS[@]}" "${DATA_ARGS[@]}" anchor-bench
else
  echo "[INFO] Running command in container: $*"
  ANCHOR_BENCH_IMAGE="${IMAGE}" ANCHOR_BENCH_DOCKER_TARGET=full \
    docker compose --file "${COMPOSE_FILE}" "${X11_OVERLAY[@]}" run --rm "${USER_ARGS[@]}" "${DATA_ARGS[@]}" anchor-bench \
      bash -c 'cd /workspace/anchor-bench && exec "$@"' bash "$@"
fi
