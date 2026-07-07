#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---kitless}"
if [[ $# -gt 0 ]]; then
  shift
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIT_IMAGE="${ANCHOR_BENCH_KITLESS_IMAGE:-anchor-bench:kitless}"
FULL_IMAGE="${ANCHOR_BENCH_FULL_IMAGE:-anchor-bench:full}"
COMPATIBILITY_FILE="${ROOT}/versions/compatibility.toml"

manifest_value() {
  local section="$1"
  local key="$2"
  awk -F= -v section="${section}" -v key="${key}" '
    /^[[:space:]]*\[/ {
      current=$0
      gsub(/^[[:space:]]*\[/, "", current)
      gsub(/\][[:space:]]*$/, "", current)
      in_section=(current == section)
      next
    }
    in_section {
      lhs=$1
      gsub(/[[:space:]]/, "", lhs)
      if (lhs == key) {
        value=$2
        for (i=3; i<=NF; i++) value=value "=" $i
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
        gsub(/^"|"$/, "", value)
        print value
        exit
      }
    }
  ' "${COMPATIBILITY_FILE}"
}

PYTHON_VERSION="$(manifest_value isaac_sim python)"
ISAACSIM_VERSION="$(manifest_value isaac_sim version)"
NEWTON_PACKAGE="$(manifest_value newton package)"
NEWTON_REPO="$(manifest_value newton repo)"
NEWTON_REF="$(manifest_value newton commit)"
WARP_PACKAGE="$(manifest_value warp package)"
WARP_VERSION="$(manifest_value warp version)"
ISAACLAB_REPO="$(manifest_value isaac_lab repo)"
ISAACLAB_BRANCH="$(manifest_value isaac_lab branch)"
ISAACLAB_SHA="$(manifest_value isaac_lab commit)"
ISAACLAB_SOURCE="${ANCHOR_BENCH_ISAACLAB_SOURCE:-image}"
CONTAINER_REPO=/workspace/anchor-bench
CONTAINER_THIRD_PARTY_ISAACLAB="${CONTAINER_REPO}/third_party/IsaacLab"

: "${PYTHON_VERSION:?Missing isaac_sim.python in versions/compatibility.toml}"
: "${ISAACSIM_VERSION:?Missing isaac_sim.version in versions/compatibility.toml}"
: "${NEWTON_PACKAGE:?Missing newton.package in versions/compatibility.toml}"
: "${NEWTON_REPO:?Missing newton.repo in versions/compatibility.toml}"
: "${NEWTON_REF:?Missing newton.commit in versions/compatibility.toml}"
: "${WARP_PACKAGE:?Missing warp.package in versions/compatibility.toml}"
: "${WARP_VERSION:?Missing warp.version in versions/compatibility.toml}"
: "${ISAACLAB_REPO:?Missing isaac_lab.repo in versions/compatibility.toml}"
: "${ISAACLAB_BRANCH:?Missing isaac_lab.branch in versions/compatibility.toml}"
: "${ISAACLAB_SHA:?Missing isaac_lab.commit in versions/compatibility.toml}"

usage() {
  cat <<'USAGE'
Docker-first Anchor Bench setup.

Usage: ./setup.sh [--kitless|--full|--assets|--check|--shell|--help]

--kitless  Build anchor-bench:kitless with host-safe CLI/test dependencies.
--full     Build anchor-bench:full with Isaac Sim 6.0.0.0 and pinned Isaac Lab.
--assets   Create host asset/output directories only.
--check    Verify Docker tooling, then run check_environment.py inside kitless.
--shell    Open/reuse the persistent full Isaac Sim/Newton container with this repo mounted.
--help     Show this help.

This script does not create a host .venv or install Anchor Bench on the host.
USAGE
}

is_dry_run() {
  [[ "${ANCHOR_BENCH_SETUP_DRY_RUN:-0}" == "1" ]]
}

run_cmd() {
  if is_dry_run; then
    printf ' +'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

prepare_local_isaaclab() {
  case "${ISAACLAB_SOURCE}" in
    image|"") return 0 ;;
    third_party) ;;
    *)
      echo "ANCHOR_BENCH_ISAACLAB_SOURCE must be 'image' or 'third_party', got '${ISAACLAB_SOURCE}'." >&2
      exit 1
      ;;
  esac

  local lab_dir="${ROOT}/third_party/IsaacLab"
  echo "[INFO] Preparing third_party/IsaacLab at ${lab_dir}"
  echo "[INFO] Pinning Isaac Lab to ${ISAACLAB_SHA}"

  if is_dry_run; then
    run_cmd mkdir -p "${ROOT}/third_party"
    echo "[INFO] If ${lab_dir}/.git is missing, clone the checkout:"
    run_cmd git clone --branch "${ISAACLAB_BRANCH}" --single-branch "${ISAACLAB_REPO}" "${lab_dir}"
    run_cmd git -C "${lab_dir}" fetch origin "${ISAACLAB_SHA}"
    run_cmd git -C "${lab_dir}" switch --detach "${ISAACLAB_SHA}"
    run_cmd git -C "${lab_dir}" rev-parse HEAD
    export ANCHOR_BENCH_ISAACLAB_DIR="${CONTAINER_THIRD_PARTY_ISAACLAB}"
    echo "[INFO] ANCHOR_BENCH_ISAACLAB_DIR=${ANCHOR_BENCH_ISAACLAB_DIR}"
    return 0
  fi

  if [[ -e "${lab_dir}" && ! -d "${lab_dir}/.git" ]]; then
    echo "${lab_dir} exists but is not a git checkout." >&2
    exit 1
  fi

  if [[ ! -d "${lab_dir}/.git" ]]; then
    mkdir -p "${ROOT}/third_party"
    git clone --branch "${ISAACLAB_BRANCH}" --single-branch "${ISAACLAB_REPO}" "${lab_dir}"
  fi

  git -C "${lab_dir}" fetch origin "${ISAACLAB_SHA}"
  git -C "${lab_dir}" switch --detach "${ISAACLAB_SHA}"

  local actual
  actual="$(git -C "${lab_dir}" rev-parse HEAD)"
  if [[ "${actual}" != "${ISAACLAB_SHA}" ]]; then
    echo "Isaac Lab checkout is not pinned to ${ISAACLAB_SHA}; got ${actual}." >&2
    exit 1
  fi

  export ANCHOR_BENCH_ISAACLAB_DIR="${CONTAINER_THIRD_PARTY_ISAACLAB}"
  echo "[INFO] Using third_party Isaac Lab inside container: ${ANCHOR_BENCH_ISAACLAB_DIR}"
}

ensure_docker() {
  if is_dry_run; then
    return 0
  fi
  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is required. Install Docker Engine or Docker Desktop, then rerun setup.sh." >&2
    exit 1
  fi
  if ! docker info >/dev/null 2>&1; then
    echo "Docker is installed but the daemon is not reachable." >&2
    exit 1
  fi
}

ensure_compose() {
  if is_dry_run; then
    return 0
  fi
  if docker compose version >/dev/null 2>&1; then
    return 0
  fi
  if command -v docker-compose >/dev/null 2>&1 && docker-compose version >/dev/null 2>&1; then
    return 0
  fi
  echo "Docker Compose is required for the repo-owned docker-compose.yaml workflow." >&2
  exit 1
}

check_nvidia_runtime() {
  if is_dry_run; then
    return 0
  fi
  if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
    echo "NVIDIA container runtime: detected"
    return 0
  fi
  if docker run --help 2>/dev/null | grep -q -- '--gpus'; then
    echo "NVIDIA container runtime: Docker supports --gpus, but no nvidia runtime was listed."
    return 0
  fi
  echo "NVIDIA container runtime: not detected; full GPU runs may fail." >&2
}

build_image() {
  local target="$1"
  local image="$2"
  ensure_docker
  run_cmd docker build \
    --file docker/Dockerfile \
    --target "${target}" \
    --tag "${image}" \
    --build-arg "PYTHON_VERSION=${PYTHON_VERSION}" \
    --build-arg "NEWTON_PACKAGE=${NEWTON_PACKAGE}" \
    --build-arg "NEWTON_REPO=${NEWTON_REPO}" \
    --build-arg "NEWTON_REF=${NEWTON_REF}" \
    --build-arg "WARP_PACKAGE=${WARP_PACKAGE}" \
    --build-arg "WARP_VERSION=${WARP_VERSION}" \
    --build-arg "ISAACSIM_VERSION=${ISAACSIM_VERSION}" \
    --build-arg "ISAACLAB_REPO=${ISAACLAB_REPO}" \
    --build-arg "ISAACLAB_BRANCH=${ISAACLAB_BRANCH}" \
    --build-arg "ISAACLAB_REF=${ISAACLAB_SHA}" \
    "${ROOT}"
}

ensure_image() {
  local target="$1"
  local image="$2"
  if is_dry_run; then
    run_cmd docker image inspect "${image}"
    return 0
  fi
  if ! docker image inspect "${image}" >/dev/null 2>&1; then
    build_image "${target}" "${image}"
  fi
}

install_assets() {
  mkdir -p "${ROOT}/assets/robots" "${ROOT}/outputs" "${ROOT}/third_party/artifacts/h1"
  cat <<'ASSETS'
Asset directories are ready.

Large datasets, SO-101 benchmark outputs, neural weights, FMUs, generated plots,
third-party checkouts, and full robot assets should stay outside git. Put local
FMUs under third_party/artifacts/<robot>/, and add manifests under
assets/robots/<robot>/ or docs/artifacts.md when publishing downloadable
artifacts.
ASSETS
}

run_check() {
  ensure_docker
  ensure_compose
  check_nvidia_runtime
  ensure_image kitless "${KIT_IMAGE}"
  run_cmd docker run --rm \
    -v "${ROOT}:/workspace/anchor-bench" \
    -w /workspace/anchor-bench \
    "${KIT_IMAGE}" \
    python check_environment.py --profile kitless
}

open_shell() {
  ensure_docker
  ensure_image full "${FULL_IMAGE}"
  if ! is_dry_run; then
    install_assets >/dev/null
  fi
  prepare_local_isaaclab
  if is_dry_run; then
    run_cmd "${ROOT}/docker/run-persist.sh" "$@"
    return 0
  else
    exec "${ROOT}/docker/run-persist.sh" "$@"
  fi
}

case "${MODE}" in
  --kitless) build_image kitless "${KIT_IMAGE}" ;;
  --full) build_image full "${FULL_IMAGE}" ;;
  --assets) install_assets ;;
  --check) run_check ;;
  --shell) open_shell "$@" ;;
  --help|-h) usage ;;
  *) usage; exit 1 ;;
esac
