#!/usr/bin/env bash
# Internal helper for ./setup.sh --shell.
#
# This keeps the long-lived Anchor Bench full container running.
# Prefer:
#   ./setup.sh --shell
#   ./setup.sh --shell anchor-bench benchmark --robot so101 --motion inputs/datasets/so101/custom_figure8_with_camera
#
# Env knobs:
#   HEADLESS=1                 skip X11 forwarding
#   REBUILD=1                  rebuild the full image and recreate the persistent container
#   FORCE=1                    continue on build-input fingerprint mismatch
#   STRICT=1                   abort on build-input fingerprint mismatch
#   ANCHOR_BENCH_DATA_DIR=...  mount host read-only data at /data/anchor-bench
#   ANCHOR_BENCH_RUN_AS_ROOT=1 run container as root (default: host UID/GID)
#   ANCHOR_BENCH_GPUS=...      passed to docker --gpus (default: all)
#   ANCHOR_BENCH_NETWORK=...   passed to docker --network (default: host)

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,15p' "$0"
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${REPO_ROOT}/docker-compose.yaml"

CONTAINER=anchor-bench-persist
IMAGE="${ANCHOR_BENCH_FULL_IMAGE:-anchor-bench:full}"
FINGERPRINT_LABEL=anchor-bench.build-fingerprint
XAUTH_DIR="${XAUTH_DIR:-$(mktemp -d -t anchor-bench-persist-xauth.XXXXXX)}"
XAUTH_PATH="${XAUTH_DIR}/cookie"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
ANCHOR_BENCH_GPUS="${ANCHOR_BENCH_GPUS:-all}"
ANCHOR_BENCH_NETWORK="${ANCHOR_BENCH_NETWORK:-host}"
ANCHOR_BENCH_RUN_AS_ROOT="${ANCHOR_BENCH_RUN_AS_ROOT:-0}"

FINGERPRINT_FILES=(
  "${SCRIPT_DIR}/Dockerfile"
  "${SCRIPT_DIR}/entrypoint.sh"
  "${COMPOSE_FILE}"
  "${REPO_ROOT}/pyproject.toml"
  "${REPO_ROOT}/versions/compatibility.toml"
)

is_dry_run() {
  [[ "${ANCHOR_BENCH_SETUP_DRY_RUN:-0}" == "1" ]]
}

isaaclab_pythonpath() {
  local lab_dir="${ANCHOR_BENCH_ISAACLAB_DIR:-}"
  [[ -n "${lab_dir}" ]] || return 0
  printf '%s' \
    "${lab_dir}/source/isaaclab:${lab_dir}/source/isaaclab_assets:${lab_dir}/source/isaaclab_tasks:${lab_dir}/source/isaaclab_newton:${lab_dir}/source/isaaclab_visualizers"
}

EXEC_ENV=()
build_exec_env() {
  EXEC_ENV=()
  if [[ -z "${HEADLESS:-}" && -n "${DISPLAY:-}" ]]; then
    EXEC_ENV=(-e "DISPLAY=$DISPLAY")
  fi
  if [[ -n "${ANCHOR_BENCH_ISAACLAB_DIR:-}" ]]; then
    local extra_pythonpath
    extra_pythonpath="$(isaaclab_pythonpath)"
    EXEC_ENV+=(
      -e "ANCHOR_BENCH_ISAACLAB_DIR=${ANCHOR_BENCH_ISAACLAB_DIR}"
      -e "ANCHOR_BENCH_EXTRA_PYTHONPATH=${extra_pythonpath}"
    )
  fi
}

print_cmd() {
  printf ' +'
  printf ' %q' "$@"
  printf '\n'
}

dry_run_shell() {
  build_exec_env
  print_cmd docker ps --format '{{.Names}}'
  print_cmd docker exec "${EXEC_ENV[@]}" -w /workspace/anchor-bench "$CONTAINER" python check_environment.py
  echo "[INFO] Container ${CONTAINER} loaded properly"
  if [[ "$#" -eq 0 ]]; then
    print_cmd docker exec -i "${EXEC_ENV[@]}" -w /workspace/anchor-bench "$CONTAINER" bash
  else
    print_cmd docker exec -i "${EXEC_ENV[@]}" -w /workspace/anchor-bench "$CONTAINER" "$@"
  fi
}

if is_dry_run; then
  dry_run_shell "$@"
  exit 0
fi

mkdir -p "${REPO_ROOT}/assets" "${REPO_ROOT}/outputs" "${REPO_ROOT}/third_party/artifacts/h1"

compute_fingerprint() {
  local f
  for f in "${FINGERPRINT_FILES[@]}"; do
    if [[ -f "$f" ]]; then
      printf '%s\n' "$f"
      sha256sum "$f" | awk '{print $1}'
    else
      printf '%s MISSING\n' "$f"
    fi
  done | sha256sum | awk '{print $1}'
}

per_file_fingerprint_blob() {
  local f cur
  for f in "${FINGERPRINT_FILES[@]}"; do
    if [[ -f "$f" ]]; then
      cur="$(sha256sum "$f" | awk '{print $1}')"
    else
      cur="MISSING"
    fi
    printf '%s=%s\n' "$f" "$cur"
  done
}

list_changed_files() {
  local stored_per_file f cur
  stored_per_file="$(docker inspect "$CONTAINER" --format "{{ index .Config.Labels \"${FINGERPRINT_LABEL}.files\" }}" 2>/dev/null || true)"
  for f in "${FINGERPRINT_FILES[@]}"; do
    if [[ -f "$f" ]]; then
      cur="$(sha256sum "$f" | awk '{print $1}')"
    else
      cur="MISSING"
    fi
    if ! grep -Fxq "${f}=${cur}" <<<"${stored_per_file}"; then
      echo "  - $f"
    fi
  done
}

refresh_x11_cookie() {
  [[ -n "${HEADLESS:-}" ]] && return 0
  [[ -z "${DISPLAY:-}" ]] && return 0
  mkdir -p "${XAUTH_DIR}"
  chmod 700 "${XAUTH_DIR}"
  local cookie
  cookie="$(xauth nlist "$DISPLAY" 2>/dev/null || true)"
  [[ -z "$cookie" ]] && return 0
  local tmp
  tmp="$(mktemp "${XAUTH_DIR}/cookie.XXXXXX")"
  chmod 600 "$tmp"
  echo "$cookie" | sed 's/ffff//' | xauth -f "$tmp" nmerge -
  mv -f "$tmp" "$XAUTH_PATH"
}

build_image() {
  echo "[INFO] Rebuilding full image..."
  ANCHOR_BENCH_IMAGE="${IMAGE}" ANCHOR_BENCH_DOCKER_TARGET=full \
    docker compose --file "${COMPOSE_FILE}" build anchor-bench
}

run_environment_check() {
  local output
  if output="$(docker exec "${EXEC_ENV[@]}" -w /workspace/anchor-bench "$CONTAINER" python check_environment.py 2>&1)"; then
    echo "[INFO] Container ${CONTAINER} loaded properly"
  else
    if [[ -n "$output" ]]; then
      printf '%s\n' "$output" >&2
    fi
    echo "[ERROR] Container ${CONTAINER} failed environment check." >&2
    exit 1
  fi
}

start_persistent_container() {
  local current_fp="$1"
  local per_file_blob="$2"

  docker rm -f "$CONTAINER" 2>/dev/null || true

  local xauth_args=()
  if [[ -z "${HEADLESS:-}" && -n "${DISPLAY:-}" ]]; then
    rm -rf "$XAUTH_DIR"
    refresh_x11_cookie
    if [[ -s "$XAUTH_PATH" ]]; then
      echo "[INFO] X11 forwarding enabled"
    else
      echo "[WARN] No X11 cookie for DISPLAY=$DISPLAY; viewer windows will not appear"
    fi
    xauth_args=(
      -e DISPLAY
      -e TERM
      -e QT_X11_NO_MITSHM=1
      -e XAUTHORITY="$XAUTH_PATH"
      -v "$XAUTH_DIR:$XAUTH_DIR"
      -v /tmp/.X11-unix:/tmp/.X11-unix
      -v /etc/localtime:/etc/localtime:ro
    )
  else
    echo "[INFO] Headless (no X11)"
  fi

  local data_env=()
  local data_args=()
  if [[ -n "${ANCHOR_BENCH_DATA_DIR:-}" ]]; then
    data_env=(-e ANCHOR_BENCH_DATA_DIR=/data/anchor-bench)
    data_args=(-v "${ANCHOR_BENCH_DATA_DIR}:/data/anchor-bench:ro")
  else
    data_env=(-e ANCHOR_BENCH_DATA_DIR=)
  fi

  local user_args=(
    --user "${HOST_UID}:${HOST_GID}"
  )
  if [[ "${ANCHOR_BENCH_RUN_AS_ROOT}" == "1" ]]; then
    user_args=(--user 0:0)
    echo "[INFO] ANCHOR_BENCH_RUN_AS_ROOT=1; container will run as root"
  else
    echo "[INFO] Container user mapped to host UID:GID ${HOST_UID}:${HOST_GID}"
  fi

  echo "[INFO] Starting persistent container: $CONTAINER"
  docker run -d \
    --name "$CONTAINER" \
    --network "$ANCHOR_BENCH_NETWORK" \
    --gpus "$ANCHOR_BENCH_GPUS" \
    "${user_args[@]}" \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}" \
    -e ACCEPT_EULA=Y \
    -e OMNI_KIT_ACCEPT_EULA=YES \
    -e HOME=/tmp/anchor-bench-home \
    -e XDG_CACHE_HOME=/tmp/anchor-bench-cache \
    -e MPLCONFIGDIR=/tmp/anchor-bench-cache/matplotlib \
    -e ANCHOR_BENCH_ASSET_DIR=/workspace/anchor-bench/assets/robots \
    "${data_env[@]}" \
    --label "${FINGERPRINT_LABEL}=${current_fp}" \
    --label "${FINGERPRINT_LABEL}.files=${per_file_blob}" \
    -v "${REPO_ROOT}:/workspace/anchor-bench" \
    "${data_args[@]}" \
    -v anchor-bench-cache-kit:/isaac-sim/kit/cache \
    -v anchor-bench-cache-ov:/home/anchor/.cache/ov \
    -v anchor-bench-cache-pip:/home/anchor/.cache/pip \
    -v anchor-bench-cache-gl:/home/anchor/.cache/nvidia/GLCache \
    -v anchor-bench-cache-compute:/home/anchor/.nv/ComputeCache \
    -v anchor-bench-logs:/home/anchor/.nvidia-omniverse/logs \
    -v anchor-bench-data:/home/anchor/.local/share/ov/data \
    -v anchor-bench-docs:/home/anchor/Documents \
    --entrypoint /bin/bash \
    "${xauth_args[@]}" \
    "$IMAGE" \
    -c 'mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}" && sleep infinity' \
    >/dev/null
}

CURRENT_FP="$(compute_fingerprint)"
PER_FILE_BLOB="$(per_file_fingerprint_blob)"

if [[ -n "${REBUILD:-}" ]]; then
  build_image
  start_persistent_container "$CURRENT_FP" "$PER_FILE_BLOB"
elif ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[ERROR] Image '$IMAGE' not found locally. Build it once with:"
    echo "        ./setup.sh --full"
    echo "        or rerun with REBUILD=1 ./setup.sh --shell ..."
    exit 1
  fi
  start_persistent_container "$CURRENT_FP" "$PER_FILE_BLOB"
else
  STORED_FP="$(docker inspect "$CONTAINER" --format "{{ index .Config.Labels \"${FINGERPRINT_LABEL}\" }}" 2>/dev/null || true)"
  if [[ "$STORED_FP" != "$CURRENT_FP" ]]; then
    echo ""
    echo "[WARN] Persistent container is stale relative to current build inputs:"
    list_changed_files
    echo ""
    echo "Options:"
    echo "  REBUILD=1 ./setup.sh --shell ...   # rebuild + recreate"
    echo "  FORCE=1   ./setup.sh --shell ...   # use current container"
    if [[ -n "${STRICT:-}" ]]; then
      echo "[ABORT] STRICT=1 set; refusing to exec into stale container."
      exit 1
    fi
    if [[ -z "${FORCE:-}" ]]; then
      if [[ ! -t 0 ]]; then
        echo "[ABORT] Non-interactive shell and FORCE not set."
        exit 1
      fi
      read -r -p "Continue with stale container? [y/N] " ans
      case "$ans" in
        y|Y|yes|YES) ;;
        *) echo "Aborted."; exit 1 ;;
      esac
    fi
  fi
fi

refresh_x11_cookie
build_exec_env

EXEC_FLAGS=(-i)
if [[ -t 0 && -t 1 ]]; then
  EXEC_FLAGS+=(-t)
fi

run_environment_check

set +e
if [[ "$#" -eq 0 ]]; then
  docker exec "${EXEC_FLAGS[@]}" "${EXEC_ENV[@]}" "$CONTAINER" \
    bash -c 'cd /workspace/anchor-bench && if [[ -n "${ANCHOR_BENCH_EXTRA_PYTHONPATH:-}" ]]; then export PYTHONPATH=/workspace/anchor-bench:${ANCHOR_BENCH_EXTRA_PYTHONPATH}:${PYTHONPATH:-}; else export PYTHONPATH=/workspace/anchor-bench:${PYTHONPATH:-}; fi && exec bash'
else
  docker exec "${EXEC_FLAGS[@]}" "${EXEC_ENV[@]}" "$CONTAINER" \
    bash -c 'cd /workspace/anchor-bench && if [[ -n "${ANCHOR_BENCH_EXTRA_PYTHONPATH:-}" ]]; then export PYTHONPATH=/workspace/anchor-bench:${ANCHOR_BENCH_EXTRA_PYTHONPATH}:${PYTHONPATH:-}; else export PYTHONPATH=/workspace/anchor-bench:${PYTHONPATH:-}; fi && exec "$@"' bash "$@"
fi
EXIT_CODE=$?
exit "$EXIT_CODE"
