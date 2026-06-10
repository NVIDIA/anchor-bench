#!/usr/bin/env bash
set -euo pipefail

: "${HOME:=/tmp/anchor-bench-home}"
: "${XDG_CACHE_HOME:=${HOME}/.cache}"
: "${MPLCONFIGDIR:=${XDG_CACHE_HOME}/matplotlib}"
export HOME XDG_CACHE_HOME MPLCONFIGDIR
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"

if [[ -d /workspace/anchor-bench/anchor_bench ]]; then
  export PYTHONPATH="/workspace/anchor-bench:${PYTHONPATH:-}"
  cd /workspace/anchor-bench
else
  cd /opt/anchor-bench
fi

if [[ "$#" -eq 0 ]]; then
  exec bash
fi

exec "$@"
