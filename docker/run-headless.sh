#!/usr/bin/env bash
# Run the Anchor Bench full container without GUI/X11 forwarding.
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,2p' "$0"
  echo "#"
  echo "# Usage:"
  echo "#   ./docker/run-headless.sh"
  echo "#   ./docker/run-headless.sh anchor-bench benchmark --robot so101 --motion inputs/datasets/so101/custom_figure8_with_camera --headless"
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HEADLESS=1
exec "${SCRIPT_DIR}/run-gui.sh" "$@"
