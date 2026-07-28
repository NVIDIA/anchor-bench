#!/usr/bin/env bash
# Generate a CycloneDX 1.6 SBOM for anchor-bench (nSpect-compatible JSON).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VERSION="$(python3 -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")"
OUT_DIR="${OUT_DIR:-$ROOT/sbom}"
TOOLS_VENV="${TOOLS_VENV:-/tmp/anchor-bench-sbom-tools}"
RUNTIME_VENV="${RUNTIME_VENV:-/tmp/anchor-bench-sbom-runtime}"
CDX_JSON="$OUT_DIR/anchor-bench-${VERSION}-python.cdx.json"
CSV_OUT="$OUT_DIR/anchor-bench-${VERSION}-python.csv"
FREEZE_OUT="$OUT_DIR/requirements-freeze.txt"

# Public installable "full" extras (skip isaacsim / newton / mujoco-warp on host).
FULLISH_DEPS=(
  cmaes
  optuna
  scipy
  pandas
  pyarrow
  "huggingface_hub>=0.34"
  matplotlib
  mujoco
)

mkdir -p "$OUT_DIR"

echo "Creating tools venv at $TOOLS_VENV"
python3 -m venv "$TOOLS_VENV"
"$TOOLS_VENV/bin/pip" install -q -U pip cyclonedx-bom

echo "Creating runtime venv at $RUNTIME_VENV"
python3 -m venv "$RUNTIME_VENV"
"$RUNTIME_VENV/bin/pip" install -q -U pip setuptools wheel
"$RUNTIME_VENV/bin/pip" install -q -e .
"$RUNTIME_VENV/bin/pip" install -q "${FULLISH_DEPS[@]}"

echo "Writing CycloneDX 1.6 SBOM to $CDX_JSON"
"$TOOLS_VENV/bin/cyclonedx-py" environment \
  --pyproject pyproject.toml \
  --mc-type application \
  --sv 1.6 \
  --of JSON \
  -o "$CDX_JSON" \
  "$RUNTIME_VENV/bin/python"

"$RUNTIME_VENV/bin/pip" freeze | grep -v '^-e' > "$FREEZE_OUT"

python3 - <<PY
import csv
import json
import tomllib
from pathlib import Path

root = Path(${ROOT@Q})
version = ${VERSION@Q}
cdx_path = Path(${CDX_JSON@Q})
csv_path = Path(${CSV_OUT@Q})
compat = tomllib.loads((root / "versions" / "compatibility.toml").read_text())
sbom = json.loads(cdx_path.read_text())
comps = sorted(sbom.get("components") or [], key=lambda c: c.get("name", "").lower())

declared_missing = [
    ("isaacsim", compat["isaac_sim"]["version"], "library", f"pkg:pypi/isaacsim@{compat['isaac_sim']['version']}", "", "", "full extra; host SBOM env could not install"),
    ("newton", compat["newton"]["version"], "library", f"pkg:pypi/newton@{compat['newton']['version']}", "", "", "full extra; host SBOM env could not install"),
    ("mujoco-warp", ">=3.5.0", "library", "pkg:pypi/mujoco-warp", "", "", "full extra; host SBOM env could not install"),
    (
        "IsaacLab",
        compat["isaac_lab"]["commit"],
        "library",
        f"pkg:github/isaac-sim/IsaacLab@{compat['isaac_lab']['commit']}",
        "",
        "",
        "pinned in versions/compatibility.toml; Docker full image",
    ),
]

with csv_path.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["name", "version", "type", "purl", "licenses", "bom-ref", "source"])
    for c in comps:
        licenses = []
        for entry in c.get("licenses") or []:
            lic = entry.get("license") or {}
            licenses.append(lic.get("id") or lic.get("name") or "")
        w.writerow(
            [
                c.get("name", ""),
                c.get("version", ""),
                c.get("type", ""),
                c.get("purl", ""),
                ";".join(licenses),
                c.get("bom-ref", ""),
                "installed-runtime-env",
            ]
        )
    for row in declared_missing:
        w.writerow(list(row))

print(f"components={len(comps)} + {len(declared_missing)} declared-missing")
print(f"wrote {cdx_path}")
print(f"wrote {csv_path}")
print(f"wrote {Path(${FREEZE_OUT@Q})}")
PY

echo "Done."
