
# Anchor Bench Agent Guide

This repo targets the Isaac Sim, Newton, and Isaac Lab versions pinned in
`versions/compatibility.toml` — the single source of truth, consumed by
`setup.sh` and `anchor_bench.core.versions`. Do not duplicate the pinned
versions here; read them from that manifest.

## Rules

- Keep robot support declarative under `configs/robots/<robot>/`.
- Do not add legacy `model_type` actuator configs.
- Keep Isaac Sim, Isaac Lab, and Newton imports lazy so dry-run tests run on a host-only Python.
- Do not commit large datasets, neural weights, FMUs, generated benchmark outputs, or plots.
- Prefer `anchor-bench ... --dry-run` for command construction tests.

## Main Workflows

- Add a robot: follow `docs/robot-plugin.md`.
- Run SysID: use `anchor-bench sysid ... --dry-run` first; see `docs/cli.md`.
- Benchmark an exact candidate: use `anchor-bench benchmark ... --dry-run`
  first; see `docs/cli.md`.
- Record run results as generated artifacts under `outputs/`; see `docs/artifacts.md`.
