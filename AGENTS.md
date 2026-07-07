
# Anchor Bench Agent Guide

This repo targets Isaac Sim 6.0, Newton `1.4.0.dev0` at commit
`ae365981f0ba40371d2b19489e4ba07a5e5767ce`, and Isaac Lab `develop` at
`5f2db9902baa724427b7aa654835aa559fce64e8`.

## Rules

- Keep robot support declarative under `configs/robots/<robot>/`.
- Do not add legacy `model_type` actuator configs.
- Keep Isaac Sim, Isaac Lab, and Newton imports lazy so dry-run tests run on a host-only Python.
- Do not commit large datasets, neural weights, FMUs, generated benchmark outputs, or plots.
- Prefer `anchor-bench ... --dry-run` for command construction tests.

## Main Workflows

- Add a robot: follow `docs/robot-plugin.md`.
- Run SysID: use `anchor-bench sysid ... --dry-run` first; see `docs/cli.md`.
- Run benchmark/analysis: use `anchor-bench benchmark`, `anchor-bench analyze`, or
  `anchor-bench experiment ... --dry-run`; see `docs/cli.md`.
- Record run results as generated artifacts under `outputs/`; see `docs/artifacts.md`.
