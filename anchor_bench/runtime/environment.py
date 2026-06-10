from __future__ import annotations

import importlib.util
from collections.abc import Iterable


class RuntimeUnavailableError(RuntimeError):
    """Raised when a command needs the full Isaac/Newton runtime."""


def require_full_runtime(modules: Iterable[str], workflow: str) -> None:
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if not missing:
        return
    raise RuntimeUnavailableError(
        f"{workflow} requires the full Anchor Bench Docker runtime, but these modules are missing: "
        f"{', '.join(missing)}. Build it with `bash setup.sh --full`, then run inside "
        "`anchor-bench:full` with GPU/runtime mounts."
    )
