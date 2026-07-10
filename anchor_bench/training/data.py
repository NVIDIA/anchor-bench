"""Training source discovery, episode splitting, and stats serialization."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from anchor_bench.data.sage import discover_sage_motions


@dataclass(frozen=True)
class TrainingSource:
    """One recorded episode, before physical mappings are expanded."""

    path: Path


def discover_training_sources(root: str | Path) -> list[TrainingSource]:
    """Discover valid SAGE episode directories below one dataset root."""

    path = Path(root).expanduser()
    motions = discover_sage_motions(path)
    if not motions:
        raise ValueError(f"no SAGE motion folders found under {path}")
    return [TrainingSource(motion) for motion in motions]


def split_training_sources(
    sources: Sequence[TrainingSource],
    *,
    seed: int,
    val_fraction: float = 0.2,
) -> tuple[list[TrainingSource], list[TrainingSource]]:
    """Split recorded episodes deterministically before mapping expansion."""

    if not sources:
        raise ValueError("cannot split an empty training source list")
    groups: dict[Path, list[TrainingSource]] = {}
    for source in sources:
        groups.setdefault(source.path, []).append(source)
    paths = list(groups)
    random.Random(seed).shuffle(paths)
    if len(paths) == 1:
        return list(groups[paths[0]]), []
    val_count = max(1, int(round(len(paths) * val_fraction)))
    val_count = min(val_count, len(paths) - 1)
    val_paths = set(paths[:val_count])

    def source_key(source: TrainingSource) -> str:
        """Return a deterministic ordering key for one episode."""

        return str(source.path)

    train = [source for path in paths if path not in val_paths for source in groups[path]]
    val = [source for path in paths if path in val_paths for source in groups[path]]
    return sorted(train, key=source_key), sorted(val, key=source_key)


def write_stats_json(path: str | Path, stats: dict[str, Any]) -> Path:
    """Write deterministic human-readable training statistics as JSON."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output
