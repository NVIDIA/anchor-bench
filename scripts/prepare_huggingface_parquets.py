#!/usr/bin/env python3
"""Download a Hugging Face parquet folder and convert robot telemetry to SAGE motions."""

from __future__ import annotations

import argparse
import shutil
import hashlib
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from urllib.parse import quote, unquote, urlparse

from anchor_bench.data.sage import write_sage_motion


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_COMPONENT = re.compile(r"[^a-zA-Z0-9._-]+")
_SPLIT_MARKER = re.compile(r"(?:^|[-_])(heldout|test|validation|val|train)(?:[-_]|$)", re.IGNORECASE)
_SPLIT_NAMES = {
    "train": "train",
    "validation": "validation",
    "val": "validation",
    "heldout": "heldout",
    "test": "heldout",
}
_SIGNAL_CANDIDATES = {
    "command": ("command_q", "target_q", "desired_q", "q_des"),
    "position": ("actual_q", "q"),
    "velocity": ("actual_dq", "dq"),
    "effort": ("tau_est", "tau", "tau_abs"),
    "effort_sign": ("present_load_raw",),
}


@dataclass(frozen=True)
class HuggingFaceFolder:
    repo_id: str
    revision: str
    path: str

    @property
    def dataset_name(self) -> str:
        return _safe_component_name(PurePosixPath(self.path).name)


@dataclass(frozen=True)
class RemoteParquet:
    path: str
    relative_path: str
    url: str
    sha256: str | None
    split: str
    motion: str


@dataclass(frozen=True)
class SignalFields:
    source_joints: tuple[str, ...]
    command: Mapping[str, str]
    position: Mapping[str, str]
    velocity: Mapping[str, str]
    effort: Mapping[str, str]
    effort_sign: Mapping[str, str]
    effort_kind: str
    control_fallback: bool


@dataclass(frozen=True)
class ImportLayout:
    robot: str
    dataset: str
    download_dir: Path
    output_root: Path


def _safe_component_name(value: str) -> str:
    result = _SAFE_COMPONENT.sub("-", value).strip("._-").lower()
    if not result:
        raise ValueError(f"cannot derive a safe path component from {value!r}")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_huggingface_folder(url: str) -> HuggingFaceFolder:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.netloc != "huggingface.co":
        raise ValueError("hf_dir must be a https://huggingface.co/datasets/.../tree/... URL")
    parts = [unquote(part) for part in parsed.path.strip("/").split("/")]
    if len(parts) < 6 or parts[0] != "datasets" or parts[3] != "tree":
        raise ValueError("hf_dir must identify a Hugging Face dataset folder")
    repo_id = "/".join(parts[1:3])
    revision = parts[4]
    folder_path = "/".join(parts[5:])
    if not repo_id or not revision or not folder_path:
        raise ValueError("hf_dir is missing the dataset, revision, or folder path")
    return HuggingFaceFolder(repo_id=repo_id, revision=revision, path=folder_path)


def _motion_identity(remote_path: str, folder_path: str) -> tuple[str, str, str]:
    remote = PurePosixPath(remote_path)
    try:
        relative = remote.relative_to(PurePosixPath(folder_path))
    except ValueError as exc:
        raise ValueError(f"remote parquet is outside the requested folder: {remote_path}") from exc
    stem = relative.stem
    marker = _SPLIT_MARKER.search(stem)
    if marker is None:
        split = "train"
        motion_stem = stem
    else:
        split = _SPLIT_NAMES[marker.group(1).lower()]
        motion_stem = stem[marker.end():].strip("-_") or stem
    parent = "--".join(relative.parent.parts) if relative.parent != PurePosixPath(".") else ""
    motion = _safe_component_name("--".join(part for part in (parent, motion_stem) if part))
    return split, motion, relative.as_posix()


def _import_huggingface_hub():
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ModuleNotFoundError as exc:
        raise RuntimeError("Hugging Face import requires huggingface_hub; install anchor-bench[full]") from exc
    return HfApi, hf_hub_download


def _item_value(item: Any, name: str, default: Any = None) -> Any:
    return item.get(name, default) if isinstance(item, dict) else getattr(item, name, default)


def _lfs_sha256(item: Any) -> str | None:
    lfs = _item_value(item, "lfs")
    if lfs is None:
        return None
    digest = _item_value(lfs, "sha256", _item_value(lfs, "oid", ""))
    return str(digest) if _SHA256.fullmatch(str(digest)) else None


def _list_huggingface_parquets(folder_url: str, *, api: Any | None = None) -> list[RemoteParquet]:
    folder = _parse_huggingface_folder(folder_url)
    if api is None:
        HfApi, _ = _import_huggingface_hub()
        api = HfApi()
    payload = api.list_repo_tree(
        folder.repo_id,
        path_in_repo=folder.path,
        recursive=True,
        expand=False,
        revision=folder.revision,
        repo_type="dataset",
    )

    entries: list[RemoteParquet] = []
    identities: set[tuple[str, str]] = set()
    for item in payload:
        remote_path = str(_item_value(item, "path", ""))
        if not remote_path.lower().endswith(".parquet"):
            continue
        split, motion, relative_path = _motion_identity(remote_path, folder.path)
        identity = (split, motion)
        if identity in identities:
            raise ValueError(f"multiple parquet files map to {split}/{motion}")
        identities.add(identity)
        entries.append(
            RemoteParquet(
                path=remote_path,
                relative_path=relative_path,
                url=(
                    f"https://huggingface.co/datasets/{quote(folder.repo_id, safe='/')}/resolve/"
                    f"{quote(folder.revision, safe='')}/{quote(remote_path, safe='/')}"
                ),
                sha256=_lfs_sha256(item),
                split=split,
                motion=motion,
            )
        )
    if not entries:
        raise ValueError(f"no parquet files found in {folder_url}")
    return sorted(entries, key=lambda entry: entry.path)


def _verified_source(
    entry: RemoteParquet,
    folder: HuggingFaceFolder,
    download_dir: Path,
    *,
    no_download: bool,
    downloader: Any | None = None,
) -> Path:
    source = download_dir / PurePosixPath(entry.relative_path)
    if not source.exists():
        if no_download:
            raise ValueError(f"source parquet does not exist: {source}")
        if downloader is None:
            _, downloader = _import_huggingface_hub()
        cached = Path(
            downloader(
                repo_id=folder.repo_id,
                filename=entry.path,
                repo_type="dataset",
                revision=folder.revision,
            )
        )
        source.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, source)
    if entry.sha256 is not None:
        digest = _sha256(source)
        if digest != entry.sha256:
            raise ValueError(f"source SHA-256 mismatch for {source}: expected {entry.sha256}, got {digest}")
    return source


def _field_map(fields: Sequence[str], suffix: str) -> dict[str, str]:
    result: dict[str, str] = {}
    slash_suffix = f"/{suffix}"
    flat_suffix = f"_{suffix}"
    longer_flat_suffixes = tuple(
        f"_{candidate}"
        for candidates in _SIGNAL_CANDIDATES.values()
        for candidate in candidates
        if candidate != suffix and candidate.endswith(suffix)
    )
    for field in fields:
        joint: str | None = None
        if field.endswith(slash_suffix):
            joint = field[: -len(slash_suffix)]
        elif field.endswith(flat_suffix) and not field.endswith(longer_flat_suffixes):
            joint = field[: -len(flat_suffix)]
        if joint:
            result[joint] = field
    return result


def _first_signal(fields: Sequence[str], candidates: Sequence[str]) -> tuple[str, dict[str, str]]:
    for candidate in candidates:
        mapping = _field_map(fields, candidate)
        if mapping:
            return candidate, mapping
    return "", {}


def _discover_signal_fields(fields: Sequence[str]) -> SignalFields:
    _, command = _first_signal(fields, _SIGNAL_CANDIDATES["command"])
    _, position = _first_signal(fields, _SIGNAL_CANDIDATES["position"])
    _, velocity = _first_signal(fields, _SIGNAL_CANDIDATES["velocity"])
    effort_kind, effort = _first_signal(fields, _SIGNAL_CANDIDATES["effort"])
    _, effort_sign = _first_signal(fields, _SIGNAL_CANDIDATES["effort_sign"])
    missing = [
        name
        for name, mapping in (("position", position), ("velocity", velocity), ("effort", effort))
        if not mapping
    ]
    if missing:
        raise ValueError(f"parquet is missing recognizable {', '.join(missing)} signals")

    source_joints = tuple(sorted(set(position) & set(velocity) & set(effort)))
    if not source_joints:
        raise ValueError("position, velocity, and effort signals have no common joints")
    control_fallback = not set(source_joints).issubset(command)
    if control_fallback:
        command = {joint: position[joint] for joint in source_joints}
    return SignalFields(
        source_joints=source_joints,
        command={joint: command[joint] for joint in source_joints},
        position={joint: position[joint] for joint in source_joints},
        velocity={joint: velocity[joint] for joint in source_joints},
        effort={joint: effort[joint] for joint in source_joints},
        effort_sign={joint: effort_sign[joint] for joint in source_joints if joint in effort_sign},
        effort_kind=effort_kind,
        control_fallback=control_fallback,
    )


@lru_cache(maxsize=None)
def _robot_joint_configuration(robot: str) -> tuple[tuple[str, ...], Mapping[str, str]] | None:
    try:
        from anchor_bench.robots import get_robot, list_robots

        if robot not in list_robots():
            return None
        spec = get_robot(robot).spec
        return tuple(spec.joint_names), dict(spec.joint_aliases)
    except (KeyError, OSError, ValueError):
        return None


def _normalized_joint(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _canonical_joint_pairs(source_joints: Sequence[str], robot: str) -> list[tuple[str, str]]:
    configuration = _robot_joint_configuration(robot)
    if configuration is None:
        return [(joint, joint) for joint in source_joints]
    canonical_joints, aliases = configuration
    lookup: dict[str, str] = {}
    for canonical in canonical_joints:
        lookup[_normalized_joint(canonical)] = canonical
    for alias, canonical in aliases.items():
        lookup[_normalized_joint(alias)] = canonical

    pairs = [(source, lookup.get(_normalized_joint(source), source)) for source in source_joints]
    canonical_order = {joint: index for index, joint in enumerate(canonical_joints)}
    pairs.sort(key=lambda pair: (canonical_order.get(pair[1], len(canonical_order)), pair[1]))
    resolved = [canonical for _, canonical in pairs]
    duplicates = sorted({joint for joint in resolved if resolved.count(joint) > 1})
    if duplicates:
        raise ValueError(f"source joints map to duplicate canonical joints: {', '.join(duplicates)}")
    return pairs


def _pivot_signal(frame: Any, field_map: Mapping[str, str], joint_pairs: Sequence[tuple[str, str]], label: str) -> Any:
    ordered_fields = [field_map[source] for source, _ in joint_pairs]
    selected = frame[frame["field"].isin(ordered_fields)].copy()
    if selected.duplicated(["time_ns", "field"]).any():
        raise ValueError(f"source parquet has duplicate {label} samples")
    wide = selected.pivot(index="time_ns", columns="field", values="value").sort_index()
    missing = [field for field in ordered_fields if field not in wide]
    if missing:
        raise ValueError(f"source parquet is missing {label} fields: {', '.join(missing)}")
    wide = wide[ordered_fields]
    wide.columns = [canonical for _, canonical in joint_pairs]
    if wide.isna().any().any():
        raise ValueError(f"source parquet {label} signals contain incomplete samples")
    return wide


def _vector_rows(frame: Any, row_type: str, origin_ns: int, *signals: Any) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for timestamp_ns, values in frame.iterrows():
        timestamp_s = round((int(timestamp_ns) - origin_ns) / 1e9, 12)
        row: list[Any] = [row_type, timestamp_s, [float(value) for value in values.tolist()]]
        for signal in signals:
            row.append([float(value) for value in signal.loc[timestamp_ns].tolist()])
        rows.append(row)
    return rows


def convert_parquet(
    source: Path,
    output: Path,
    *,
    robot: str,
    source_url: str,
    source_revision: str,
    source_split: str,
    source_motion: str,
    source_sha256: str | None,
) -> None:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("conversion requires pandas and pyarrow; run it inside ./setup.sh --shell") from exc

    frame = pd.read_parquet(source, columns=["time_ns", "experiment", "field", "value"])
    experiments = sorted(str(value) for value in frame["experiment"].dropna().unique())
    if len(experiments) != 1:
        raise ValueError(f"expected one experiment in {source}, found {experiments}")
    fields = [str(value) for value in frame["field"].dropna().unique()]
    signals = _discover_signal_fields(fields)
    joint_pairs = _canonical_joint_pairs(signals.source_joints, robot)

    commands = _pivot_signal(frame, signals.command, joint_pairs, "command position")
    positions = _pivot_signal(frame, signals.position, joint_pairs, "position")
    velocities = _pivot_signal(frame, signals.velocity, joint_pairs, "velocity")
    efforts = _pivot_signal(frame, signals.effort, joint_pairs, "effort")
    sign_applied = signals.effort_kind == "tau_abs" and len(signals.effort_sign) == len(signals.source_joints)
    sign_signal = (
        _pivot_signal(frame, signals.effort_sign, joint_pairs, "effort sign")
        if sign_applied
        else None
    )

    state_index = positions.index.intersection(velocities.index).intersection(efforts.index)
    if sign_signal is not None:
        state_index = state_index.intersection(sign_signal.index)
    if state_index.empty:
        raise ValueError("position, velocity, and effort signals have no common timestamps")
    positions = positions.loc[state_index]
    velocities = velocities.loc[state_index]
    efforts = efforts.loc[state_index]
    if sign_signal is not None:
        sign_signal = sign_signal.loc[state_index]
        efforts = efforts * sign_signal.map(lambda value: -1.0 if value < 0.0 else 1.0 if value > 0.0 else 0.0)
    origin_ns = int(min(commands.index.min(), state_index.min()))

    canonical_joints = [canonical for _, canonical in joint_pairs]
    effort_semantics = signals.effort_kind
    if signals.effort_kind == "tau_abs":
        effort_semantics += "_signed_by_present_load_raw" if sign_applied else "_unsigned"
    metadata = {
        "schema": "sage",
        "time_unit": "seconds",
        "position_unit": "rad",
        "velocity_unit": "rad/s",
        "torque_unit": "N*m",
        "robot": robot,
        "source": experiments[0],
        "source_url": source_url,
        "source_revision": source_revision,
        "source_sha256": source_sha256 or _sha256(source),
        "source_split": source_split,
        "source_motion": source_motion,
        "source_joint_names": list(signals.source_joints),
        "source_fields": {
            "control_positions": [signals.command[source] for source, _ in joint_pairs],
            "state_positions": [signals.position[source] for source, _ in joint_pairs],
            "state_velocities": [signals.velocity[source] for source, _ in joint_pairs],
            "state_torques": [signals.effort[source] for source, _ in joint_pairs],
        },
        "control_semantics": (
            "measured_position_fallback"
            if signals.control_fallback
            else "recorded_command_position"
        ),
        "effort_semantics": effort_semantics,
        "unit_note": (
            "Units follow the source field conventions and must be checked against "
            "the source robot's hardware calibration before physical conclusions."
        ),
    }
    write_sage_motion(
        output,
        canonical_joints,
        _vector_rows(commands, "CONTROL", origin_ns),
        _vector_rows(positions, "STATE_MOTOR", origin_ns, velocities, efforts),
        metadata,
    )
    print(f"wrote {len(commands)} control rows and {len(positions)} state rows to {output}")


def _infer_robot_name(folder: HuggingFaceFolder, explicit_robot: str | None) -> str:
    if explicit_robot:
        return _safe_component_name(explicit_robot)
    folder_name = folder.dataset_name
    first_token = re.split(r"[-_]", folder_name, maxsplit=1)[0]
    try:
        from anchor_bench.robots import list_robots

        configured = list_robots()
    except (OSError, ValueError):
        configured = []
    if first_token in configured:
        return first_token
    matches = [name for name in configured if re.search(rf"(?:^|[-_]){re.escape(name)}(?:[-_]|$)", folder_name)]
    if len(matches) == 1:
        return matches[0]
    return first_token


def _import_layout(
    folder: HuggingFaceFolder,
    *,
    robot: str | None,
    download_dir: Path | None,
    output_root: Path | None,
) -> ImportLayout:
    robot_name = _infer_robot_name(folder, robot)
    dataset = folder.dataset_name
    base = Path("inputs/datasets") / robot_name
    return ImportLayout(
        robot=robot_name,
        dataset=dataset,
        download_dir=download_dir or base / "_parquet" / dataset,
        output_root=output_root or base / dataset,
    )


def import_huggingface_folder(
    folder_url: str,
    *,
    robot: str | None = None,
    download_dir: Path | None = None,
    output_root: Path | None = None,
    no_download: bool = False,
) -> list[Path]:
    folder = _parse_huggingface_folder(folder_url)
    layout = _import_layout(folder, robot=robot, download_dir=download_dir, output_root=output_root)
    entries = _list_huggingface_parquets(folder_url)
    outputs: list[Path] = []
    for index, entry in enumerate(entries, start=1):
        print(f"[{index}/{len(entries)}] {entry.split}/{entry.motion}", flush=True)
        source = _verified_source(entry, folder, layout.download_dir, no_download=no_download)
        output = layout.output_root / entry.split / entry.motion
        convert_parquet(
            source,
            output,
            robot=layout.robot,
            source_url=entry.url,
            source_revision=folder.revision,
            source_split=entry.split,
            source_motion=entry.motion,
            source_sha256=entry.sha256,
        )
        outputs.append(output)
    counts = {split: sum(path.parent.name == split for path in outputs) for split in _SPLIT_NAMES.values()}
    summary = ", ".join(f"{count} {split}" for split, count in sorted(counts.items()) if count)
    print(f"converted {summary} motions for {layout.robot} into {layout.output_root}")
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hf_dir", help="Hugging Face dataset folder URL")
    parser.add_argument("--robot", help="output robot name; inferred from the folder name by default")
    parser.add_argument("--output", type=Path, help="SAGE dataset output root")
    parser.add_argument("--download-dir", type=Path, help="local parquet cache root")
    parser.add_argument("--no-download", action="store_true", help="require every parquet to exist in the cache")
    args = parser.parse_args(argv)
    import_huggingface_folder(
        args.hf_dir,
        robot=args.robot,
        download_dir=args.download_dir,
        output_root=args.output,
        no_download=args.no_download,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
