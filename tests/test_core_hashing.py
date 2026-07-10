from __future__ import annotations

import math
from pathlib import PurePosixPath, PureWindowsPath

import pytest

from anchor_bench.core.hashing import (
    CanonicalizationError,
    UnsafePathError,
    canonical_bytes,
    canonical_sha256,
    contained_path,
    repo_relative_posix,
    require_relative_posix_path,
    sha256_file,
    sha256_tree,
    tree_fingerprint,
    tree_manifest,
)


def test_canonical_json_is_compact_order_independent_and_order_preserving() -> None:
    first = {"b": 1, "a": [1, 2, {"y": 2.5, "x": None}]}
    second = {"a": (1, 2, {"x": None, "y": 2.5}), "b": 1}
    assert canonical_sha256(first) == canonical_sha256(second)
    assert canonical_bytes({"k": [True, "v"]}) == b'{"k":[true,"v"]}'
    assert canonical_sha256([1, 2]) != canonical_sha256([2, 1])


def test_pure_paths_normalize_to_posix_strings() -> None:
    expected = canonical_sha256({"p": "outputs/sysid/g1"})
    assert canonical_sha256({"p": PurePosixPath("outputs/sysid/g1")}) == expected
    assert canonical_sha256({"p": PureWindowsPath("outputs\\sysid\\g1")}) == expected


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_floats_are_rejected(bad: float) -> None:
    with pytest.raises(CanonicalizationError, match="non-finite"):
        canonical_bytes({"value": bad})


@pytest.mark.parametrize("bad", [{1, 2}, b"bytes", object()])
def test_unsupported_values_are_rejected(bad: object) -> None:
    with pytest.raises(CanonicalizationError):
        canonical_bytes({"value": bad})
    with pytest.raises(CanonicalizationError, match="non-string"):
        canonical_bytes({1: "value"})


def test_raw_file_and_tree_hashes_are_content_based(tmp_path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "b.csv").write_bytes(b"abc")
    (tmp_path / "sub" / "a.csv").write_bytes(b"1")
    assert sha256_file(tmp_path / "b.csv") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    assert list(tree_manifest(tmp_path)) == ["b.csv", "sub/a.csv"]
    fingerprint = tree_fingerprint(tmp_path)
    assert fingerprint["tree_sha256"] == sha256_tree(tmp_path)
    before = sha256_tree(tmp_path)
    (tmp_path / "sub" / "a.csv").write_bytes(b"2")
    assert sha256_tree(tmp_path) != before


def test_tree_hash_rejects_symlinks(tmp_path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (tmp_path / "escape").symlink_to(outside)
    with pytest.raises(UnsafePathError, match="symlink"):
        tree_manifest(tmp_path)


def test_containment_rejects_traversal_and_symlink_escape(tmp_path) -> None:
    inside = tmp_path / "inside"
    inside.mkdir()
    assert contained_path(tmp_path, "inside") == inside.resolve()
    with pytest.raises(UnsafePathError, match="escapes"):
        contained_path(tmp_path, "../escape", must_exist=False)

    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir(exist_ok=True)
    (tmp_path / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(UnsafePathError, match="escapes"):
        contained_path(tmp_path, "link")


def test_repo_relative_paths_are_normalized_and_contained(tmp_path) -> None:
    artifact = tmp_path / "outputs" / "run" / "network.pt"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"network")
    assert (
        repo_relative_posix(artifact, repository_root=tmp_path)
        == "outputs/run/network.pt"
    )
    assert require_relative_posix_path("outputs/run/network.pt") == "outputs/run/network.pt"
    for unsafe in ("../run", "/absolute", "a//b", "./a", "a\\b", ""):
        with pytest.raises(UnsafePathError):
            require_relative_posix_path(unsafe)
