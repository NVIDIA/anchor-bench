from __future__ import annotations


def resolve_positive_int(name: str, value: int | None, default: int) -> int:
    resolved = default if value is None else value
    resolved = int(resolved)
    if resolved < 1:
        raise ValueError(f"{name} must be >= 1")
    return resolved


def resolve_optional_positive_int(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    return resolve_positive_int(name, value, 1)
