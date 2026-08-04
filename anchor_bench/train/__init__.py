"""Training-task interfaces."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anchor_bench.tasks.train.preprocessor_runner import PreprocessorRunner


from anchor_bench.train.preprocessors import BasePreprocessor


def __getattr__(name: str) -> Any:
    """Load task orchestration exports only when requested."""

    if name == "PreprocessorRunner":
        from anchor_bench.tasks.train.preprocessor_runner import PreprocessorRunner

        return PreprocessorRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BasePreprocessor",
    "PreprocessorRunner",
]
