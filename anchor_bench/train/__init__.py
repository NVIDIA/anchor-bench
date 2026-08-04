"""Training-task interfaces."""

from anchor_bench.tasks.train import PreprocessorRunner
from anchor_bench.train.preprocessors import BasePreprocessor

__all__ = [
    "BasePreprocessor",
    "PreprocessorRunner",
]
