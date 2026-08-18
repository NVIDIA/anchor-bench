# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extension interfaces for training-dataset preprocessing."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from anchor_bench.core.simulation import SimulationContext


class BaseDataProcessor(ABC):
    """Base class for generating features from a resampled dataset."""

    def __init__(self, dataset: pd.DataFrame) -> None:
        self.dataset = dataset
        self.simulation_context = SimulationContext.current()

    @abstractmethod
    def process(self) -> pd.DataFrame:
        """Return generated feature columns aligned to the dataset index."""
        raise NotImplementedError
