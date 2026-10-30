# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How candidate PD gains reach a Newton simulation during sysID."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

import newton
import numpy as np

from anchor_bench.tasks.actuator_sys_id.model import Candidate


class NewtonActuation(ABC):
    """Apply candidate PD gains for a subset of joints across all worlds."""

    # True when before_substep must run inside the physics substep loop.
    runs_before_substep: ClassVar[bool] = False

    @abstractmethod
    def apply_population(self, population: Sequence[Candidate]) -> None:
        """Write each world's candidate stiffness and damping."""

    def before_substep(
        self, state: newton.State, control: newton.Control | None, dt: float
    ) -> None:
        """Update actuator efforts before a physics substep."""

    def reset(self, control: newton.Control) -> None:
        """Clear per-episode actuator state when a trajectory restarts."""

    def is_graphable(self) -> bool:
        """Return whether before_substep can run inside a captured graph."""
        return True


class NativePD(NewtonActuation):
    """Fit gains through the solver's own joint position drive."""

    _ARRAYS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("stiffness", "joint_target_ke"),
        ("damping", "joint_target_kd"),
    )

    def __init__(
        self,
        model: newton.Model,
        *,
        joint_names: Sequence[str],
        dof_indices: np.ndarray,
    ) -> None:
        self._model = model
        self._joint_names = tuple(joint_names)
        self._dof_indices = dof_indices
        self._base = {
            array_name: np.array(getattr(model, array_name).numpy(), copy=True)
            for _, array_name in self._ARRAYS
        }

    def apply_population(self, population: Sequence[Candidate]) -> None:
        for parameter, array_name in self._ARRAYS:
            values = np.array(self._base[array_name], copy=True)
            for world, candidate in enumerate(population):
                for column, joint in enumerate(self._joint_names):
                    values[self._dof_indices[world, column]] = candidate[joint][
                        parameter
                    ]
            getattr(self._model, array_name).assign(values)
