# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Newton (c) 2026 The Newton Developers, Apache-2.0.

import numpy as np


def create_funnel_mesh(
    aperture_radius, top_radius, height, z_offset, thickness=0.005, num_segments=64
):
    """Generate a thick-walled funnel mesh open at both ends.

    The funnel is a truncated cone extruded radially outward by *thickness*
    to form a closed shell.  Four rings of vertices define the inner and
    outer surfaces, connected by top and bottom rims.  All face normals
    point outward from the solid wall so the collider works correctly.

    Args:
        aperture_radius: Radius of the narrow opening at the bottom [m].
        top_radius: Radius of the wide opening at the top [m].
        height: Vertical extent of the funnel [m].
        z_offset: Z position of the funnel bottom [m].
        thickness: Radial wall thickness [m].
        num_segments: Number of segments around the circumference.

    Returns:
        Tuple of (vertices, indices) suitable for :class:`newton.Mesh`.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, num_segments, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    n = num_segments

    def ring(radius, z):
        return np.column_stack([radius * cos_t, radius * sin_t, np.full(n, z)])

    # Ring 0: inner bottom (aperture)
    # Ring 1: inner top
    # Ring 2: outer top
    # Ring 3: outer bottom (aperture)
    vertices = np.vstack(
        [
            ring(aperture_radius, z_offset),
            ring(top_radius, z_offset + height),
            ring(top_radius + thickness, z_offset + height),
            ring(aperture_radius + thickness, z_offset),
        ]
    ).astype(np.float32)

    indices = []
    for i in range(n):
        j = (i + 1) % n
        r0_i, r0_j = i, j
        r1_i, r1_j = i + n, j + n
        r2_i, r2_j = i + 2 * n, j + 2 * n
        r3_i, r3_j = i + 3 * n, j + 3 * n

        # Inner wall (ring0 -> ring1): normals face inward (toward axis)
        indices.extend([r0_i, r1_i, r0_j])
        indices.extend([r0_j, r1_i, r1_j])

        # Outer wall (ring3 -> ring2): normals face outward (away from axis)
        indices.extend([r3_i, r3_j, r2_i])
        indices.extend([r2_i, r3_j, r2_j])

        # Top rim (ring1 -> ring2): normals face up
        indices.extend([r1_i, r2_i, r1_j])
        indices.extend([r1_j, r2_i, r2_j])

        # Bottom rim (ring3 -> ring0): normals face down
        indices.extend([r3_i, r0_i, r3_j])
        indices.extend([r3_j, r0_i, r0_j])

    indices = np.array(indices, dtype=np.int32)
    return vertices, indices
