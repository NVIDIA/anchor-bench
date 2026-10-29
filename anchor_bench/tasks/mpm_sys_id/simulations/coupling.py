# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Newton (c) 2026 The Newton Developers, Apache-2.0.

# Warp kernels are compiled DSL functions. Warp's Python stubs do not model
# scalar array indexing or spatial-vector constructors inside a kernel.
# Validate these operations with the CUDA integration tests instead.
# pyright: reportGeneralTypeIssues=false, reportIndexIssue=false, reportArgumentType=false, reportOperatorIssue=false, reportCallIssue=false
import warp as wp


@wp.kernel
def compute_body_forces(
    dt: float,
    collider_ids: wp.array[int],
    collider_impulses: wp.array[wp.vec3],
    collider_impulse_pos: wp.array[wp.vec3],
    body_ids: wp.array[int],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_f: wp.array[wp.spatial_vector],
):
    """Compute forces applied by sand to rigid bodies.

    Sum the impulses applied on each mpm grid node and convert to
    forces and torques at the body's center of mass.
    """

    i = wp.tid()

    cid = collider_ids[i]
    if cid >= 0 and cid < body_ids.shape[0]:
        body_index = body_ids[cid]
        if body_index == -1:
            return

        f_world = collider_impulses[i] / dt

        X_wb = body_q[body_index]
        X_com = body_com[body_index]
        r = collider_impulse_pos[i] - wp.transform_point(X_wb, X_com)
        wp.atomic_add(
            body_f, body_index, wp.spatial_vector(f_world, wp.cross(r, f_world))
        )


@wp.kernel
def subtract_body_force(
    dt: float,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    body_inv_inertia: wp.array[wp.mat33],
    body_inv_mass: wp.array[float],
    body_q_res: wp.array[wp.transform],
    body_qd_res: wp.array[wp.spatial_vector],
):
    """Update the rigid bodies velocity to remove the forces applied by sand at the last step.

    This is necessary to compute the total impulses that are required to enforce the complementarity-based
    frictional contact boundary conditions.
    """

    body_id = wp.tid()

    # Remove previously applied force
    f = body_f[body_id]
    delta_v = dt * body_inv_mass[body_id] * wp.spatial_top(f)
    r = wp.transform_get_rotation(body_q[body_id])

    delta_w = dt * wp.quat_rotate(
        r, body_inv_inertia[body_id] * wp.quat_rotate_inv(r, wp.spatial_bottom(f))
    )

    body_q_res[body_id] = body_q[body_id]
    body_qd_res[body_id] = body_qd[body_id] - wp.spatial_vector(delta_v, delta_w)


@wp.kernel
def _lift_body(
    dt: float,
    speed: float,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    i = wp.tid()
    q = body_q[i]
    body_q[i] = wp.transform(
        wp.transform_get_translation(q) + wp.vec3(0.0, 0.0, speed * dt),
        wp.transform_get_rotation(q),
    )
    body_qd[i] = wp.spatial_vector(wp.vec3(0.0, 0.0, speed), wp.vec3(0.0))
