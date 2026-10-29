# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Drop a water-filled rigid bottle cap-first into dry MPM sand, then remove it.

Provisional bottle geometry and vertical extraction are configurable. Water is
represented by the bottle's rigid mass and inertia; slosh and plastic deformation
are not modeled. Two-way contact follows the MPM two-way coupling example.
"""

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverImplicitMPM

from .arrays import require_array
from .coupling import _lift_body, compute_body_forces, subtract_body_force
from .material import apply_material


def surface_grid(positions, radii, *, extent, cell_size):
    """Rasterize the highest sand sample surface in each XY cell [m]."""
    count = int(np.ceil(2 * extent / cell_size))
    spacing = 2 * extent / count
    indices = np.floor((positions[:, :2] + extent) / spacing).astype(int)
    valid = np.all((indices >= 0) & (indices < count), axis=1)
    heights = np.zeros(count * count)
    np.maximum.at(
        heights,
        indices[valid, 0] * count + indices[valid, 1],
        positions[valid, 2] + radii[valid],
    )
    return heights.reshape(count, count), spacing


def measure_crater(reference, current, *, cell_size, threshold):
    """Measure the central crater's footprint and post-removal rim-to-floor depth.

    Width and length follow fixed X/Y axes. Depth is positive downward. A 3x3
    average suppresses sampling noise; the depression boundary uses an explicit
    threshold [m]. Missing surface bins are ground level, so fully excavated cells
    remain measurable. The reference locates the depression only. Depth uses the
    highest current surface in a three-cell rim band minus the lowest current
    crater-floor surface. Peripheral mound edges are outside the sampled region.
    """
    if reference.shape != current.shape or reference.ndim != 2:
        raise ValueError(
            "Reference and current heightfields must have the same 2D shape"
        )
    if not np.isfinite(reference).all() or not np.isfinite(current).all():
        raise ValueError("Heightfields must be finite")
    if cell_size <= 0 or threshold <= 0:
        raise ValueError("Cell size and crater threshold must be positive")
    delta = reference - current
    padded = np.pad(delta, 1, mode="edge")
    smooth = (
        np.asarray(
            sum(
                padded[i : i + delta.shape[0], j : j + delta.shape[1]]
                for i in range(3)
                for j in range(3)
            )
        )
        / 9
    )
    mask = smooth >= threshold
    # Select the deepest central depression; ignore detached peripheral sampling artifacts.
    center = np.array(delta.shape) // 2
    radius = max(1, int(0.10 / cell_size))
    lo = np.maximum(center - radius, 0)
    hi = np.minimum(center + radius + 1, delta.shape)
    region = smooth[lo[0] : hi[0], lo[1] : hi[1]]
    start = tuple(lo + np.array(np.unravel_index(np.argmax(region), region.shape)))
    if not mask[start]:
        return {
            "width_m": 0.0,
            "length_m": 0.0,
            "depth_m": 0.0,
            "touches_measurement_boundary": False,
        }
    pending = [start]
    visited = {start}
    while pending:
        i, j = pending.pop()
        for neighbor in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            x, y = neighbor
            if (
                0 <= x < mask.shape[0]
                and 0 <= y < mask.shape[1]
                and mask[x, y]
                and neighbor not in visited
            ):
                visited.add(neighbor)
                pending.append(neighbor)
    cells = np.array(sorted(visited))
    lower, upper = cells.min(axis=0), cells.max(axis=0)
    component = np.zeros_like(mask)
    component[cells[:, 0], cells[:, 1]] = True
    expanded = component.copy()
    for _ in range(3):
        padding = np.pad(expanded, 1)
        expanded = np.logical_or.reduce(
            [
                padding[i : i + mask.shape[0], j : j + mask.shape[1]]
                for i in range(3)
                for j in range(3)
            ]
        )
    rim = expanded & ~component
    if not np.any(rim):
        raise ValueError(
            "Crater occupies the whole measurement region; enlarge the region"
        )
    padding = np.pad(current, 1, mode="edge")
    surface = (
        np.asarray(
            sum(
                padding[i : i + mask.shape[0], j : j + mask.shape[1]]
                for i in range(3)
                for j in range(3)
            )
        )
        / 9
    )
    rim_height = float(np.max(surface[rim]))
    floor_height = float(np.min(surface[component]))
    return {
        "width_m": float((upper[0] - lower[0] + 1) * cell_size),
        "length_m": float((upper[1] - lower[1] + 1) * cell_size),
        "depth_m": max(0.0, rim_height - floor_height),
        "rim_height_m": rim_height,
        "floor_height_m": floor_height,
        "touches_measurement_boundary": bool(
            expanded[0].any()
            or expanded[-1].any()
            or expanded[:, 0].any()
            or expanded[:, -1].any()
        ),
    }


def bottle_mesh(radius, height, cap_radius, cap_height):
    """Build a capped axisymmetric bottle mesh with its cap tip at local z=0."""
    segments = 32
    profile = [
        (0.0, cap_radius),
        (cap_height, cap_radius),
        (cap_height + 0.01, cap_radius * 0.8),
        (cap_height + 0.04, radius),
        (height, radius),
    ]
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    vertices = [[r * np.cos(a), r * np.sin(a), z] for z, r in profile for a in angles]
    vertices += [[0, 0, 0], [0, 0, height]]
    indices = []
    for ring in range(len(profile) - 1):
        for i in range(segments):
            j = (i + 1) % segments
            a, b = ring * segments + i, ring * segments + j
            indices += [a, b, a + segments, b, b + segments, a + segments]
    bottom, top = len(vertices) - 2, len(vertices) - 1
    for i in range(segments):
        j = (i + 1) % segments
        indices += [
            bottom,
            j,
            i,
            top,
            (len(profile) - 1) * segments + i,
            (len(profile) - 1) * segments + j,
        ]
    # Integrate disk slices to obtain the water-filled solid's center and inertia per kg.
    z = (np.arange(256) + 0.5) * height / 256
    r = np.interp(z, [v[0] for v in profile], [v[1] for v in profile])
    weights = r**2 / np.sum(r**2)
    com_z = float(np.sum(weights * z))
    axial = float(np.sum(weights * r**2 / 2))
    transverse = float(np.sum(weights * (r**2 / 4 + (z - com_z) ** 2)))
    return (
        np.asarray(vertices, dtype=np.float32),
        np.asarray(indices, dtype=np.int32),
        com_z,
        (transverse, transverse, axial),
    )


class Simulation:
    def __init__(self, args):
        for name in (
            "drop_height_m",
            "water_volume_m3",
            "density",
            "voxel_size",
            "fps",
            "substeps",
            "bed_width",
            "bed_length",
            "bed_depth",
            "bottle_radius",
            "bottle_height",
            "cap_radius",
            "cap_height",
            "pre_settle_seconds",
            "impact_settle_seconds",
            "lift_speed",
            "lift_distance",
            "crater_threshold",
        ):
            if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if (
            args.bottle_height <= args.cap_height + 0.04
            or args.bottle_radius <= args.cap_radius
        ):
            raise ValueError(
                "Bottle must be taller and wider than its cap and shoulder"
            )
        if (
            args.bed_depth < max(0.30, 4 * args.voxel_size)
            or min(args.bed_width, args.bed_length) < 0.4
        ):
            raise ValueError(
                "Use at least 0.30 m of sand, four voxels deep, and a bed at least 0.4 m wide"
            )
        if args.voxel_size > args.cap_radius:
            raise ValueError(
                "Resolve the cap with at least two voxels across its diameter"
            )
        self.args = args
        self.sim_time = 0.0
        self.frame_dt = 1.0 / args.fps
        self.sim_dt = self.frame_dt / args.substeps
        self.reference = None
        self.phase = "settling bed"
        self.removal_start = args.pre_settle_seconds + args.impact_settle_seconds
        self.removal_end = self.removal_start + args.lift_distance / args.lift_speed
        self.mass = args.water_volume_m3 * 1000.0
        self.min_cap_z = float("inf")
        builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(builder)
        spacing = args.voxel_size / 2
        self.footprint = np.array([args.bed_width, args.bed_length])
        axes = [
            np.arange(-size / 2 + spacing / 2, size / 2, spacing)
            for size in self.footprint
        ]
        z = np.arange(spacing / 2, args.bed_depth, spacing)
        points = np.stack(np.meshgrid(*axes, z, indexing="ij"), axis=-1).reshape(-1, 3)
        # Static walls retain a deep bed without simulating a wide sloped mound.
        wall = 0.025
        wall_height = args.bed_depth + 0.05
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=args.ground_friction)
        for axis in (0, 1):
            for sign in (-1, 1):
                center = [0.0, 0.0, wall_height / 2]
                center[axis] = sign * (self.footprint[axis] / 2 + wall / 2)
                half = [
                    args.bed_width / 2 + wall,
                    args.bed_length / 2 + wall,
                    wall_height / 2,
                ]
                half[axis] = wall / 2
                builder.add_shape_box(
                    body=-1,
                    xform=wp.transform(wp.vec3(*center), wp.quat_identity()),
                    hx=half[0],
                    hy=half[1],
                    hz=half[2],
                    cfg=cfg,
                    color=(0.25, 0.32, 0.38),
                )
        rng = np.random.default_rng(42)
        points += rng.uniform(-0.12 * spacing, 0.12 * spacing, points.shape)
        builder.add_particles(
            pos=[wp.vec3(*point) for point in points],
            vel=[wp.vec3(0.0)] * len(points),
            mass=[args.density * spacing**3] * len(points),
            radius=[spacing / 2] * len(points),
        )
        vertices, indices, com_z, inertia = bottle_mesh(
            args.bottle_radius, args.bottle_height, args.cap_radius, args.cap_height
        )
        builder.add_body(
            xform=wp.transform(
                wp.vec3(0, 0, args.bed_depth + args.drop_height_m), wp.quat_identity()
            ),
            com=wp.vec3(0, 0, com_z),
            mass=self.mass,
            inertia=wp.mat33(np.diag(inertia) * self.mass),
            lock_inertia=True,
            label="water_bottle",
        )
        builder.add_shape_mesh(
            body=0,
            mesh=newton.Mesh(vertices, indices, compute_inertia=False),
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=args.bottle_friction),
            color=(0.15, 0.5, 0.8),
        )
        builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(mu=args.ground_friction)
        )
        self.builder = builder
        self.model = builder.finalize()
        self.model.set_gravity((0, 0, -9.81))
        apply_material(self.model, args)
        config = SolverImplicitMPM.Config()
        config.voxel_size = args.voxel_size
        config.max_iterations = 100
        config.max_active_cell_count = max(8192, len(points) // 4)
        self.solver = SolverImplicitMPM(self.model, config=config)
        self.state_0 = self.model.state()
        self.rigid_state = self.model.state()
        self.kinematic_mass = wp.zeros_like(require_array(self.model.body_mass))
        self.kinematic_inertia = wp.zeros_like(
            require_array(self.model.body_inv_inertia)
        )
        self.solver.setup_collider(
            body_mass=self.kinematic_mass, body_inv_inertia=self.kinematic_inertia
        )
        self.colors = wp.array(
            rng.uniform(0.8, 1.0, (len(points), 1)) * np.array([0.82, 0.64, 0.36]),
            dtype=wp.vec3,
        )
        self.radii = require_array(self.model.particle_radius).numpy()
        self.surface_extent = min(args.bed_width, args.bed_length) * 0.45
        self.initial_mass = float(require_array(self.model.particle_mass).numpy().sum())
        self.metrics = None
        self.use_cuda_graph = (
            self.model.device.is_cuda
            and wp.is_conditional_graph_supported()
            and wp.is_mempool_enabled(self.model.device)
            and not args.no_cuda_graph
        )
        self.graphs = {}
        self.warmed_phases = set()

    def _surface(self):
        return surface_grid(
            require_array(self.state_0.particle_q).numpy(),
            self.radii,
            extent=self.surface_extent,
            cell_size=self.args.voxel_size,
        )

    def step(self):
        args = self.args
        for _ in range(args.substeps):
            if (
                self.phase == "settling bed"
                and self.sim_time >= args.pre_settle_seconds - 1e-8
            ):
                self.reference, self.surface_spacing = self._surface()
                self.solver.setup_collider()
                self.phase = "falling / impact"
            if (
                self.phase == "falling / impact"
                and self.sim_time >= self.removal_start - 1e-8
            ):
                self.solver.setup_collider(
                    body_mass=self.kinematic_mass,
                    body_inv_inertia=self.kinematic_inertia,
                )
                require_array(self.state_0.body_qd).zero_()
                self.phase = "removing bottle"
            if (
                self.phase == "removing bottle"
                and self.sim_time >= self.removal_end - 1e-8
            ):
                require_array(self.state_0.body_qd).zero_()
                self.phase = "settling crater"
            if self.use_cuda_graph and self.phase in self.warmed_phases:
                if self.phase not in self.graphs:
                    with wp.ScopedCapture(device=self.model.device) as capture:
                        self._substep()
                    self.graphs[self.phase] = capture.graph
                wp.capture_launch(self.graphs[self.phase])
            else:
                # Warm the solver's internal graph without adding an extra timestep.
                self._substep()
                self.warmed_phases.add(self.phase)
            self.sim_time += self.sim_dt
        self.solver.check_sparse_grid_rebuild_status()
        cap_z = float(require_array(self.state_0.body_q).numpy()[0, 2])
        self.min_cap_z = min(self.min_cap_z, cap_z)

    def _substep(self):
        args = self.args

        if self.phase == "falling / impact":
            self.solver.integrate_bodies(
                self.model, self.state_0, self.rigid_state, self.sim_dt
            )
            wp.launch(
                subtract_body_force,
                dim=1,
                inputs=[
                    self.sim_dt,
                    self.rigid_state.body_q,
                    self.rigid_state.body_qd,
                    self.state_0.body_f,
                    self.model.body_inv_inertia,
                    self.model.body_inv_mass,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                ],
            )
        elif self.phase == "removing bottle":
            wp.launch(
                _lift_body,
                dim=1,
                inputs=[
                    self.sim_dt,
                    args.lift_speed,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                ],
            )
        # Implicit MPM accepts absent controls/contacts; its stubs require both.
        self.solver.step(self.state_0, self.state_0, None, None, self.sim_dt)  # pyright: ignore[reportArgumentType]
        # Enforce containment at box/floor contacts, including corner crossings.
        self.solver.project_outside(self.state_0, self.state_0, self.sim_dt)
        if self.phase == "falling / impact":
            require_array(self.state_0.body_qd).assign(self.rigid_state.body_qd)
            # Capture the current solve's impulse fields into a persistent force
            # buffer; capturing prior-step field handles can freeze feedback.
            self.state_0.clear_forces()
            impulses, positions, ids = self.solver.collect_collider_impulses(
                self.state_0
            )
            wp.launch(
                compute_body_forces,
                dim=ids.shape[0],
                inputs=[
                    self.sim_dt,
                    ids,
                    impulses,
                    positions,
                    self.solver.collider_body_index,
                    self.state_0.body_q,
                    self.model.body_com,
                    self.state_0.body_f,
                ],
            )

    def measure(self):
        positions = require_array(self.state_0.particle_q).numpy()
        velocities = require_array(self.state_0.particle_qd).numpy()
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            raise ValueError("Non-finite sand state")
        result = {
            "rms_speed_m_s": float(np.sqrt(np.mean(np.sum(velocities**2, axis=1)))),
            "phase": self.phase,
            "valid": False,
            "min_cap_z_m": self.min_cap_z,
        }
        if self.reference is not None:
            current, _ = self._surface()
            result.update(
                measure_crater(
                    self.reference,
                    current,
                    cell_size=self.surface_spacing,
                    threshold=self.args.crater_threshold,
                )
            )
            result["valid"] = (
                self.phase == "settling crater"
                and self.sim_time >= self.removal_end + 1.0
                and result["rms_speed_m_s"] < 0.03
                and not result["touches_measurement_boundary"]
            )
        return result

    def test_final(self):
        """Check finite two-way impact states, floor containment, and sand mass."""
        assert np.isfinite(require_array(self.state_0.body_q).numpy()).all(), (
            "Bottle pose is non-finite"
        )
        assert np.isfinite(require_array(self.state_0.body_qd).numpy()).all(), (
            "Bottle velocity is non-finite"
        )
        assert np.isfinite(require_array(self.state_0.particle_q).numpy()).all(), (
            "Sand position is non-finite"
        )
        assert np.isfinite(require_array(self.state_0.particle_qd).numpy()).all(), (
            "Sand velocity is non-finite"
        )
        assert (
            require_array(self.state_0.particle_q).numpy()[:, 2].min()
            > -self.args.voxel_size
        ), "Sand penetrated the floor"
        np.testing.assert_allclose(
            require_array(self.model.body_mass).numpy()[0], self.mass, rtol=1e-6
        )
        np.testing.assert_allclose(
            require_array(self.model.particle_mass).numpy().sum(),
            self.initial_mass,
            rtol=1e-6,
        )
        if self.sim_time > self.removal_end + 1:
            assert self.min_cap_z < self.args.bed_depth, (
                "Bottle never indented the sand"
            )
            assert self.min_cap_z > 0, "Bottle passed through the sand bed"
            assert self.measure()["valid"], "Crater not ready for measurement"
