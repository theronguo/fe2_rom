"""Full-order first-order (CH1) periodic hyperelastic homogenization solver."""
import os
import logging
from typing import Callable

import numpy as np
from dolfinx import fem, io, mesh as dmesh
from petsc4py import PETSc
from mpi4py import MPI
import ufl
import dolfinx_mpc

from fe2_rom.ch1.averages import (
    EffectiveAbar,
    EffectiveFbar,
    EffectivePbar,
    HomogenizationContext,
    resolve_average_quantities,
)
from fe2_rom.ch1.constraints import LinearConstraint, ZeroVolumeAverage
from fe2_rom.ch1.exceptions import RVEConvergenceError
from fe2_rom.ch1.solvers import NewtonSolverFE2
from fe2_rom.hyperelastic_solver.forms import basis_tensor_ufl, build_homogenization_weak_form
from fe2_rom.hyperelastic_solver.logging_utils import silence_c_stdout
from fe2_rom.hyperelastic_solver.material import MaterialModel
from fe2_rom.hyperelastic_solver.output import VTXManager
from fe2_rom.hyperelastic_solver.stability import (
    StabilityAnalyzer,
    apply_eigenmode_perturbation,
    mesh_characteristic_length,
)
from fe2_rom.hyperelastic_solver.timestepping import TimeStepper

logger = logging.getLogger(__name__)


class MicroSolver:
    """Periodic hyperelastic homogenization with pluggable effective quantities.

    Periodic ties are enforced via ``dolfinx_mpc`` for faces and corners
    (all non-max corners are slaved to the max corner). The Newton step is
    the standard ``NewtonSolverFE2`` path with CG/MINRES + GAMG
    (no saddle-point augmentation).

    Effective quantities and tangent moduli are computed through the modular
    ``AverageQuantity`` interface; ``__call__`` returns
    ``list[dict[str, ...]]`` — one dict per accepted load step, keyed by
    each quantity's ``name``.

    Subclass hooks (``_setup_phi``, ``_setup_macro_vars``,
    ``_build_u_total_extra``, ``_build_macro_var_rhs_forms``,
    ``_make_default_average_quantities``, etc.) allow derived classes to add
    additional macro variables, ansatz contributions, constraints, and
    effective quantities — used by the micromorphic subclass.
    """

    def __init__(self, mesh_path, comm, gdim,
                 material: MaterialModel, *,
                 degree: int = 1,
                 output_dir: str = "output",
                 check_stability: bool = True,
                 visualize_fields: list[str] | None = None,
                 average_quantities: list | None = None,
                 stability_options: dict | None = None,
                 newton_options: dict | None = None,
                 timestepper_options: dict | None = None,
                 save_snapshots: list[str] | None = None,
                 averages_only_final: bool = False,
                 corner_periodic: bool = False,
                 constraints: "list[LinearConstraint] | None" = None,
                 rve_volume: float | None = None,
                 ) -> None:

        newton_options = newton_options if newton_options is not None else {
            "rel_tol": 1e-8, "abs_tol": 1e-6, "max_iter": 10, "max_iter_instab": 30,
            "switch_to_minres": False, "div_rel_tol": 10.0,
        }
        timestepper_options = timestepper_options if timestepper_options is not None else {
            "t_end": 1.0, "dt_init": 1.0, "dt_min": 1e-5, "dt_max": 1.0, "good_newton_steps": 7,
        }
        stability_options = stability_options if stability_options is not None else {
            "nev": 5, "neg_tol": -1e-12,
        }
        if visualize_fields is None:
            visualize_fields = ["u_fluc"]
        if save_snapshots is None:
            save_snapshots = []

        # ---- Mesh ----
        self.comm = comm
        with silence_c_stdout():
            mesh_data = io.gmsh.read_from_msh(mesh_path, self.comm, 0, gdim=gdim)
        self._mesh = mesh_data.mesh
        self._cell_tags = mesh_data.cell_tags
        self._facet_tags = mesh_data.facet_tags
        self._mesh.topology.create_connectivity(self._mesh.topology.dim - 1, self._mesh.topology.dim)
        self.dx = ufl.Measure("dx", domain=self._mesh, subdomain_data=self._cell_tags)
        self.gdim = gdim

        self.mins, self.maxs = self._compute_domain_bounds()
        self.length_scale = (self.maxs - self.mins).max()
        logger.debug("Mesh loaded: %d cells, %d facets, gdim=%d",
                     self._mesh.topology.index_map(self._mesh.topology.dim).size_global,
                     self._mesh.topology.index_map(self._mesh.topology.dim - 1).size_global,
                     gdim)
        logger.debug("Domain bounds: x [%.3f, %.3f]", self.mins[0], self.maxs[0])
        logger.debug("Domain bounds: y [%.3f, %.3f]", self.mins[1], self.maxs[1])
        if gdim == 3:
            logger.debug("Domain bounds: z [%.3f, %.3f]", self.mins[2], self.maxs[2])

        self._averages_only_final = averages_only_final
        self._material = material

        # ---- Function space and state fields ----
        self._degree = degree
        self.V = fem.functionspace(self._mesh, ("Lagrange", degree, (self.gdim,)))
        n_dofs = self.V.dofmap.index_map.size_global * self.V.dofmap.index_map_bs

        self.u = fem.Function(self.V)
        self._u_last = fem.Function(self.V)
        self._u_conv = fem.Function(self.V)
        self._du = fem.Function(self.V)
        self._eigenfunction = fem.Function(self.V)
        self.F_bar = fem.Constant(self._mesh, np.eye(self.gdim, dtype=PETSc.ScalarType))
        self.F_bar_conv = np.eye(self.gdim, dtype=PETSc.ScalarType)
        logger.debug("Functions set up with %d global DOFs", n_dofs)

        # ---- Periodic BCs + MPC ----
        self._corner_periodic = corner_periodic
        bcs, self.mpc = self._setup_periodic_bcs_and_mpc()
        self._bcs = bcs
        logger.debug("Periodic BCs set up with %d slave points",
                     self.comm.allreduce(len(self.mpc.slaves), op=MPI.SUM))

        # ---- Subclass hooks: φᵢ and the macro-variable dict ----
        self._setup_phi()
        self._setup_macro_vars()

        # Characteristic mesh extent — fallback length scale for the
        # eigenmode perturbation when |u| is ~0 (e.g. first load step).
        self._char_length = mesh_characteristic_length(self._mesh)

        # ---- Stability analyzer (optional) ----
        if check_stability:
            n_skip_default = self._count_zero_modes()
            n_skip = stability_options.pop("n_skip_eigenvalues", n_skip_default)
            self._stability = StabilityAnalyzer(
                self.comm, n_skip_eigenvalues=n_skip, **stability_options,
            )
            logger.debug("Stability checks enabled (skipping %d gauge eigenvalues); options: %s",
                         n_skip, stability_options)
            if "switch_to_minres" in newton_options:
                if newton_options["switch_to_minres"] is False:
                    logger.debug("Overriding newton_options['switch_to_minres'] to True for stability checks.")
            newton_options["switch_to_minres"] = True
        else:
            self._stability = None

        # ---- Weak form (with optional u_total_extra from subclass) ----
        u_total_extra = self._build_u_total_extra()
        (R_form, J_form, F_var, P_ufl, J_ufl, W_ufl, A_ufl, u_total,
         build_tangent_rhs_forms) = build_homogenization_weak_form(
            self._mesh, self.V, self.u, self.F_bar, self._material,
            u_total_extra=u_total_extra, dx=self.dx,
        )

        # ---- Adjoint-RHS forms per macro variable (subclass extends) ----
        self._macro_var_rhs_forms = self._build_macro_var_rhs_forms(build_tangent_rhs_forms)

        # ---- Constraint forms (projected Newton solve) ----
        constraint_forms = self._build_constraint_forms(constraints)

        # ---- Newton solver (NewtonSolverFE2 with periodic MPC ties) ----
        self._newton = NewtonSolverFE2(
            self.comm, R_form, J_form, self.u, self._du, self._bcs, self.mpc,
            constraint_forms=constraint_forms,
            **newton_options,
        )
        logger.debug("Newton solver initialized with options: %s", newton_options)

        # ---- Time stepper ----
        self._timestepper = TimeStepper(**timestepper_options)
        logger.debug("Time stepper initialized with options: %s", timestepper_options)

        # ---- Visualization ----
        self._setup_visualization(visualize_fields, F_var, P_ufl, J_ufl, W_ufl, u_total, output_dir)

        # ---- Volume + homogenization context ----
        # Macroscopic averaging is (1/|Q|) ∫_Ω · dX with |Q| the periodic-cell
        # volume — voids counted.  For arbitrary (e.g. hexagonal) cells the
        # caller MUST pass ``rve_volume``; the bounding box would be wrong.
        # If not provided we fall back to ∫_Ω 1 dx = |Ω_solid|, which equals
        # |Q| only for non-porous RVEs.
        if rve_volume is None:
            vol_local = fem.assemble_scalar(fem.form(1.0 * self.dx))
            self._vol_global = float(self.comm.allreduce(vol_local, op=MPI.SUM))
        else:
            self._vol_global = float(rve_volume)
        self._context = HomogenizationContext(
            mesh=self._mesh, V=self.V, dx=self.dx, comm=self.comm,
            vol_global=self._vol_global,
            F_var=F_var, P_ufl=P_ufl, A_ufl=A_ufl, W_ufl=W_ufl,
            u=self.u, u_total=u_total,
            macro_vars=self.macro_vars,
            phi=self._phi,
        )

        # ---- Average quantities ----
        if average_quantities is None:
            average_quantities = self._make_default_average_quantities()
        self._average_quantities = resolve_average_quantities(average_quantities)
        for q in self._average_quantities:
            q.setup(self._context)

        # ---- Snapshot saving ----
        self.output_dir = output_dir
        self.save_snapshots = save_snapshots
        logger.debug("Snapshot fields: %s", save_snapshots)
        logger.debug("Setup complete (n_dofs=%d)", n_dofs)

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _setup_phi(self) -> None:
        self._phi: list[fem.Function] = []

    def _setup_macro_vars(self) -> None:
        self.macro_vars: dict = {"Fbar": self.F_bar}

    def _build_u_total_extra(self):
        return None

    def _build_macro_var_rhs_forms(self, build_tangent_rhs_forms) -> dict:
        gdim = self.gdim
        dF_dFbar_list = [basis_tensor_ufl(gdim, i, j)
                         for i in range(gdim) for j in range(gdim)]
        return {"Fbar": build_tangent_rhs_forms(dF_dFbar_list)}

    def _make_default_average_quantities(self) -> list:
        return [EffectiveFbar(), EffectivePbar(), EffectiveAbar()]

    def _build_constraint_forms(self, constraints: "list[LinearConstraint] | None") -> list:
        if constraints is None:
            constraints = [ZeroVolumeAverage()] if self._corner_periodic else []
        if not constraints:
            return []
        forms = []
        for c in constraints:
            c_forms, _ = c.build(self.V, self.dx, self._mesh, self.mpc)
            forms.extend(c_forms)
        logger.debug("Constraint forms: %d rows from %d constraint object(s)",
                     len(forms), len(constraints))
        return forms

    def _restore_trial_state(self) -> None:
        return

    def _update_macro_load_schedule(self, t: float) -> None:
        return

    def _commit_extra_state(self) -> None:
        return

    def _count_zero_modes(self) -> int:
        return self.gdim + 1 if self._corner_periodic else 0

    # ------------------------------------------------------------------
    # Mesh / MPC helpers
    # ------------------------------------------------------------------

    def _compute_domain_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        coords = self._mesh.geometry.x
        mins_local = np.min(coords[:, :self.gdim], axis=0)
        maxs_local = np.max(coords[:, :self.gdim], axis=0)
        mins = np.array(
            [self.comm.allreduce(float(mins_local[i]), op=MPI.MIN) for i in range(self.gdim)],
            dtype=float,
        )
        maxs = np.array(
            [self.comm.allreduce(float(maxs_local[i]), op=MPI.MAX) for i in range(self.gdim)],
            dtype=float,
        )
        return mins, maxs

    @staticmethod
    def _make_axis_map(axis: int, target_value: float) -> Callable:
        def axis_map(x):
            y = x.copy()
            y[axis] = target_value
            return y
        return axis_map

    @staticmethod
    def _make_periodic_slave_selector(axis: int, mins: np.ndarray, maxs: np.ndarray,
                                      tol: float, exclude_axes: tuple[int, ...],
                                      corners: list[np.ndarray]) -> Callable:
        def selector(x):
            mask = np.isclose(x[axis], mins[axis], atol=tol, rtol=0.0)
            for ex_axis in exclude_axes:
                mask &= (x[ex_axis] > mins[ex_axis] + tol)
                mask &= (x[ex_axis] < maxs[ex_axis] - tol)
            for corner in corners:
                on_corner = np.ones(x.shape[1], dtype=bool)
                for ax in range(len(corner)):
                    on_corner &= np.isclose(x[ax], corner[ax], atol=tol, rtol=0.0)
                mask &= ~on_corner
            return mask
        return selector

    @staticmethod
    def _make_corner_selector(corner_coords: np.ndarray, tol: float) -> Callable:
        def selector(x):
            mask = np.ones(x.shape[1], dtype=bool)
            for axis, value in enumerate(corner_coords):
                mask &= np.isclose(x[axis], value, atol=tol, rtol=0.0)
            return mask
        return selector

    @staticmethod
    def _make_corner_map(master_coords: np.ndarray) -> Callable:
        def corner_map(x):
            y = x.copy()
            for axis, value in enumerate(master_coords):
                y[axis] = value
            return y
        return corner_map

    @staticmethod
    def _corner_coordinates(mins: np.ndarray, maxs: np.ndarray) -> list[np.ndarray]:
        dim = len(mins)
        corners: list[np.ndarray] = []
        for bits in range(1 << dim):
            coord = np.empty(dim, dtype=float)
            for axis in range(dim):
                coord[axis] = maxs[axis] if (bits >> axis) & 1 else mins[axis]
            corners.append(coord)
        return corners

    @staticmethod
    def _locate_corner_dofs(V, mins: np.ndarray, maxs: np.ndarray, tol: float) -> np.ndarray:
        dim = len(mins)

        def corner_selector(x):
            mask = np.ones(x.shape[1], dtype=bool)
            for axis in range(dim):
                on_min = np.isclose(x[axis], mins[axis], atol=tol, rtol=0.0)
                on_max = np.isclose(x[axis], maxs[axis], atol=tol, rtol=0.0)
                mask &= on_min | on_max
            return mask

        return fem.locate_dofs_geometrical(V, corner_selector)

    def _setup_periodic_bcs_and_mpc(self) -> tuple[list, dolfinx_mpc.MultiPointConstraint]:
        if self.gdim not in (2, 3):
            raise ValueError(
                f"Periodic homogenization supports only 2D rectangle or 3D cuboid, got dim={self.gdim}."
            )
        tol = 1e-8 * max(1.0, float(np.max(self.maxs - self.mins)))

        if self._corner_periodic:
            bcs: list = []
        else:
            corner_dofs = self._locate_corner_dofs(self.V, self.mins, self.maxs, tol)
            u_zero = fem.Constant(self._mesh, np.zeros(self.gdim, dtype=PETSc.ScalarType))
            bcs = [fem.dirichletbc(u_zero, corner_dofs, self.V)]

        mpc = dolfinx_mpc.MultiPointConstraint(self.V)
        for axis in range(self.gdim):
            selector = self._make_periodic_slave_selector(
                axis=axis, mins=self.mins, maxs=self.maxs, tol=tol,
                exclude_axes=tuple(range(axis)), corners=self._corner_coordinates(self.mins, self.maxs)
            )
            axis_map = self._make_axis_map(axis, self.maxs[axis])
            mpc.create_periodic_constraint_geometrical(self.V, selector, axis_map, bcs)

        if self._corner_periodic:
            master_corner = self.maxs.copy()
            corner_map = self._make_corner_map(master_corner)
            for corner_coords in self._corner_coordinates(self.mins, self.maxs):
                if np.allclose(corner_coords, master_corner, atol=tol, rtol=0.0):
                    continue
                selector = self._make_corner_selector(corner_coords, tol)
                mpc.create_periodic_constraint_geometrical(self.V, selector, corner_map, bcs)

        mpc.finalize()

        return bcs, mpc

    # ------------------------------------------------------------------
    # Visualization and snapshots
    # ------------------------------------------------------------------

    def _setup_visualization(self, visualize_fields, F_var, P_ufl, J_ufl, W_ufl, u_total, output_dir) -> None:
        fields = []
        for field in visualize_fields:
            TT = fem.functionspace(self._mesh, ("DG", 1, (self.gdim, self.gdim)))
            V1 = fem.functionspace(self._mesh, ("DG", 1, (self.gdim,)))
            SS = fem.functionspace(self._mesh, ("DG", 1))
            if field == "u_fluc":
                self.u_int = fem.Function(V1, name="u_fluc")
                fields.append(self.u_int)
            elif field == "u_total":
                self.u_total = fem.Function(V1, name="u_total")
                self._u_total_expr = fem.Expression(u_total, V1.element.interpolation_points)
                fields.append(self.u_total)
            elif field == "F":
                self.F_func = fem.Function(TT, name="F")
                self._F_expr = fem.Expression(F_var, TT.element.interpolation_points)
                fields.append(self.F_func)
            elif field == "P":
                self.P_func = fem.Function(TT, name="P")
                self._P_expr = fem.Expression(P_ufl, TT.element.interpolation_points)
                fields.append(self.P_func)
            elif field == "J":
                self.J_func = fem.Function(SS, name="J")
                self._J_expr = fem.Expression(J_ufl, SS.element.interpolation_points)
                fields.append(self.J_func)
            elif field == "W":
                self.W_func = fem.Function(SS, name="W")
                self._W_expr = fem.Expression(W_ufl, SS.element.interpolation_points)
                fields.append(self.W_func)
        if fields:
            self.vtx = VTXManager(self.comm, f"{output_dir}/solution.bp", fields)
        else:
            self.vtx = None
        self.visualize_fields = visualize_fields
        logger.debug("Visualization fields: %s", visualize_fields)

    def _write_fields(self, t: float) -> None:
        if self.vtx is not None:
            for field in self.visualize_fields:
                if field == "u_fluc":
                    self.u_int.interpolate(self.u)
                elif field == "u_total":
                    self.u_total.interpolate(self._u_total_expr)
                elif field == "F":
                    self.F_func.interpolate(self._F_expr)
                elif field == "P":
                    self.P_func.interpolate(self._P_expr)
                elif field == "J":
                    self.J_func.interpolate(self._J_expr)
                elif field == "W":
                    self.W_func.interpolate(self._W_expr)
            self.vtx.write(t)
        else:
            logger.warning("No fields to write at t=%.5f (vtx is None)", t)

    def _save_snapshot(self, field_name: str, func, t_save: float) -> None:
        imap = func.function_space.dofmap.index_map
        bs = func.function_space.dofmap.index_map_bs
        n_local = imap.size_local
        owned_vals = func.x.array[:n_local * bs].copy()
        owned_coords = func.function_space.tabulate_dof_coordinates()[:n_local].copy()
        all_vals = self.comm.gather(owned_vals, root=0)
        all_coords = self.comm.gather(owned_coords, root=0)
        if self.comm.rank == 0:
            vals = np.concatenate(all_vals)
            coords = np.concatenate(all_coords, axis=0)
            snap_dir = f"{self.output_dir}/snapshots"
            np.save(f"{snap_dir}/{field_name}_{t_save:.5f}.npy", vals)
            coords_path = f"{snap_dir}/{field_name}_dof_coords.npy"
            if not os.path.exists(coords_path):
                np.save(coords_path, coords)

    # ------------------------------------------------------------------
    # Averaging and __call__
    # ------------------------------------------------------------------

    def _collect_averages(self) -> dict:
        needed: set[str] = set()
        for q in self._average_quantities:
            for name in q.required_macro_adjoints:
                needed.add(name)
        adjoints: dict | None = None
        if needed:
            rhs_dict = {name: self._macro_var_rhs_forms[name] for name in needed}
            adjoints = self._newton.solve_macro_sensitivities(rhs_dict)
        out: dict = {}
        for q in self._average_quantities:
            out[q.name] = q.compute(self._context, adjoints)
        return out

    def __call__(self, Fbar: np.ndarray, *,
                 pert_amplitude_init: float = 1e-2,
                 plot_time_start: float = 0.0) -> list[dict]:
        assert self._newton is not None, "Setup not complete."

        Fbar_prev = self.F_bar_conv.copy()
        self.F_bar.value[:] = Fbar_prev
        self.u.x.array[:] = self._u_conv.x.array
        self.u.x.scatter_forward()
        self._restore_trial_state()

        target_F = np.asarray(Fbar, dtype=PETSc.ScalarType)

        def load_schedule(t: float) -> None:
            for i in range(target_F.shape[0]):
                for j in range(target_F.shape[1]):
                    self.F_bar.value[i, j] = (
                        t * (target_F[i, j] - Fbar_prev[i, j]) + Fbar_prev[i, j]
                    )
            self._update_macro_load_schedule(t)

        u = self.u
        if self.vtx is not None:
            self._write_fields(0.0 + plot_time_start)

        output_quantities: list[dict] = []
        if not self._averages_only_final:
            output_quantities.append(self._collect_averages())

        self._timestepper.reset()
        while not self._timestepper.finished:
            trial_time = self._timestepper.step_forward()
            logger.info("── Step  t=%.5f  dt=%.2e", trial_time, self._timestepper.dt)

            load_schedule(trial_time)

            stable_configuration = False
            pert_amplitude = pert_amplitude_init
            iter_newton = 0
            self._newton.reset_for_new_timestep()

            while not stable_configuration:
                converged, iter_newton = self._newton.solve(iter_start=iter_newton)

                if converged:
                    if self._stability is not None:
                        K = self._newton.assemble_stiffness()
                        try:
                            is_stable, eigenvalues = self._stability.check(K, self._eigenfunction)
                        except (PETSc.Error, SystemError):
                            logger.error("Stability check failed.")
                            ok = self._timestepper.reject()
                            if not ok:
                                logger.error(
                                    "Minimum time step dt=%.2e reached — stopping.",
                                    self._timestepper.dt_min,
                                )
                                u.x.array[:] = self._u_last.x.array[:]
                                u.x.scatter_forward()
                                raise RVEConvergenceError(
                                    f"Stability check failed and dt_min={self._timestepper.dt_min:.2e} "
                                    f"reached at t={self._timestepper.t_current:.4f}"
                                )
                            else:
                                logger.warning(
                                    "Eigensolver crashed — halving dt to %.2e",
                                    self._timestepper.dt,
                                )
                            u.x.array[:] = self._u_last.x.array[:]
                            u.x.scatter_forward()
                            break
                    else:
                        is_stable, eigenvalues = True, np.array([])

                    if not is_stable:
                        target = np.where(eigenvalues < self._stability._neg_tol)[0]
                        scale, info = apply_eigenmode_perturbation(
                            u, self._eigenfunction, pert_amplitude, self.comm,
                            char_length=self._char_length,
                        )
                        u_ref, phi_max = info[0]
                        logger.warning(
                            "Unstable equilibrium (λ_min=%.4e) — "
                            "perturbing with eigenvector "
                            "(factor=%.2e, |u|=%.2e, ‖perturbation‖_∞=%.2e)",
                            eigenvalues[target[0]], pert_amplitude, u_ref,
                            scale * phi_max,
                        )
                        pert_amplitude *= 2
                    else:
                        stable_configuration = True
                        self._timestepper.accept(iter_newton)
                        self._u_last.x.array[:] = u.x.array[:]
                        self._u_last.x.scatter_forward()

                        if not self._averages_only_final:
                            output_quantities.append(self._collect_averages())

                        t_save = self._timestepper.t_current + plot_time_start
                        if self.vtx is not None:
                            self._write_fields(t_save)

                        if self.save_snapshots:
                            os.makedirs(f"{self.output_dir}/snapshots", exist_ok=True)
                        for field in self.save_snapshots:
                            if field == "u_fluc":
                                self._save_snapshot("u_fluc", u, t_save)
                            elif field == "P":
                                self.P_func.interpolate(self._P_expr)
                                self._save_snapshot("P", self.P_func, t_save)

                else:
                    ok = self._timestepper.reject()
                    if not ok:
                        logger.error(
                            "Minimum time step dt=%.2e reached — stopping.", self._timestepper.dt_min
                        )
                        raise RVEConvergenceError(
                            f"Newton did not converge and dt_min={self._timestepper.dt_min:.2e} "
                            f"reached at t={self._timestepper.t_current:.4f}"
                        )
                    else:
                        logger.warning(
                            "Newton did not converge — halving dt to %.2e", self._timestepper.dt
                        )
                    u.x.array[:] = self._u_last.x.array[:]
                    u.x.scatter_forward()
                    break

        if self._averages_only_final:
            output_quantities.append(self._collect_averages())

        return output_quantities

    def commit(self) -> None:
        self.F_bar_conv[:] = self.F_bar.value
        self._u_conv.x.array[:] = self.u.x.array
        self._u_conv.x.scatter_forward()
        self._commit_extra_state()

    # ------------------------------------------------------------------
    # Checkpoint hooks (used by the FE² macro driver for full two-scale
    # runs; per-RVE state, one COMM_SELF mesh per qp).
    # ------------------------------------------------------------------

    def dump_state(self) -> dict[str, np.ndarray]:
        """Return the converged restart state of this RVE as a dict of
        numpy arrays (single-rank, since RVEs live on COMM_SELF)."""
        d = {
            "F_bar_conv": np.asarray(self.F_bar_conv, dtype=np.float64).copy(),
            "u_conv": np.asarray(self._u_conv.x.array, dtype=np.float64).copy(),
        }
        extra = self._dump_extra_state()
        if extra:
            d.update(extra)
        return d

    def load_state(self, d: dict[str, np.ndarray]) -> None:
        """Load the converged restart state from a dict (inverse of
        :meth:`dump_state`). Seeds the trial state from the converged one
        so the next solve warm-starts correctly."""
        F = np.asarray(d["F_bar_conv"], dtype=PETSc.ScalarType)
        if F.shape != self.F_bar_conv.shape:
            raise RuntimeError(
                f"F_bar_conv shape mismatch: got {F.shape}, expected "
                f"{self.F_bar_conv.shape}"
            )
        u = np.asarray(d["u_conv"])
        if u.shape != self._u_conv.x.array.shape:
            raise RuntimeError(
                f"u_conv shape mismatch: got {u.shape}, expected "
                f"{self._u_conv.x.array.shape}"
            )
        self.F_bar_conv[:] = F
        self._u_conv.x.array[:] = u
        self._u_conv.x.scatter_forward()
        # Seed trial state from converged state so the next solve starts
        # at the restart point.
        self.F_bar.value[:] = self.F_bar_conv
        self.u.x.array[:] = self._u_conv.x.array
        self.u.x.scatter_forward()
        self._load_extra_state(d)

    def _dump_extra_state(self) -> dict[str, np.ndarray]:
        """Subclass hook: extra per-RVE state for checkpoint."""
        return {}

    def _load_extra_state(self, d: dict[str, np.ndarray]) -> None:
        """Subclass hook: consume extra state added by ``_dump_extra_state``."""
        return
