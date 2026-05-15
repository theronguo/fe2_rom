import os
import logging
from typing import Callable

import numpy as np
from dolfinx import fem, io, mesh as dmesh
from petsc4py import PETSc
from mpi4py import MPI
import ufl
import dolfinx_mpc

from .averages import (
    EffectiveAbar,
    EffectiveFbar,
    EffectivePbar,
    HomogenizationContext,
    resolve_average_quantities,
)
from .boundary import ReactionProbe
from .exceptions import RVEConvergenceError
from .forms import basis_tensor_ufl, build_homogenization_weak_form, build_weak_forms
from .logging_utils import silence_c_stdout
from .material import MaterialModel
from .output import ReactionForceLogger, VTXManager
from .solvers import CylindricalArcLength, NewtonSolver, NewtonSolverFE2
from .stability import StabilityAnalyzer
from .timestepping import TimeStepper

logger = logging.getLogger(__name__)


class HyperelasticStabilitySolver:
    """Modular hyperelastic stability solver.

    Usage pattern (two-phase init):
        solver = HyperelasticStabilitySolver(mesh, cell_tags, facet_tags, material)
        solver.add_bc(...)      # call for each BC component
        solver.setup()          # compiles UFL forms (collective)
        solver.run(load_schedule, ...)

    Attributes exposed for VTXManager (only when enable_viz_fields=True):
        u_int, F_func, P_func, J_func
    """

    def __init__(self, mesh, cell_tags, facet_tags, material: MaterialModel, *,
                 degree: int = 1, body_force=None, neumann_terms=None,
                 enable_viz_fields: bool = True):
        self.comm = mesh.comm
        self._mesh = mesh
        self._cell_tags = cell_tags
        self._facet_tags = facet_tags
        self._material = material
        self._degree = degree
        self._body_force = body_force
        self._neumann_terms = neumann_terms
        self._enable_viz_fields = enable_viz_fields

        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        space_dims = mesh.geometry.dim

        self.V = fem.functionspace(mesh, ("Lagrange", degree, (space_dims,)))
        self.u = fem.Function(self.V)
        self._u_last = fem.Function(self.V)
        self._du = fem.Function(self.V)
        self._eigenfunction = fem.Function(self.V)

        if enable_viz_fields:
            V1 = fem.functionspace(mesh, ("DG", 1, (space_dims,)))
            TT = fem.functionspace(mesh, ("DG", 1, (space_dims, space_dims)))
            SS = fem.functionspace(mesh, ("DG", 1))
            self.u_int = fem.Function(V1, name="u")
            self.F_func = fem.Function(TT, name="DeformationGradient")
            self.P_func = fem.Function(TT, name="Stress1PK")
            self.J_func = fem.Function(SS, name="JacobianDet")

        n_dofs = self.V.dofmap.index_map.size_global * self.V.dofmap.index_map_bs
        logger.info("Global DOFs: %d", n_dofs)

        self._bc_specs: list = []
        self._bcs: list = []
        self._reaction_probes: list[ReactionProbe] = []
        self._newton: NewtonSolver | None = None
        self._stability: StabilityAnalyzer | None = None
        self._F_var = None
        self._P_ufl = None

    def add_bc(self, subspace_index: int, locate_fn: Callable,
               value: fem.Constant, *,
               measure_reaction: bool = False,
               reaction_direction: tuple = (0.0, 0.0, 1.0)) -> None:
        """Register a Dirichlet BC component.

        subspace_index: 0=x, 1=y, 2=z
        locate_fn: callable x -> bool array (geometric boundary detection)
        value: fem.Constant whose .value is updated by the load_schedule
        measure_reaction: if True, a ReactionProbe is created for this surface
        reaction_direction: unit vector for the reaction force projection
        """
        self._bc_specs.append((subspace_index, locate_fn, value,
                                measure_reaction, reaction_direction))

    def setup(self, check_stability: bool = True,
              newton_options: dict | None = None) -> None:
        """Freeze BCs, compile UFL forms, and instantiate sub-solvers.

        Must be called once after all add_bc() calls and before run().
        Collective: calls fem.form() on all MPI ranks.
        """
        newton_options = newton_options if newton_options is not None else {}

        mesh = self._mesh
        fdim = mesh.topology.dim - 1
        V = self.V
        dx = ufl.Measure("dx", domain=mesh, subdomain_data=self._cell_tags)

        R_form, J_form, F_var, P_ufl, J_ufl = build_weak_forms(
            mesh, V, self.u, self._material,
            body_force=self._body_force, dx=dx,
            neumann_terms=self._neumann_terms,
        )
        self._R_form = R_form
        self._J_form = J_form
        self._F_var = F_var
        self._P_ufl = P_ufl
        self._J_ufl = J_ufl

        bcs = []
        probes = []
        for subspace_index, locate_fn, value, measure_reaction, reaction_dir in self._bc_specs:
            V_sub = V.sub(subspace_index)
            facets = dmesh.locate_entities_boundary(mesh, fdim, locate_fn)
            dofs = fem.locate_dofs_topological(V_sub, fdim, facets)
            bc = fem.dirichletbc(value, dofs, V_sub)
            bcs.append(bc)
            if measure_reaction:
                probe = ReactionProbe(mesh, facets, P_ufl,
                                      direction=reaction_dir, bc_value=value)
                probes.append(probe)

        self._bcs = bcs
        self._reaction_probes = probes

        if self._enable_viz_fields:
            TT = self.F_func.function_space
            SS = self.J_func.function_space
            self._F_expr = fem.Expression(F_var, TT.element.interpolation_points)
            self._P_expr = fem.Expression(P_ufl, TT.element.interpolation_points)
            self._J_expr = fem.Expression(J_ufl, SS.element.interpolation_points)

        if check_stability:
            self._stability = StabilityAnalyzer(self.comm)
            if "switch_to_minres" in newton_options:
                if newton_options["switch_to_minres"] is False:
                    logger.info("Overriding provided newton_options['switch_to_minres'] to True for stability checks.")
            newton_options["switch_to_minres"] = True
        self._newton = NewtonSolver(
            self.comm, R_form, J_form, self.u, self._du, bcs,
            **newton_options,
        )
        logger.info("Setup complete — %d BCs, %d reaction probe(s)",
                    len(bcs), len(probes))

    def _write_fields(self, output_manager: VTXManager | None, t: float) -> None:
        if self._enable_viz_fields:
            self.u_int.interpolate(self.u)
            self.F_func.interpolate(self._F_expr)
            self.P_func.interpolate(self._P_expr)
            self.J_func.interpolate(self._J_expr)
        if output_manager is not None:
            output_manager.write(t)

    def run(self, load_schedule: Callable[[float], None], *,
            timestepper: TimeStepper | None = None,
            output_manager: VTXManager | None = None,
            reaction_logger: ReactionForceLogger | None = None,
            pert_amplitude_init: float = 1e1) -> None:
        """Main time-stepping loop.

        load_schedule(t) is called once per trial time step to update any
        time-varying fem.Constants (e.g. prescribed displacements).

        pert_amplitude_init: initial eigenvector perturbation amplitude.
        Doubles on each stability retry; reset to this value each new time step.
        TODO: improve by normalising eigenvector relative to mesh size h.
        """
        assert self._newton is not None, "Call setup() before run()"

        if timestepper is None:
            timestepper = TimeStepper()

        comm = self.comm
        u = self.u

        self._write_fields(output_manager, 0.0)
        if reaction_logger is not None:
            reaction_logger.record(0.0, 0.0)

        simulation_finished = False
        while not timestepper.finished:
            trial_time = timestepper.step_forward()
            logger.info("── Step  t=%.5f  dt=%.2e", trial_time, timestepper.dt)

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
                        # _stability.check() takes ownership of K and destroys it.
                        is_stable, eigenvalues = self._stability.check(K, self._eigenfunction)
                    else:
                        is_stable, eigenvalues = True, np.array([])

                    if not is_stable:
                        target = np.where(eigenvalues < 1e-12)[0]
                        u.x.petsc_vec.axpy(pert_amplitude, self._eigenfunction.x.petsc_vec)
                        u.x.scatter_forward()
                        pert_amplitude *= 2
                        logger.warning(
                            "Unstable equilibrium (λ_min=%.4e) — "
                            "perturbing with eigenvector (amplitude=%.2e)",
                            eigenvalues[target[0]], pert_amplitude,
                        )
                    else:
                        stable_configuration = True
                        timestepper.accept(iter_newton)
                        self._u_last.x.array[:] = u.x.array[:]
                        self._u_last.x.scatter_forward()

                        self._write_fields(output_manager, timestepper.t_current)

                        for probe in self._reaction_probes:
                            rf = probe.assemble(comm)
                            if reaction_logger is not None:
                                reaction_logger.record(probe.displacement, rf)
                            logger.info(
                                "   disp=% .6f  reaction_z=% .6f",
                                probe.displacement, rf,
                            )

                else:
                    ok = timestepper.reject()
                    if not ok:
                        logger.error(
                            "Minimum time step dt=%.2e reached — stopping.", timestepper.dt_min
                        )
                        simulation_finished = True
                    else:
                        logger.warning(
                            "Newton did not converge — halving dt to %.2e", timestepper.dt
                        )
                    u.x.array[:] = self._u_last.x.array[:]
                    u.x.scatter_forward()
                    break

            if simulation_finished:
                break

    def run_arc_length(self, arc_solver: CylindricalArcLength,
                       load_fn: Callable[[float], None], *,
                       lambda_init: float = 0.0,
                       lambda_max: float = 1.0,
                       output_manager: VTXManager | None = None,
                       reaction_logger: ReactionForceLogger | None = None,
                       step_callback: "Callable[[float], None] | None" = None) -> None:
        """Crisfield cylindrical arc-length continuation loop.

        load_fn(lam) must update all load-controlling fem.Constants so that
        the assembled forms reflect the load at multiplier lam.

        lambda_init: starting load factor.
        lambda_max:  loop stops when lam ≥ lambda_max.
        step_callback(lam): called after every accepted step with the current
            load factor.  Use it to record quantities from self.u without
            modifying the solver (e.g. midspan displacement for snap-through).

        On corrector failure the arc-length is halved and the step retried
        (up to 4 times). If all retries fail the loop terminates early.
        """
        assert self._newton is not None, "Call setup() before run_arc_length()"

        comm = self.comm
        u = self.u
        newton = self._newton

        load_fn(lambda_init)
        lam = lambda_init
        ds = arc_solver.arc_length

        du_prev: PETSc.Vec | None = None
        dlambda_prev: float = 1.0

        self._write_fields(output_manager, 0)
        if reaction_logger is not None:
            reaction_logger.record(lam, 0.0)

        for step in range(arc_solver.max_arc_steps):
            if lam >= lambda_max:
                logger.info("Arc-length: reached λ_max=%.4f — done.", lambda_max)
                break

            logger.info("── Arc step %3d  λ=%.4f  Δs=%.3e", step, lam, ds)
            u_base = u.x.array.copy()

            # Predictor + corrector with up to 5 retries at halved arc-length.
            # du_accepted is None if all retries fail, a live Vec otherwise.
            du_accepted: PETSc.Vec | None = None
            dlambda_accepted: float = 0.0
            n_iter_accepted: int = 0

            for retry in range(5):
                if retry > 0:
                    ds *= 0.5
                    arc_solver.arc_length = ds
                    logger.warning("  retry %d  Δs → %.3e", retry, ds)
                    u.x.array[:] = u_base
                    u.x.scatter_forward()
                    load_fn(lam)

                du_pred, dlambda_pred, f_ref = arc_solver.predictor(
                    newton, load_fn, lam, du_prev, dlambda_prev
                )
                converged, n_iter, dlambda_final = arc_solver.corrector(
                    newton, load_fn,
                    lam_0=lam,
                    u_base=u_base,
                    du_total=du_pred,
                    dlambda_total=dlambda_pred,
                    f_ref=f_ref,
                    ds=ds,
                )
                PETSc.Vec.destroy(f_ref)

                if converged:
                    du_accepted = du_pred        # ownership transfers here
                    dlambda_accepted = dlambda_final
                    n_iter_accepted = n_iter
                    break

                # Failed: release this step's Vec, restore base state
                PETSc.Vec.destroy(du_pred)
                u.x.array[:] = u_base
                u.x.scatter_forward()
                load_fn(lam)

            if du_accepted is None:
                logger.error("Arc step %d failed after all retries — stopping.", step)
                break

            lam += dlambda_accepted
            logger.info("   converged in %d iter  λ=%.4f", n_iter_accepted, lam)

            # u is already at u_base + ΔU (set by the last corrector iteration);
            # just synchronise the load constant with the accepted λ
            load_fn(lam)
            self._write_fields(output_manager, step + 1)

            for probe in self._reaction_probes:
                rf = probe.assemble(comm)
                if reaction_logger is not None:
                    reaction_logger.record(lam, rf)
                logger.info("   λ=% .4f  reaction=% .6f", lam, rf)

            if step_callback is not None:
                step_callback(lam)

            if du_prev is not None:
                PETSc.Vec.destroy(du_prev)
            du_prev = du_accepted.copy()
            dlambda_prev = dlambda_accepted
            PETSc.Vec.destroy(du_accepted)

        if du_prev is not None:
            PETSc.Vec.destroy(du_prev)


class PeriodicHyperelasticHomogenizationSolver:
    """Periodic hyperelastic homogenization with pluggable effective quantities.

    Rigid-body modes are removed by Dirichlet-pinning the corner nodes (the
    original gauge fix) and the periodic ties are enforced via ``dolfinx_mpc``.
    The Newton step is the standard ``NewtonSolverFE2`` path with CG/MINRES
    + GAMG (no saddle-point augmentation).

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

        # ---- Periodic BCs + MPC (corner-pinning makes K SPD) ----
        bcs, self.mpc = self._setup_periodic_bcs_and_mpc()
        self._bcs = bcs
        logger.debug("Periodic BCs set up with %d slave points",
                     self.comm.allreduce(len(self.mpc.slaves), op=MPI.SUM))

        # ---- Subclass hooks: φᵢ and the macro-variable dict ----
        self._setup_phi()
        self._setup_macro_vars()

        # ---- Stability analyzer (optional) ----
        if check_stability:
            # With corner-pinning K is SPD (no zero modes); subclasses that
            # introduce additional zero modes override ``_count_zero_modes``.
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

        # ---- Newton solver (NewtonSolverFE2 with corner-pinning BCs) ----
        self._newton = NewtonSolverFE2(
            self.comm, R_form, J_form, self.u, self._du, self._bcs, self.mpc,
            **newton_options,
        )
        logger.debug("Newton solver initialized with options: %s", newton_options)

        # ---- Time stepper ----
        self._timestepper = TimeStepper(**timestepper_options)
        logger.debug("Time stepper initialized with options: %s", timestepper_options)

        # ---- Visualization ----
        self._setup_visualization(visualize_fields, F_var, P_ufl, J_ufl, W_ufl, u_total, output_dir)

        # ---- Volume + homogenization context ----
        vol_local = fem.assemble_scalar(fem.form(1.0 * self.dx))
        self._vol_global = float(self.comm.allreduce(vol_local, op=MPI.SUM))
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
        """Register the user-provided global modes φᵢ. Default: empty list."""
        self._phi: list[fem.Function] = []

    def _setup_macro_vars(self) -> None:
        """Initialize ``self.macro_vars``. Default: ``{"Fbar": self.F_bar}``."""
        self.macro_vars: dict = {"Fbar": self.F_bar}

    def _build_u_total_extra(self):
        """Return the additional UFL contribution to ``u_total`` beyond
        ``(F̄ − I)·X + w``. Default: ``None`` (no extra contribution)."""
        return None

    def _build_macro_var_rhs_forms(self, build_tangent_rhs_forms) -> dict:
        """Build adjoint-RHS forms per macro variable.

        Returned dict maps macro variable name -> flat list of compiled
        ``fem.Form`` (one per scalar component, in C order).
        """
        gdim = self.gdim
        dF_dFbar_list = [basis_tensor_ufl(gdim, i, j)
                         for i in range(gdim) for j in range(gdim)]
        return {"Fbar": build_tangent_rhs_forms(dF_dFbar_list)}

    def _make_default_average_quantities(self) -> list:
        """Default effective quantities. Base: F̄ (echo), P̄, dP̄/dF̄."""
        return [EffectiveFbar(), EffectivePbar(), EffectiveAbar()]

    def _restore_trial_state(self) -> None:
        """Hook: restore subclass-specific trial state from converged. Base: no-op."""
        return

    def _update_macro_load_schedule(self, t: float) -> None:
        """Hook: update subclass-specific load constants at trial time ``t``. Base: no-op."""
        return

    def _commit_extra_state(self) -> None:
        """Hook: commit subclass-specific trial state to the converged restart. Base: no-op."""
        return

    def _count_zero_modes(self) -> int:
        """Number of near-zero modes of K that the stability check should skip.

        With corner-pinning Dirichlet BCs (this base class), K is SPD and has
        no zero modes — return 0. Subclasses that swap corner-pinning for an
        integral gauge or otherwise introduce gauge directions in K's null
        space must override; e.g. for ``⟨w⟩=0`` the value is ``gdim``.
        """
        return 0

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
                                      tol: float, exclude_axes: tuple[int, ...]) -> Callable:
        def selector(x):
            mask = np.isclose(x[axis], mins[axis], atol=tol, rtol=0.0)
            for ex_axis in exclude_axes:
                mask &= (x[ex_axis] > mins[ex_axis] + tol)
                mask &= (x[ex_axis] < maxs[ex_axis] - tol)
            return mask
        return selector

    @staticmethod
    def _locate_corner_dofs(V, mins: np.ndarray, maxs: np.ndarray, tol: float) -> np.ndarray:
        """Locate all corner DOFs (4 in 2D, 8 in 3D; times vector components)."""
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
        """Build periodic-tie MPC and corner-pinning Dirichlet BCs for a
        rectangular / cuboid domain.

        Corner-pinning removes the rigid-body translation null space so K is
        SPD and CG + GAMG works out of the box. The Lagrange-multiplier
        gauge alternative (``⟨w⟩=0`` etc.) lives in ``SaddlePointNewtonSolver``
        and is used by the micromorphic subclass.
        """
        if self.gdim not in (2, 3):
            raise ValueError(
                f"Periodic homogenization supports only 2D rectangle or 3D cuboid, got dim={self.gdim}."
            )
        tol = 1e-8 * max(1.0, float(np.max(self.maxs - self.mins)))

        corner = self._locate_corner_dofs(self.V, self.mins, self.maxs, tol)
        u_zero = fem.Constant(self._mesh, np.zeros(self.gdim, dtype=PETSc.ScalarType))
        bcs = [fem.dirichletbc(u_zero, corner, self.V)]

        mpc = dolfinx_mpc.MultiPointConstraint(self.V)
        for axis in range(self.gdim):
            selector = self._make_periodic_slave_selector(
                axis=axis, mins=self.mins, maxs=self.maxs, tol=tol,
                exclude_axes=tuple(range(axis)),
            )
            axis_map = self._make_axis_map(axis, self.maxs[axis])
            mpc.create_periodic_constraint_geometrical(self.V, selector, axis_map, bcs)
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
        """Compute all configured ``average_quantities`` at the current trial state."""
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
                 pert_amplitude_init: float = 1e1,
                 plot_time_start: float = 0.0) -> list[dict]:
        """Main time-stepping loop.

        Linearly ramps ``F̄`` from the last committed state to the target.
        Subclasses extend the ramp via the ``_update_macro_load_schedule`` hook.

        ``pert_amplitude_init`` is the initial eigenvector perturbation
        amplitude — doubles on each stability retry; reset at each new step.
        """
        assert self._newton is not None, "Setup not complete."

        # Restart from the last committed state.
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
                        u.x.petsc_vec.axpy(pert_amplitude, self._eigenfunction.x.petsc_vec)
                        u.x.scatter_forward()
                        pert_amplitude *= 2
                        logger.warning(
                            "Unstable equilibrium (λ_min=%.4e) — "
                            "perturbing with eigenvector (amplitude=%.2e)",
                            eigenvalues[target[0]], pert_amplitude,
                        )
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
        """Promote trial state to the converged restart point.

        Call once from the macro driver after a successful outer time step.
        """
        self.F_bar_conv[:] = self.F_bar.value
        self._u_conv.x.array[:] = self.u.x.array
        self._u_conv.x.scatter_forward()
        self._commit_extra_state()
