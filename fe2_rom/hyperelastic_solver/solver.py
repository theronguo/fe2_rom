import os
import logging
from typing import Callable

import numpy as np
from dolfinx import fem, io, mesh as dmesh
from petsc4py import PETSc
from mpi4py import MPI
import ufl
import dolfinx_mpc

from .boundary import ReactionProbe
from .forms import build_homogenization_weak_form, build_weak_forms
from .exceptions import RVEConvergenceError
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
                    K = self._newton.assemble_stiffness()
                    if self._stability is not None:
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
    """Modular hyperelastic periodic homogenization solver.

    Usage pattern (two-phase init):
        solver = HyperelasticStabilitySolver(mesh, cell_tags, facet_tags, material)
        solver.add_bc(...)      # call for each BC component
        solver.setup()          # compiles UFL forms (collective)
        solver.run(load_schedule, ...)

    Attributes exposed for VTXManager (only when enable_viz_fields=True):
        u_int, F_func, P_func, J_func
    """

    def __init__(self, mesh_path, comm, gdim, 
                 material: MaterialModel, *,
                 degree: int = 1, 
                 output_dir: str = "output",
                 check_stability: bool = True,
                 visualize_fields: list[str] | None = None,
                 average_fields: list[str] | None = None,
                 stability_options: dict | None = None,
                 newton_options: dict | None = None,
                 timestepper_options: dict | None = None,
                 save_snapshots: list[str] | None = None,
                 averages_only_final: bool = False,
                 ) -> None:
        
        ### Default Newton, Timestepper, Visualization, Averaging options ###
        newton_options = newton_options if newton_options is not None else {
            "rel_tol": 1e-8, "abs_tol": 1e-6, "max_iter": 10, "max_iter_instab": 30, "switch_to_minres": False
        }
        timestepper_options = timestepper_options if timestepper_options is not None else {
            "t_end": 1.0, "dt_init": 1.0, "dt_min": 1e-5, "dt_max": 1.0, "good_newton_steps": 7
        }
        stability_options = stability_options if stability_options is not None else {
            "nev": 5, "neg_tol": -1e-12
        }
        if visualize_fields is None:
            visualize_fields = ["u_fluc"]
        if average_fields is None:
            average_fields = ["P"]
        if save_snapshots is None:
            save_snapshots = []

        ### Read mesh ###
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
                    gdim,
        )
        logger.debug("Domain bounds: x_min=%.3f  x_max=%.3f", self.mins[0], self.maxs[0])
        logger.debug("Domain bounds: y_min=%.3f  y_max=%.3f", self.mins[1], self.maxs[1])
        if gdim == 3:
            logger.debug("Domain bounds: z_min=%.3f  z_max=%.3f", self.mins[2], self.maxs[2])
        
        # If True, ``__call__`` runs the (expensive) effective-quantity
        # reductions only at the final converged endpoint of the F_bar ramp.
        # Intermediate accepted steps still solve Newton + (optional)
        # stability, but ``output_quantities`` only contains the final entry.
        self._averages_only_final = averages_only_final

        ### Material model ###
        self._material = material
        
        ### Function space and fields ###
        self._degree = degree
        self.V = fem.functionspace(self._mesh, ("Lagrange", degree, (self.gdim,)))
        n_dofs = self.V.dofmap.index_map.size_global * self.V.dofmap.index_map_bs
        
        self.u = fem.Function(self.V)
        self._u_last = fem.Function(self.V)   # within-call revert (per-step)
        self._u_conv = fem.Function(self.V)   # cross-call commit point
        self._du = fem.Function(self.V)
        self._eigenfunction = fem.Function(self.V)
        self.F_bar = fem.Constant(self._mesh, np.eye(self.gdim, dtype=PETSc.ScalarType))
        # Last *converged* F_bar.  ``F_bar`` itself tracks the trial state
        # (possibly mid-ramp / failed); ``F_bar_conv`` and ``_u_conv`` are
        # updated only on ``commit()`` and are the canonical restart point
        # for the next call.
        self.F_bar_conv = np.eye(self.gdim, dtype=PETSc.ScalarType)
        logger.debug("Functions set up with %d global DOFs", n_dofs)

        ### Periodic BCs ###
        bcs, mpc = self._setup_periodic_bcs_and_mpc()
        self.mpc = mpc
        self._bcs = bcs
        logger.debug("Periodic BCs set up with %d slave points", self.comm.allreduce(len(mpc.slaves), op=MPI.SUM))
        
        ### Stability analyzer (optional) ###
        if check_stability:
            self._stability = StabilityAnalyzer(self.comm, **stability_options)
            logger.debug("Stability checks enabled with options: %s", stability_options)
            if "switch_to_minres" in newton_options:
                if newton_options["switch_to_minres"] is False:
                    logger.debug("Overriding provided newton_options['switch_to_minres'] to True for stability checks.")
            newton_options["switch_to_minres"] = True
        else:
            self._stability = None

        ### Newton solver ###
        R_form, J_form, Jij_forms, F_var, P_ufl, J_ufl, W_ufl, A_ufl, u_total = build_homogenization_weak_form(
            self._mesh, self.V, self.u, self.F_bar, self._material, dx=self.dx
        )
        self._newton = NewtonSolverFE2(
            self.comm, R_form, J_form, Jij_forms, self.u, self._du, bcs, self.mpc,
            **newton_options,
        )
        logger.debug("Newton solver initialized with options: %s", newton_options)

        ### Time stepper ###
        self._timestepper = TimeStepper(**timestepper_options)
        logger.debug("Time stepper initialized with options: %s", timestepper_options)
        
        ### Visualization setup ###
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

        # Averaging setup
        self.average_fields = average_fields
        if "P" in average_fields:
            self._P_ufl = P_ufl
        if "W" in average_fields:
            self._W_ufl = W_ufl
        if "A" in average_fields:
            self._A_ufl = A_ufl
        logger.debug("Averaging fields: %s", average_fields)

        # Snapshot saving setup
        self.output_dir = output_dir
        self.save_snapshots = save_snapshots
        logger.debug("Saving snapshots for fields: %s", save_snapshots)
        logger.debug("Setup complete")

    def _compute_domain_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return global min/max coordinates along each geometric axis."""
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
        """Return x -> y map that sets y[axis] to target_value."""
        def axis_map(x):
            y = x.copy()
            y[axis] = target_value
            return y

        return axis_map

    @staticmethod
    def _make_periodic_slave_selector(axis: int,
                                      mins: np.ndarray,
                                      maxs: np.ndarray,
                                      tol: float,
                                      exclude_axes: tuple[int, ...]) -> Callable:
        """Select lower-side boundary DOFs for one axis, excluding previous-axis boundaries."""
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
        """Build periodic corner constraints and MPC ties for rectangular/cuboid domains."""
        mesh = self._mesh
        V = self.V
        if self.gdim not in (2, 3):
            raise ValueError(
                f"Periodic homogenization supports only 2D rectangle or 3D cuboid, got dim={self.gdim}."
            )
        tol = 1e-8 * max(1.0, float(np.max(self.maxs - self.mins)))

        # Fix all corner DOFs to remove rigid modes and avoid over-constraining periodic ties.
        corner = self._locate_corner_dofs(V, self.mins, self.maxs, tol)
        u_zero = fem.Constant(mesh, np.zeros(self.gdim, dtype=PETSc.ScalarType))
        bcs = [fem.dirichletbc(u_zero, corner, V)]

        mpc = dolfinx_mpc.MultiPointConstraint(V)
        # Build one non-overlapping periodic slave set per axis:
        # axis 0 uses all x_min points, axis 1 excludes x-boundaries,
        # axis 2 (3D only) excludes x/y-boundaries.
        for axis in range(self.gdim):
            selector = self._make_periodic_slave_selector(
                axis=axis,
                mins=self.mins,
                maxs=self.maxs,
                tol=tol,
                exclude_axes=tuple(range(axis)),
            )
            axis_map = self._make_axis_map(axis, self.maxs[axis])
            mpc.create_periodic_constraint_geometrical(V, selector, axis_map, bcs)
        mpc.finalize()

        return bcs, mpc

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
        """Gather owned DOF values and coordinates to rank 0 and save.

        Values are gathered in global-DOF-index order (rank 0's owned DOFs,
        then rank 1's, …).  DOF coordinates are saved once alongside the
        snapshots so that a serial post-processing tool (e.g. build_rom.py)
        can compute the permutation needed to convert from the parallel DOF
        ordering to the serial DOF ordering.

        Coordinates are written to
            <output_dir>/snapshots/<field_name>_dof_coords.npy
        and are not matched by the time-stamped glob pattern used to load
        snapshots (which contains a decimal point in the stem).
        """
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

    def compute_effective_strain_energy_density(self) -> float:
        """Compute domain-average effective strain energy density from the current converged state."""

        W_local = fem.assemble_scalar(fem.form(self._W_ufl * self.dx))
        W_global = self.comm.allreduce(W_local, op=MPI.SUM)

        vol_local = fem.assemble_scalar(fem.form(1.0 * self.dx))
        vol_global = self.comm.allreduce(vol_local, op=MPI.SUM)

        return W_global / vol_global

    def compute_average_first_pk_stress(self) -> np.ndarray:
        """Compute domain-average first Piola-Kirchhoff stress tensor from the current converged state."""

        vol_local = fem.assemble_scalar(fem.form(1.0 * self.dx))
        vol_global = self.comm.allreduce(vol_local, op=MPI.SUM)

        P_eff = np.zeros((self.gdim, self.gdim), dtype=float)
        for i in range(self.gdim):
            for j in range(self.gdim):
                P_ij_local = fem.assemble_scalar(fem.form(self._P_ufl[i, j] * self.dx))
                P_ij_global = self.comm.allreduce(P_ij_local, op=MPI.SUM)
                P_eff[i, j] = P_ij_global / vol_global

        return P_eff

    def compute_effective_tangent_moduli(self) -> np.ndarray:
        """Compute effective tangent moduli tensor from the current converged state with adjoints."""
        vol_local = fem.assemble_scalar(fem.form(1.0 * self.dx))
        vol_global = self.comm.allreduce(vol_local, op=MPI.SUM)

        adjoints = self._newton.solve_adjoint()
        A_eff = np.zeros((self.gdim, self.gdim, self.gdim, self.gdim), dtype=float)
        for i in range(self.gdim):
            for j in range(self.gdim):
                for k in range(self.gdim):
                    for l in range(self.gdim):
                        A_avg_local = fem.assemble_scalar(fem.form(self._A_ufl[i, j, k, l] * self.dx))
                        A_fluc_local = fem.assemble_scalar(
                            fem.form(ufl.inner(self._A_ufl[i, j, :, :], ufl.grad(adjoints[k][l])) * self.dx)
                            )
                        
                        A_avg_global = self.comm.allreduce(A_avg_local, op=MPI.SUM)
                        A_fluc_global = self.comm.allreduce(A_fluc_local, op=MPI.SUM)
                        
                        A_eff[i, j, k, l] = (A_avg_global + A_fluc_global) / vol_global

        return A_eff

    def __call__(self, Fbar: np.array,
            pert_amplitude_init: float = 1e1,
            plot_time_start: float = 0.0
            ) -> list:
        """Main time-stepping loop.

        load_schedule(t) is called once per trial time step to update any
        time-varying fem.Constants (e.g. prescribed displacements).

        pert_amplitude_init: initial eigenvector perturbation amplitude.
        Doubles on each stability retry; reset to this value each new time step.
        TODO: improve by normalising eigenvector relative to mesh size h.
        """
        assert self._newton is not None, "Call setup() before run()"
        
        # Restart from the last *committed* state — F_bar and the
        # displacement may have been left at a trial value the outer macro
        # SNES rejected.
        Fbar_prev = self.F_bar_conv.copy()
        self.F_bar.value[:] = Fbar_prev
        self.u.x.array[:] = self._u_conv.x.array
        self.u.x.scatter_forward()
        def load_schedule(t: float) -> None:
            for i in range(Fbar.shape[0]):
                for j in range(Fbar.shape[1]):
                    self.F_bar.value[i, j] = t * (Fbar[i, j] - Fbar_prev[i, j])  + Fbar_prev[i, j]  # Linear ramp from 0 to Fbar

        u = self.u
        if self.vtx is not None:
            self._write_fields(0.0+plot_time_start)

        output_quantities = []
        if not self._averages_only_final:
            # initial-state snapshot
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
                    K = self._newton.assemble_stiffness()
                    if self._stability is not None:
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

                        t_save = self._timestepper.t_current+plot_time_start
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
        """Promote trial state (F_bar, u) to the converged restart point.

        Call once from the macro driver after a successful outer time step.
        Calling it after a *failed* macro step would poison the restart, so
        the inner ``__call__`` never does this automatically.
        """
        self.F_bar_conv[:] = self.F_bar.value
        self._u_conv.x.array[:] = self.u.x.array
        self._u_conv.x.scatter_forward()

    def _collect_averages(self) -> list:
        """Snapshot all configured ``average_fields`` at the current state."""
        quantities = []
        for quantity in self.average_fields:
            if quantity == "F":
                quantities.append(self.F_bar.value.copy())
            elif quantity == "W":
                quantities.append(self.compute_effective_strain_energy_density())
            elif quantity == "P":
                quantities.append(self.compute_average_first_pk_stress().copy())
            elif quantity == "A":
                quantities.append(self.compute_effective_tangent_moduli().copy())
        return quantities