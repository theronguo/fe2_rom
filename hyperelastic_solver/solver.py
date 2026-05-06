import logging
from typing import Callable

import numpy as np
from dolfinx import fem, mesh as dmesh
from petsc4py import PETSc

from .boundary import ReactionProbe
from .forms import build_weak_forms
from .material import MaterialModel
from .output import ReactionForceLogger, VTXManager
from .solvers import ArcLengthSolver, NewtonSolver
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
                 degree: int = 1, body_force=None, enable_viz_fields: bool = True):
        self.comm = mesh.comm
        self._mesh = mesh
        self._cell_tags = cell_tags
        self._facet_tags = facet_tags
        self._material = material
        self._degree = degree
        self._body_force = body_force
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

    def setup(self) -> None:
        """Freeze BCs, compile UFL forms, and instantiate sub-solvers.

        Must be called once after all add_bc() calls and before run().
        Collective: calls fem.form() on all MPI ranks.
        """
        import ufl

        mesh = self._mesh
        fdim = mesh.topology.dim - 1
        V = self.V
        dx = ufl.Measure("dx", domain=mesh, subdomain_data=self._cell_tags)

        R_form, J_form, F_var, P_ufl, J_ufl = build_weak_forms(
            mesh, V, self.u, self._material,
            body_force=self._body_force, dx=dx,
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
            self._F_expr = fem.Expression(F_var, TT.element.interpolation_points())
            self._P_expr = fem.Expression(P_ufl, TT.element.interpolation_points())
            self._J_expr = fem.Expression(J_ufl, SS.element.interpolation_points())

        self._newton = NewtonSolver(
            self.comm, R_form, J_form, self.u, self._du, bcs,
        )
        self._stability = StabilityAnalyzer(self.comm)
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
                    is_stable, eigenvalues = self._stability.check(K, self._eigenfunction)

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

    def run_arc_length(self, arc_length_solver: ArcLengthSolver, *,
                       output_manager: VTXManager | None = None) -> None:
        """Arc-length continuation loop. Not yet implemented."""
        raise NotImplementedError(
            "Arc-length solver is not yet implemented. "
            "Implement CylindricalArcLength.predictor / constraint / corrector "
            "in hyperelastic_solver/solvers.py and wire it here."
        )
