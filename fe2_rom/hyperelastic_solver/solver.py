import logging
from typing import Callable

import numpy as np
from dolfinx import fem, mesh as dmesh
from petsc4py import PETSc
from mpi4py import MPI
import ufl

from .boundary import ReactionProbe
from .forms import build_weak_forms
from .material import MaterialModel
from .output import ReactionForceLogger, VTXManager
from .solvers import CylindricalArcLength, NewtonSolver
from .stability import (
    StabilityAnalyzer,
    apply_eigenmode_perturbation,
    mesh_characteristic_length,
)
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
               reaction_direction: tuple = (0.0, 0.0, 1.0),
               pointwise: bool = False) -> None:
        """Register a Dirichlet BC component.

        subspace_index: 0=x, 1=y, 2=z
        locate_fn: callable x -> bool array (geometric boundary detection)
        value: fem.Constant whose .value is updated by the load_schedule
        measure_reaction: if True, a ReactionProbe is created for this surface
            (consistent residual-based reaction at the constrained dofs)
        reaction_direction: unit vector for the reaction force projection.
            A component BC only transmits force along its own axis, so only
            entry [subspace_index] enters (as a projection weight).
        pointwise: if True, dofs are located *geometrically* (``locate_fn`` may
            select isolated nodes, not whole boundary facets) — used for minimal
            rigid-body pins in uniaxial-stress setups. Incompatible with
            ``measure_reaction`` (a point has no surface to integrate over).
        """
        if pointwise and measure_reaction:
            raise ValueError("pointwise BCs cannot be reaction probes.")
        self._bc_specs.append((subspace_index, locate_fn, value,
                                measure_reaction, reaction_direction, pointwise))

    def setup(self, check_stability: bool = True,
              newton_options: dict | None = None,
              stability_options: dict | None = None) -> None:
        """Freeze BCs, compile UFL forms, and instantiate sub-solvers.

        Must be called once after all add_bc() calls and before run().
        Collective: calls fem.form() on all MPI ranks.

        stability_options: kwargs forwarded to StabilityAnalyzer (nev, neg_tol,
            tol, petsc_options, n_skip_eigenvalues).  Ignored when
            check_stability=False.
        """
        newton_options = newton_options if newton_options is not None else {}
        stability_options = stability_options if stability_options is not None else {}

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
        for subspace_index, locate_fn, value, measure_reaction, reaction_dir, pointwise in self._bc_specs:
            V_sub = V.sub(subspace_index)
            if pointwise:
                dofs = fem.locate_dofs_geometrical(
                    (V_sub, V_sub.collapse()[0]), locate_fn)[0]
                facets = None
            else:
                facets = dmesh.locate_entities_boundary(mesh, fdim, locate_fn)
                dofs = fem.locate_dofs_topological(V_sub, fdim, facets)
            bc = fem.dirichletbc(value, dofs, V_sub)
            bcs.append(bc)
            if measure_reaction:
                # The subspace-only `dofs` above is what fem.dirichletbc needs;
                # the probe indexes the assembled residual, so it needs the
                # same dofs in the *parent* numbering of V.
                parent_dofs, _ = fem.locate_dofs_topological(
                    (V_sub, V_sub.collapse()[0]), fdim, facets)
                probe = ReactionProbe(V, parent_dofs, R_form, subspace_index,
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

        self._char_length = mesh_characteristic_length(mesh)

        if check_stability:
            self._stability = StabilityAnalyzer(self.comm, **stability_options)
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
            pert_amplitude_init: float = 1e-2,
            output_dir: str | None = None,
            enable_restart: bool = False) -> None:
        """Main time-stepping loop.

        load_schedule(t) is called once per trial time step to update any
        time-varying fem.Constants (e.g. prescribed displacements).

        pert_amplitude_init: dimensionless eigenvector-perturbation factor.
        First perturbation magnitude on instability is
        ``pert_amplitude_init * max|u|`` (or ``pert_amplitude_init *
        char_length`` if ``|u|`` is still ~0). Doubles on each stability
        retry; reset to this value each new time step.

        output_dir / enable_restart: opt-in checkpoint/restart, mirroring the
        full two-scale macros. When ``enable_restart=True`` (requires
        ``output_dir``), a rolling checkpoint is written under
        ``output_dir/checkpoint/`` after every accepted load step (atomic
        write: ``checkpoint.tmp/`` → rename) holding the displacement field,
        timestepper state, and reaction history. If that checkpoint exists on
        entry it is loaded and the run resumes from the saved ``t`` — the
        constitutive law is stateless, so the displacement field is the only
        per-cell state. Restart requires the **same MPI rank count** (a
        per-rank partition fingerprint is verified on load). The caller is
        responsible for VTX (.bp) continuity on resume — see
        :meth:`checkpoint_exists` to pick a fresh segment path before opening
        the ``output_manager``.
        """
        assert self._newton is not None, "Call setup() before run()"

        if timestepper is None:
            timestepper = TimeStepper()

        comm = self.comm
        u = self.u

        if enable_restart and output_dir is None:
            raise ValueError("enable_restart=True requires output_dir.")

        self._step_index = 0
        resumed = False
        if enable_restart and self._checkpoint_complete(output_dir):
            self._restore_checkpoint(output_dir, timestepper, reaction_logger,
                                     load_schedule)
            resumed = True

        if not resumed:
            self._write_fields(output_manager, 0.0)
            if reaction_logger is not None:
                reaction_logger.record(0.0, 0.0)

        simulation_finished = False
        while not timestepper.finished:
            trial_time = timestepper.step_forward()
            logger.info("── Step  t=%.8f  dt=%.2e", trial_time, timestepper.dt)

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
                        scale, info = apply_eigenmode_perturbation(
                            u, self._eigenfunction, pert_amplitude, comm,
                            char_length=self._char_length,
                        )
                        u_ref, phi_max = info[0]
                        logger.warning(
                            "Unstable equilibrium (λ_min=%.4e) — "
                            "perturbing with eigenvector "
                            "(factor=%.2e, |u|=%.2e, ‖perturbation‖_∞=%.2e)",
                            eigenvalues.min(), pert_amplitude, u_ref,
                            scale * phi_max,
                        )
                        pert_amplitude *= 2
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

                        self._step_index += 1
                        if enable_restart:
                            self._write_checkpoint(output_dir, timestepper,
                                                   reaction_logger)

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

    # ------------------------------------------------------------------
    # Checkpoint / restart (single-scale; stateless constitutive law)
    # ------------------------------------------------------------------

    @staticmethod
    def _restart():
        """Deferred import of the shared restart helpers.

        ``fe2_rom.ch1.restart`` carries no ``fe2_rom`` dependencies, but
        importing it triggers ``fe2_rom.ch1.__init__`` which pulls in the
        macro solvers (and thus this layer back) — so it must be imported
        lazily, after both packages are fully loaded, to avoid a circular
        import.
        """
        from fe2_rom.ch1 import restart as _restart
        return _restart

    def _fingerprint(self) -> str:
        if not hasattr(self, "_cached_fingerprint"):
            self._cached_fingerprint = self._restart().compute_partition_fingerprint(
                self._mesh)
        return self._cached_fingerprint

    def checkpoint_exists(self, output_dir: str) -> bool:
        """Collective: True if a complete restart checkpoint lives under
        ``output_dir``.

        Useful to the caller *before* opening a VTX manager so it can pick a
        fresh ``.bp`` segment on resume (the rolling checkpoint preserves the
        physics; ParaView frames written before the last checkpoint live in
        the prior ``.bp``).
        """
        return self._checkpoint_complete(output_dir)

    def _checkpoint_complete(self, output_dir: str) -> bool:
        return self._restart().checkpoint_complete(
            self.comm, output_dir, require_rves=False)

    def _write_checkpoint(self, output_dir, timestepper, reaction_logger):
        rst = self._restart()
        tmp = rst.prepare_tmp(self.comm, output_dir)
        fp = self._fingerprint()
        rst.save_meta(
            self.comm, tmp,
            t_current=float(timestepper.t_current),
            dt=float(timestepper.dt),
            step_index=int(getattr(self, "_step_index", 0)),
            gdim=int(self._mesh.geometry.dim),
            kind="dns",
        )
        rst.save_reaction(self.comm, reaction_logger, tmp)
        rst.save_macro_field(self.comm, self.u, tmp, fp)
        rst.atomic_finalize(self.comm, output_dir)

    def _restore_checkpoint(self, output_dir, timestepper, reaction_logger,
                            load_schedule):
        rst = self._restart()
        ckpt_dir, _ = rst.checkpoint_dirs(output_dir)
        meta = rst.load_meta(self.comm, ckpt_dir)
        if int(meta.get("n_ranks", -1)) != self.comm.size:
            raise RuntimeError(
                f"Checkpoint was written with n_ranks={meta.get('n_ranks')}, "
                f"current run uses {self.comm.size}. Restart requires the "
                "same MPI rank count."
            )
        gdim = self._mesh.geometry.dim
        if int(meta.get("gdim", -1)) != gdim:
            raise RuntimeError(
                f"Checkpoint gdim={meta.get('gdim')} != current {gdim}."
            )
        timestepper.t_current = float(meta["t_current"])
        self._step_index = int(meta.get("step_index", 0))
        # Restore dt, but fall back to the configured dt if the saved value
        # was clamped to ~0 by the previous run ending at its t_end (so a
        # resume to a larger t_end can still make progress).
        saved_dt = float(meta["dt"])
        remaining = timestepper.t_end - timestepper.t_current
        if saved_dt > 1e-15 and saved_dt <= remaining + 1e-15:
            timestepper.dt = saved_dt
        else:
            timestepper.dt = min(timestepper.dt, max(remaining, 0.0))

        fp = self._fingerprint()
        rst.load_macro_field(self.comm, self.u, ckpt_dir, fp)
        self._u_last.x.array[:] = self.u.x.array
        self._u_last.x.scatter_forward()

        rst.load_reaction(self.comm, reaction_logger, ckpt_dir)

        load_schedule(timestepper.t_current)
        logger.info(
            "Resumed from checkpoint at t=%.6f, dt=%.2e",
            timestepper.t_current, timestepper.dt,
        )

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


