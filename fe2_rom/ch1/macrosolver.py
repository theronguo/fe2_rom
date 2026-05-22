"""FE² macro solver.

A continuum body in which the constitutive response at every macro quadrature
point is supplied by a nested RVE solver — either the full periodic
:class:`fe2_rom.ch1.microsolver.MicroSolver`
or the reduced-order :class:`fe2_rom.rom.ReducedMicroSolver` — selected by a
single ``full`` boolean flag at construction time.

Usage pattern (mirrors
:class:`fe2_rom.hyperelastic_solver.HyperelasticStabilitySolver`):

    solver = MacroSolver(mesh, full=False, n_qp=2,
                         rve_mesh_path=..., rve_material=..., rom_dir=...)
    solver.add_bc(0, lambda x: np.isclose(x[2], 0.0), zero_const)
    solver.add_bc(2, lambda x: np.isclose(x[2], 1.0), disp_const,
                  measure_reaction=True, reaction_direction=(0, 0, 1))
    solver.setup()
    solver.solve(output_dir="output", timestepper=..., loadhistory=...,
                 output_variables=[solver.u], reaction_logger=...)
"""
from __future__ import annotations

import logging
import os
from typing import Callable

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import ufl
from dolfinx import fem, mesh as dmesh

from dolfinx_materials.quadrature_map import QuadratureMap
from dolfinx_materials.solvers import NonlinearMaterialProblem
from dolfinx_materials.utils import nonsymmetric_tensor_to_vector

from fe2_rom.ch1.averages import AverageQuantity, STRING_KEY_MAP
from fe2_rom.hyperelastic_solver.output import ReactionForceLogger, VTXManager
from fe2_rom.ch1.microsolver import MicroSolver
from fe2_rom.hyperelastic_solver.stability import (
    StabilityAnalyzer,
    apply_eigenmode_perturbation,
    mesh_characteristic_length,
)
from fe2_rom.hyperelastic_solver.timestepping import TimeStepper
from fe2_rom.ch1.material import RVEMaterial
from fe2_rom.ch1 import restart as _restart

logger = logging.getLogger(__name__)


def _select_local_qps(qps, my_rank: int, n_local: int) -> list[int]:
    """Resolve a user ``rve_history_qps`` spec to this rank's qps.

    Accepted forms:
      * ``[3, 5]`` — bare ints; every rank that has those local
        indices picks them (legacy behaviour).
      * ``[(0, 3), (2, 5)]`` — (rank, qp_local) tuples; only the
        matching rank picks them.
      * ``{0: [3, 5], 2: [1]}`` — dict form, same semantics as tuples.
    """
    if not qps:
        return []
    out: list[int] = []
    if isinstance(qps, dict):
        for qp in qps.get(my_rank, ()):
            if 0 <= int(qp) < n_local:
                out.append(int(qp))
        return out
    for item in qps:
        if isinstance(item, (tuple, list)):
            r, qp = item
            if int(r) == my_rank and 0 <= int(qp) < n_local:
                out.append(int(qp))
        else:
            qp = int(item)
            if 0 <= qp < n_local:
                out.append(qp)
    return out


def _vtx_segment_path(comm, output_dir: str, stem: str, segmented: bool) -> str:
    """Return the .bp filename for the upcoming VTX segment.

    When ``segmented`` is False the legacy single-file behaviour is kept
    (``<stem>.bp``). When True, pick the next non-existing
    ``<stem>_NNN.bp`` so a resumed run doesn't clobber prior frames; the
    set of .bp files can be loaded together as one time series in
    ParaView.
    """
    if not segmented:
        return os.path.join(output_dir, f"{stem}.bp")
    if comm.rank == 0:
        i = 1
        while os.path.exists(os.path.join(output_dir, f"{stem}_{i:03d}.bp")):
            i += 1
        pick = os.path.join(output_dir, f"{stem}_{i:03d}.bp")
    else:
        pick = None
    return comm.bcast(pick, root=0)


def _resolve_quantity_names(items) -> set[str]:
    """Return the set of dict keys an ``average_quantities`` list will emit.

    Accepts string keys (resolved via ``STRING_KEY_MAP``) and ``AverageQuantity``
    instances. Used by ``MacroSolver`` to validate the inner-RVE config statically
    without instantiating an RVE.
    """
    names: set[str] = set()
    for q in items:
        if isinstance(q, str):
            cls = STRING_KEY_MAP.get(q)
            if cls is not None:
                names.add(cls.name)
        elif isinstance(q, AverageQuantity):
            names.add(q.name)
    return names


class MacroSolver:
    """Two-scale FE² macro solver with optional stability check.

    The macro residual is assembled with dolfinx_materials' ``QuadratureMap``;
    the SNES driver is wrapped via ``NonlinearMaterialProblem`` so the nested
    RVE solves at each macro qp run inside the residual/Jacobian callbacks.

    Two-phase init: ``__init__`` builds geometry, function spaces, material
    bridge, qmap and weak forms; ``add_bc(...)`` registers Dirichlet BCs and
    (optionally) reaction probes; ``setup()`` freezes the configuration and
    constructs the SNES problem; ``solve(...)`` runs the adaptive load loop.
    """

    def __init__(
        self,
        mesh,
        full: bool,
        n_qp: int,
        *,
        # --- inner RVE configuration (required) ---
        rve_mesh_path: str,
        rve_material,
        # --- common to MicroSolver and ReducedMicroSolver ---
        gdim: int = 3,
        rve_degree: int = 2,
        rve_output_dir: str = "output",
        rve_visualize_fields: list | None = None,
        rve_average_quantities: list | None = None,
        rve_newton_options: dict | None = None,
        rve_timestepper_options: dict | None = None,
        rve_averages_only_final: bool = True,
        rve_volume: float | None = None,
        # --- reduced-specific (full=False) ---
        rom_dir: str | None = None,
        # --- full-specific (full=True) ---
        rve_check_stability: bool = False,
        rve_stability_options: dict | None = None,
        rve_save_snapshots: list | None = None,
        # --- macro-level ---
        degree: int = 1,
        snes_options: dict | None = None,
        check_stability: bool = False,
        stability_options: dict | None = None,
    ):
        if not full and rom_dir is None:
            raise ValueError("rom_dir is required when full=False (reduced inner RVE).")
        if mesh.geometry.dim != gdim:
            raise ValueError(
                f"mesh.geometry.dim ({mesh.geometry.dim}) != gdim ({gdim})."
            )

        self._mesh = mesh
        self.comm = mesh.comm
        self.gdim = gdim
        self._full = bool(full)
        self._n_qp_per_cell = int(n_qp)
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

        # Function spaces and fields
        self._degree = degree
        self.V = fem.functionspace(mesh, ("Lagrange", degree, (gdim,)))
        self.u = fem.Function(self.V, name="displacement")
        self._u_last = fem.Function(self.V)
        self._du = ufl.TrialFunction(self.V)
        self._v = ufl.TestFunction(self.V)
        self._eigenfunction = fem.Function(self.V)
        n_dofs = self.V.dofmap.index_map.size_global * self.V.dofmap.index_map_bs
        logger.debug("Macro DOFs: %d", n_dofs)

        # RVE factory: one fresh inner solver per macro qp, always on COMM_SELF
        # so each macro rank handles its own qp population independently.
        # ``rve_average_quantities`` accepts string keys ("P", "A", ...) or
        # AverageQuantity instances — same as the inner solver's kwarg.
        _rve_average_quantities = (
            rve_average_quantities if rve_average_quantities is not None else ["P", "A"]
        )
        provided_names = _resolve_quantity_names(_rve_average_quantities)
        for required in ("Pbar", "dPbar_dFbar"):
            if required not in provided_names:
                raise ValueError(
                    f"rve_average_quantities must produce {required!r} "
                    f"(RVEMaterial reads out[-1]['Pbar'] and out[-1]['dPbar_dFbar'])."
                )

        if full:
            def _make_rve(rank: int, index: int):
                return MicroSolver(
                    mesh_path=rve_mesh_path,
                    comm=MPI.COMM_SELF,
                    gdim=gdim,
                    material=rve_material,
                    degree=rve_degree,
                    output_dir=f"{rve_output_dir}/rve_{rank}_{index}",
                    check_stability=rve_check_stability,
                    visualize_fields=rve_visualize_fields,
                    average_quantities=_rve_average_quantities,
                    stability_options=rve_stability_options,
                    newton_options=rve_newton_options,
                    timestepper_options=rve_timestepper_options,
                    save_snapshots=rve_save_snapshots,
                    averages_only_final=rve_averages_only_final,
                    rve_volume=rve_volume,
                )
        else:
            def _make_rve(rank: int, index: int):
                from fe2_rom.rom.solver_ch1 import ReducedMicroSolver
                return ReducedMicroSolver(
                    mesh_path=rve_mesh_path,
                    rom_dir=rom_dir,
                    material=rve_material,
                    comm=MPI.COMM_SELF,
                    gdim=gdim,
                    degree=rve_degree,
                    output_dir=f"{rve_output_dir}/rve_{rank}_{index}",
                    visualize_fields=rve_visualize_fields,
                    average_quantities=_rve_average_quantities,
                    newton_options=rve_newton_options,
                    timestepper_options=rve_timestepper_options,
                    averages_only_final=rve_averages_only_final,
                    rve_volume=rve_volume,
                )
        self._make_rve = _make_rve

        # Material bridge + qmap (RVEMaterial needs gdim so its F-vector
        # convention matches the macro grad(u) — 5 entries in 2D, 9 in 3D).
        self.material = RVEMaterial(self._make_rve, gdim=gdim)
        self.qmap = QuadratureMap(mesh, n_qp, self.material)

        Id = ufl.Identity(gdim)
        F_ufl = nonsymmetric_tensor_to_vector(Id + ufl.grad(self.u))
        dF_ufl = lambda w: ufl.derivative(F_ufl, self.u, w)
        self.qmap.register_gradient("F", F_ufl)

        P_vec = self.qmap.fluxes["PK1"]
        self.Res = ufl.dot(P_vec, dF_ufl(self._v)) * self.qmap.dx
        self.Jac = self.qmap.derivative(self.Res, self.u, self._du)

        # SNES options (mirror legacy run_macro.py)
        self._snes_options = snes_options if snes_options is not None else {
            "snes_type": "newtonls",
            "snes_linesearch_type": "none",
            "snes_rtol": 1e-6,
            "snes_atol": 1e-8,
            "snes_max_it": 25,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }

        # Macro stability config
        self._check_stability = check_stability
        self._stability_options = stability_options if stability_options is not None else {
            "nev": 5, "neg_tol": -1e-12,
        }

        # Populated in setup()
        self._bc_specs: list = []
        self._bcs: list = []
        # Reaction wiring: each entry is (parent_dofs, value, direction_label),
        # where parent_dofs are the dofs in self.V at the constrained boundary.
        # The reaction along the BC's subspace direction is obtained by summing
        # owned entries of the BC-free residual at those dofs.
        self._reaction_specs: list = []
        self._problem: NonlinearMaterialProblem | None = None
        self._stability: StabilityAnalyzer | None = None
        self._Jac_form: fem.Form | None = None
        self._Res_form: fem.Form | None = None

    # ------------------------------------------------------------------
    # BC + reaction probe registration
    # ------------------------------------------------------------------

    def add_bc(
        self,
        subspace_index: int,
        locate_fn: Callable,
        value: fem.Constant,
        *,
        measure_reaction: bool = False,
        reaction_direction: tuple = (0.0, 0.0, 1.0),
    ) -> None:
        """Register a Dirichlet BC on one displacement component.

        Parameters
        ----------
        subspace_index
            Component index in ``self.V`` (0=x, 1=y, 2=z).
        locate_fn
            Callable ``x -> bool array`` for geometric boundary detection.
        value
            Scalar ``fem.Constant`` whose ``.value`` is mutated by the
            ``loadhistory(t)`` callback passed to :meth:`solve`.
        measure_reaction
            If True, a :class:`ReactionProbe` is created on the same facets
            and its assembly is reported each accepted load step.
        reaction_direction
            Unit vector along which the reaction force is projected.
        """
        self._bc_specs.append(
            (subspace_index, locate_fn, value, measure_reaction, reaction_direction)
        )

    # ------------------------------------------------------------------
    # Two-phase setup (collective)
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Materialise BCs, reaction probes, the SNES problem and (optionally)
        the stability analyzer. Collective: must be called on all MPI ranks
        after every :meth:`add_bc` call and before :meth:`solve`.
        """
        mesh = self._mesh
        fdim = mesh.topology.dim - 1
        V = self.V

        bcs: list = []
        reaction_specs: list = []
        for sub_idx, locate_fn, value, measure_reaction, direction in self._bc_specs:
            V_sub = V.sub(sub_idx)
            facets = dmesh.locate_entities_boundary(mesh, fdim, locate_fn)
            dofs = fem.locate_dofs_topological(V_sub, fdim, facets)
            bcs.append(fem.dirichletbc(value, dofs, V_sub))
            if measure_reaction:
                # Locate the SAME dofs but in the *parent* numbering of V, so
                # we can index into the assembled residual vector. The
                # subspace-only `dofs` above is what fem.dirichletbc needs;
                # for residual extraction we need the V-level indices.
                V_sub_collapsed, _ = V_sub.collapse()
                parent_dofs, _ = fem.locate_dofs_topological(
                    (V_sub, V_sub_collapsed), fdim, facets,
                )
                reaction_specs.append((np.asarray(parent_dofs, dtype=np.int32),
                                       value, direction))

        self._bcs = bcs
        self._reaction_specs = reaction_specs

        self._problem = NonlinearMaterialProblem(
            self.qmap, self.Res, self.u,
            bcs=self._bcs, J=self.Jac,
            petsc_options_prefix="fe2_macro_",
            petsc_options=self._snes_options,
        )
        self._problem.solver.setMonitor(
            lambda _snes, i, rnorm: logger.info("   iter %2d  ||F|| = %.6e", i, rnorm)
        )

        if self._check_stability:
            self._stability = StabilityAnalyzer(self.comm, **self._stability_options)

        self._Jac_form = fem.form(self.Jac)
        self._Res_form = fem.form(self.Res)

        self._char_length = mesh_characteristic_length(mesh)

        logger.debug(
            "Setup complete — %d BCs, %d reaction probe(s), stability=%s",
            len(self._bcs), len(self._reaction_specs),
            "on" if self._stability is not None else "off",
        )

    # ------------------------------------------------------------------
    # Main load-stepping loop
    # ------------------------------------------------------------------

    def solve(
        self,
        *,
        output_dir: str,
        timestepper: TimeStepper,
        loadhistory: Callable[[float], None],
        output_variables: list | None = None,
        reaction_logger: ReactionForceLogger | None = None,
        pert_amplitude_init: float = 1e-2,
        max_iter_per_step: int = 25,
        save_macro_history: bool = False,
        vtx_segment_per_resume: bool = False,
        rve_history_qps: list[int] | None = None,
    ) -> None:
        """Run the adaptive load-stepping outer loop.

        Each accepted macro step: writes the visualisation fields, commits
        every RVE's state, refreshes the DG-projected PK1 stress and assembles
        every reaction probe.

        On SNES or RVE failure the displacement is reverted, the timestepper
        halves dt, and the step is retried until dt < dt_min.

        On macro instability (smallest tangent eigenvalue < neg_tol), the
        displacement is perturbed along the eigenvector and SNES is re-driven.
        ``pert_amplitude_init`` is a dimensionless factor: the actual
        perturbation magnitude on the first retry is
        ``pert_amplitude_init * max|u|`` (or ``pert_amplitude_init *
        char_length`` if ``|u|`` is still ~0). The factor doubles on each
        unstable iteration within a step.

        ``max_iter_per_step`` caps the *total* number of Newton iterations
        spent across all perturb-and-retry SNES calls within a single load
        step. When exceeded, the step is rejected like a SNES failure and
        the timestepper halves dt.
        """
        if self._problem is None:
            raise RuntimeError("Call setup() before solve().")

        os.makedirs(output_dir, exist_ok=True)
        fields = output_variables if output_variables is not None else [self.u]

        u = self.u

        self._step_index = 0
        resumed = False
        if self._full and _restart.checkpoint_complete(self.comm, output_dir):
            self._restore_checkpoint(output_dir, timestepper, reaction_logger,
                                     loadhistory)
            resumed = True

        vtx_path = _vtx_segment_path(self.comm, output_dir, "macro",
                                     resumed and vtx_segment_per_resume)
        vtx = VTXManager(self.comm, vtx_path, fields)

        if not resumed:
            loadhistory(0.0)
            vtx.write(0.0)
            if reaction_logger is not None:
                reaction_logger.record(0.0, 0.0)
            if save_macro_history and self._full:
                _restart.save_macro_snapshot(
                    self.comm, u, output_dir, self._step_index, 0.0,
                    self._fingerprint(),
                )

        simulation_finished = False
        try:
            while not timestepper.finished:
                trial_t = timestepper.step_forward()
                logger.info("── Step  t=%.8f  dt=%.2e", trial_t, timestepper.dt)
                loadhistory(trial_t)

                stable_configuration = False
                pert_amplitude = pert_amplitude_init
                iters_in_step = 0

                while not stable_configuration:
                    self.material.step_failed = False
                    self.material.failure_reason = ""
                    try:
                        self._problem.solve()
                        reason = self._problem.solver.getConvergedReason()
                        n_iters = self._problem.solver.getIterationNumber()
                    except PETSc.Error:
                        reason, n_iters = -1, 0

                    if self.material.step_failed:
                        logger.warning("%s", self.material.failure_reason)
                        reason = -1

                    iters_in_step += max(n_iters, 0)

                    if reason <= 0:
                        # SNES or RVE failure → reject and shrink dt.
                        ok = timestepper.reject()
                        u.x.array[:] = self._u_last.x.array
                        u.x.scatter_forward()
                        if not ok:
                            logger.error(
                                "Minimum time step dt=%.2e reached — stopping.",
                                timestepper.dt_min,
                            )
                            simulation_finished = True
                        else:
                            logger.warning(
                                "SNES did not converge in %d iter (reason=%d) "
                                "— halving dt to %.2e",
                                n_iters, reason, timestepper.dt,
                            )
                        break  # exit inner retry loop; outer loop ends or retries

                    # SNES converged — check macro stability if enabled.
                    if self._stability is not None:
                        K = fem.petsc.assemble_matrix(self._Jac_form, bcs=self._bcs)
                        K.assemble()
                        try:
                            is_stable, eigenvalues = self._stability.check(K, self._eigenfunction)
                            # StabilityAnalyzer.check destroys K internally.
                        except (PETSc.Error, SystemError):
                            logger.error("Stability check failed.")
                            ok = timestepper.reject()
                            u.x.array[:] = self._u_last.x.array
                            u.x.scatter_forward()
                            if not ok:
                                logger.error(
                                    "Minimum time step dt=%.2e reached — stopping.",
                                    timestepper.dt_min,
                                )
                                simulation_finished = True
                            else:
                                logger.warning(
                                    "SNES did not converge in %d iter (reason=%d) "
                                    "— halving dt to %.2e",
                                    n_iters, reason, timestepper.dt,
                                )
                            break
                        logger.info(
                            "   λ = [%s]",
                            ", ".join(f"{ev:.4e}" for ev in eigenvalues),
                        )
                    else:
                        is_stable, eigenvalues = True, np.array([])

                    if not is_stable:
                        if iters_in_step >= max_iter_per_step:
                            ok = timestepper.reject()
                            u.x.array[:] = self._u_last.x.array
                            u.x.scatter_forward()
                            if not ok:
                                logger.error(
                                    "Iteration budget exhausted (%d ≥ %d) "
                                    "and dt=%.2e at dt_min — stopping.",
                                    iters_in_step, max_iter_per_step,
                                    timestepper.dt_min,
                                )
                                simulation_finished = True
                            else:
                                logger.warning(
                                    "Iteration budget exhausted (%d ≥ %d) "
                                    "within step — halving dt to %.2e",
                                    iters_in_step, max_iter_per_step,
                                    timestepper.dt,
                                )
                            break
                        scale, info = apply_eigenmode_perturbation(
                            u, self._eigenfunction, pert_amplitude, self.comm,
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
                        continue

                    # Stable + converged — accept.
                    stable_configuration = True
                    timestepper.accept(n_iters)
                    self._u_last.x.array[:] = u.x.array
                    self._u_last.x.scatter_forward()
                    self.material.commit()
                    logger.info(
                        "   SNES converged in %d iteration(s)  ||F|| = %.2e",
                        n_iters, self._problem.solver.getFunctionNorm(),
                    )

                    vtx.write(trial_t)
                    if self._reaction_specs:
                        self._record_reactions(reaction_logger)

                    self._step_index += 1
                    if self._full:
                        if save_macro_history:
                            _restart.save_macro_snapshot(
                                self.comm, u, output_dir, self._step_index,
                                trial_t, self._fingerprint(),
                            )
                        if rve_history_qps:
                            self._dump_rve_history(
                                output_dir, self._step_index, trial_t,
                                rve_history_qps,
                            )
                        self._write_checkpoint(output_dir, timestepper,
                                               reaction_logger)

                if simulation_finished:
                    break
        finally:
            vtx.close()

    # ------------------------------------------------------------------
    # Reaction-force extraction from the BC-free residual
    # ------------------------------------------------------------------

    def _record_reactions(
        self,
        reaction_logger: ReactionForceLogger | None,
    ) -> None:
        """Compute and log the reaction at every registered probe.

        Assembles ``self.Res`` *without* applying BCs, then sums residual
        entries at the constrained dofs (owned dofs only; ghosts are filtered
        out before the MPI reduction). By divergence theorem, this equals
        ∫(P·n)·e_k ds on the constrained face, where ``e_k`` is the basis
        vector of the BC's subspace.

        ``reaction_direction`` from add_bc is treated as informational only —
        the value reported is always along the BC's component direction.
        """
        b = fem.petsc.assemble_vector(self._Res_form)
        try:
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            n_local = (
                self.V.dofmap.index_map.size_local
                * self.V.dofmap.index_map_bs
            )
            local = b.array_r
            for parent_dofs, value, _direction in self._reaction_specs:
                owned = parent_dofs[parent_dofs < n_local]
                rxn_local = float(local[owned].sum()) if owned.size else 0.0
                rxn = self.comm.allreduce(rxn_local, op=MPI.SUM)
                disp = float(value.value)
                logger.info("   disp=%+.6f  reaction=%+.6f", disp, rxn)
                if reaction_logger is not None:
                    reaction_logger.record(disp, rxn)
        finally:
            b.destroy()

    # ------------------------------------------------------------------
    # Checkpoint / restart (full two-scale only)
    # ------------------------------------------------------------------

    def _local_n_qp(self) -> int:
        # QuadratureMap takes a *quadrature degree*, not a point count —
        # the actual number of qps per cell depends on the basix scheme.
        # Read it off one of the qmap's quadrature Functions.
        fn = next(iter(self.qmap.fluxes.values()))
        return fn.function_space.dofmap.index_map.size_local

    def _fingerprint(self) -> str:
        if not hasattr(self, "_cached_fingerprint"):
            self._cached_fingerprint = _restart.compute_partition_fingerprint(
                self._mesh)
        return self._cached_fingerprint

    def quadrature_point_info(self, *, gather: bool = True):
        """Inspect macro qp layout for choosing ``rve_history_qps``.

        See :func:`fe2_rom.ch1.restart.quadrature_point_info`.
        """
        return _restart.quadrature_point_info(self.comm, self.qmap, gather=gather)

    def _dump_rve_history(self, output_dir, step_index, t, qps):
        rves = self.material._rves
        if rves is None:
            return
        local_qps = _select_local_qps(qps, self.comm.rank, len(rves))
        local = {qp: rves[qp].dump_state() for qp in local_qps}
        _restart.save_rve_history(self.comm.rank, output_dir, step_index, t, local)

    def _write_checkpoint(self, output_dir, timestepper, reaction_logger):
        tmp = _restart.prepare_tmp(self.comm, output_dir)
        fp = self._fingerprint()
        _restart.save_meta(
            self.comm, tmp,
            t_current=float(timestepper.t_current),
            dt=float(timestepper.dt),
            step_index=int(getattr(self, "_step_index", 0)),
            gdim=int(self.gdim),
            kind="ch1",
        )
        _restart.save_reaction(self.comm, reaction_logger, tmp)
        _restart.save_macro_field(self.comm, self.u, tmp, fp)
        self.material.save_rves(tmp, fp)
        _restart.atomic_finalize(self.comm, output_dir)

    def _restore_checkpoint(self, output_dir, timestepper, reaction_logger,
                            loadhistory):
        ckpt_dir, _ = _restart.checkpoint_dirs(output_dir)
        meta = _restart.load_meta(self.comm, ckpt_dir)
        if int(meta.get("n_ranks", -1)) != self.comm.size:
            raise RuntimeError(
                f"Checkpoint was written with n_ranks={meta.get('n_ranks')}, "
                f"current run uses {self.comm.size}. Restart requires the "
                "same MPI rank count."
            )
        if int(meta.get("gdim", -1)) != self.gdim:
            raise RuntimeError(
                f"Checkpoint gdim={meta.get('gdim')} != current {self.gdim}."
            )
        timestepper.t_current = float(meta["t_current"])
        self._step_index = int(meta.get("step_index", 0))
        # Restore dt, but fall back to the user-configured dt if the
        # saved value has been clamped to ~0 by the previous run ending
        # at its t_end (so a resume to a larger t_end can make progress).
        saved_dt = float(meta["dt"])
        remaining = timestepper.t_end - timestepper.t_current
        if saved_dt > 1e-15 and saved_dt <= remaining + 1e-15:
            timestepper.dt = saved_dt
        else:
            timestepper.dt = min(timestepper.dt, max(remaining, 0.0))

        fp = self._fingerprint()
        _restart.load_macro_field(self.comm, self.u, ckpt_dir, fp)
        self._u_last.x.array[:] = self.u.x.array
        self._u_last.x.scatter_forward()

        self.material.load_rves(ckpt_dir, self._local_n_qp(), fp)

        _restart.load_reaction(self.comm, reaction_logger, ckpt_dir)

        loadhistory(timestepper.t_current)
        logger.info(
            "Resumed from checkpoint at t=%.6f, dt=%.2e",
            timestepper.t_current, timestepper.dt,
        )
