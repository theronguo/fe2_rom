"""Two-scale FE² macro solver for the micromorphic continuum.

The macroscopic problem couples the standard displacement field ``u`` with
``N_modes`` scalar enrichment-amplitude fields ``v_i`` (i = 1..N).  The weak
form is:

    ∫ P : ∇(δu) dX = 0          (u-block)
    ∫ Π_i δv_i dX + ∫ Λ_i · ∇(δv_i) dX = 0   ∀i   (v-block, weak form of Div Λ_i - Π_i = 0)

where ``P``, ``Π_i``, ``Λ_i`` come from the constitutive law at each macro qp
(either a dummy linear model or a nested micromorphic RVE).

Usage pattern::

    material = DummyMicromorphicMaterial(N_modes=1, mu=1.0, alpha=1.0, beta=1.0)
    solver = MacroMicromorphicSolver(mesh, n_qp=2, N_modes=1, material=material)
    solver.add_bc((0, 0), lambda x: np.isclose(x[0], 0.0), zero_const)
    solver.add_bc((0, 1), lambda x: np.isclose(x[0], 0.0), zero_const)
    solver.add_bc((0, 0), lambda x: np.isclose(x[0], 1.0), disp_const,
                  measure_reaction=True)
    solver.add_bc((1,),   lambda x: np.ones(x.shape[1], dtype=bool), zero_v)

Works for 2D and 3D meshes; the material classes must be constructed with the
matching ``gdim``.
    solver.setup()
    solver.solve(output_dir="output", timestepper=..., loadhistory=...)
"""
from __future__ import annotations

import logging
import os
from typing import Callable

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import basix.ufl
import ufl
from dolfinx import fem, mesh as dmesh

from fe2_rom.ch1.quadrature import OwnedCellQuadratureMap as QuadratureMap
from dolfinx_materials.solvers import NonlinearMaterialProblem
from dolfinx_materials.utils import nonsymmetric_tensor_to_vector

from fe2_rom.hyperelastic_solver.output import ReactionForceLogger, VTXManager
from fe2_rom.hyperelastic_solver.stability import (
    StabilityAnalyzer,
    apply_eigenmode_perturbation,
    mesh_characteristic_length,
)
from fe2_rom.hyperelastic_solver.timestepping import TimeStepper
from fe2_rom.ch1 import restart as _restart
from fe2_rom.ch1.macrosolver import _vtx_segment_path, _select_local_qps
from fe2_rom.mm.material import MicromorphicRVEMaterial

logger = logging.getLogger(__name__)


class MacroMicromorphicSolver:
    """Two-scale FE² solver for the micromorphic macro continuum.

    Uses a mixed function space ``[Q_deg^gdim, Q_deg, ..., Q_deg]``
    (displacement + N_modes scalar enrichment amplitudes).

    Two-phase init: ``__init__`` builds spaces, registers gradients, builds
    weak form; ``add_bc`` registers Dirichlet conditions; ``setup`` freezes
    configuration; ``solve`` runs the adaptive load loop.
    """

    def __init__(
        self,
        mesh,
        n_qp: int,
        N_modes: int,
        material,
        *,
        degree: int = 1,
        snes_options: dict | None = None,
        check_stability: bool = False,
        stability_options: dict | None = None,
        enable_restart: bool = False,
    ):
        self._mesh = mesh
        self.comm = mesh.comm
        self.gdim = mesh.geometry.dim
        self.N_modes = N_modes
        self._n_qp_per_cell = int(n_qp)
        if enable_restart and not isinstance(material, MicromorphicRVEMaterial):
            raise ValueError(
                "enable_restart=True requires material to be "
                "MicromorphicRVEMaterial (FOM inner). Dummy / ROM-inner runs "
                "are not checkpointed."
            )
        self._full_two_scale = bool(enable_restart)
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

        # ---- Mixed function space: [Q_u (vector), Q_v1, ..., Q_vN] --------
        cell = mesh.topology.cell_type.name
        P_u = basix.ufl.element("Lagrange", cell, degree, shape=(self.gdim,))
        P_v = basix.ufl.element("Lagrange", cell, degree)
        mixed_el = basix.ufl.mixed_element([P_u] + [P_v] * N_modes)
        self.V = fem.functionspace(mesh, mixed_el)

        self.w = fem.Function(self.V, name="solution")
        self._w_last = fem.Function(self.V)
        self._dw = ufl.TrialFunction(self.V)
        self._test = ufl.TestFunction(self.V)
        self._eigenfunction = fem.Function(self.V)

        n_dofs = self.V.dofmap.index_map.size_global * self.V.dofmap.index_map_bs
        logger.debug("Macro micromorphic DOFs: %d", n_dofs)

        # ---- Material + QuadratureMap --------------------------------------
        self.material = material
        self.qmap = QuadratureMap(mesh, n_qp, material)

        # Split symbolic fields for gradient registration
        split_w = ufl.split(self.w)
        u_sym = split_w[0]
        vs_sym = list(split_w[1:])  # length N_modes

        Id = ufl.Identity(self.gdim)
        F_expr = nonsymmetric_tensor_to_vector(Id + ufl.grad(u_sym))

        # UFL registration: use scalar when N=1 to stay consistent with
        # dolfinx_materials' create_quadrature_function (dim=1 → scalar QF).
        # For N>1 use as_vector so variation() returns the right-rank expression.
        if N_modes == 1:
            v_expr = vs_sym[0]   # scalar
        elif N_modes > 1:
            v_expr = ufl.as_vector([vs_sym[i] for i in range(N_modes)])

        if N_modes == 1:
            g_expr = ufl.as_vector(
                [ufl.grad(vs_sym[0])[d] for d in range(self.gdim)]
            )
        elif N_modes > 1:
            g_expr = ufl.as_vector(
                [ufl.grad(vs_sym[i])[d]
                 for i in range(N_modes) for d in range(self.gdim)]
            )

        self.qmap.register_gradient("F", F_expr)
        if N_modes > 0:
            self.qmap.register_gradient("v", v_expr)
            self.qmap.register_gradient("g", g_expr)

        # ---- Weak form -----------------------------------------------------
        P_qf      = self.qmap.fluxes["P"]
        Pi_qf     = self.qmap.fluxes["Pi"]      if N_modes > 0 else None
        Lambda_qf = self.qmap.fluxes["Lambda"]  if N_modes > 0 else None

        split_test = ufl.split(self._test)
        du_test = split_test[0]
        dvs_test = list(split_test[1:])

        test_dF = nonsymmetric_tensor_to_vector(ufl.grad(du_test))
        self.Res = ufl.dot(P_qf, test_dF) * self.qmap.dx

        if N_modes > 0:
            # Build Pi / Lambda contributions component-wise to avoid UFL
            # scalar-vs-vector issues when N_modes=1 (Pi_qf becomes scalar).
            pi_lam_res = 0
            for i in range(N_modes):
                # Pi_i term
                pi_i = Pi_qf if N_modes == 1 else Pi_qf[i]
                pi_lam_res = pi_lam_res + pi_i * dvs_test[i]
                # Lambda_{i,d} term
                for d in range(self.gdim):
                    lam_idx = i * self.gdim + d
                    lam_id = Lambda_qf[lam_idx]
                    pi_lam_res = pi_lam_res + lam_id * ufl.grad(dvs_test[i])[d]
            self.Res = self.Res + pi_lam_res * self.qmap.dx

        self.Jac = self.qmap.derivative(self.Res, self.w, self._dw)

        # ---- SNES options --------------------------------------------------
        self._snes_options = snes_options if snes_options is not None else {
            "snes_type": "newtonls",
            "snes_linesearch_type": "none",
            "snes_rtol": 1e-6,
            "snes_atol": 1e-8,
            "snes_max_it": 25,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps"
        }

        self._check_stability = check_stability
        self._stability_options = stability_options if stability_options is not None else {
            "nev": 5, "neg_tol": -1e-12,
        }

        # Populated in setup()
        self._bc_specs: list = []
        self._bcs: list = []
        self._reaction_specs: list = []
        self._problem: NonlinearMaterialProblem | None = None
        self._stability: StabilityAnalyzer | None = None
        self._Jac_form: fem.Form | None = None
        self._Res_form: fem.Form | None = None

    # ------------------------------------------------------------------
    # BC registration
    # ------------------------------------------------------------------

    def add_bc(
        self,
        component,
        locate_fn: Callable,
        value: fem.Constant,
        *,
        measure_reaction: bool = False,
        reaction_direction: tuple | None = None,
        pointwise: bool = False,
    ) -> None:
        """Register a Dirichlet BC on any component of the mixed space.

        Parameters
        ----------
        component
            ``int`` (0, 1, 2, …) → displacement component u_x, u_y, u_z.
            ``tuple`` → enrichment amplitude: ``(i+1,)`` pins v_{i+1}.
        locate_fn
            Callable ``x -> bool array`` for geometric detection.
        value
            Scalar ``fem.Constant`` mutated by ``loadhistory(t)``.
        measure_reaction
            If True, residual is summed at the constrained DOFs after each
            accepted step.
        pointwise
            If True, dofs are located *geometrically* (``locate_fn`` may select
            isolated nodes, not whole boundary facets) — used for minimal
            rigid-body pins in uniaxial-stress setups. Incompatible with
            ``measure_reaction``.
        """
        if pointwise and measure_reaction:
            raise ValueError("pointwise BCs cannot be reaction probes.")
        if isinstance(component, int):
            path = (0, component)
        else:
            path = tuple(component)
        self._bc_specs.append(
            (path, locate_fn, value, measure_reaction, reaction_direction, pointwise)
        )

    # ------------------------------------------------------------------
    # Two-phase setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Materialise BCs, create the SNES problem and stability analyser."""
        mesh = self._mesh
        fdim = mesh.topology.dim - 1
        V = self.V

        bcs: list = []
        reaction_specs: list = []

        for path, locate_fn, value, measure_reaction, direction, pointwise in self._bc_specs:
            # Traverse subspace path
            V_sub = V.sub(path[0])
            for idx in path[1:]:
                V_sub = V_sub.sub(idx)

            if pointwise:
                dofs = fem.locate_dofs_geometrical(
                    (V_sub, V_sub.collapse()[0]), locate_fn)[0]
                facets = None
            else:
                facets = dmesh.locate_entities_boundary(mesh, fdim, locate_fn)
                dofs = fem.locate_dofs_topological(V_sub, fdim, facets)
            bcs.append(fem.dirichletbc(value, dofs, V_sub))

            if measure_reaction:
                V_sub_collapsed, _ = V_sub.collapse()
                parent_dofs, _ = fem.locate_dofs_topological(
                    (V_sub, V_sub_collapsed), fdim, facets
                )
                reaction_specs.append(
                    (np.asarray(parent_dofs, dtype=np.int32), value, direction)
                )

        self._bcs = bcs
        self._reaction_specs = reaction_specs

        self._problem = NonlinearMaterialProblem(
            self.qmap, self.Res, self.w,
            bcs=self._bcs, J=self.Jac,
            petsc_options_prefix="fe2_micro_macro_",
            petsc_options=self._snes_options,
        )
        self._problem.solver.setMonitor(
            lambda _snes, i, rnorm: logger.info("   iter %2d  ||F|| = %.6e", i, rnorm)
        )

        if self._check_stability:
            self._stability = StabilityAnalyzer(self.comm, **self._stability_options)

        self._Jac_form = fem.form(self.Jac)
        self._Res_form = fem.form(self.Res)

        # Parent-space dof indices, per subspace. The eigenmode perturbation
        # caps each block independently: u against |u| (length-scaled),
        # each v_i against |v_i| (with fallback 1.0, since v_i is the
        # amplitude of a unit-normalised RVE buckling mode and naturally
        # lives in O(1)).
        _, u_collapse_map = self.V.sub(0).collapse()
        self._u_parent_dofs = np.asarray(u_collapse_map, dtype=np.int32)
        self._v_parent_dofs: list[np.ndarray] = []
        for i in range(1, 1 + self.N_modes):
            _, v_map = self.V.sub(i).collapse()
            self._v_parent_dofs.append(np.asarray(v_map, dtype=np.int32))

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
        max_negative_eigenvalues: int = 1,
        save_macro_history: bool = False,
        vtx_segment_per_resume: bool = False,
        rve_history_qps: list[int] | None = None,
        save_qp_history: bool = False,
    ) -> None:
        """Run the adaptive load-stepping outer loop.

        ``pert_amplitude_init`` is a dimensionless factor: the first
        perturbation magnitude on instability is
        ``pert_amplitude_init * max|u|`` (or ``pert_amplitude_init *
        char_length`` if ``|u|`` is ~0), measured over the u-subspace only.
        The factor doubles on each retry within a step.

        ``max_iter_per_step`` caps the *total* number of Newton iterations
        spent across all perturb-and-retry SNES calls within a single load
        step. When exceeded, the step is rejected like a SNES failure and
        the timestepper halves dt.

        ``max_negative_eigenvalues`` bounds how many negative tangent
        eigenvalues (below ``neg_tol``) the eigenmode-perturbation path will
        try to resolve. A single-mode perturbation cannot stabilise several
        simultaneous buckling modes, so when the count exceeds this threshold
        the step is rejected outright (dt halved) instead of perturbed. The
        default ``1`` rejects as soon as two or more negative eigenvalues
        appear; raise it to fall back to the perturb-and-retry behaviour.
        """
        if self._problem is None:
            raise RuntimeError("Call setup() before solve().")

        os.makedirs(output_dir, exist_ok=True)

        self._step_index = 0
        resumed = False
        if self._full_two_scale and _restart.checkpoint_complete(self.comm, output_dir):
            self._restore_checkpoint(output_dir, timestepper, reaction_logger,
                                     loadhistory)
            resumed = True

        seg = resumed and vtx_segment_per_resume
        # Default: collapse each sub-space into a separate named Function.
        # Pass output_variables=[] to suppress VTX output entirely.
        # VTXWriter does not support mixed Functions directly.
        self._vtxs: list = []
        if output_variables is None:
            cell = self._mesh.topology.cell_type.name
            P_u = basix.ufl.element("Lagrange", cell, 1, shape=(self.gdim,))
            P_v = basix.ufl.element("Lagrange", cell, 1)
            Vu_out = fem.functionspace(self._mesh, P_u)
            Vv_out = fem.functionspace(self._mesh, P_v)
            fn_u = fem.Function(Vu_out, name="u")
            fn_u.interpolate(self.w.sub(0))
            self._vtxs.append((
                VTXManager(self.comm,
                           _vtx_segment_path(self.comm, output_dir, "macro_u", seg),
                           [fn_u]),
                fn_u, 0,
            ))
            for i in range(1, 1 + self.N_modes):
                fn_v = fem.Function(Vv_out, name=f"v{i}")
                fn_v.interpolate(self.w.sub(i))
                self._vtxs.append((
                    VTXManager(self.comm,
                               _vtx_segment_path(self.comm, output_dir,
                                                 f"macro_v{i}", seg),
                               [fn_v]),
                    fn_v, i,
                ))
        elif output_variables:
            self._vtxs.append((
                VTXManager(self.comm,
                           _vtx_segment_path(self.comm, output_dir,
                                             "macro_micromorphic", seg),
                           output_variables),
                None, None,
            ))

        if resumed:
            for vtx, fn, idx in self._vtxs:
                if fn is not None:
                    fn.interpolate(self.w.sub(idx))
        else:
            loadhistory(0.0)
            for vtx, fn, idx in self._vtxs:
                if fn is not None:
                    fn.interpolate(self.w.sub(idx))
                vtx.write(0.0)
            if save_macro_history and self._full_two_scale:
                _restart.save_macro_snapshot(
                    self.comm, self.w, output_dir, self._step_index, 0.0,
                    self._fingerprint(),
                )
            if save_qp_history:
                self._save_qp_history(output_dir, self._step_index, 0.0)

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
                        ok = timestepper.reject()
                        self.w.x.array[:] = self._w_last.x.array
                        self.w.x.scatter_forward()
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

                    # Optional macro stability check
                    if self._stability is not None:
                        K = fem.petsc.assemble_matrix(self._Jac_form, bcs=self._bcs)
                        K.assemble()
                        try:
                            is_stable, eigenvalues = self._stability.check(
                                K, self._eigenfunction
                            )
                        except (PETSc.Error, SystemError):
                            logger.error("Stability check failed.")
                            ok = timestepper.reject()
                            self.w.x.array[:] = self._w_last.x.array
                            self.w.x.scatter_forward()
                            if not ok:
                                logger.error(
                                    "Minimum time step dt=%.2e reached — stopping.",
                                    timestepper.dt_min,
                                )
                                simulation_finished = True
                            else:
                                logger.warning(
                                    "Stability check failure — halving dt to %.2e",
                                    timestepper.dt,
                                )
                            break
                        logger.info(
                            "   λ = [%s]",
                            ", ".join(f"{ev:.4e}" for ev in eigenvalues),
                        )
                    else:
                        is_stable, eigenvalues = True, np.array([])

                    if not is_stable:
                        n_neg = int(np.sum(eigenvalues < self._stability._neg_tol))
                        if n_neg > max_negative_eigenvalues:
                            ok = timestepper.reject()
                            self.w.x.array[:] = self._w_last.x.array
                            self.w.x.scatter_forward()
                            if not ok:
                                logger.error(
                                    "%d negative eigenvalues (> %d) and dt=%.2e "
                                    "at dt_min — stopping.",
                                    n_neg, max_negative_eigenvalues,
                                    timestepper.dt_min,
                                )
                                simulation_finished = True
                            else:
                                logger.warning(
                                    "%d negative eigenvalues (> %d) — too many to "
                                    "perturb a single mode; rejecting step, "
                                    "halving dt to %.2e",
                                    n_neg, max_negative_eigenvalues,
                                    timestepper.dt,
                                )
                            break
                        if iters_in_step >= max_iter_per_step:
                            ok = timestepper.reject()
                            self.w.x.array[:] = self._w_last.x.array
                            self.w.x.scatter_forward()
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
                        blocks = [(self._u_parent_dofs, self._char_length)]
                        for v_dofs in self._v_parent_dofs:
                            blocks.append((v_dofs, 1.0))
                        scale, info = apply_eigenmode_perturbation(
                            self.w, self._eigenfunction, pert_amplitude,
                            self.comm, blocks=blocks,
                        )
                        u_ref, phi_u_max = info[0]
                        v_log = ", ".join(
                            f"|v{i+1}|={info[i+1][0]:.2e}→Δ={scale*info[i+1][1]:.2e}"
                            for i in range(self.N_modes)
                        )
                        logger.warning(
                            "Unstable equilibrium (λ_min=%.4e) — "
                            "perturbing with eigenvector "
                            "(factor=%.2e, |u|=%.2e, Δu=%.2e; %s)",
                            eigenvalues.min(), pert_amplitude, u_ref,
                            scale * phi_u_max, v_log,
                        )
                        pert_amplitude *= 2
                        continue

                    # Accept
                    stable_configuration = True
                    timestepper.accept(n_iters)
                    self._w_last.x.array[:] = self.w.x.array
                    self._w_last.x.scatter_forward()
                    self.material.commit()
                    logger.info(
                        "   SNES converged in %d iteration(s)  ||F|| = %.2e",
                        n_iters, self._problem.solver.getFunctionNorm(),
                    )

                    for vtx, fn, idx in self._vtxs:
                        if fn is not None:
                            fn.interpolate(self.w.sub(idx))
                        vtx.write(trial_t)
                    if self._reaction_specs:
                        self._record_reactions(reaction_logger, trial_t)

                    self._step_index += 1
                    if save_qp_history:
                        self._save_qp_history(
                            output_dir, self._step_index, trial_t,
                        )
                    if self._full_two_scale:
                        if save_macro_history:
                            _restart.save_macro_snapshot(
                                self.comm, self.w, output_dir,
                                self._step_index, trial_t, self._fingerprint(),
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
            for vtx, _fn, _idx in self._vtxs:
                vtx.close()

    # ------------------------------------------------------------------
    # Reaction-force extraction
    # ------------------------------------------------------------------

    def _record_reactions(
        self,
        reaction_logger: ReactionForceLogger | None,
        t: float,
    ) -> None:
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
                disp = float(np.asarray(value.value).ravel()[0])
                logger.info("   disp=%+.6f  reaction=%+.6f", disp, rxn)
                if reaction_logger is not None:
                    reaction_logger.record(disp, rxn)
        finally:
            b.destroy()

    # ------------------------------------------------------------------
    # Checkpoint / restart (full two-scale only)
    # ------------------------------------------------------------------

    def _local_n_qp(self) -> int:
        # ``n_qp`` is a quadrature *degree*, not a point count — the
        # per-cell point count depends on the basix scheme. Read the
        # actual local qp count off one of the qmap's quadrature
        # Functions.
        fn = next(iter(self.qmap.fluxes.values()))
        return fn.function_space.dofmap.index_map.size_local

    def _fingerprint(self) -> str:
        if not hasattr(self, "_cached_fingerprint"):
            self._cached_fingerprint = _restart.compute_partition_fingerprint(
                self._mesh)
        return self._cached_fingerprint

    def quadrature_point_info(self, *, gather: bool = True):
        """See :func:`fe2_rom.ch1.restart.quadrature_point_info`."""
        return _restart.quadrature_point_info(self.comm, self.qmap, gather=gather)

    def _save_qp_history(self, output_dir, step_index, t):
        _restart.save_qp_history(
            self.comm, self.qmap, output_dir, step_index, t,
        )

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
            N_modes=int(self.N_modes),
            kind="mm",
        )
        _restart.save_reaction(self.comm, reaction_logger, tmp)
        _restart.save_macro_field(self.comm, self.w, tmp, fp)
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
        if int(meta.get("N_modes", -1)) != self.N_modes:
            raise RuntimeError(
                f"Checkpoint N_modes={meta.get('N_modes')} != "
                f"current {self.N_modes}."
            )
        timestepper.t_current = float(meta["t_current"])
        self._step_index = int(meta.get("step_index", 0))
        saved_dt = float(meta["dt"])
        remaining = timestepper.t_end - timestepper.t_current
        if saved_dt > 1e-15 and saved_dt <= remaining + 1e-15:
            timestepper.dt = saved_dt
        else:
            timestepper.dt = min(timestepper.dt, max(remaining, 0.0))

        fp = self._fingerprint()
        _restart.load_macro_field(self.comm, self.w, ckpt_dir, fp)
        self._w_last.x.array[:] = self.w.x.array
        self._w_last.x.scatter_forward()

        self.material.load_rves(ckpt_dir, self._local_n_qp(), fp)

        _restart.load_reaction(self.comm, reaction_logger, ckpt_dir)

        loadhistory(timestepper.t_current)
        logger.info(
            "Resumed from checkpoint at t=%.6f, dt=%.2e",
            timestepper.t_current, timestepper.dt,
        )
