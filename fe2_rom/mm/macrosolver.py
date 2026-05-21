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

from dolfinx_materials.quadrature_map import QuadratureMap
from dolfinx_materials.solvers import NonlinearMaterialProblem
from dolfinx_materials.utils import nonsymmetric_tensor_to_vector

from fe2_rom.hyperelastic_solver.output import ReactionForceLogger, VTXManager
from fe2_rom.hyperelastic_solver.stability import StabilityAnalyzer
from fe2_rom.hyperelastic_solver.timestepping import TimeStepper

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
    ):
        self._mesh = mesh
        self.comm = mesh.comm
        self.gdim = mesh.geometry.dim
        self.N_modes = N_modes
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
        """
        if isinstance(component, int):
            path = (0, component)
        else:
            path = tuple(component)
        self._bc_specs.append(
            (path, locate_fn, value, measure_reaction, reaction_direction)
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

        for path, locate_fn, value, measure_reaction, direction in self._bc_specs:
            # Traverse subspace path
            V_sub = V.sub(path[0])
            for idx in path[1:]:
                V_sub = V_sub.sub(idx)

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
        pert_amplitude_init: float = 1e1,
    ) -> None:
        """Run the adaptive load-stepping outer loop."""
        if self._problem is None:
            raise RuntimeError("Call setup() before solve().")

        os.makedirs(output_dir, exist_ok=True)
        # Default: collapse each sub-space into a separate named Function.
        # Pass output_variables=[] to suppress VTX output entirely.
        # VTXWriter does not support mixed Functions directly.
        if output_variables is None:
            fields = []
            for i in range(1 + self.N_modes):
                Vs, _ = self.V.sub(i).collapse()
                name = "u" if i == 0 else f"v{i}"
                fn = fem.Function(Vs, name=name)
                fn.interpolate(self.w.sub(i))
                fields.append(fn)
            self._vtx_fields = fields  # keep alive; updated each step below
        elif output_variables:
            fields = output_variables
            self._vtx_fields = None
        else:
            fields = []
            self._vtx_fields = None
        vtx = VTXManager(self.comm, os.path.join(output_dir, "macro_micromorphic.bp"), fields) \
            if fields else None

        loadhistory(0.0)
        if vtx is not None:
            if self._vtx_fields is not None:
                for i, fn in enumerate(self._vtx_fields):
                    fn.interpolate(self.w.sub(i))
            vtx.write(0.0)

        simulation_finished = False
        try:
            while not timestepper.finished:
                trial_t = timestepper.step_forward()
                logger.info("── Step  t=%.5f  dt=%.2e", trial_t, timestepper.dt)
                loadhistory(trial_t)

                stable_configuration = False
                pert_amplitude = pert_amplitude_init

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
                    else:
                        is_stable, eigenvalues = True, np.array([])

                    if not is_stable:
                        self.w.x.petsc_vec.axpy(
                            pert_amplitude, self._eigenfunction.x.petsc_vec
                        )
                        self.w.x.scatter_forward()
                        logger.warning(
                            "Unstable equilibrium (λ_min=%.4e) — "
                            "perturbing (amplitude=%.2e)",
                            eigenvalues[0], pert_amplitude,
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

                    if vtx is not None:
                        if self._vtx_fields is not None:
                            for i, fn in enumerate(self._vtx_fields):
                                fn.interpolate(self.w.sub(i))
                        vtx.write(trial_t)
                    if self._reaction_specs:
                        self._record_reactions(reaction_logger, trial_t)

                if simulation_finished:
                    break
        finally:
            if vtx is not None:
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
