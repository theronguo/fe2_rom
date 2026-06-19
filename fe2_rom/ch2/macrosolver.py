"""Two-scale FE² macro solver for second-order computational homogenization.

The macroscopic problem is the mixed (saddle-point) formulation of Eqs. 3-6,
coupling the displacement ``u``, an independent deformation-gradient field ``F̂``
and a Lagrange multiplier ``L̄`` enforcing ``F̂ = F̄ := I + ∇u``. Internally the
gradient field is stored as the *fluctuation* ``H := F̂ − I`` (so every field
starts at zero). The weak form (all contractions are standard inner products in
the ``∇F̂`` index convention) reads:

    ∫ (P̄ − L̄) : ∇δu  dX                       = 0     (u-block)
    ∫ ( L̄ : δH + Q̄ : ∇δH ) dX                 = 0     (F̂-block)
    ∫ (H − ∇u) : δL̄  dX                        = 0     (L̄-block)

with ``F̄ = I + ∇u``, ``F̂ = I + H``, ``Ḡ = ∇F̂ = ∇H``. The constitutive law
(``Ch2RVEMaterial`` or ``DummyCh2Material``) provides ``P̄``, ``Q̄`` and the four
tangents ``d{P̄,Q̄}/d{F̄,Ḡ}`` at each macro quadrature point.

Discretization (paper §2.1): inf-sup-stable Taylor-Hood-like quadrilateral
element — quadratic displacement ``u`` (Q2), bilinear deformation gradient
``F̂`` (Q1) and one piecewise-constant multiplier ``L̄`` per element (Q0/DG0).

Stability of the saddle system: a *stable* equilibrium has inertia ``(n₊, m, 0)``
with ``m`` = the global number of ``L̄`` DOFs; a buckling instability shows up as
``n_neg > m`` (one reduced-Hessian eigenvalue crossing zero). When
``check_stability`` is on, the number of negative eigenvalues is read from the
MUMPS LDLᵀ factorization (INFOG(12)) and compared against ``m``.
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

from dolfinx_materials.solvers import NonlinearMaterialProblem
from dolfinx_materials.utils import nonsymmetric_tensor_to_vector, vector_to_tensor

from fe2_rom.ch1.quadrature import OwnedCellQuadratureMap as QuadratureMap
from fe2_rom.ch1 import restart as _restart
from fe2_rom.ch2.material import Ch2RVEMaterial
from fe2_rom.hyperelastic_solver.output import ReactionForceLogger, VTXManager
from fe2_rom.hyperelastic_solver.stability import (
    apply_eigenmode_perturbation,
    mesh_characteristic_length,
    solve_smallest_eigenpairs,
)
from fe2_rom.hyperelastic_solver.timestepping import TimeStepper

logger = logging.getLogger(__name__)


class MacroSecondOrderSolver:
    """Mixed FE² driver for the second-order (CH2) macro continuum.

    Two-phase init: ``__init__`` builds the mixed space and weak form;
    ``add_bc`` registers Dirichlet conditions; ``setup`` freezes the problem;
    ``solve`` runs the adaptive load loop.
    """

    def __init__(
        self,
        mesh,
        n_qp: int,
        material,
        *,
        degree: int = 1,
        u_degree: int | None = None,
        lagrange_degree: int | None = None,
        lagrange_discontinuous: bool | None = None,
        snes_options: dict | None = None,
        check_stability: bool = False,
        perturb_post_buckling: bool = False,
        neg_eig_tol: float = -1e-8,
        compat_penalty: float = 0.0,
        lagrange_stab: float = 0.0,
        enable_restart: bool = False,
    ):
        self._mesh = mesh
        self.comm = mesh.comm
        self.gdim = mesh.geometry.dim
        g = self.gdim
        self._n_qp_per_cell = int(n_qp)
        self._compat_penalty = float(compat_penalty)
        self._lagrange_stab = float(lagrange_stab)
        # Checkpoint/restart is only meaningful for a full two-scale run whose
        # RVEs carry restartable state (Ch2RVEMaterial). ROM- and dummy-inner
        # runs are cheap to redo and deliberately not checkpointed.
        if enable_restart and not isinstance(material, Ch2RVEMaterial):
            raise ValueError(
                "enable_restart=True requires material to be Ch2RVEMaterial "
                "(full / reduced RVE inner); dummy runs are not checkpointed.")
        self._full_two_scale = bool(enable_restart)
        self._step_index = 0
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

        # ---- Mixed space [u, H = F̂−I, L̄ (multiplier)] ----
        # ``degree`` is the F̂ (P1) degree; ``u_degree`` defaults to ``degree + 1``
        # (P2) — the displacement one order above F̂ makes ⟨L̄:∇u⟩ rich.
        #
        # Multiplier (L̄) default: a *continuous* P1 multiplier in the same space
        # as F̂  ⇒  P2-P1-P1, both 2D and 3D (override via ``lagrange_degree`` /
        # ``lagrange_discontinuous``). With L̄ in the F̂ space the ⟨L̄:F̂⟩ coupling
        # is a coercive mass matrix (multiplier dofs ∝ vertices), which is
        # inf-sup (LBB) stable in 2D and 3D with no ``lagrange_stab``, and is
        # better-conditioned at fine meshes than the DG0 alternative.
        #
        # The original paper element (Guo et al.) is P2-P1-DG0 — recover it with
        # ``lagrange_degree=0, lagrange_discontinuous=True``. DG0 is inf-sup
        # stable on 2D triangles but UNSTABLE on 3D tetrahedra (the DG0
        # multiplier has more modes, ∝ cells, than (u, F̂) can pin, and the
        # spurious-mode count grows with refinement). For unstable choices the
        # ``lagrange_stab`` regularization (−ε⟨L̄:δL̄⟩) is a fallback, but it
        # biases the solution by O(ε).
        u_deg = degree + 1 if u_degree is None else int(u_degree)
        if lagrange_degree is None:
            lagrange_degree = degree
        if lagrange_discontinuous is None:
            lagrange_discontinuous = False
        cell = mesh.topology.cell_type.name
        P_u = basix.ufl.element("Lagrange", cell, u_deg, shape=(g,))
        P_H = basix.ufl.element("Lagrange", cell, degree, shape=(g, g))
        P_L = basix.ufl.element(
            "Lagrange", cell, lagrange_degree, shape=(g, g),
            discontinuous=lagrange_discontinuous,
        )
        self.V = fem.functionspace(mesh, basix.ufl.mixed_element([P_u, P_H, P_L]))

        self.w = fem.Function(self.V, name="solution")
        self._w_last = fem.Function(self.V)
        self._dw = ufl.TrialFunction(self.V)
        self._test = ufl.TestFunction(self.V)
        self._eigenfunction = fem.Function(self.V)

        n_dofs = self.V.dofmap.index_map.size_global * self.V.dofmap.index_map_bs
        logger.debug("Macro CH2 DOFs: %d", n_dofs)

        # ---- Material + QuadratureMap ----
        self.material = material
        self.qmap = QuadratureMap(mesh, n_qp, material)

        u_sym, H_sym, L_sym = ufl.split(self.w)
        Id = ufl.Identity(g)
        Fhat = Id + H_sym
        Fbar = Id + ufl.grad(u_sym)

        # The RVE's FIRST-order input is the displacement gradient F̄ = I + ∇u
        # (directly available from u), NOT the independent field F̂. F̂ exists only
        # to supply the recoverable SECOND gradient Ḡ = ∇F̂, with the constraint
        # F̂ = F̄ tying ∇F̂ ≈ ∇∇u. Driving the first order with F̄ gives the
        # displacement its own stiffness (∂P̄/∂(∇u) ≠ 0); driving it with F̂ leaves
        # the u-block of the Jacobian zero and the high-frequency u modes
        # (invisible to the DG0 multiplier) unconstrained.
        F_expr = nonsymmetric_tensor_to_vector(Fbar)   # F̄ = I + ∇u
        gradH = ufl.grad(H_sym)  # (g,g,g): Ḡ_iJK = ∂F̂_iJ/∂X_K = ∂H_iJ/∂X_K
        G_expr = ufl.as_vector(
            [gradH[i, j, k] for i in range(g) for j in range(g) for k in range(g)]
        )
        self.qmap.register_gradient("F", F_expr)
        self.qmap.register_gradient("G", G_expr)

        # ---- Weak form ----
        P_qf = self.qmap.fluxes["P"]        # (F_dim,) vector
        Q_qf = self.qmap.fluxes["Q"]        # (g³,) vector
        Pbar = vector_to_tensor(P_qf)       # (g,g)
        Qbar = ufl.as_tensor(
            [[[Q_qf[i * g * g + j * g + k] for k in range(g)]
              for j in range(g)] for i in range(g)]
        )                                   # (g,g,g)

        du, dH, dL = ufl.split(self._test)
        dx = self.qmap.dx
        self.Res = (
            ufl.inner(Pbar - L_sym, ufl.grad(du)) * dx
            + ufl.inner(L_sym, dH) * dx
            + ufl.inner(Qbar, ufl.grad(dH)) * dx
            + ufl.inner(H_sym - ufl.grad(u_sym), dL) * dx
        )

        # Consistent multiplier regularization (removes the Q1–P0 checkerboard
        # null modes of the DG0 L̄). −ε⟨L̄:δL̄⟩ gives every multiplier mode a tiny
        # stiffness; ε small ⇒ the physical fields are unaffected.
        if self._lagrange_stab != 0.0:
            self.Res = self.Res - self._lagrange_stab * ufl.inner(L_sym, dL) * dx

        # Consistent compatibility stabilization. The double stress Q̄ depends on
        # ∇F̂ only through its part symmetric in the last two indices, so the
        # *incompatible* part skew_JK(∇F̂) is a stiffness nullspace (F̂ is meant to
        # be a gradient, ⇒ curl F̂ = 0 ⇒ ∇F̂ symmetric). The DG0 multiplier only
        # ties cell-averages of F̂ = F̄, leaving the sub-cell incompatibility
        # unconstrained. This penalty removes the nullspace and vanishes at the
        # physical solution (skew(∇F̂) = skew(∇∇u) = 0), so it is consistent.
        if self._compat_penalty != 0.0:
            gH = ufl.grad(H_sym)    # (g,g,g): ∂H_iJ/∂X_K
            gdH = ufl.grad(dH)
            skew_H = ufl.as_tensor(
                [[[0.5 * (gH[i, j, k] - gH[i, k, j]) for k in range(g)]
                  for j in range(g)] for i in range(g)])
            skew_dH = ufl.as_tensor(
                [[[0.5 * (gdH[i, j, k] - gdH[i, k, j]) for k in range(g)]
                  for j in range(g)] for i in range(g)])
            self.Res = self.Res + self._compat_penalty * ufl.inner(skew_H, skew_dH) * dx

        self.Jac = self.qmap.derivative(self.Res, self.w, self._dw)

        # ---- SNES options ----
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

        self._check_stability = check_stability
        self._perturb_post_buckling = perturb_post_buckling
        self._neg_eig_tol = neg_eig_tol

        # Populated in setup()
        self._bc_specs: list = []
        self._bcs: list = []
        self._reaction_specs: list = []
        self._problem: NonlinearMaterialProblem | None = None
        self._Jac_form: fem.Form | None = None
        self._Res_form: fem.Form | None = None
        self._m_lagrange: int | None = None

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
        """Register a Dirichlet BC on a component of the mixed space.

        ``component``: ``int`` → displacement component ``u_i`` (subspace 0);
        ``(1, c)`` → flattened component ``c`` (row-major ``i*gdim+j``) of the
        deformation-gradient fluctuation ``H = F̂ − I`` (so a value of ``0`` pins
        ``F̂_ij`` to ``δ_ij``); ``(2, c)`` → ``L̄`` (rarely needed).
        ``pointwise=True`` locates isolated nodes geometrically (e.g. the
        corner ``F̂ = I`` pin).
        """
        if pointwise and measure_reaction:
            raise ValueError("pointwise BCs cannot be reaction probes.")
        path = (0, component) if isinstance(component, int) else tuple(component)
        self._bc_specs.append(
            (path, locate_fn, value, measure_reaction, reaction_direction, pointwise)
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        mesh = self._mesh
        fdim = mesh.topology.dim - 1
        V = self.V
        bcs: list = []
        reaction_specs: list = []

        for path, locate_fn, value, measure_reaction, direction, pointwise in self._bc_specs:
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
                V_sub_coll, _ = V_sub.collapse()
                parent_dofs, _ = fem.locate_dofs_topological(
                    (V_sub, V_sub_coll), fdim, facets)
                reaction_specs.append(
                    (np.asarray(parent_dofs, dtype=np.int32), value, direction))

        self._bcs = bcs
        self._reaction_specs = reaction_specs

        self._problem = NonlinearMaterialProblem(
            self.qmap, self.Res, self.w, bcs=self._bcs, J=self.Jac,
            petsc_options_prefix="fe2_ch2_macro_",
            petsc_options=self._snes_options,
        )
        self._problem.solver.setMonitor(
            lambda _s, i, rnorm: logger.info("   iter %2d  ||F|| = %.6e", i, rnorm))

        self._Jac_form = fem.form(self.Jac)
        self._Res_form = fem.form(self.Res)

        # m = global number of Lagrange-multiplier (L̄) DOFs — the baseline
        # negative-eigenvalue count of the stable saddle system.
        L_coll, _ = self.V.sub(2).collapse()
        self._m_lagrange = (
            L_coll.dofmap.index_map.size_global * L_coll.dofmap.index_map_bs)

        # Parent-space dof indices per primal block, for the eigenmode
        # perturbation: the kick is capped per field (u against the mesh
        # characteristic length, F̂ against 1). The L̄ block is not perturbed.
        self._u_parent_dofs = np.asarray(
            self.V.sub(0).collapse()[1], dtype=np.int32)
        self._fhat_parent_dofs = np.asarray(
            self.V.sub(1).collapse()[1], dtype=np.int32)
        self._char_length = mesh_characteristic_length(self._mesh)

        logger.debug("Setup complete — %d BCs, %d reaction probe(s), m_L=%d",
                     len(self._bcs), len(self._reaction_specs), self._m_lagrange)

    # ------------------------------------------------------------------
    # Saddle-point stability via factorization inertia
    # ------------------------------------------------------------------

    def _num_negative_eigenvalues(self) -> int | None:
        """Number of negative eigenvalues of the BC-applied Jacobian, read from
        the MUMPS symmetric-indefinite factorization (INFOG(12)). Returns
        ``None`` if the inertia could not be obtained."""
        A = fem.petsc.assemble_matrix(self._Jac_form, bcs=self._bcs)
        A.assemble()
        try:
            ksp = PETSc.KSP().create(self.comm)
            ksp.setOperators(A)
            ksp.setType("preonly")
            pc = ksp.getPC()
            pc.setType("cholesky")
            pc.setFactorSolverType("mumps")
            pc.setUp()
            Fmat = pc.getFactorMatrix()
            neg = int(Fmat.getMumpsInfog(12))
            ksp.destroy()
            return neg
        except Exception:  # pragma: no cover - depends on PETSc/MUMPS build
            logger.exception("Could not read MUMPS inertia; skipping stability.")
            return None
        finally:
            A.destroy()

    def _set_buckling_eigenvector(self, neg_tol: float, nev: int = 12) -> float | None:
        """Write the buckling eigenvector into ``self._eigenfunction``.

        For the saddle system the ``m`` structural multiplier modes are deeply
        negative; the buckling mode is the negative eigenvalue *closest to zero*
        (the reduced-Hessian eigenvalue that just crossed). Solve the smallest-
        |λ| eigenpairs (shift-invert at 0, direct inner solve for the indefinite
        K) and pick the least-negative one. Returns its eigenvalue, or ``None``
        if no negative eigenvalue is found among the computed set.
        """
        A = fem.petsc.assemble_matrix(self._Jac_form, bcs=self._bcs)
        A.assemble()
        try:
            eps, n_conv = solve_smallest_eigenpairs(
                A, self.comm, nev=nev, tol=1e-4,
                petsc_options={"st_ksp_type": "preonly", "st_pc_type": "lu",
                               "st_pc_factor_mat_solver_type": "mumps"})
            try:
                negs = [(eps.getEigenvalue(i).real, i) for i in range(n_conv)
                        if eps.getEigenvalue(i).real < neg_tol]
                if not negs:
                    return None
                lam, idx = max(negs, key=lambda t: t[0])  # least negative ≈ just crossed
                eps.getEigenvector(idx, self._eigenfunction.x.petsc_vec)
                self._eigenfunction.x.scatter_forward()
                return lam
            finally:
                eps.destroy()
        finally:
            A.destroy()

    # ------------------------------------------------------------------
    # Main load-stepping loop
    # ------------------------------------------------------------------

    def solve(
        self,
        *,
        output_dir: str,
        timestepper: TimeStepper,
        loadhistory: Callable[[float], None],
        reaction_logger: ReactionForceLogger | None = None,
        write_vtx: bool = True,
        pert_amplitude_init: float = 1e-2,
        max_iter_per_step: int = 30,
    ) -> None:
        if self._problem is None:
            raise RuntimeError("Call setup() before solve().")
        os.makedirs(output_dir, exist_ok=True)

        self._step_index = 0
        resumed = False
        if self._full_two_scale and _restart.checkpoint_complete(self.comm, output_dir):
            self._restore_checkpoint(output_dir, timestepper, reaction_logger,
                                     loadhistory)
            resumed = True

        # VTX output: u (P1 vector), F̂ = I + H (P1 tensor), L̄ (DG1 tensor).
        # One .bp per field (VTX cannot mix value shapes / CG+DG in one writer).
        vtxs = self._setup_vtx(output_dir) if write_vtx else []
        if resumed:
            self._update_vtx(vtxs, timestepper.t_current)
        else:
            loadhistory(0.0)
            self._update_vtx(vtxs, 0.0)

        simulation_finished = False
        try:
            while not timestepper.finished:
                trial_t = timestepper.step_forward()
                logger.info("── Step  t=%.8f  dt=%.2e", trial_t, timestepper.dt)
                loadhistory(trial_t)

                stable_configuration = False
                rejected = False
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
                        logger.warning("SNES did not converge (reason=%d).", reason)
                        rejected = True
                        break

                    # Saddle-point inertia stability: stable system has exactly
                    # m negative eigenvalues (m = L̄ DOFs). n_neg = m+1 → one
                    # buckling mode (perturb along it); n_neg ≥ m+2 → several
                    # simultaneous modes (a single eigenmode kick can't fix them,
                    # so reject).
                    if not self._check_stability:
                        stable_configuration = True
                        break
                    n_neg = self._num_negative_eigenvalues()
                    if n_neg is None:
                        stable_configuration = True   # inertia unavailable → accept
                        break
                    n_extra = n_neg - self._m_lagrange
                    logger.info("   inertia: n_neg=%d  m_L=%d  → %d extra negative",
                                n_neg, self._m_lagrange, n_extra)
                    if n_extra <= 0:
                        stable_configuration = True
                        break
                    if n_extra >= 2:
                        logger.warning(
                            "Unstable: %d negative eigenvalues beyond the %d "
                            "Lagrange multipliers — too many to perturb a single "
                            "mode; rejecting step.", n_extra, self._m_lagrange)
                        rejected = True
                        break
                    if iters_in_step >= max_iter_per_step:
                        logger.warning(
                            "Iteration budget exhausted (%d ≥ %d) within step — "
                            "rejecting.", iters_in_step, max_iter_per_step)
                        rejected = True
                        break
                    # n_extra == 1: perturb along the buckling eigenmode.
                    lam = self._set_buckling_eigenvector(self._neg_eig_tol)
                    if lam is None:
                        logger.warning(
                            "n_neg = m+1 but no buckling eigenvector found — "
                            "rejecting step.")
                        rejected = True
                        break
                    scale, info = apply_eigenmode_perturbation(
                        self.w, self._eigenfunction, pert_amplitude, self.comm,
                        blocks=[(self._u_parent_dofs, self._char_length),
                                (self._fhat_parent_dofs, 1.0)])
                    logger.warning(
                        "Unstable equilibrium (λ_buckle=%.4e) — perturbing along "
                        "the buckling eigenmode (factor=%.2e, Δu=%.2e, ΔF̂=%.2e)",
                        lam, pert_amplitude, scale * info[0][1], scale * info[1][1])
                    pert_amplitude *= 2
                    # loop: re-solve from the perturbed state

                if rejected:
                    ok = timestepper.reject()
                    self.w.x.array[:] = self._w_last.x.array
                    self.w.x.scatter_forward()
                    if not ok:
                        logger.error("Minimum dt=%.2e reached — stopping.",
                                     timestepper.dt_min)
                        simulation_finished = True
                    else:
                        logger.warning("Step rejected — halving dt to %.2e",
                                       timestepper.dt)
                    if simulation_finished:
                        break
                    continue

                # Accept
                timestepper.accept(n_iters)
                self._w_last.x.array[:] = self.w.x.array
                self._w_last.x.scatter_forward()
                self.material.commit()
                logger.info("   SNES converged in %d iteration(s)  ||F|| = %.2e",
                            n_iters, self._problem.solver.getFunctionNorm())

                self._update_vtx(vtxs, trial_t)
                if self._reaction_specs:
                    self._record_reactions(reaction_logger, trial_t)

                self._step_index += 1
                if self._full_two_scale:
                    self._write_checkpoint(output_dir, timestepper, reaction_logger)
        finally:
            for vtx, _fn, _sub, _add_I in vtxs:
                vtx.close()

    # ------------------------------------------------------------------
    # Reaction-force extraction
    # ------------------------------------------------------------------

    def _record_reactions(self, reaction_logger, t: float) -> None:
        b = fem.petsc.assemble_vector(self._Res_form)
        try:
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            n_local = self.V.dofmap.index_map.size_local * self.V.dofmap.index_map_bs
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
    # VTX output: u, F̂ = I + H, L̄
    # ------------------------------------------------------------------

    def _setup_vtx(self, output_dir: str) -> list:
        """One VTX writer per macro field (VTX cannot mix value shapes or
        CG+DG in a single writer). Returns
        ``[(VTXManager, Function, subspace, add_identity), ...]``."""
        g = self.gdim
        cell = self._mesh.topology.cell_type.name
        P_u = basix.ufl.element("Lagrange", cell, 1, shape=(g,))
        P_t = basix.ufl.element("Lagrange", cell, 1, shape=(g, g))
        P_t_dg = basix.ufl.element("DG", cell, 1, shape=(g, g))
        fn_u = fem.Function(fem.functionspace(self._mesh, P_u), name="u")
        fn_F = fem.Function(fem.functionspace(self._mesh, P_t), name="Fhat")
        fn_L = fem.Function(fem.functionspace(self._mesh, P_t_dg), name="L")
        return [
            (VTXManager(self.comm, os.path.join(output_dir, "macro_u.bp"), [fn_u]),
             fn_u, 0, False),
            (VTXManager(self.comm, os.path.join(output_dir, "macro_Fhat.bp"), [fn_F]),
             fn_F, 1, True),   # add_identity: F̂ = I + H
            (VTXManager(self.comm, os.path.join(output_dir, "macro_L.bp"), [fn_L]),
             fn_L, 2, False),
        ]

    def _update_vtx(self, vtxs: list, t: float) -> None:
        g = self.gdim
        diag = [i * g + i for i in range(g)]
        for vtx, fn, sub, add_I in vtxs:
            fn.interpolate(self.w.sub(sub))
            if add_I:  # F̂ = I + H (add 1 on the diagonal tensor components)
                fn.x.array.reshape(-1, g * g)[:, diag] += 1.0
            fn.x.scatter_forward()
            vtx.write(t)

    # ------------------------------------------------------------------
    # Checkpoint / restart (full two-scale) — shares fe2_rom.ch1.restart
    # ------------------------------------------------------------------

    def _local_n_qp(self) -> int:
        fn = next(iter(self.qmap.fluxes.values()))
        return fn.function_space.dofmap.index_map.size_local

    def _fingerprint(self) -> str:
        if not hasattr(self, "_cached_fingerprint"):
            self._cached_fingerprint = _restart.compute_partition_fingerprint(self._mesh)
        return self._cached_fingerprint

    def quadrature_point_info(self, *, gather: bool = True):
        """See :func:`fe2_rom.ch1.restart.quadrature_point_info`."""
        return _restart.quadrature_point_info(self.comm, self.qmap, gather=gather)

    def _write_checkpoint(self, output_dir, timestepper, reaction_logger) -> None:
        tmp = _restart.prepare_tmp(self.comm, output_dir)
        fp = self._fingerprint()
        _restart.save_meta(
            self.comm, tmp,
            t_current=float(timestepper.t_current), dt=float(timestepper.dt),
            step_index=int(self._step_index), gdim=int(self.gdim), kind="ch2")
        _restart.save_reaction(self.comm, reaction_logger, tmp)
        _restart.save_macro_field(self.comm, self.w, tmp, fp)
        self.material.save_rves(tmp, fp)
        _restart.atomic_finalize(self.comm, output_dir)

    def _restore_checkpoint(self, output_dir, timestepper, reaction_logger,
                            loadhistory) -> None:
        ckpt_dir, _ = _restart.checkpoint_dirs(output_dir)
        meta = _restart.load_meta(self.comm, ckpt_dir)
        if int(meta.get("n_ranks", -1)) != self.comm.size:
            raise RuntimeError(
                f"Checkpoint n_ranks={meta.get('n_ranks')} != current "
                f"{self.comm.size}; restart needs the same MPI rank count.")
        if int(meta.get("gdim", -1)) != self.gdim:
            raise RuntimeError(
                f"Checkpoint gdim={meta.get('gdim')} != current {self.gdim}.")
        timestepper.t_current = float(meta["t_current"])
        self._step_index = int(meta.get("step_index", 0))
        saved_dt = float(meta["dt"])
        remaining = timestepper.t_end - timestepper.t_current
        if 1e-15 < saved_dt <= remaining + 1e-15:
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
        logger.info("Resumed from checkpoint at t=%.6f, dt=%.2e",
                    timestepper.t_current, timestepper.dt)
