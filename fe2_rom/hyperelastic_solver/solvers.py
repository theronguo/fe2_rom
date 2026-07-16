"""Newton-Raphson and arc-length solvers share this module because arc-length
is an augmented Newton method — the corrector loop reuses the same PETSc
assembly machinery and only adds an arc-length constraint equation."""

import logging
from abc import ABC, abstractmethod
from typing import Callable

import dolfinx_mpc
import numpy as np
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
from mpi4py import MPI
from petsc4py import PETSc

logger = logging.getLogger(__name__)


class NewtonSolver:
    """Newton-Raphson solver with configurable linear solver and optional MINRES fallback.

    Default primary solver: CG + GAMG with PETSc default smoothers.

    Pass *petsc_options* (dict of option-key → value, without the leading '-') to
    replace the default solver entirely with any PETSc KSP/PC combination.

    When *switch_to_minres* is True and the primary solver diverges, the solver
    switches to a hardcoded MINRES + GAMG setup (ignoring *petsc_options*) and
    expands *effective_max_iter* to *max_iter_instab*.  This is useful for
    following the solution through buckling where K becomes indefinite.

    Designed to be called multiple times per time step (once per stability retry)
    via solve(iter_start=...).  Call reset_for_new_timestep() at the start of
    each time step.
    """

    def __init__(self, comm, R_form, J_form, u, du, bcs, mpc=None, *,
                 rel_tol=1e-8, abs_tol=1e-6, max_iter=10, max_iter_instab=30,
                 div_rel_tol=10.0,
                 petsc_options: dict | None = None,
                 switch_to_minres=False,
                 line_search=False,
                 constraint_forms: "list | None" = None,
                 ):
        self._comm = comm
        self._R_form = R_form
        self._J_form = J_form
        self._u = u
        self._du = du
        self._bcs = bcs
        self._rel_tol = rel_tol
        self._abs_tol = abs_tol
        self._max_iter = max_iter
        self._max_iter_instab = max_iter_instab
        self._div_rel_tol = div_rel_tol
        self._petsc_options = petsc_options
        self._switch_to_minres = switch_to_minres
        self._line_search = line_search
        self.mpc = mpc

        if mpc is not None:
            self._du = fem.Function(mpc.function_space, name="du")  # shadow du with MPC-aware version

        self._using_minres_fallback = False
        self.effective_max_iter = max_iter
        self._abs_b_norm_init = 1.0

        # Rows of the constraint matrix C.  Forms are stored so they can be
        # reassembled later (e.g. after φ-modes are updated in the micromorphic
        # solver).  Call rebuild_constraint_vecs() to refresh after any change.
        self._constraint_forms_raw: list = list(constraint_forms or [])
        self._constraint_vecs: list[PETSc.Vec] = []
        self.rebuild_constraint_vecs()

    def _make_projector(self):
        """Build and return (G_pinv, apply_P) from the current constraint vecs.

        ``apply_P(v)`` projects v in-place onto the constraint null space:
        ``v <- P v = v - C^T (C C^T)^+ (C v)``.

        The constraint basis is **normalised** before forming the Gram matrix,
        so ``G`` is a well-scaled correlation matrix (unit diagonal). This
        matters for the second-order (ch2) solver, which mixes volume integrals
        ``⟨w⟩`` with face integrals ``∫_Γ w``: their raw magnitudes differ by
        orders of magnitude, so the unnormalised ``C C^T`` is severely
        ill-conditioned and its explicit inverse **overflows** — projecting a
        non-constraint-satisfying vector (e.g. a raw buckling eigenmode, where
        ``C v = O(1)``) then blows it up to ~1e308 and injects NaNs downstream.
        A Moore–Penrose pseudo-inverse additionally tolerates a genuinely
        redundant / linearly-dependent constraint (it projects onto the actual
        span instead of dividing by ~0). Scaling the rows of ``C`` leaves the
        projector ``P`` mathematically unchanged.
        """
        c_vecs = self._constraint_vecs
        m = len(c_vecs)
        norms = np.array([float(np.sqrt(max(ci.dot(ci), 0.0))) for ci in c_vecs])
        norms = np.where(norms > 0.0, norms, 1.0)
        G = np.empty((m, m))
        for i, ci in enumerate(c_vecs):
            for j, cj in enumerate(c_vecs):
                G[i, j] = ci.dot(cj) / (norms[i] * norms[j])
        G_pinv = np.linalg.pinv(G, rcond=1e-10)

        def apply_P(v: PETSc.Vec) -> None:
            Cv = np.array([c_vecs[i].dot(v) / norms[i] for i in range(m)])
            alpha = G_pinv @ Cv
            for i, ci in enumerate(c_vecs):
                v.axpy(-alpha[i] / norms[i], ci)

        return G_pinv, apply_P

    def project_direction(self, func) -> float:
        """Make a perturbation direction consistent with the constraint
        manifold and the periodic ties, in place; return the fraction of the
        direction that survives projection, ``‖P v‖ / ‖v‖``.

        The projected Newton step only ever adds C-orthogonal increments
        (``du`` is re-projected, so ``C·du = 0``); ``C·u`` is therefore a
        conserved quantity of the solve and is never restored to zero. An
        eigenmode kick ``u += s·φ`` taken from the raw tangent generally has
        ``C·φ ≠ 0`` (and is not periodic at the MPC slaves), so it pushes
        ``u`` off the manifold and the projected Newton then freezes that
        violation in. Projecting ``φ`` into ``range(P)`` and MPC-back-
        substituting it — in the same order the Newton increment is handled
        (project → forward-scatter → backsubstitute) — keeps the kicked state
        on the manifold.

        The returned ratio is the cheap Mechanism-2 test: a raw-tangent
        eigenmode that lies (almost) entirely in the constraint subspace —
        a *constraint-forbidden* direction, not a physical instability —
        survives projection as ``≈ 0``, so a tiny ratio flags a spurious mode
        the caller should not perturb along. It is measured as the ratio of
        owned ``‖·‖_∞`` *after* the full project→backsubstitute (the exact
        direction the kick will use) to the raw ``‖·‖_∞`` *before* — so it
        agrees with the magnitude ``apply_eigenmode_perturbation`` sees
        (measuring before backsubstitution can disagree, since the periodic
        backsubstitution relocates the projected mode's energy at the slave
        dofs).

        Returns ``1.0`` (no-op) when there are no integral constraints (e.g.
        the default CH1 corner-pinned regime), so that well-tested
        unconstrained path is left untouched.
        """
        if not self._constraint_vecs:
            return 1.0
        imap = func.function_space.dofmap.index_map
        bs = func.function_space.dofmap.index_map_bs
        n_local = imap.size_local * bs
        comm = func.function_space.mesh.comm

        def owned_inf_norm() -> float:
            a = func.x.array[:n_local]
            local = float(np.max(np.abs(a))) if a.size else 0.0
            return comm.allreduce(local, op=MPI.MAX)

        raw_inf = owned_inf_norm()
        vec = func.x.petsc_vec
        # Project the (condensed) eigenmode onto the integral-constraint null
        # space — and stop there. We deliberately do NOT MPC-backsubstitute:
        # composing the constraint projector with the periodic backsubstitution
        # on a raw buckling eigenmode overflows to ~1e308 (in either order —
        # apply_P's constraint correction at a master fans out through the
        # periodic ties), injecting NaNs into the kick. apply_P alone is stable,
        # and keeping the kick C-orthogonal is the property that matters (the
        # original solver never backsubstituted the eigenmode at all); the
        # subsequent Newton solve enforces the periodic ties on its increments.
        _, apply_P = self._make_projector()
        apply_P(vec)
        func.x.scatter_forward()
        proj_inf = owned_inf_norm()
        return float(proj_inf / raw_inf) if raw_inf > 0.0 else 0.0

    def _solve_projected(self, K: PETSc.Mat, residual: PETSc.Vec) -> int:
        """Solve P K P du = -P R via MINRES on a shell matrix; recover λ.

        Implements the projected formulation for the constrained Newton step:

            [K   C^T] [du]   [-R]
            [C    0 ] [λ ] = [ 0]

        without forming the KKT matrix.  The orthogonal projector
            P = I - C^T (C C^T)^{-1} C
        is applied matrix-free.  G = C C^T is tiny (m×m, m ≤ gdim) and
        inverted with numpy.

        Steps:
          1. Build G and G_inv once.
          2. Define apply_P(v) as a closure: v -= C^T G^{-1} (C v).
          3. Compute b = P(-R) (projected RHS).
          4. Wrap K_proj(v) = P(K(Pv)) as a PETSc shell matrix.
          5. Solve K_proj du = b with MINRES + PC=NONE.
          6. Re-project du for numerical precision.
          7. Recover λ = G^{-1} C(-R - K du).

        Writes result into self._du.x.petsc_vec.
        Returns the KSP convergence reason (positive = converged).
        """
        c_vecs = self._constraint_vecs
        G_inv, apply_P = self._make_projector()

        # b = P(-R)
        b = residual.copy()
        b.scale(-1.0)
        apply_P(b)

        # Shell matrix: K_proj(v) = P(K(Pv))
        # The inner ghost update propagates the P-modified values across
        # MPI ranks before K.mult reads ghost DOFs.
        class _PKP:
            def mult(self_inner, mat, x, y):
                Px = x.copy()
                apply_P(Px)
                K.mult(Px, y)
                PETSc.Vec.destroy(Px)
                apply_P(y)

            def multTranspose(self_inner, mat, x, y):
                self_inner.mult(mat, x, y)  # K_proj is symmetric

        sizes = K.getSizes()
        K_proj = PETSc.Mat().create(self._comm)
        K_proj.setSizes(sizes)
        K_proj.setType(PETSc.Mat.Type.PYTHON)
        K_proj.setPythonContext(_PKP())
        K_proj.setUp()

        du_vec = self._du.x.petsc_vec
        n_global = sizes[0][1]  # global rows
        ksp_proj = PETSc.KSP().create(self._comm)
        # Use K as the PC matrix so the preconditioner builds from the assembled
        # operator even though the matvec operator K_proj is a shell matrix.
        ksp_proj.setOperators(K_proj, K)
        rtol = min(self._rel_tol * 1e-2, 1e-10)
        atol = min(self._abs_tol * 1e-2, 1e-12)
        po = self._petsc_options
        inner_ksp = None
        K_reg = None
        if po is not None and po.get("pc_type") in ("lu", "cholesky"):
            # Projected direct preconditioner  M⁻¹ = P (K+σI)⁻¹ P.
            sigma = float(po.get("proj_pc_shift", 1e-8))
            K_reg = K.copy()
            K_reg.shift(sigma)
            inner_ksp = PETSc.KSP().create(self._comm)
            inner_ksp.setOperators(K_reg)
            inner_ksp.setType(PETSc.KSP.Type.PREONLY)
            ipc = inner_ksp.getPC()
            ipc.setType(po["pc_type"])
            ipc.setFactorSolverType(po.get("pc_factor_mat_solver_type", "mumps"))

            class _ProjPC:
                def apply(self_pc, pc, x, y):
                    Px = x.copy()
                    apply_P(Px)
                    inner_ksp.solve(Px, y)
                    apply_P(y)
                    PETSc.Vec.destroy(Px)

            ksp_proj.setType(po.get("ksp_type", "fgmres"))
            pc = ksp_proj.getPC()
            pc.setType(PETSc.PC.Type.PYTHON)
            pc.setPythonContext(_ProjPC())
        elif po is not None:
            opts = PETSc.Options()
            for key, val in po.items():
                opts[key] = val
            ksp_proj.setFromOptions()
        else:
            ksp_proj.setType(PETSc.KSP.Type.MINRES)
            ksp_proj.getPC().setType(PETSc.PC.Type.GAMG)
            ksp_proj.setFromOptions()
        ksp_proj.setTolerances(rtol=rtol, atol=atol, max_it=min(n_global, 50_000))
        ksp_proj.solve(b, du_vec)
        reason = ksp_proj.getConvergedReason()
        n_iter = ksp_proj.getIterationNumber()
        ksp_proj.destroy()
        K_proj.destroy()
        if inner_ksp is not None:
            inner_ksp.destroy()
        if K_reg is not None:
            K_reg.destroy()

        logger.debug("Projected solve: reason=%d  n_iter=%d  |du|=%.3e",
                     reason, n_iter, du_vec.norm())

        # Re-project for numerical precision
        apply_P(du_vec)

        # Recover λ: G λ = C(-R - K du)
        Kdu = residual.duplicate()
        K.mult(du_vec, Kdu)
        r_kkt = residual.copy()
        r_kkt.scale(-1.0)       # -R
        r_kkt.axpy(-1.0, Kdu)  # -R - K du
        PETSc.Vec.destroy(Kdu)
        Cr = np.array([ci.dot(r_kkt) for ci in c_vecs])
        lam = G_inv @ Cr
        PETSc.Vec.destroy(r_kkt)
        PETSc.Vec.destroy(b)

        logger.debug("Lagrange multipliers: %s", np.array2string(lam, precision=4))
        return reason

    def rebuild_constraint_vecs(self) -> None:
        """Reassemble constraint vectors from stored forms.

        Must be called collectively on all MPI ranks.  The micromorphic solver
        calls this after ``compute_linear_buckling_modes()`` updates the φ
        functions so that ⟨w·φᵢ⟩ and ⟨(w·φᵢ)X⟩ rows reflect the new modes.
        """
        for c in self._constraint_vecs:
            PETSc.Vec.destroy(c)
        self._constraint_vecs = []
        for form in self._constraint_forms_raw:
            if self.mpc is not None:
                c = dolfinx_mpc.assemble_vector(form, self.mpc)
            else:
                c = fem_petsc.assemble_vector(form)
            c.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            self._constraint_vecs.append(c)
        logger.debug("Constraint vecs rebuilt: %d rows", len(self._constraint_vecs))

    def reset_for_new_timestep(self):
        """Reset solver state for a fresh time step."""
        self._using_minres_fallback = False
        self.effective_max_iter = self._max_iter
        self._abs_b_norm_init = 1.0

    def _make_ksp(self, K: PETSc.Mat) -> PETSc.KSP:
        """Build and return a KSP for matrix K.

        Priority:
          1. MINRES + GAMG when _using_minres_fallback is True (ignores petsc_options).
          2. User-supplied petsc_options via PETSc.Options + setFromOptions().
          3. Default: CG + GAMG with PETSc default smoother.
        """
        ksp = PETSc.KSP().create(self._comm)
        ksp.setOperators(K)

        if self._using_minres_fallback:
            ksp.setType(PETSc.KSP.Type.MINRES)
            ksp.getPC().setType(PETSc.PC.Type.GAMG)
        elif self._petsc_options is not None:
            opts = PETSc.Options()
            for key, val in self._petsc_options.items():
                opts[key] = val
        else:
            ksp.setType(PETSc.KSP.Type.CG)
            ksp.getPC().setType(PETSc.PC.Type.GAMG)

        ksp.setFromOptions()
        return ksp

    def _scatter_update(self, vec):
        vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    def solve(self, iter_start: int = 0) -> tuple[bool, int]:
        """Run Newton iterations from iter_start up to effective_max_iter.

        Returns (converged, final_iter_count).
        May switch from CG to MINRES internally and update effective_max_iter.
        The initial residual norm is set only when iter_newton == 0 to preserve
        accumulation behaviour across stability retries.
        """
        iter_newton = iter_start
        is_converged = False
        convergence_reason = ""

        while iter_newton < self.effective_max_iter:
            is_converged = False

            if self.mpc is not None:
                residual = dolfinx_mpc.assemble_vector(self._R_form, self.mpc)
                dolfinx_mpc.apply_lifting(residual, [self._J_form], [self._bcs], self.mpc,
                                          x0=[self._u.x.petsc_vec], scale=-1.0)
            else:
                residual = fem_petsc.assemble_vector(self._R_form)
                fem_petsc.apply_lifting(residual, [self._J_form], [self._bcs],
                                        x0=[self._u.x.petsc_vec], alpha=-1.0)
            residual.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            fem_petsc.set_bc(residual, self._bcs, x0=self._u.x.petsc_vec, alpha=-1.0)

            if self._constraint_vecs:
                # KKT convergence: check ||P R|| — the C^T λ component of R
                # is balanced by the Lagrange multiplier and must not count.
                _, apply_P_conv = self._make_projector()
                pr = residual.copy()
                apply_P_conv(pr)
                abs_b_norm = pr.norm()
                PETSc.Vec.destroy(pr)
            else:
                abs_b_norm = residual.norm()
            if iter_newton == 0:
                self._abs_b_norm_init = abs_b_norm

            rel = abs_b_norm / max(self._abs_b_norm_init, 1e-16)
            logger.debug("  iter %2d  rel=%.3e  abs=%.3e", iter_newton, rel, abs_b_norm)

            if rel < self._rel_tol or abs_b_norm < self._abs_tol:
                if rel < self._rel_tol:
                    convergence_reason = f"rel={rel:.3e} < tol={self._rel_tol:.3e}"
                else:
                    convergence_reason = f"abs={abs_b_norm:.3e} < tol={self._abs_tol:.3e}"
                PETSc.Vec.destroy(residual)
                is_converged = True
                break

            if rel > self._div_rel_tol or np.isnan(abs_b_norm):
                PETSc.Vec.destroy(residual)
                break
            
            if self.mpc is not None:
                K = dolfinx_mpc.assemble_matrix(self._J_form, self.mpc, bcs=self._bcs)
            else:
                K = fem_petsc.assemble_matrix(self._J_form, bcs=self._bcs)
            K.assemble()

            if self._constraint_vecs:
                # Projected constrained solve: P K P du = -P R via MINRES shell.
                # _switch_to_minres / fallback logic does not apply here because
                # we always use MINRES on the projected (symmetric) system.
                reason = self._solve_projected(K, residual)
                PETSc.Mat.destroy(K)
                if reason < 0:
                    logger.warning("Projected MINRES did not converge (reason %d)", reason)
                    PETSc.Vec.destroy(residual)
                    return is_converged, iter_newton
            else:
                # Standard unconstrained solve (CG/MINRES + GAMG).
                ksp = self._make_ksp(K)
                ksp.solve(-residual, self._du.x.petsc_vec)
                logger.debug("du norm: %.3e", self._du.x.petsc_vec.norm())
                reason = ksp.getConvergedReason()
                if reason < 0 and not self._using_minres_fallback:
                    if self._switch_to_minres:
                        logger.warning("Primary solver did not converge (reason %d) — switching to MINRES+GAMG", reason)
                        self._using_minres_fallback = True
                        self.effective_max_iter = self._max_iter_instab
                        ksp.destroy()
                        PETSc.Vec.destroy(residual)
                        PETSc.Mat.destroy(K)
                        continue
                    else:
                        logger.warning("Primary solver did not converge (reason %d)", reason)
                        ksp.destroy()
                        PETSc.Vec.destroy(residual)
                        PETSc.Mat.destroy(K)
                        return is_converged, iter_newton
                elif reason < 0 and self._using_minres_fallback:
                    logger.warning("MINRES did not converge (reason %d) — reducing time step", reason)
                    ksp.destroy()
                    PETSc.Vec.destroy(residual)
                    PETSc.Mat.destroy(K)
                    break
                else:
                    ksp.destroy()
                PETSc.Mat.destroy(K)

            self._du.x.petsc_vec.ghostUpdate(
                addv=PETSc.InsertMode.INSERT,  # type: ignore
                mode=PETSc.ScatterMode.FORWARD,  # type: ignore
            )
            if self.mpc is not None:
                self.mpc.backsubstitution(self._du.x.petsc_vec)
            if not self._line_search:
                self._u.x.petsc_vec.axpy(1.0, self._du.x.petsc_vec)
                self._u.x.scatter_forward()
            else:
                # Backtracking line search: accept u + α·du only if the
                # residual norm decreases. A full Newton step from a state
                # kicked off a bifurcation saddle overshoots on the shallow
                # post-buckling landscape and diverges geometrically; damping
                # to a descent step keeps the walk inside the basin of the
                # buckled root. If even α_min gives no decrease, the α_min
                # step is taken anyway — max_iter/div_rel_tol then govern.
                alpha, applied, alpha_min = 1.0, 0.0, 1.0 / 256.0
                while True:
                    self._u.x.petsc_vec.axpy(alpha - applied,
                                             self._du.x.petsc_vec)
                    self._u.x.scatter_forward()
                    applied = alpha
                    trial_norm = self._residual_norm()
                    if trial_norm < abs_b_norm or alpha <= alpha_min:
                        logger.debug("  line search: alpha=%.4f  |R| %.3e -> %.3e",
                                     alpha, abs_b_norm, trial_norm)
                        break
                    alpha *= 0.5
            iter_newton += 1

            PETSc.Vec.destroy(residual)
            PETSc.Mat.destroy(K)

        if is_converged:
            logger.info("Newton converged in %d iteration(s) [%s]",
                        iter_newton, convergence_reason)
        return is_converged, iter_newton

    def _residual_norm(self) -> float:
        """Assemble the residual at the current state and return the norm used
        for convergence (projected onto the constraint manifold when integral
        constraints are present) — mirrors the assembly at the top of solve().
        """
        if self.mpc is not None:
            r = dolfinx_mpc.assemble_vector(self._R_form, self.mpc)
            dolfinx_mpc.apply_lifting(r, [self._J_form], [self._bcs], self.mpc,
                                      x0=[self._u.x.petsc_vec], scale=-1.0)
        else:
            r = fem_petsc.assemble_vector(self._R_form)
            fem_petsc.apply_lifting(r, [self._J_form], [self._bcs],
                                    x0=[self._u.x.petsc_vec], alpha=-1.0)
        r.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem_petsc.set_bc(r, self._bcs, x0=self._u.x.petsc_vec, alpha=-1.0)
        if self._constraint_vecs:
            _, apply_P = self._make_projector()
            apply_P(r)
        norm = r.norm()
        PETSc.Vec.destroy(r)
        return norm

    def assemble_stiffness(self) -> PETSc.Mat:
        """Assemble and return tangent stiffness K. Caller is responsible for destroying it."""
        if self.mpc is not None:
            K = dolfinx_mpc.assemble_matrix(self._J_form, self.mpc, bcs=self._bcs)
        else:
            K = fem_petsc.assemble_matrix(self._J_form, bcs=self._bcs)
        K.assemble()
        return K

# ---------------------------------------------------------------------------
# Arc-length solvers
# ---------------------------------------------------------------------------

class ArcLengthSolver(ABC):
    """Abstract base for arc-length (path-following) solvers.

    The key design principle: K is never augmented with extra rows.
    Instead the corrector solves TWO systems against the same symmetric K
    and computes δλ analytically from the constraint — preserving the SPD
    structure so CG+GAMG remains valid throughout.
    """

    arc_length: float = 0.01
    max_arc_steps: int = 100
    max_newton_iter: int = 10
    abs_tol: float = 1e-6

    @abstractmethod
    def constraint(self, du_vec: PETSc.Vec, dlambda: float, ds: float) -> float:
        """Scalar arc-length constraint g(ΔU, Δλ, Δs)."""
        ...


class CylindricalArcLength(ArcLengthSolver):
    """Crisfield cylindrical arc-length (path-following) solver.

    Constraint:  g = ‖ΔU‖² + Δλ² − Δs² = 0

    Partitioned corrector (no extra row in K):
        K δu_I  = −R           (standard Newton correction)
        K δu_II = f_ref        (load-direction correction)
        δλ solved analytically from linearised constraint
        δu = δu_I + δλ·δu_II

    K retains its original symmetry at every corrector iteration, so
    CG+GAMG can be reused without modification.

    load_fn(lambda_val) must update all load-controlling fem.Constants
    before the residual is assembled for that load level.
    """

    def __init__(self, *, arc_length: float = 0.01, max_arc_steps: int = 100,
                 max_newton_iter: int = 10, abs_tol: float = 1e-6):
        self.arc_length = arc_length
        self.max_arc_steps = max_arc_steps
        self.max_newton_iter = max_newton_iter
        self.abs_tol = abs_tol

    # ------------------------------------------------------------------
    # Constraint
    # ------------------------------------------------------------------

    def constraint(self, du_vec: PETSc.Vec, dlambda: float, ds: float) -> float:
        """g = ‖ΔU‖² + Δλ² − Δs²"""
        return float(du_vec.dot(du_vec)) + dlambda ** 2 - ds ** 2

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assemble_residual(self, newton: NewtonSolver) -> PETSc.Vec:
        """Assemble R with lifting and BC enforcement at newton._u."""
        r = fem_petsc.assemble_vector(newton._R_form)
        fem_petsc.apply_lifting(r, [newton._J_form], [newton._bcs],
                                x0=[newton._u.x.petsc_vec], alpha=-1.0)
        r.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem_petsc.set_bc(r, newton._bcs, x0=newton._u.x.petsc_vec, alpha=-1.0)
        return r

    def _build_ksp(self, K: PETSc.Mat, comm) -> PETSc.KSP:
        """Set up a KSP with CG+GAMG for the given K.

        Caller must destroy the returned KSP.
        """
        ksp = PETSc.KSP().create(comm)
        ksp.setOperators(K)
        ksp.setType(PETSc.KSP.Type.CG)
        ksp.getPC().setType(PETSc.PC.Type.GAMG)
        ksp.setUp()
        return ksp

    def _ksp_solve_one(self, ksp: PETSc.KSP, rhs: PETSc.Vec,
                       label: str = "") -> PETSc.Vec:
        """Solve ksp·x = rhs; returns a new Vec.

        label identifies the call site in log messages (e.g. "predictor δu_T").
        If CG diverges (K indefinite) the KSP is switched in place to
        MINRES — subsequent calls on the same object use MINRES directly
        without an extra warning.
        """
        x = rhs.duplicate()
        ksp.solve(rhs, x)
        if ksp.getConvergedReason() < 0:
            tag = f" [{label}]" if label else ""
            logger.warning("CG did not converge (K indefinite)%s — switching to MINRES", tag)
            ksp.setType(PETSc.KSP.Type.MINRES)
            ksp.setUp()
            ksp.solve(rhs, x)
        return x

    def _load_direction(self, newton: NewtonSolver,
                        load_fn: Callable[[float], None],
                        lam: float) -> PETSc.Vec:
        """Reference load vector f_ref = −∂R/∂λ via central finite difference.

        Evaluated at the current newton._u state.
        Returns a new Vec; caller must destroy it.
        Central differences give O(ε²) accuracy at cost of two residual assembles.
        """
        eps = 1e-6
        load_fn(lam + eps)
        r_fwd = self._assemble_residual(newton)
        load_fn(lam - eps)
        r_bwd = self._assemble_residual(newton)
        load_fn(lam)                      # restore

        # f_ref = −dR/dλ ≈ −(R(λ+ε) − R(λ−ε)) / 2ε
        r_fwd.axpy(-1.0, r_bwd)          # r_fwd = R(λ+ε) − R(λ−ε)
        r_fwd.scale(-1.0 / (2.0 * eps))  # f_ref
        PETSc.Vec.destroy(r_bwd)
        return r_fwd

    # ------------------------------------------------------------------
    # Predictor
    # ------------------------------------------------------------------

    def predictor(self, newton: NewtonSolver,
                  load_fn: Callable[[float], None],
                  lam: float,
                  du_prev: "PETSc.Vec | None",
                  dlambda_prev: float) -> "tuple[PETSc.Vec, float, PETSc.Vec]":
        """Tangent predictor step.

        Solves K·δu_T = f_ref with the current (symmetric) K.
        Scales (δu_T, 1) to lie exactly on the arc of radius Δs.
        Sign is chosen to continue in the same direction as the previous step.

        Returns (du_pred, dlambda_pred, f_ref).
        du_pred and f_ref are new Vecs — caller must destroy them.
        """
        ds = self.arc_length
        f_ref = self._load_direction(newton, load_fn, lam)

        K = newton.assemble_stiffness()
        ksp = self._build_ksp(K, newton._comm)
        du_T = self._ksp_solve_one(ksp, f_ref, label="predictor δu_T")
        ksp.destroy()
        PETSc.Mat.destroy(K)

        # ‖(δu_T, 1)‖² = δu_T·δu_T + 1  →  |Δλ| = Δs/‖(δu_T,1)‖
        beta = float(du_T.dot(du_T)) + 1.0
        dlambda_scale = ds / np.sqrt(beta)

        # Sign: use augmented inner product with previous converged increment
        # so that both displacement and load components agree in direction
        if du_prev is not None:
            dot = float(du_T.dot(du_prev)) + dlambda_scale * dlambda_prev
            dlambda_pred = dlambda_scale if dot >= 0.0 else -dlambda_scale
        else:
            dlambda_pred = dlambda_scale

        du_pred = du_T.copy()
        du_pred.scale(dlambda_pred)
        PETSc.Vec.destroy(du_T)

        return du_pred, dlambda_pred, f_ref

    # ------------------------------------------------------------------
    # Corrector
    # ------------------------------------------------------------------

    def corrector(self, newton: NewtonSolver,
                  load_fn: Callable[[float], None],
                  lam_0: float,
                  u_base: np.ndarray,
                  du_total: PETSc.Vec,
                  dlambda_total: float,
                  f_ref: PETSc.Vec,
                  ds: float) -> "tuple[bool, int, float]":
        """Partitioned augmented Newton corrector (no extra row in K).

        At each iteration:
          1. Set trial state: u ← u_base + ΔU,  λ ← λ₀ + Δλ
          2. Assemble R at this state
          3. Solve K δu_I  = −R          with the current symmetric K
          4. Solve K δu_II = f_ref       with the same factorised K
          5. Compute δλ from linearised cylindrical constraint (analytical)
          6. Update:  ΔU ← ΔU + δu_I + δλ·δu_II,   Δλ ← Δλ + δλ

        K is assembled and destroyed each iteration; it is never modified.
        Returns (converged, n_iters, dlambda_total_final).
        du_total is modified in place.
        """
        u = newton._u
        comm = newton._comm

        for i in range(self.max_newton_iter):
            # --- set trial displacement and load ---
            u.x.array[:] = u_base
            u.x.petsc_vec.axpy(1.0, du_total)
            u.x.scatter_forward()
            load_fn(lam_0 + dlambda_total)

            residual = self._assemble_residual(newton)
            abs_r = float(residual.norm())
            logger.info("    corrector iter %d  |R|=%.3e  Δλ=%.6f",
                        i, abs_r, dlambda_total)

            if abs_r < self.abs_tol:
                PETSc.Vec.destroy(residual)
                return True, i, dlambda_total

            # --- assemble K once; build KSP once; solve twice with the
            #     same preconditioner (one GAMG setup, two solve calls) ---
            K = newton.assemble_stiffness()
            ksp = self._build_ksp(K, comm)

            # Step 3:  K δu_I = −R  (standard Newton direction)
            neg_r = residual.copy()
            neg_r.scale(-1.0)
            du_I = self._ksp_solve_one(ksp, neg_r,
                                       label=f"corrector iter {i} δu_I")
            PETSc.Vec.destroy(neg_r)
            PETSc.Vec.destroy(residual)

            # Step 4:  K δu_II = f_ref  (load direction, same KSP/preconditioner)
            # If CG failed above, ksp is already MINRES — no extra warning.
            du_II = self._ksp_solve_one(ksp, f_ref,
                                        label=f"corrector iter {i} δu_II")
            ksp.destroy()
            PETSc.Mat.destroy(K)

            # Step 5:  linearise g = ‖ΔU‖² + Δλ² − Δs² = 0
            #   2 ΔU·(δu_I + δλ δu_II) + 2 Δλ δλ = −g
            #   δλ = (−g − 2 ΔU·δu_I) / (2(ΔU·δu_II + Δλ))
            g = self.constraint(du_total, dlambda_total, ds)
            numer = -g - 2.0 * float(du_total.dot(du_I))
            denom = 2.0 * (float(du_total.dot(du_II)) + dlambda_total)

            if abs(denom) < 1e-20:
                logger.warning("Arc corrector: denominator ≈ 0 at iter %d", i)
                PETSc.Vec.destroy(du_I)
                PETSc.Vec.destroy(du_II)
                return False, i + 1, dlambda_total

            d_dlambda = numer / denom

            # Step 6:  update increments
            du_total.axpy(1.0, du_I)
            du_total.axpy(d_dlambda, du_II)
            dlambda_total += d_dlambda

            PETSc.Vec.destroy(du_I)
            PETSc.Vec.destroy(du_II)

        return False, self.max_newton_iter, dlambda_total
