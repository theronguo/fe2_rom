"""Newton-Raphson and arc-length solvers share this module because arc-length
is an augmented Newton method — the corrector loop reuses the same PETSc
assembly machinery and only adds an arc-length constraint equation."""

import logging
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
from petsc4py import PETSc

logger = logging.getLogger(__name__)


class NewtonSolver:
    """Newton-Raphson solver with CG+GAMG primary solver and MINRES fallback.

    Designed to be called multiple times per time step (once per stability retry)
    via solve(iter_start=...). The iter_start parameter allows iter_newton to
    accumulate across stability retries, matching the original solver's behaviour.

    Call reset_for_new_timestep() at the start of each time step to reset
    solver_type, effective_max_iter, and the stored initial residual norm.
    """

    def __init__(self, comm, R_form, J_form, u, du, bcs, *,
                 rel_tol=1e-8, abs_tol=1e-6, max_iter=10, max_iter_instab=30,
                 switch_to_minres=False
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
        self._switch_to_minres = switch_to_minres

        self._solver_type = PETSc.KSP.Type.CG
        self.effective_max_iter = max_iter
        self._abs_b_norm_init = 1.0

    def reset_for_new_timestep(self):
        """Reset solver type and iteration state for a fresh time step."""
        self._solver_type = PETSc.KSP.Type.CG
        self.effective_max_iter = self._max_iter
        self._abs_b_norm_init = 1.0

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

            residual = fem_petsc.assemble_vector(self._R_form)
            fem.apply_lifting(residual, [self._J_form], [self._bcs],
                              x0=[self._u.x.petsc_vec], alpha=-1.0)
            self._scatter_update(residual)
            fem.set_bc(residual, self._bcs, x0=self._u.x.petsc_vec, scale=-1.0)

            abs_b_norm = residual.norm()
            if iter_newton == 0:
                self._abs_b_norm_init = abs_b_norm

            rel = abs_b_norm / self._abs_b_norm_init
            logger.debug("  iter %2d  rel=%.3e  abs=%.3e", iter_newton, rel, abs_b_norm)

            if rel < self._rel_tol or abs_b_norm < self._abs_tol:
                if rel < self._rel_tol:
                    convergence_reason = f"rel={rel:.3e} < tol={self._rel_tol:.3e}"
                else:
                    convergence_reason = f"abs={abs_b_norm:.3e} < tol={self._abs_tol:.3e}"
                PETSc.Vec.destroy(residual)
                is_converged = True
                break

            if rel > 10 or np.isnan(abs_b_norm):
                PETSc.Vec.destroy(residual)
                break

            K = fem_petsc.assemble_matrix(self._J_form, bcs=self._bcs)
            K.assemble()
            ksp = PETSc.KSP().create(self._comm)
            ksp.setOperators(K)
            ksp.setType(self._solver_type)
            ksp.getPC().setType(PETSc.PC.Type.GAMG)
            ksp.solve(-residual, self._du.x.petsc_vec)

            reason = ksp.getConvergedReason()
            if reason < 0 and self._solver_type == PETSc.KSP.Type.CG:
                if self._switch_to_minres:
                    logger.warning("CG did not converge (reason %d) — switching to MINRES", reason)
                    self._solver_type = PETSc.KSP.Type.MINRES
                    self.effective_max_iter = self._max_iter_instab
                    ksp.destroy()
                    PETSc.Vec.destroy(residual)
                    PETSc.Mat.destroy(K)
                    continue
                else:
                    logger.warning("CG did not converge (reason %d)", reason)
                    ksp.destroy()
                    PETSc.Vec.destroy(residual)
                    PETSc.Mat.destroy(K)
                    return is_converged, iter_newton
            elif reason < 0 and self._solver_type == PETSc.KSP.Type.MINRES:
                logger.warning("MINRES did not converge (reason %d) — reducing time step", reason)
                ksp.destroy()
                PETSc.Vec.destroy(residual)
                PETSc.Mat.destroy(K)
                break
            else:
                ksp.destroy()

            self._u.x.petsc_vec.axpy(1.0, self._du.x.petsc_vec)
            self._u.x.scatter_forward()
            iter_newton += 1

            PETSc.Vec.destroy(residual)
            PETSc.Mat.destroy(K)

        if is_converged:
            logger.info("Newton converged in %d iteration(s) [%s]",
                        iter_newton, convergence_reason)
        return is_converged, iter_newton

    def assemble_stiffness(self) -> PETSc.Mat:
        """Assemble and return tangent stiffness K. Caller is responsible for destroying it."""
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
        fem.apply_lifting(r, [newton._J_form], [newton._bcs],
                          x0=[newton._u.x.petsc_vec], alpha=-1.0)
        r.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(r, newton._bcs, x0=newton._u.x.petsc_vec, scale=-1.0)
        return r

    def _build_ksp(self, K: PETSc.Mat, comm) -> PETSc.KSP:
        """Set up a KSP with CG+GAMG for the given symmetric K.

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
