"""Newton-Raphson and arc-length solvers share this module because arc-length
is an augmented Newton method — the corrector loop reuses the same PETSc
assembly machinery and only adds an arc-length constraint equation."""

import logging
from abc import ABC, abstractmethod

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
                 rel_tol=1e-8, abs_tol=1e-6, max_iter=10, max_iter_instab=30):
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
                logger.warning("CG did not converge (reason %d) — switching to MINRES", reason)
                self._solver_type = PETSc.KSP.Type.MINRES
                self.effective_max_iter = self._max_iter_instab
                ksp.destroy()
                PETSc.Vec.destroy(residual)
                PETSc.Mat.destroy(K)
                continue
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
    """Abstract arc-length (continuation) solver.

    Arc-length is an augmented Newton method: the system [R; g] = 0 is solved,
    where g is the arc-length constraint that closes the additional unknown Δλ
    (load factor increment). Constraint, predictor, and corrector are separate
    abstract methods so cylindrical vs. Riks variants differ only in those bodies.

    The corrector receives a NewtonSolver so it can call assemble_stiffness()
    and reuse assembled forms without duplicating PETSc plumbing.
    """

    arc_length: float = 0.01
    max_arc_steps: int = 100
    max_newton_iter: int = 10
    rel_tol: float = 1e-8

    @abstractmethod
    def constraint(self, du_vec: PETSc.Vec, dlambda: float, ds: float) -> float:
        """Scalar arc-length constraint residual g(ΔU, Δλ, Δs).

        Cylindrical:  ||ΔU||² + Δλ² − Δs²
        Riks (normalised): ΔU·ΔU_prev/||ΔU_prev|| + Δλ − Δs
        """
        ...

    @abstractmethod
    def predictor(self, newton: "NewtonSolver",
                  lambda_current: float) -> tuple[PETSc.Vec, float]:
        """Tangent predictor step.

        Receives the assembled NewtonSolver (so K can be obtained via
        newton.assemble_stiffness()) and the current load factor.
        Returns (du_pred, dlambda_pred).
        """
        ...

    @abstractmethod
    def corrector(self, newton: "NewtonSolver",
                  du_pred: PETSc.Vec, dlambda_pred: float,
                  ds: float) -> tuple[bool, int]:
        """Augmented Newton corrector.

        Updates newton._u in-place via petsc_vec.axpy().
        Returns (converged, n_iters).
        """
        ...


class CylindricalArcLength(ArcLengthSolver):
    """Crisfield cylindrical arc-length.

    Constraint: ||ΔU||² + Δλ² = Δs²

    All methods raise NotImplementedError — replace the bodies to activate.
    TODO: implement predictor, constraint, and corrector.
    """

    def __init__(self, arc_length: float = 0.01, **kwargs):
        self.arc_length = arc_length
        for k, v in kwargs.items():
            setattr(self, k, v)

    def constraint(self, du_vec, dlambda, ds):
        raise NotImplementedError("CylindricalArcLength.constraint not yet implemented")

    def predictor(self, newton, lambda_current):
        raise NotImplementedError("CylindricalArcLength.predictor not yet implemented")

    def corrector(self, newton, du_pred, dlambda_pred, ds):
        raise NotImplementedError("CylindricalArcLength.corrector not yet implemented")
