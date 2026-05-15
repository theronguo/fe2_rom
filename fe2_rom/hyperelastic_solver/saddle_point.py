"""Schur-complement Newton solver for periodic homogenization with integral
linear constraints on the fluctuation w.

At each Newton step the augmented (saddle-point) system

    [ K  Cᵀ ] [Δw]   [ -R ]
    [ C  0  ] [ λ ] = [ -G ]

is solved by block elimination of the multipliers:

    1. X       = K⁻¹(-R)                   (1 KSP solve)
    2. Y_α     = K⁻¹(c_αᵀ)  for α = 1..m   (m KSP solves, same KSP)
    3. S_pos   = C @ Y                     (m × m, dense, replicated)
    4. λ       = S_pos⁻¹ (C @ X + G)
    5. Δw      = X − Σ_α λ_α Y_α

The (m+1) inner K-solves share one KSP, so the CG+GAMG (or MINRES+GAMG)
preconditioner is set up once. The m × m dense system is solved redundantly
on each rank (m is small).

Because corner-pinning has been removed in favour of integral constraints,
K has a non-trivial null space — the constant translations of the
displacement field. We attach this null space to K so the inner KSP
(``PETSc CG / MINRES + GAMG``) handles consistent right-hand sides
correctly. The constraints in C fix the gauge in the augmented system.
"""

from __future__ import annotations

import logging

import dolfinx_mpc
import numpy as np
import scipy.linalg
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
from petsc4py import PETSc

from .solvers import NewtonSolver

logger = logging.getLogger(__name__)


class SaddlePointNewtonSolver(NewtonSolver):
    """Newton solver for the constrained homogenization problem."""

    def __init__(self, comm, R_form, J_form, u, du, mpc,
                 constraint_forms: list, constraint_rhs: np.ndarray,
                 *, bcs=None, **newton_kwargs):
        bcs = list(bcs) if bcs is not None else []
        super().__init__(comm, R_form, J_form, u, du, bcs, mpc=mpc, **newton_kwargs)
        self._constraint_forms = list(constraint_forms)
        self._constraint_rhs = np.asarray(constraint_rhs, dtype=float).copy()
        self._m = len(self._constraint_forms)
        if self._constraint_rhs.shape != (self._m,):
            raise ValueError(
                "constraint_rhs shape must match the number of constraint forms "
                f"(got {self._constraint_rhs.shape} for m={self._m})."
            )
        self._c_rows = self._assemble_constraint_rows()
        self._nullspace = self._build_null_space()

    def _assemble_constraint_rows(self) -> list[PETSc.Vec]:
        rows = []
        for form in self._constraint_forms:
            if self.mpc is not None:
                v = dolfinx_mpc.assemble_vector(form, self.mpc)
            else:
                v = fem_petsc.assemble_vector(form)
            v.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            rows.append(v)
        return rows

    def _build_null_space(self) -> PETSc.NullSpace:
        V = self._u.function_space
        bs = V.dofmap.index_map_bs
        vectors = []
        for d in range(bs):
            f = fem.Function(V)
            f.x.array[d::bs] = 1.0
            f.x.scatter_forward()
            vec = f.x.petsc_vec.copy()
            vec.normalize()
            vectors.append(vec)
        return PETSc.NullSpace().create(constant=False, vectors=vectors, comm=self._comm)

    def _regularize_for_inner_solve(self, K: PETSc.Mat) -> None:
        """Make K usable by the inner Krylov solver (CG/MINRES + GAMG).

        Without corner-pinning K has a ``gdim``-dim null space of constant
        translations. CG/MINRES + GAMG do not deal with this robustly
        (``KSPSetNullSpace`` projects the RHS but GAMG's coarsening flags an
        indefinite preconditioner). A small diagonal shift makes K SPD; the
        integral constraints in C exactly fix the gauge in the augmented
        system, so the introduced error scales linearly with the shift and
        is well below the Newton tolerance.

        Translation modes are also passed as a near-null-space hint so GAMG
        coarsens correctly for elasticity-shaped problems.

        Must NOT be called on the K returned from ``assemble_stiffness``,
        which is consumed by the stability analyzer and must reflect the
        unperturbed tangent.
        """
        try:
            kappa = float(K.norm(PETSc.NormType.FROBENIUS))
        except Exception:
            kappa = 1.0
        eps = 1e-10 * max(kappa, 1.0)
        K.shift(eps)
        K.setNearNullSpace(self._nullspace)

    def _constraint_residual(self) -> np.ndarray:
        """G = C w - rhs (length m)."""
        if self._m == 0:
            return np.zeros(0)
        u_vec = self._u.x.petsc_vec
        Cw = np.array([float(c.dot(u_vec)) for c in self._c_rows])
        return Cw - self._constraint_rhs

    def _schur_block_solve(self, ksp: PETSc.KSP,
                           primary_rhs: PETSc.Vec,
                           constraint_rhs: np.ndarray) -> PETSc.Vec | None:
        """Solve [K Cᵀ; C 0] [x; λ] = [primary_rhs; constraint_rhs] and return x.

        Returns ``None`` if any inner KSP solve fails to converge or the dense
        Schur system is singular — caller should treat as a failed step.
        """
        X = self._du.x.petsc_vec.duplicate()
        ksp.solve(primary_rhs, X)
        if ksp.getConvergedReason() < 0:
            PETSc.Vec.destroy(X)
            return None

        if self._m == 0:
            return X

        Y: list[PETSc.Vec] = []
        for c in self._c_rows:
            y = self._du.x.petsc_vec.duplicate()
            ksp.solve(c, y)
            if ksp.getConvergedReason() < 0:
                PETSc.Vec.destroy(y)
                for prev in Y:
                    PETSc.Vec.destroy(prev)
                PETSc.Vec.destroy(X)
                return None
            Y.append(y)

        m = self._m
        S_pos = np.empty((m, m))
        b = np.empty(m)
        for alpha in range(m):
            b[alpha] = float(self._c_rows[alpha].dot(X)) - constraint_rhs[alpha]
            for beta in range(m):
                S_pos[alpha, beta] = float(self._c_rows[alpha].dot(Y[beta]))

        try:
            lam = scipy.linalg.solve(S_pos, b, assume_a="sym")
        except (np.linalg.LinAlgError, scipy.linalg.LinAlgError):
            for y in Y:
                PETSc.Vec.destroy(y)
            PETSc.Vec.destroy(X)
            return None

        for alpha in range(m):
            X.axpy(-float(lam[alpha]), Y[alpha])
            PETSc.Vec.destroy(Y[alpha])
        return X

    def solve(self, iter_start: int = 0) -> tuple[bool, int]:
        """Run Newton iterations with Schur-complement constraint elimination.

        Convergence: combined norm sqrt(|R|² + |G|²) compared against tolerances.
        """
        iter_newton = iter_start
        is_converged = False
        convergence_reason = ""

        while iter_newton < self.effective_max_iter:
            is_converged = False

            if self.mpc is not None:
                residual = dolfinx_mpc.assemble_vector(self._R_form, self.mpc)
                if self._bcs:
                    dolfinx_mpc.apply_lifting(
                        residual, [self._J_form], [self._bcs], self.mpc,
                        x0=[self._u.x.petsc_vec], scale=-1.0,
                    )
            else:
                residual = fem_petsc.assemble_vector(self._R_form)
                if self._bcs:
                    fem_petsc.apply_lifting(
                        residual, [self._J_form], [self._bcs],
                        x0=[self._u.x.petsc_vec], alpha=-1.0,
                    )
            residual.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            if self._bcs:
                fem_petsc.set_bc(residual, self._bcs, x0=self._u.x.petsc_vec, alpha=-1.0)

            abs_R = float(residual.norm())
            G = self._constraint_residual()
            abs_G = float(np.linalg.norm(G))
            total = float(np.sqrt(abs_R * abs_R + abs_G * abs_G))

            if iter_newton == 0:
                self._abs_b_norm_init = total if total > 0 else 1.0

            rel = total / self._abs_b_norm_init
            logger.debug("  iter %2d  rel=%.3e  |R|=%.3e  |G|=%.3e",
                         iter_newton, rel, abs_R, abs_G)

            if rel < self._rel_tol or total < self._abs_tol:
                convergence_reason = (
                    f"rel={rel:.3e} < tol={self._rel_tol:.3e}"
                    if rel < self._rel_tol
                    else f"abs={total:.3e} < tol={self._abs_tol:.3e}"
                )
                PETSc.Vec.destroy(residual)
                is_converged = True
                break

            if rel > self._div_rel_tol or np.isnan(total):
                PETSc.Vec.destroy(residual)
                break

            if self.mpc is not None:
                K = dolfinx_mpc.assemble_matrix(self._J_form, self.mpc, bcs=self._bcs)
            else:
                K = fem_petsc.assemble_matrix(self._J_form, bcs=self._bcs)
            K.assemble()
            self._regularize_for_inner_solve(K)
            ksp = self._make_ksp(K)

            primary_rhs = residual.copy()
            primary_rhs.scale(-1.0)
            delta = self._schur_block_solve(ksp, primary_rhs, -G)
            PETSc.Vec.destroy(primary_rhs)

            if delta is None:
                reason = ksp.getConvergedReason()
                if not self._using_minres_fallback and self._switch_to_minres:
                    logger.warning(
                        "Primary solver did not converge (reason %d) — switching to MINRES+GAMG",
                        reason,
                    )
                    self._using_minres_fallback = True
                    self.effective_max_iter = self._max_iter_instab
                    ksp.destroy()
                    PETSc.Vec.destroy(residual)
                    PETSc.Mat.destroy(K)
                    continue
                if not self._using_minres_fallback:
                    logger.warning("Primary solver did not converge (reason %d)", reason)
                    ksp.destroy()
                    PETSc.Vec.destroy(residual)
                    PETSc.Mat.destroy(K)
                    return is_converged, iter_newton
                logger.warning("MINRES did not converge (reason %d) — reducing time step", reason)
                ksp.destroy()
                PETSc.Vec.destroy(residual)
                PETSc.Mat.destroy(K)
                break

            ksp.destroy()

            # Copy delta into self._du, backsubstitute MPC, apply increment to u.
            delta.copy(self._du.x.petsc_vec)
            PETSc.Vec.destroy(delta)
            self._du.x.petsc_vec.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD,
            )
            if self.mpc is not None:
                self.mpc.backsubstitution(self._du.x.petsc_vec)
            self._u.x.petsc_vec.axpy(1.0, self._du.x.petsc_vec)
            self._u.x.scatter_forward()
            iter_newton += 1

            PETSc.Vec.destroy(residual)
            PETSc.Mat.destroy(K)

        if is_converged:
            logger.info("Newton converged in %d iteration(s) [%s]", iter_newton, convergence_reason)
        return is_converged, iter_newton

    def assemble_stiffness(self) -> PETSc.Mat:
        """Assemble K for stability analysis. Caller destroys K.

        Returns the unperturbed K (no shift, no near-null-space hint) so that
        the stability eigenvalues reflect the true tangent operator and the
        ``gdim`` translation modes show up as exact zeros.
        """
        return super().assemble_stiffness()

    def solve_macro_sensitivities(
        self,
        rhs_forms_dict: dict[str, list],
    ) -> dict[str, list[fem.Function]]:
        """For each macro variable in ``rhs_forms_dict``, solve the constrained
        sensitivity equation
            [ K  Cᵀ ] [ p ]   [ rhs_k ]
            [ C  0  ] [ ν ] = [ 0     ]
        for each scalar component k, returning the forward sensitivity p = ∂w/∂μ_k.

        The K-factorisation and constraint Y vectors are shared across all
        macro variables; each component costs one extra K-solve + a small
        dense Schur solve.
        """
        if self.mpc is not None:
            K = dolfinx_mpc.assemble_matrix(self._J_form, self.mpc, bcs=self._bcs)
        else:
            K = fem_petsc.assemble_matrix(self._J_form, bcs=self._bcs)
        K.assemble()
        self._regularize_for_inner_solve(K)
        ksp = self._make_ksp(K)

        Y: list[PETSc.Vec] = []
        for c in self._c_rows:
            y = self._du.x.petsc_vec.duplicate()
            ksp.solve(c, y)
            Y.append(y)

        m = self._m
        S_lu = None
        if m > 0:
            S_pos = np.empty((m, m))
            for alpha in range(m):
                for beta in range(m):
                    S_pos[alpha, beta] = float(self._c_rows[alpha].dot(Y[beta]))
            S_lu = scipy.linalg.lu_factor(S_pos)

        results: dict[str, list[fem.Function]] = {}
        for name, forms in rhs_forms_dict.items():
            sensitivities: list[fem.Function] = []
            for rhs_form in forms:
                if self.mpc is not None:
                    rhs = dolfinx_mpc.assemble_vector(rhs_form, self.mpc)
                else:
                    rhs = fem_petsc.assemble_vector(rhs_form)
                rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

                X = self._du.x.petsc_vec.duplicate()
                ksp.solve(rhs, X)

                if m > 0:
                    b = np.empty(m)
                    for alpha in range(m):
                        b[alpha] = float(self._c_rows[alpha].dot(X))
                    lam = scipy.linalg.lu_solve(S_lu, b)
                    for alpha in range(m):
                        X.axpy(-float(lam[alpha]), Y[alpha])

                X.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                if self.mpc is not None:
                    self.mpc.backsubstitution(X)

                p = fem.Function(self._u.function_space)
                # PETSc Vec.copy is owned-only — dolfinx Function arrays include
                # ghost DOFs, so we copy through PETSc and then scatter_forward
                # to populate the ghost layer.
                X.copy(p.x.petsc_vec)
                p.x.scatter_forward()
                sensitivities.append(p)

                PETSc.Vec.destroy(rhs)
                PETSc.Vec.destroy(X)
            results[name] = sensitivities

        for y in Y:
            PETSc.Vec.destroy(y)
        ksp.destroy()
        PETSc.Mat.destroy(K)
        return results

    def __del__(self):
        for v in getattr(self, "_c_rows", []):
            try:
                PETSc.Vec.destroy(v)
            except Exception:
                pass
