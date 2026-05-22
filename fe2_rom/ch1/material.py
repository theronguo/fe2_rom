"""RVE-backed constitutive law for dolfinx_materials.

The macroscopic problem is assembled with dolfinx_materials' usual machinery
(:class:`QuadratureMap` + :class:`NonlinearMaterialProblem`).  The only piece
that changes vs. a closed-form material is the constitutive law: at each macro
quadrature point we run an RVE solve and read back its volume-averaged first
Piola-Kirchhoff stress and tangent moduli.

The class is RVE-agnostic — both
:class:`fe2_rom.hyperelastic_solver.MicroSolver`
and :class:`fe2_rom.rom.ReducedMicroSolver` work, provided their per-call dict
output exposes ``Pbar`` (3×3) and ``dPbar_dFbar`` (3×3×3×3).

Each macro qp owns its own RVE instance.  The RVEs are created lazily on the
first ``integrate()`` call (when the QuadratureMap reports its quadrature-point
count) and kept across macro Newton iterations and load steps — every RVE
warm-starts from its own previous converged state.  Run the RVE on
``MPI.COMM_SELF`` so each macro rank handles its own qp population
independently.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from dolfinx_materials.generic import Material

from fe2_rom.hyperelastic_solver.logging_utils import qp_context
from fe2_rom.ch1.exceptions import RVEConvergenceError

logger = logging.getLogger(__name__)


# MFront non-symmetric tensor → vector orderings used by
# ``dolfinx_materials.utils.nonsymmetric_tensor_to_vector``.  In 2D the third
# slot is a *dummy* plane-strain entry (set to zero here) so positions 0..4
# correspond to (T_00, T_11, _, T_01, T_10).
_F_ORDER_2D: tuple[tuple[int | None, int | None], ...] = (
    (0, 0), (1, 1), (None, None), (0, 1), (1, 0),
)
_F_ORDER_3D: tuple[tuple[int, int], ...] = (
    (0, 0), (1, 1), (2, 2),
    (0, 1), (1, 0),
    (0, 2), (2, 0),
    (1, 2), (2, 1),
)


def _f_order(gdim: int):
    if gdim == 2:
        return _F_ORDER_2D, 5
    if gdim == 3:
        return _F_ORDER_3D, 9
    raise ValueError(f"gdim must be 2 or 3, got {gdim}")


def _vec_to_tensor(v: np.ndarray, gdim: int) -> np.ndarray:
    order, _ = _f_order(gdim)
    T = np.zeros((gdim, gdim), dtype=float)
    for k, (i, j) in enumerate(order):
        if i is None:
            continue
        T[i, j] = v[k]
    return T


def _tensor_to_vec(T: np.ndarray, gdim: int) -> np.ndarray:
    order, n = _f_order(gdim)
    out = np.zeros(n, dtype=float)
    for k, (i, j) in enumerate(order):
        if i is None:
            continue
        out[k] = T[i, j]
    return out


def _tangent4_to_mat(A: np.ndarray, gdim: int) -> np.ndarray:
    order, n = _f_order(gdim)
    M = np.zeros((n, n), dtype=float)
    for p, (i, j) in enumerate(order):
        if i is None:
            continue
        for q, (k, l) in enumerate(order):
            if k is None:
                continue
            M[p, q] = A[i, j, k, l]
    return M

from mpi4py import MPI
class RVEMaterial(Material):
    """dolfinx_materials Material whose response comes from a nested RVE.

    Parameters
    ----------
    rve_factory
        Zero-argument callable returning a fresh RVE solver.  The RVE must
        (a) expose ``F_bar.value`` of shape ``(3, 3)`` and (b) be callable as
        ``out = rve(Fbar)`` advancing its internal state to ``Fbar`` and
        returning a list with one entry (a dict) per accepted load step; the
        final entry must contain keys ``"Pbar"`` (3×3 stress) and
        ``"dPbar_dFbar"`` (3×3×3×3 tangent).  Configure the RVE with
        ``average_quantities=["P", "A"]`` (or include ``EffectivePbar`` and
        ``EffectiveAbar`` instances).
    """

    def __init__(self, rve_factory: Callable[[int, int], object], *, gdim: int = 3):
        super().__init__()
        self._rve_factory = rve_factory
        self._gdim = int(gdim)
        _, self._F_dim = _f_order(self._gdim)
        self._rves: list | None = None
        self._n_qp: int | None = None
        self._rank = MPI.COMM_WORLD.Get_rank()
        self.step_failed: bool = False
        self.failure_reason: str = ""

    # ------------------------------------------------------------------
    # dolfinx_materials Material interface
    # ------------------------------------------------------------------

    @property
    def gradients(self):
        return {"F": self._F_dim}

    @property
    def fluxes(self):
        return {"PK1": self._F_dim}

    def _ensure_rves(self, n_qp: int) -> None:
        if self._rves is None:
            logger.info(
                "RVEMaterial: instantiating %d RVE(s) (one per macro qp)",
                n_qp, extra={"all_ranks": True},
            )
            self._rves = []
            for i in range(n_qp):
                with qp_context(i):
                    self._rves.append(self._rve_factory(self._rank, i))
            self._n_qp = n_qp
        elif n_qp != self._n_qp:
            raise RuntimeError(
                f"RVEMaterial: quadrature-point count changed "
                f"({self._n_qp} → {n_qp}); RVE bookkeeping would be invalidated."
            )

    def integrate(self, gradients: np.ndarray, dt: float = 0.0):
        """Run an RVE solve for every macro quadrature point.

        Parameters
        ----------
        gradients : (n_qp, F_dim) array
            Macroscopic deformation gradient at each qp in the nonsymmetric
            vector convention used by dolfinx_materials.  ``F_dim`` is 5 in
            2D (plane strain, with a dummy zz-slot) and 9 in 3D.
        dt : float
            Unused (RVE solvers do their own substepping).

        Returns
        -------
        fluxes, isvs, tangent
            ``fluxes`` is ``(n_qp, F_dim)`` PK1; ``tangent`` is
            ``(n_qp, F_dim, F_dim)`` material tangent dP/dF;
            ``isvs`` is ``(n_qp, 0)``.
        """
        n_qp = gradients.shape[0]
        self._ensure_rves(n_qp)

        F_dim = self._F_dim
        gdim = self._gdim
        P_flat = np.zeros((n_qp, F_dim), dtype=float)
        A_flat = np.zeros((n_qp, F_dim, F_dim), dtype=float)

        local_failure = 0
        failed_qp = -1
        for i, rve in enumerate(self._rves):
            F_qp = _vec_to_tensor(gradients[i], gdim)
            with qp_context(i):
                try:
                    out = rve(F_qp)
                except RVEConvergenceError as exc:
                    logger.warning(
                        "RVE did not converge: %s. Step will be rejected collectively.",
                        exc,
                    )
                    local_failure = 1
                    if failed_qp < 0:
                        failed_qp = i
                    P_flat[i] = 0.0
                    A_flat[i] = np.eye(F_dim)
                    continue
                except Exception:
                    logger.exception(
                        "RVE failed for F =\n%s\n(previous F_bar =\n%s)",
                        F_qp, getattr(rve, "F_bar").value,
                    )
                    raise
            P_qp, A_qp = out[-1]["Pbar"], out[-1]["dPbar_dFbar"]
            P_flat[i] = _tensor_to_vec(P_qp, gdim)
            A_flat[i] = _tangent4_to_mat(A_qp, gdim)

        # If any RVE failed on any rank, reject entire macro step by setting very high stress.
        any_failure = MPI.COMM_WORLD.allreduce(local_failure, op=MPI.LOR)
        if any_failure:
            self.step_failed = True
            if failed_qp >= 0:
                self.failure_reason = (
                    f"RVE solve failed (rank {self._rank} "
                    f"local_failure={bool(local_failure)}, first failed qp={failed_qp})"
                )
            else:
                self.failure_reason = ""
            
            # Setting to high value to trigger PETSc-side SNES non-convergence and step rejection.
            P_flat[:] = 999999999999999

        self.data_manager.s1.set_item({"PK1": P_flat})

        return (
            self.data_manager.s1.fluxes,
            self.data_manager.s1.internal_state_variables,
            A_flat,
        )

    def commit(self) -> None:
        """Promote every RVE's current ``F_bar`` to its converged restart state.

        Call once from the macro driver after the outer SNES converges *and*
        the macro time stepper accepts the step.  Calling this after a
        rejected step would poison each RVE's restart point.
        """
        if self._rves is None:
            return
        for rve in self._rves:
            rve.commit()

    def constitutive_update(self, F_flat, state, dt):
        # Required by the Material API but unused — we override integrate().
        pass
