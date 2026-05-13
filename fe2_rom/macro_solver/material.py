"""RVE-backed constitutive law for dolfinx_materials.

The macroscopic problem is assembled with dolfinx_materials' usual machinery
(:class:`QuadratureMap` + :class:`NonlinearMaterialProblem`).  The only piece
that changes vs. a closed-form material is the constitutive law: at each macro
quadrature point we run an RVE solve and read back its volume-averaged first
Piola-Kirchhoff stress and tangent moduli.

The class is RVE-agnostic — both
:class:`fe2_rom.hyperelastic_solver.PeriodicHyperelasticHomogenizationSolver`
and :class:`fe2_rom.rve_rom.RVESolver` work, provided they are constructed with
``average_fields=["P", "A"]`` so the trailing entry of their per-call output is
``[Pbar, Abar]``.

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

logger = logging.getLogger(__name__)


# 9-vector ordering used by dolfinx_materials.utils.nonsymmetric_tensor_to_vector
# for a 3×3 nonsymmetric tensor.  Index k of the 9-vector corresponds to entry
# (i, j) of the tensor.
_F9_ORDER: tuple[tuple[int, int], ...] = (
    (0, 0), (1, 1), (2, 2),
    (0, 1), (1, 0),
    (0, 2), (2, 0),
    (1, 2), (2, 1),
)


def _vec9_to_tensor3(v: np.ndarray) -> np.ndarray:
    T = np.zeros((3, 3), dtype=float)
    for k, (i, j) in enumerate(_F9_ORDER):
        T[i, j] = v[k]
    return T


def _tensor3_to_vec9(T: np.ndarray) -> np.ndarray:
    return np.array([T[i, j] for (i, j) in _F9_ORDER], dtype=float)


def _tangent4_to_mat99(A: np.ndarray) -> np.ndarray:
    M = np.empty((9, 9), dtype=float)
    for p, (i, j) in enumerate(_F9_ORDER):
        for q, (k, l) in enumerate(_F9_ORDER):
            M[p, q] = A[i, j, k, l]
    return M


class RVEMaterial(Material):
    """dolfinx_materials Material whose response comes from a nested RVE.

    Parameters
    ----------
    rve_factory
        Zero-argument callable returning a fresh RVE solver.  The RVE must
        (a) expose ``F_bar.value`` of shape ``(3, 3)`` and (b) be callable as
        ``out = rve(Fbar)`` advancing its internal state to ``Fbar`` and
        returning a list with one entry per accepted load step; the final
        entry is ``[Pbar, Abar]`` (3×3 stress, 3×3×3×3 tangent).  Configure
        the RVE with ``average_fields=["P", "A"]``.
    """

    def __init__(self, rve_factory: Callable[[], object]):
        super().__init__()
        self._rve_factory = rve_factory
        self._rves: list | None = None
        self._n_qp: int | None = None

    # ------------------------------------------------------------------
    # dolfinx_materials Material interface
    # ------------------------------------------------------------------

    @property
    def gradients(self):
        return {"F": 9}

    @property
    def fluxes(self):
        return {"PK1": 9}

    def _ensure_rves(self, n_qp: int) -> None:
        if self._rves is None:
            logger.info("RVEMaterial: instantiating %d RVE(s) (one per macro qp)", n_qp)
            self._rves = [self._rve_factory() for _ in range(n_qp)]
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
        gradients : (n_qp, 9) array
            Macroscopic deformation gradient at each qp in the
            nonsymmetric 9-vector convention used by dolfinx_materials.
        dt : float
            Unused (RVE solvers do their own substepping).

        Returns
        -------
        fluxes, isvs, tangent
            ``fluxes`` is ``(n_qp, 9)`` PK1; ``tangent`` is ``(n_qp, 9, 9)``
            material tangent dP/dF; ``isvs`` is ``(n_qp, 0)``.
        """
        n_qp = gradients.shape[0]
        self._ensure_rves(n_qp)

        P_flat = np.zeros((n_qp, 9), dtype=float)
        A_flat = np.zeros((n_qp, 9, 9), dtype=float)

        for i, rve in enumerate(self._rves):
            F_qp = _vec9_to_tensor3(gradients[i])
            try:
                out = rve(F_qp)
            except Exception:
                logger.exception(
                    "RVE %d failed for F =\n%s\n(previous F_bar =\n%s)",
                    i, F_qp, getattr(rve, "F_bar").value,
                )
                raise
            P_qp, A_qp = out[-1][0], out[-1][1]
            P_flat[i] = _tensor3_to_vec9(P_qp)
            A_flat[i] = _tangent4_to_mat99(A_qp)

        self.data_manager.s1.set_item({"PK1": P_flat})

        return (
            self.data_manager.s1.fluxes,
            self.data_manager.s1.internal_state_variables,
            A_flat,
        )

    def constitutive_update(self, F_flat, state, dt):
        # Required by the Material API but unused — we override integrate().
        pass
