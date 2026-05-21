"""Micromorphic constitutive laws for the two-scale macro driver.

Two classes are provided:

``DummyMicromorphicMaterial``
    Linear, decoupled constitutive law for verifying the mixed macro
    formulation without running any RVE:
        P = µ (F_vec - I_vec),  Π_i = α v_i,  Λ_i = β g_i

``MicromorphicRVEMaterial``
    Wraps a :class:`~fe2_rom.rve_rom.solver_micromorphic.MicromorphicRVESolver`
    (or :class:`~fe2_rom.hyperelastic_solver.solver_micromorphic.MicromorphicHyperelasticHomogenizationSolver`).
    One RVE per macro quadrature point, lazy-initialised on the first
    ``integrate()`` call.

Both classes expose the same ``gradients``, ``fluxes``, and ``tangent_blocks``
so they are drop-in substitutes for the :class:`QuadratureMap`.

Gradient / flux convention (2D, following dolfinx_materials
``nonsymmetric_tensor_to_vector`` for a 2×2 tensor):
    F_vec  = [F00, F11, 0, F01, F10]          → F_dim = 5
    v_vec  = [v_1, ..., v_N]                  → shape N
    g_vec  = [g_1[0], g_1[1], ..., g_N[1]]   → shape N*gdim (mode-major)

Tangent blocks (3 × 3 grid):
    ("P",      "F"), ("P",      "v"), ("P",      "g")
    ("Pi",     "F"), ("Pi",     "v"), ("Pi",     "g")
    ("Lambda", "F"), ("Lambda", "v"), ("Lambda", "g")
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from mpi4py import MPI

from dolfinx_materials.generic import Material

from fe2_rom.hyperelastic_solver.logging_utils import qp_context
from fe2_rom.hyperelastic_solver.exceptions import RVEConvergenceError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 2D tensor ↔ 5-vector helpers
# F5 ordering: [F00, F11, F22(=0), F01, F10]
# Index 2 is always zero (out-of-plane), column/row 2 of any tangent is zero.
# ---------------------------------------------------------------------------

_F5_ORDER_2D: tuple[tuple[int, int] | None, ...] = (
    (0, 0), (1, 1), None, (0, 1), (1, 0)
)


def _f5_to_2x2(v: np.ndarray) -> np.ndarray:
    """5-vector → 2×2 F.  Entry v[2] (F22) is ignored."""
    return np.array([[v[0], v[3]], [v[4], v[1]]], dtype=float)


def _tensor2x2_to_f5(T: np.ndarray) -> np.ndarray:
    """2×2 tensor → 5-vector.  Entry [2] (F22) set to 0."""
    return np.array([T[0, 0], T[1, 1], 0.0, T[0, 1], T[1, 0]], dtype=float)


def _tangent2x2x2x2_to_mat55(A: np.ndarray) -> np.ndarray:
    """(2,2,2,2) tangent → 5×5 matrix in F5 convention.
    Rows/cols corresponding to the out-of-plane (index 2) entry remain zero.
    """
    M = np.zeros((5, 5), dtype=float)
    for p, ij in enumerate(_F5_ORDER_2D):
        if ij is None:
            continue
        i, j = ij
        for q, kl in enumerate(_F5_ORDER_2D):
            if kl is None:
                continue
            k, l = kl
            M[p, q] = A[i, j, k, l]
    return M


def _pbar_v_tangent_to_mat5N(A: np.ndarray, N: int) -> np.ndarray:
    """(2,2,N) dPbar/dv → (5, N) matrix in F5 convention."""
    M = np.zeros((5, N), dtype=float)
    for p, ij in enumerate(_F5_ORDER_2D):
        if ij is None:
            continue
        i, j = ij
        M[p, :] = A[i, j, :]
    return M


def _pbar_g_tangent_to_mat5Ng(A: np.ndarray, N: int, gdim: int) -> np.ndarray:
    """(2,2,N,gdim) dPbar/dg → (5, N*gdim) matrix in F5 convention."""
    M = np.zeros((5, N * gdim), dtype=float)
    for p, ij in enumerate(_F5_ORDER_2D):
        if ij is None:
            continue
        i, j = ij
        for n in range(N):
            for d in range(gdim):
                M[p, n * gdim + d] = A[i, j, n, d]
    return M


def _pi_fbar_tangent_to_matN5(A: np.ndarray, N: int) -> np.ndarray:
    """(N,2,2) dPi/dFbar → (N, 5) matrix in F5 convention."""
    M = np.zeros((N, 5), dtype=float)
    for q, kl in enumerate(_F5_ORDER_2D):
        if kl is None:
            continue
        k, l = kl
        M[:, q] = A[:, k, l]
    return M


def _pi_g_tangent_to_matNNg(A: np.ndarray, N: int, gdim: int) -> np.ndarray:
    """(N,N,gdim) dPi/dg → (N, N*gdim) matrix."""
    return A.reshape(N, N * gdim)


def _lambda_fbar_tangent_to_matNg5(A: np.ndarray, N: int, gdim: int) -> np.ndarray:
    """(N,gdim,2,2) dLambda/dFbar → (N*gdim, 5) matrix in F5 convention."""
    M = np.zeros((N * gdim, 5), dtype=float)
    for n in range(N):
        for d in range(gdim):
            for q, kl in enumerate(_F5_ORDER_2D):
                if kl is None:
                    continue
                k, l = kl
                M[n * gdim + d, q] = A[n, d, k, l]
    return M


def _lambda_v_tangent_to_matNgN(A: np.ndarray, N: int, gdim: int) -> np.ndarray:
    """(N,gdim,N) dLambda/dv → (N*gdim, N) matrix."""
    return A.reshape(N * gdim, N)


def _lambda_g_tangent_to_matNgNg(A: np.ndarray, N: int, gdim: int) -> np.ndarray:
    """(N,gdim,N,gdim) dLambda/dg → (N*gdim, N*gdim) matrix."""
    return A.reshape(N * gdim, N * gdim)


# ---------------------------------------------------------------------------
# Shared tangent_blocks definition
# ---------------------------------------------------------------------------

def _build_tangent_blocks(F_dim: int, N: int, gdim: int) -> dict:
    Ng = N * gdim
    blocks = {("P", "F"): (F_dim, F_dim)}
    if N > 0:
        blocks.update({
            ("P",      "v"): (F_dim, N),
            ("P",      "g"): (F_dim, Ng),
            ("Pi",     "F"): (N,     F_dim),
            ("Pi",     "v"): (N,     N),
            ("Pi",     "g"): (N,     Ng),
            ("Lambda", "F"): (Ng,    F_dim),
            ("Lambda", "v"): (Ng,    N),
            ("Lambda", "g"): (Ng,    Ng),
        })
    return blocks


def _pack_ct(blocks: list[np.ndarray], n_qp: int) -> np.ndarray:
    """Concatenate (n_qp, m, n) blocks into (n_qp, total) Ct_vals row-major."""
    return np.concatenate([b.reshape(n_qp, -1) for b in blocks], axis=1)


# ---------------------------------------------------------------------------
# DummyMicromorphicMaterial
# ---------------------------------------------------------------------------

class DummyMicromorphicMaterial(Material):
    """Linear, decoupled micromorphic constitutive law for testing.

    Parameters
    ----------
    N_modes : int
        Number of enrichment modes (must match the macro solver).
    gdim : int
        Geometric dimension (2 or 3; currently only 2 is implemented).
    mu : float
        Elastic modulus for P = mu * (F_vec - I_vec).
    alpha : float
        Coupling coefficient: Pi_i = alpha * v_i.
    beta : float
        Gradient coupling: Lambda_i = beta * g_i.
    """

    def __init__(self, N_modes: int, gdim: int = 2, mu: float = 1.0,
                 alpha: float = 1.0, beta: float = 1.0):
        if gdim != 2:
            raise NotImplementedError("DummyMicromorphicMaterial only supports gdim=2.")
        self._N = N_modes
        self._gdim = gdim
        self._F_dim = 5  # nonsymmetric_tensor_to_vector for 2D → 5-vector
        self._mu = mu
        self._alpha = alpha
        self._beta = beta
        # I_vec: identity deformation gradient as 5-vector [1, 1, 0, 0, 0]
        self._I_vec = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
        self.step_failed: bool = False
        self.failure_reason: str = ""
        super().__init__()

    @property
    def gradients(self):
        d = {"F": self._F_dim}
        if self._N > 0:
            d["v"] = self._N
            d["g"] = self._N * self._gdim
        return d

    @property
    def fluxes(self):
        d = {"P": self._F_dim}
        if self._N > 0:
            d["Pi"] = self._N
            d["Lambda"] = self._N * self._gdim
        return d

    @property
    def tangent_blocks(self):
        return _build_tangent_blocks(self._F_dim, self._N, self._gdim)

    def integrate(self, gradients: np.ndarray, dt: float = 0.0):
        F_dim = self._F_dim
        N = self._N
        gdim = self._gdim
        Ng = N * gdim
        n_qp = gradients.shape[0]

        F_all = gradients[:, :F_dim]                     # (n_qp, 5)
        v_all = gradients[:, F_dim:F_dim + N] if N > 0 else np.zeros((n_qp, 0))
        g_all = gradients[:, F_dim + N:] if N > 0 else np.zeros((n_qp, 0))

        # Fluxes
        P_flat = self._mu * (F_all - self._I_vec)   # (n_qp, 5)
        flux_parts = [P_flat]
        if N > 0:
            Pi_flat     = self._alpha * v_all
            Lambda_flat = self._beta * g_all
            flux_parts.extend([Pi_flat, Lambda_flat])
        flux_vals = np.concatenate(flux_parts, axis=1)

        # Tangents — constant, broadcast over qps
        dP_dF = np.broadcast_to(
            self._mu * np.eye(F_dim), (n_qp, F_dim, F_dim)).copy()
        ct_blocks = [dP_dF]
        if N > 0:
            ct_blocks.extend([
                np.zeros((n_qp, F_dim, N)),
                np.zeros((n_qp, F_dim, Ng)),
                np.zeros((n_qp, N, F_dim)),
                np.broadcast_to(self._alpha * np.eye(N), (n_qp, N, N)).copy(),
                np.zeros((n_qp, N, Ng)),
                np.zeros((n_qp, Ng, F_dim)),
                np.zeros((n_qp, Ng, N)),
                np.broadcast_to(self._beta * np.eye(Ng), (n_qp, Ng, Ng)).copy(),
            ])
        Ct_vals = _pack_ct(ct_blocks, n_qp)

        # Keep data_manager in sync so qmap.advance() propagates correct values.
        self.data_manager.s1.fluxes[:] = flux_vals
        self.data_manager.s0.fluxes[:] = flux_vals

        return flux_vals, np.zeros((n_qp, 0)), Ct_vals

    def constitutive_update(self, gradients, state, dt):
        pass  # unused — we override integrate()

    def commit(self) -> None:
        pass  # stateless


# ---------------------------------------------------------------------------
# MicromorphicRVEMaterial
# ---------------------------------------------------------------------------

class MicromorphicRVEMaterial(Material):
    """dolfinx_materials Material backed by a ``MicromorphicRVESolver``.

    One RVE instance per macro quadrature point.  Created lazily on the first
    ``integrate()`` call.  Each RVE warm-starts from its own previous converged
    state across Newton iterations and load steps.

    Parameters
    ----------
    rve_factory : callable(rank, index) → solver
        Returns a fresh :class:`~fe2_rom.rve_rom.solver_micromorphic.MicromorphicRVESolver`
        (or full-order counterpart).  The solver must be callable as
        ``out = solver(Fbar, v=v_arr, g=g_arr)`` and the final step dict must
        contain keys ``Pbar``, ``Pi``, ``Lambda`` and all nine tangent blocks
        ``dPbar_dFbar``, ``dPbar_dv``, ``dPbar_dg``, ``dPi_dFbar``,
        ``dPi_dv``, ``dPi_dg``, ``dLambda_dFbar``, ``dLambda_dv``,
        ``dLambda_dg``.
    N_modes : int
        Number of enrichment modes.
    gdim : int
        Geometric dimension (currently only 2 supported).
    """

    def __init__(self, rve_factory: Callable[[int, int], object],
                 N_modes: int, gdim: int = 2):
        if gdim != 2:
            raise NotImplementedError("MicromorphicRVEMaterial only supports gdim=2.")
        self._rve_factory = rve_factory
        self._N = N_modes
        self._gdim = gdim
        self._F_dim = 5
        self._rves: list | None = None
        self._n_qp: int | None = None
        self._rank = MPI.COMM_WORLD.Get_rank()
        self.step_failed: bool = False
        self.failure_reason: str = ""
        super().__init__()

    @property
    def gradients(self):
        d = {"F": self._F_dim}
        if self._N > 0:
            d["v"] = self._N
            d["g"] = self._N * self._gdim
        return d

    @property
    def fluxes(self):
        d = {"P": self._F_dim}
        if self._N > 0:
            d["Pi"] = self._N
            d["Lambda"] = self._N * self._gdim
        return d

    @property
    def tangent_blocks(self):
        return _build_tangent_blocks(self._F_dim, self._N, self._gdim)

    def _ensure_rves(self, n_qp: int) -> None:
        if self._rves is None:
            logger.info(
                "MicromorphicRVEMaterial: instantiating %d RVE(s)", n_qp,
                extra={"all_ranks": True},
            )
            self._rves = []
            for i in range(n_qp):
                with qp_context(i):
                    self._rves.append(self._rve_factory(self._rank, i))
            self._n_qp = n_qp
        elif n_qp != self._n_qp:
            raise RuntimeError(
                f"MicromorphicRVEMaterial: qp count changed "
                f"({self._n_qp} → {n_qp})."
            )

    def integrate(self, gradients: np.ndarray, dt: float = 0.0):
        F_dim = self._F_dim
        N = self._N
        gdim = self._gdim
        Ng = N * gdim
        n_qp = gradients.shape[0]
        self._ensure_rves(n_qp)

        F_all = gradients[:, :F_dim]
        v_all = gradients[:, F_dim:F_dim + N]
        g_all = gradients[:, F_dim + N:]

        P_flat      = np.zeros((n_qp, F_dim))
        Pi_flat     = np.zeros((n_qp, N))
        Lambda_flat = np.zeros((n_qp, Ng))

        dP_dF   = np.zeros((n_qp, F_dim, F_dim))
        dP_dv   = np.zeros((n_qp, F_dim, N))
        dP_dg   = np.zeros((n_qp, F_dim, Ng))
        dPi_dF  = np.zeros((n_qp, N, F_dim))
        dPi_dv  = np.zeros((n_qp, N, N))
        dPi_dg  = np.zeros((n_qp, N, Ng))
        dLam_dF = np.zeros((n_qp, Ng, F_dim))
        dLam_dv = np.zeros((n_qp, Ng, N))
        dLam_dg = np.zeros((n_qp, Ng, Ng))

        local_failure = 0
        failed_qp = -1

        for i, rve in enumerate(self._rves):
            F_qp = _f5_to_2x2(F_all[i])
            v_qp = v_all[i]                              # (N,)
            g_qp = g_all[i].reshape(N, gdim)             # (N, gdim)
            with qp_context(i):
                try:
                    out = rve(F_qp, v=v_qp, g=g_qp)
                except RVEConvergenceError as exc:
                    logger.warning(
                        "RVE qp %d did not converge: %s. Step will be rejected.", i, exc
                    )
                    local_failure = 1
                    if failed_qp < 0:
                        failed_qp = i
                    dP_dF[i] = np.eye(F_dim)
                    dPi_dv[i] = np.eye(N)
                    dLam_dg[i] = np.eye(Ng)
                    continue
                except Exception:
                    logger.exception("RVE qp %d failed unexpectedly", i)
                    raise

            d = out[-1]
            P_flat[i]      = _tensor2x2_to_f5(d["Pbar"])
            Pi_flat[i]     = np.asarray(d["Pi"]).ravel()
            Lambda_flat[i] = np.asarray(d["Lambda"]).ravel()

            dP_dF[i]   = _tangent2x2x2x2_to_mat55(d["dPbar_dFbar"])
            dP_dv[i]   = _pbar_v_tangent_to_mat5N(d["dPbar_dv"], N)
            dP_dg[i]   = _pbar_g_tangent_to_mat5Ng(d["dPbar_dg"], N, gdim)
            dPi_dF[i]  = _pi_fbar_tangent_to_matN5(d["dPi_dFbar"], N)
            dPi_dv[i]  = np.asarray(d["dPi_dv"]).reshape(N, N)
            dPi_dg[i]  = _pi_g_tangent_to_matNNg(d["dPi_dg"], N, gdim)
            dLam_dF[i] = _lambda_fbar_tangent_to_matNg5(d["dLambda_dFbar"], N, gdim)
            dLam_dv[i] = _lambda_v_tangent_to_matNgN(d["dLambda_dv"], N, gdim)
            dLam_dg[i] = _lambda_g_tangent_to_matNgNg(d["dLambda_dg"], N, gdim)

        any_failure = MPI.COMM_WORLD.allreduce(local_failure, op=MPI.LOR)
        if any_failure:
            self.step_failed = True
            self.failure_reason = (
                f"Micromorphic RVE failed (rank {self._rank}, "
                f"local={bool(local_failure)}, first qp={failed_qp})"
            )
            P_flat[:] = 999999999999999.0
        else:
            self.step_failed = False

        flux_parts = [P_flat]
        ct_blocks = [dP_dF]
        if N > 0:
            flux_parts.extend([Pi_flat, Lambda_flat])
            ct_blocks.extend([
                dP_dv, dP_dg,
                dPi_dF, dPi_dv, dPi_dg,
                dLam_dF, dLam_dv, dLam_dg,
            ])
        flux_vals = np.concatenate(flux_parts, axis=1)
        Ct_vals = _pack_ct(ct_blocks, n_qp)

        # Keep data_manager in sync so qmap.advance() propagates correct values.
        self.data_manager.s1.fluxes[:] = flux_vals
        self.data_manager.s0.fluxes[:] = flux_vals

        return flux_vals, np.zeros((n_qp, 0)), Ct_vals

    def commit(self) -> None:
        if self._rves is None:
            return
        for rve in self._rves:
            rve.commit()

    def constitutive_update(self, gradients, state, dt):
        pass
