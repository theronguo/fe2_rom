"""Second-order (CH2) constitutive bridge for the mixed FE² macro driver.

Two classes:

``DummyCh2Material``
    Linear, decoupled gradient-elasticity law for verifying the mixed
    ``[u, F̂, L̄]`` assembly without running any RVE::

        P̄ = µ (F̂ − I),     Q̄ = η Ḡ.

``Ch2RVEMaterial``
    Wraps a second-order RVE solver (:class:`fe2_rom.ch2.MicroSolver` or the
    reduced counterpart). One RVE per macro quadrature point, lazily created on
    the first ``integrate()`` call and warm-started across Newton iterations /
    load steps.

Conventions
-----------
``F̂`` (2nd order) uses the MFront nonsymmetric vector ordering shared with the
first-order / micromorphic bridges (``F_dim`` = 5 in 2D, 9 in 3D; the 2D zz slot
is a zero placeholder). ``Ḡ`` and the double stress ``Q̄`` (3rd order, ``gdim³``
entries) use a plain row-major flattening

    G_vec[i·g² + J·g + K] = Ḡ_iJK

matching :mod:`fe2_rom.ch2.averages` (``EffectiveQ`` / ``x_weighted_directions``).

    gradients = {"F": F_dim, "G": gdim³},   fluxes = {"P": F_dim, "Q": gdim³}
    tangent_blocks = {("P","F"), ("P","G"), ("Q","F"), ("Q","G")}.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from mpi4py import MPI

from dolfinx_materials.generic import Material

from fe2_rom.ch1.exceptions import RVEConvergenceError
from fe2_rom.ch1.material import (
    _f_order,
    _tangent4_to_mat,
    _tensor_to_vec,
    _vec_to_tensor,
)
from fe2_rom.hyperelastic_solver.logging_utils import qp_context

logger = logging.getLogger(__name__)

_I_VEC = {
    2: np.array([1.0, 1.0, 0.0, 0.0, 0.0]),
    3: np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
}


# ---------------------------------------------------------------------------
# Third-order (Ḡ / Q̄) row-major vector <-> tensor helpers and tangent packers
# ---------------------------------------------------------------------------

def _gvec_to_tensor3(v: np.ndarray, g: int) -> np.ndarray:
    return np.asarray(v, dtype=float).reshape(g, g, g)


def _tensor3_to_gvec(T: np.ndarray, g: int) -> np.ndarray:
    return np.asarray(T, dtype=float).reshape(g * g * g)


def _tangent_PG(A: np.ndarray, order: tuple, F_dim: int, g: int) -> np.ndarray:
    """(g,g,g,g,g) dP̄/dḠ → (F_dim, g³) row-major in the Ḡ index."""
    M = np.zeros((F_dim, g ** 3), dtype=float)
    for p, (i, j) in enumerate(order):
        if i is None:  # 2D out-of-plane placeholder slot
            continue
        M[p, :] = A[i, j].reshape(g ** 3)
    return M


def _tangent_QF(A: np.ndarray, order: tuple, F_dim: int, g: int) -> np.ndarray:
    """(g,g,g,g,g) dQ̄/dF̄ → (g³, F_dim); rows row-major in the Q̄ index."""
    M = np.zeros((g ** 3, F_dim), dtype=float)
    A2 = A.reshape(g ** 3, g, g)  # (q3, k, l)
    for q, (k, l) in enumerate(order):
        if k is None:  # 2D out-of-plane placeholder slot
            continue
        M[:, q] = A2[:, k, l]
    return M


def _tangent_QG(A: np.ndarray, g: int) -> np.ndarray:
    """(g,g,g,g,g,g) dQ̄/dḠ → (g³, g³) row-major."""
    return np.asarray(A, dtype=float).reshape(g ** 3, g ** 3)


def _build_tangent_blocks(F_dim: int, g: int) -> dict:
    G_dim = g ** 3
    return {
        ("P", "F"): (F_dim, F_dim),
        ("P", "G"): (F_dim, G_dim),
        ("Q", "F"): (G_dim, F_dim),
        ("Q", "G"): (G_dim, G_dim),
    }


def _pack_ct(blocks: list[np.ndarray], n_qp: int) -> np.ndarray:
    return np.concatenate([b.reshape(n_qp, -1) for b in blocks], axis=1)


# ---------------------------------------------------------------------------
# DummyCh2Material
# ---------------------------------------------------------------------------

class DummyCh2Material(Material):
    """Linear, decoupled gradient-elasticity law ``P̄ = µ(F̂−I)``, ``Q̄ = η Ḡ``.

    Used to verify the mixed second-order macro assembly and its saddle-point
    structure without any RVE. ``η`` introduces the length scale.
    """

    def __init__(self, gdim: int = 2, mu: float = 1.0, eta: float = 1.0):
        if gdim not in (2, 3):
            raise ValueError(f"gdim must be 2 or 3, got {gdim}.")
        self._gdim = gdim
        self._order, self._F_dim = _f_order(gdim)
        self._G_dim = gdim ** 3
        self._mu = mu
        self._eta = eta
        self._I_vec = _I_VEC[gdim].copy()
        self.step_failed = False
        self.failure_reason = ""
        super().__init__()

    @property
    def gradients(self):
        return {"F": self._F_dim, "G": self._G_dim}

    @property
    def fluxes(self):
        return {"P": self._F_dim, "Q": self._G_dim}

    @property
    def tangent_blocks(self):
        return _build_tangent_blocks(self._F_dim, self._gdim)

    def integrate(self, gradients: np.ndarray, dt: float = 0.0):
        F_dim, G_dim = self._F_dim, self._G_dim
        n_qp = gradients.shape[0]
        F_all = gradients[:, :F_dim]
        G_all = gradients[:, F_dim:F_dim + G_dim]

        P_flat = self._mu * (F_all - self._I_vec)
        Q_flat = self._eta * G_all
        flux_vals = np.concatenate([P_flat, Q_flat], axis=1)

        dP_dF = np.broadcast_to(self._mu * np.eye(F_dim), (n_qp, F_dim, F_dim)).copy()
        dP_dG = np.zeros((n_qp, F_dim, G_dim))
        dQ_dF = np.zeros((n_qp, G_dim, F_dim))
        dQ_dG = np.broadcast_to(self._eta * np.eye(G_dim), (n_qp, G_dim, G_dim)).copy()
        Ct_vals = _pack_ct([dP_dF, dP_dG, dQ_dF, dQ_dG], n_qp)

        self.data_manager.s1.fluxes[:] = flux_vals
        self.data_manager.s0.fluxes[:] = flux_vals
        return flux_vals, np.zeros((n_qp, 0)), Ct_vals

    def constitutive_update(self, gradients, state, dt):
        pass

    def commit(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Ch2RVEMaterial
# ---------------------------------------------------------------------------

class Ch2RVEMaterial(Material):
    """dolfinx_materials Material backed by a second-order RVE solver.

    Parameters
    ----------
    rve_factory : callable(rank, index) -> solver
        Returns a fresh :class:`fe2_rom.ch2.MicroSolver` (or reduced
        counterpart), callable as ``out = solver(Fbar, G)`` whose final step
        dict exposes ``Pbar``, ``Qbar`` and the four tangents ``dPbar_dFbar``,
        ``dPbar_dG``, ``dQbar_dFbar``, ``dQbar_dG``.
    gdim : int
        Geometric dimension (2 or 3).
    """

    def __init__(self, rve_factory: Callable[[int, int], object], gdim: int = 2):
        if gdim not in (2, 3):
            raise ValueError(f"gdim must be 2 or 3, got {gdim}.")
        self._rve_factory = rve_factory
        self._gdim = gdim
        self._order, self._F_dim = _f_order(gdim)
        self._G_dim = gdim ** 3
        self._rves: list | None = None
        self._n_qp: int | None = None
        self._rank = MPI.COMM_WORLD.Get_rank()
        self.step_failed = False
        self.failure_reason = ""
        super().__init__()

    @property
    def gradients(self):
        return {"F": self._F_dim, "G": self._G_dim}

    @property
    def fluxes(self):
        return {"P": self._F_dim, "Q": self._G_dim}

    @property
    def tangent_blocks(self):
        return _build_tangent_blocks(self._F_dim, self._gdim)

    def _ensure_rves(self, n_qp: int) -> None:
        if self._rves is None:
            logger.info(
                "Ch2RVEMaterial: instantiating %d RVE(s)", n_qp,
                extra={"all_ranks": True},
            )
            self._rves = []
            for i in range(n_qp):
                with qp_context(i):
                    self._rves.append(self._rve_factory(self._rank, i))
            self._n_qp = n_qp
        elif n_qp != self._n_qp:
            raise RuntimeError(
                f"Ch2RVEMaterial: qp count changed ({self._n_qp} -> {n_qp})."
            )

    def integrate(self, gradients: np.ndarray, dt: float = 0.0):
        F_dim, G_dim = self._F_dim, self._G_dim
        g = self._gdim
        order = self._order
        n_qp = gradients.shape[0]
        self._ensure_rves(n_qp)

        F_all = gradients[:, :F_dim]
        G_all = gradients[:, F_dim:F_dim + G_dim]

        P_flat = np.zeros((n_qp, F_dim))
        Q_flat = np.zeros((n_qp, G_dim))
        dP_dF = np.zeros((n_qp, F_dim, F_dim))
        dP_dG = np.zeros((n_qp, F_dim, G_dim))
        dQ_dF = np.zeros((n_qp, G_dim, F_dim))
        dQ_dG = np.zeros((n_qp, G_dim, G_dim))

        local_failure = 0
        failed_qp = -1
        for i, rve in enumerate(self._rves):
            F_qp = _vec_to_tensor(F_all[i], g)
            G_qp = _gvec_to_tensor3(G_all[i], g)
            with qp_context(i):
                try:
                    out = rve(F_qp, G_qp)
                except RVEConvergenceError as exc:
                    logger.warning(
                        "ch2 RVE qp %d did not converge: %s. Rejecting step.", i, exc)
                    local_failure = 1
                    if failed_qp < 0:
                        failed_qp = i
                    dP_dF[i] = np.eye(F_dim)
                    dQ_dG[i] = np.eye(G_dim)
                    continue
                except Exception:
                    logger.exception("ch2 RVE qp %d failed unexpectedly", i)
                    raise
            d = out[-1]
            P_flat[i] = _tensor_to_vec(np.asarray(d["Pbar"]), g)
            Q_flat[i] = _tensor3_to_gvec(np.asarray(d["Qbar"]), g)
            dP_dF[i] = _tangent4_to_mat(np.asarray(d["dPbar_dFbar"]), g)
            dP_dG[i] = _tangent_PG(np.asarray(d["dPbar_dG"]), order, F_dim, g)
            dQ_dF[i] = _tangent_QF(np.asarray(d["dQbar_dFbar"]), order, F_dim, g)
            dQ_dG[i] = _tangent_QG(np.asarray(d["dQbar_dG"]), g)

        any_failure = MPI.COMM_WORLD.allreduce(local_failure, op=MPI.LOR)
        if any_failure:
            self.step_failed = True
            self.failure_reason = (
                f"ch2 RVE failed (rank {self._rank}, local={bool(local_failure)}, "
                f"first qp={failed_qp})"
            )
            P_flat[:] = 999999999999999.0
        else:
            self.step_failed = False

        flux_vals = np.concatenate([P_flat, Q_flat], axis=1)
        Ct_vals = _pack_ct([dP_dF, dP_dG, dQ_dF, dQ_dG], n_qp)
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

    # ------------------------------------------------------------------
    # Checkpoint I/O (full two-scale) — mirrors ch1/mm RVEMaterial.
    # ------------------------------------------------------------------

    def save_rves(self, checkpoint_dir: str, fingerprint: str) -> None:
        from fe2_rom.ch1 import restart as _restart
        if self._rves is None or not self._rves:
            stacked: dict = {"n_qp": np.int64(0)}
        else:
            per_qp = [rve.dump_state() for rve in self._rves]
            stacked = {k: np.stack([d[k] for d in per_qp], axis=0)
                       for k in per_qp[0].keys()}
            stacked["n_qp"] = np.int64(len(per_qp))
        _restart.write_rank_state(checkpoint_dir, self._rank, fingerprint, stacked)

    def load_rves(self, checkpoint_dir: str, n_qp: int,
                  expected_fingerprint: str) -> None:
        from fe2_rom.ch1 import restart as _restart
        fp, stacked = _restart.read_rank_state(checkpoint_dir, self._rank)
        if fp != expected_fingerprint:
            raise RuntimeError(
                f"ch2 RVE partition fingerprint mismatch on rank {self._rank}.")
        ckpt_n_qp = int(stacked.pop("n_qp"))
        if ckpt_n_qp != n_qp:
            raise RuntimeError(
                f"ch2 RVE count mismatch on rank {self._rank}: checkpoint "
                f"{ckpt_n_qp}, current {n_qp}.")
        if n_qp == 0:
            return
        self._ensure_rves(n_qp)
        for i, rve in enumerate(self._rves):
            with qp_context(i):
                rve.load_state({k: v[i] for k, v in stacked.items()})
