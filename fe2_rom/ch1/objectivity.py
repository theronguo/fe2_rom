"""Objectivity (frame-indifference) reduction for homogenized constitutive laws.

Material frame indifference gives ``P(F) = R · P(U)`` for the right polar
decomposition ``F = R U`` (``R`` rotation, ``U`` symmetric SPD).  The homogenized
response therefore depends on ``F̄`` only through the symmetric stretch ``U``
(6 independent components in 3D, 3 in 2D).  This lets the RVE be driven by the
*symmetric* ``U`` — so:

* the effective stress is recovered by ``P̄ = R P̃`` (``P̃`` = response at ``U``);
* the effective tangent ``dP̄/dF̄`` is reconstructed from the *reduced* tangent
  ``dP̃/dU`` (only 6 adjoint directions, not 9) plus the analytic polar
  derivatives ``dR/dF̄`` and ``dU/dF̄``;
* for the micromorphic co-rotational ansatz ``φ → R φ`` the same machinery makes
  the law objective: ``Π`` (scalar) and ``Λ`` (material-gradient conjugate) are
  rotation-invariant, while ``P̄`` and its ``v``/``g`` derivatives rotate by ``R``.

The polar derivatives are computed via the square-root route ``C = U²``: in the
eigenbasis of ``C`` the stretch differential is ``dŨ_{ab} = dC̃_{ab}/(λ_a+λ_b)``,
whose denominators are bounded below by ``2 λ_min > 0``.  Unlike the
``1/(λ_a−λ_b)`` eigenprojection formulas this is **non-singular for repeated
stretches** (incl. ``F̄ = I``), so no special-casing is needed.

All routines here are plain NumPy, evaluated per macro quadrature point.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Symmetric basis directions (the 6/3 independent components of U)
# ---------------------------------------------------------------------------

def symmetric_index_pairs(gdim: int) -> list[tuple[int, int]]:
    """Ordered ``(p, q)`` pairs (``p <= q``) indexing the independent components
    of a symmetric ``gdim×gdim`` tensor: 3 in 2D, 6 in 3D."""
    return [(p, q) for p in range(gdim) for q in range(p, gdim)]


def symmetric_basis_tensors(gdim: int) -> list[np.ndarray]:
    """Symmetric basis 2-tensors ``S^{(pq)}`` for the independent components.

    Diagonal: ``S = e_p ⊗ e_p``.  Off-diagonal (``p<q``):
    ``S = e_p ⊗ e_q + e_q ⊗ e_p``.  Ordered like :func:`symmetric_index_pairs`.
    Used as the adjoint perturbation directions ``∂U`` so that only
    ``len`` = 6 (3D) / 3 (2D) sensitivities are solved instead of ``gdim²``.
    """
    out = []
    for p, q in symmetric_index_pairs(gdim):
        S = np.zeros((gdim, gdim), dtype=float)
        if p == q:
            S[p, p] = 1.0
        else:
            S[p, q] = 1.0
            S[q, p] = 1.0
        out.append(S)
    return out


def assemble_symmetric_tangent(dP_dir: np.ndarray, gdim: int) -> np.ndarray:
    """Map directional derivatives along the symmetric basis back to a
    ``(…, gdim, gdim)`` tensor symmetric in the last two indices.

    ``dP_dir`` has shape ``(n_sym, *lead)`` where ``dP_dir[s]`` is the response
    ``dQ̃`` to a perturbation of ``U`` along ``S^{(pq)}`` (``s``-th symmetric
    basis tensor).  Returns ``T`` of shape ``(*lead, gdim, gdim)`` with
    ``Σ_{p,q} T[...,p,q] dU_{pq} = dQ̃`` for any symmetric ``dU`` — i.e. the
    symmetric representative ``∂Q̃/∂U_{pq}``.

    Factors: diagonal direction ``S^{(pp)}`` gives ``T[...,p,p]`` directly;
    off-diagonal ``S^{(pq)} = e_p⊗e_q + e_q⊗e_p`` gives
    ``T[...,p,q] = T[...,q,p] = dP_dir[s] / 2`` (because that single solve carries
    the response to *both* ``U_{pq}`` and ``U_{qp}``).
    """
    pairs = symmetric_index_pairs(gdim)
    lead = dP_dir.shape[1:]
    T = np.zeros(lead + (gdim, gdim), dtype=float)
    for s, (p, q) in enumerate(pairs):
        if p == q:
            T[..., p, p] = dP_dir[s]
        else:
            half = 0.5 * dP_dir[s]
            T[..., p, q] = half
            T[..., q, p] = half
    return T


# ---------------------------------------------------------------------------
# Right polar decomposition and its derivatives
# ---------------------------------------------------------------------------

def right_polar(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Right polar decomposition ``F = R U`` (``R`` rotation, ``U`` symmetric SPD).

    Robust via ``C = FᵀF`` eigendecomposition (``U = sqrt(C)``,
    ``R = F U⁻¹``).
    """
    C = F.T @ F
    mu, N = np.linalg.eigh(C)                 # C = N diag(mu) Nᵀ
    mu = np.clip(mu, 1e-300, None)
    lam = np.sqrt(mu)
    U = (N * lam) @ N.T
    Uinv = (N / lam) @ N.T
    R = F @ Uinv
    return R, U


def polar_derivatives(F: np.ndarray):
    """Right polar decomposition and its derivatives w.r.t. ``F``.

    Returns ``(R, U, dR, dU)`` where ``R, U`` are ``(g,g)`` and

        ``dR[i, m, k, L] = ∂R_{im}/∂F_{kL}``
        ``dU[p, q, k, L] = ∂U_{pq}/∂F_{kL}``  (symmetric in ``p,q``)

    via ``dC = dFᵀF + FᵀdF``, ``dŨ_{ab} = (NᵀdC N)_{ab}/(λ_a+λ_b)``,
    ``dU = N dŨ Nᵀ``, ``dR = (dF − R dU) U⁻¹``.
    """
    g = F.shape[0]
    C = F.T @ F
    mu, N = np.linalg.eigh(C)
    mu = np.clip(mu, 1e-300, None)
    lam = np.sqrt(mu)
    U = (N * lam) @ N.T
    Uinv = (N / lam) @ N.T
    R = F @ Uinv

    denom = lam[:, None] + lam[None, :]       # λ_a + λ_b, strictly > 0
    dR = np.zeros((g, g, g, g), dtype=float)
    dU = np.zeros((g, g, g, g), dtype=float)

    # Apply the differential along each input basis direction dF = e_k ⊗ e_L.
    for k in range(g):
        for L in range(g):
            dF = np.zeros((g, g), dtype=float)
            dF[k, L] = 1.0
            dC = dF.T @ F + F.T @ dF
            dC_eig = N.T @ dC @ N
            dU_eig = dC_eig / denom
            dU_kL = N @ dU_eig @ N.T           # symmetric
            dR_kL = (dF - R @ dU_kL) @ Uinv
            dU[:, :, k, L] = dU_kL
            dR[:, :, k, L] = dR_kL
    return R, U, dR, dU


# ---------------------------------------------------------------------------
# Tangent reconstruction (lab frame from U-frame reduced quantities)
# ---------------------------------------------------------------------------

def reconstruct_dPbar_dFbar(R, dR, Ptilde, Atilde, dU) -> np.ndarray:
    """Full effective tangent ``dP̄/dF̄`` from U-frame quantities.

        ``A[i,J,k,L] = dR[i,m,k,L] P̃[m,J] + R[i,m] Ã[m,J,p,q] dU[p,q,k,L]``

    ``Ptilde`` = ``P̃`` (g,g), ``Atilde`` = ``dP̃/dU`` (g,g,g,g) symmetric in the
    last two indices (use :func:`assemble_symmetric_tangent`), ``R``/``dR``/``dU``
    from :func:`polar_derivatives`.
    """
    term_rot = np.einsum("imkL,mJ->iJkL", dR, Ptilde)
    term_mat = np.einsum("im,mJpq,pqkL->iJkL", R, Atilde, dU)
    return term_rot + term_mat


def objective_transform_pbar(d: dict, R, dR, dU) -> None:
    """In-place U-frame → lab-frame conversion of the ``Pbar`` / ``dPbar_dFbar``
    entries of an effective-quantity output dict (shared by the CH1 FOM and ROM
    solvers; the micromorphic solvers extend it with their own blocks).

    ``d["Pbar"]`` holds the U-frame ``P̃``; ``d["dPbar_dFbar"]`` holds the
    U-frame reduced tangent ``dP̃/dU`` (from
    :class:`~fe2_rom.ch1.averages.EffectiveAbarReduced`). On return they hold
    ``P̄ = R P̃`` and ``dP̄/dF̄ = dR·P̃ + R·(dP̃/dU)·dU``.
    """
    if "Pbar" not in d:
        return
    Ptilde = np.asarray(d["Pbar"])
    d["Pbar"] = R @ Ptilde
    if "dPbar_dFbar" in d:
        Atilde = np.asarray(d["dPbar_dFbar"])
        d["dPbar_dFbar"] = reconstruct_dPbar_dFbar(R, dR, Ptilde, Atilde, dU)


def reconstruct_dscalar_dFbar(dQtilde_dU, dU) -> np.ndarray:
    """``dQ̄/dF̄`` for a rotation-invariant quantity ``Q`` (e.g. ``Π``):

        ``dQ̄[...,k,L] = dQ̃/dU[...,p,q] dU[p,q,k,L]``

    ``dQtilde_dU`` has trailing shape ``(...,g,g)`` symmetric in ``(p,q)``.
    """
    return np.einsum("...pq,pqkL->...kL", dQtilde_dU, dU)
