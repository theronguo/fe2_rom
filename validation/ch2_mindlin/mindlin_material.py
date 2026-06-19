"""Mindlin isotropic second-gradient elastic constitutive law as a CH2
``dolfinx_materials`` material (Kouznetsova thesis §4.4.1, eq. 4.79).

Strain energy per unit reference volume (eq. 4.79)::

    W₀ = ½λ E_ii E_jj + μ E_ij E_ij
       + ½a₁(G_ijj G_ikk + G_jji G_kki) + ½a₂(G_iki G_kjj + G_iki G_jjk)
       + a₃ G_iki G_jkj + a₄ G_ikj G_ikj + ½a₅(G_ijk G_jik + G_ijk G_jki)

with the Green–Lagrange strain ``E = ½(FᵀF − I)`` and ``a₁=…=a₅ = ½μ Z²``
(``Z`` = material length scale). The first Piola–Kirchhoff stress and the
higher-order (double) stress are ``P = ∂W₀/∂F`` (St-Venant–Kirchhoff, depends on
``F`` only) and ``Q = ∂W₀/∂G`` (linear, depends on ``G`` only); the model is
decoupled (``dP/dG = 0``, ``dQ/dF = 0``).

Index convention. The thesis writes ``G_ijk = ∂F_jk/∂X_i`` (gradient index
*first*; e.g. ``G_212 = ∂F₁₂/∂X₂``), whereas the ``fe2_rom.ch2`` macro variable
is ``Ḡ_iJK = ∂F_iJ/∂X_K`` (gradient index *last*). Hence
``thesis G_ijk = code Ḡ_jik`` — a swap of the first two axes. The energy below is
transcribed in *thesis* indices and evaluated on ``swapaxes(Ḡ, 0, 1)``; ``Ḡ`` is
first symmetrised in its last two indices (its physical symmetry) so ``Q`` comes
out symmetric and conjugate to ``Ḡ`` (verified against eq. 4.86 in
``check_eq486.py``).

Everything is a torch automatic derivative of the one scalar ``W₀`` (no manual
⁶D), and ``W₀`` is polynomial so the tangent (Hessian) is exact and finite.
Same ``gradients`` / ``fluxes`` / ``tangent_blocks`` contract as
``DummyCh2Material`` / ``Ch2RVEMaterial`` → drops straight into
``MacroSecondOrderSolver(mesh, …, material=MindlinCh2Material(...))``.
"""
from __future__ import annotations

import logging

import numpy as np

from dolfinx_materials.generic import Material

from fe2_rom.ch1.material import _f_order
from fe2_rom.ch2.material import _build_tangent_blocks, _pack_ct

logger = logging.getLogger(__name__)


def mindlin_energy_grad(g, a1, a2, a3, a4, a5):
    """Gradient strain-energy density (eq. 4.79, terms in a₁…a₅), thesis indices.

    ``g`` is the thesis third-order tensor (…here passed as ``swapaxes(Ḡ,0,1)``),
    shape ``(gdim, gdim, gdim)``. Returns a scalar (torch or numpy via einsum).
    """
    import torch
    ee = torch.einsum
    t1 = ee("ijj,ikk->", g, g) + ee("jji,kki->", g, g)
    t2 = ee("iki,kjj->", g, g) + ee("iki,jjk->", g, g)
    t3 = ee("iki,jkj->", g, g)
    t4 = ee("ikj,ikj->", g, g)
    t5 = ee("ijk,jik->", g, g) + ee("ijk,jki->", g, g)
    return 0.5 * a1 * t1 + 0.5 * a2 * t2 + a3 * t3 + a4 * t4 + 0.5 * a5 * t5


class MindlinCh2Material(Material):
    """Mindlin second-gradient elastic law (eq. 4.79) for ``MacroSecondOrderSolver``.

    Parameters
    ----------
    mu : float
        Shear modulus μ.
    lmbda : float | None
        First Lamé constant λ. If ``None``, computed from ``kappa`` via
        ``λ = kappa − 2μ/3``.
    kappa : float | None
        Bulk modulus K (used only if ``lmbda`` is None).
    Z : float
        Material length scale; sets ``a₁=…=a₅ = ½ μ Z²``.
    gdim : int
        Geometric dimension (2 or 3).
    torch_threads : int
        Torch intra-op threads.
    """

    def __init__(self, mu: float, Z: float, *, lmbda: float | None = None,
                 kappa: float | None = None, gdim: int = 2, torch_threads: int = 1):
        import torch
        from torch.func import grad, jacfwd, vmap
        from fe2_rom.nn.model import _F_ORDER

        torch.set_num_threads(torch_threads)
        if gdim not in (2, 3):
            raise ValueError("gdim must be 2 or 3.")
        if lmbda is None:
            if kappa is None:
                raise ValueError("provide either lmbda or kappa.")
            lmbda = kappa - 2.0 * mu / 3.0
        self._gdim = gdim
        _, self._F_dim = _f_order(gdim)
        self._G_dim = gdim ** 3
        a = 0.5 * mu * Z ** 2          # a₁ = … = a₅ (eq. 4.79 / §4.4.3)
        mu_t = float(mu)
        lam_t = float(lmbda)

        basis = torch.zeros(self._F_dim, gdim, gdim, dtype=torch.float64)
        for k, ij in enumerate(_F_ORDER[gdim]):
            if ij is not None:
                basis[k, ij[0], ij[1]] = 1.0
        eye = torch.eye(gdim, dtype=torch.float64)

        def W_fn(z):
            F = (z[:self._F_dim, None, None] * basis).sum(0)        # (g,g)
            E = 0.5 * (F.T @ F - eye)
            trE = torch.einsum("ii->", E)
            W_strain = 0.5 * lam_t * trE * trE + mu_t * torch.einsum("ij,ij->", E, E)
            G = z[self._F_dim:].reshape(gdim, gdim, gdim)           # Ḡ_iJK (code)
            G = 0.5 * (G + G.transpose(-1, -2))                     # symmetrise (J,K)
            g = G.transpose(0, 1)                                   # thesis G_ijk = Ḡ_jik
            W_grad = mindlin_energy_grad(g, a, a, a, a, a)
            return W_strain + W_grad

        grad_fn = grad(W_fn)

        def grad_with_aux(z):
            G = grad_fn(z)
            return G, G

        self._flux_tangent_fn = vmap(jacfwd(grad_with_aux, has_aux=True))
        self._torch = torch

        self.step_failed: bool = False
        self.failure_reason: str = ""
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
        z = self._torch.from_numpy(np.ascontiguousarray(gradients, dtype=np.float64))
        H_t, G_t = self._flux_tangent_fn(z)
        flux_vals = G_t.numpy()           # (n_qp, F_dim + G_dim) = [P | Q]
        H = H_t.numpy()                   # (n_qp, dim, dim)

        sF = slice(0, F_dim)
        sG = slice(F_dim, F_dim + G_dim)
        ct_blocks = [H[:, sF, sF], H[:, sF, sG], H[:, sG, sF], H[:, sG, sG]]
        Ct_vals = _pack_ct(ct_blocks, n_qp)

        if getattr(self, "data_manager", None) is not None:
            self.data_manager.s1.fluxes[:] = flux_vals
            self.data_manager.s0.fluxes[:] = flux_vals
        return flux_vals, np.zeros((n_qp, 0)), Ct_vals

    def constitutive_update(self, gradients, state, dt):
        pass

    def commit(self) -> None:
        pass
