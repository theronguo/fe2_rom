"""Neural-network micromorphic constitutive law for the two-scale macro driver.

``NNMicromorphicMaterial`` is a drop-in substitute for
:class:`~fe2_rom.mm.material.DummyMicromorphicMaterial` /
:class:`~fe2_rom.mm.material.MicromorphicRVEMaterial`: same ``gradients`` /
``fluxes`` / ``tangent_blocks`` contract, but the response comes from a
trained :class:`~fe2_rom.nn.model.EnergyNet` (``flavor="mm"``) effective energy
W̄(Ū, v, g) instead of nested RVE solves — the two-scale problem becomes a
single macroscopic one.

All quantities are automatic derivatives of one scalar
``W(F̄, v, g) = W̄(U(F̄), v, g)`` (differentiable polar decomposition inside,
see :func:`fe2_rom.nn.model.make_lab_energy`):

* fluxes  ``[P | Π | Λ] = ∂W/∂z`` with ``z = [F_vec | v | g]`` (MFront packing,
  identical to the QuadratureMap gradient layout — no reordering needed),
* tangents = ∂²W/∂z², sliced into the 3×3 block grid. The Hessian is exactly
  symmetric, so the cross blocks satisfy dΠ/dF̄ = (dP̄/dv)ᵀ etc. by construction.

Stateless (no internal variables, no history), hence not checkpointed —
use ``enable_restart=False`` on the macro solver, like the Dummy law.
"""
from __future__ import annotations

import logging

import numpy as np

from dolfinx_materials.generic import Material

from fe2_rom.mm.material import _F_DIM, _build_tangent_blocks, _pack_ct

logger = logging.getLogger(__name__)


class NNMicromorphicMaterial(Material):
    """Micromorphic law backed by a trained MMEnergyNet.

    Parameters
    ----------
    model : str | fe2_rom.nn.model.EnergyNet
        Path to a ``.pt`` file written by ``EnergyNet.save`` (flavor ``"mm"``)
        or a model instance. Weights are frozen; evaluation is float64.
    N_modes, gdim : int, optional
        Validated against the model config (which is the source of truth);
        defaults are taken from the model.
    torch_threads : int
        Torch intra-op threads per MPI rank (keep 1 for FE²-style runs).
    """

    def __init__(self, model, N_modes: int | None = None,
                 gdim: int | None = None, torch_threads: int = 1):
        # fe2_rom.nn guards the dolfinx-before-torch load order.
        from fe2_rom.nn.model import EnergyNet, make_lab_energy
        import torch
        from torch.func import grad, jacfwd, vmap

        torch.set_num_threads(torch_threads)

        if isinstance(model, str):
            model = EnergyNet.load(model)
        if model.flavor != "mm":
            raise ValueError(
                f"NNMicromorphicMaterial needs a flavor='mm' EnergyNet, "
                f"got flavor={model.flavor!r}.")
        model = model.double().eval()
        model.requires_grad_(False)
        self._model = model

        if gdim is not None and gdim != model.gdim:
            raise ValueError(f"gdim={gdim} != model.gdim={model.gdim}")
        if N_modes is not None and N_modes != model.n_modes:
            raise ValueError(
                f"N_modes={N_modes} != model.n_modes={model.n_modes}")
        self._gdim = model.gdim
        self._N = model.n_modes
        self._F_dim = _F_DIM[self._gdim]

        W_fn = make_lab_energy(model)  # detached reference correction
        grad_fn = grad(W_fn)

        def grad_with_aux(z):
            G = grad_fn(z)
            return G, G

        # One batched forward-over-reverse pass returns Hessian + gradient.
        self._flux_tangent_fn = vmap(jacfwd(grad_with_aux, has_aux=True))
        self._torch = torch

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
        F_dim, N, gdim = self._F_dim, self._N, self._gdim
        Ng = N * gdim
        n_qp = gradients.shape[0]

        z = self._torch.from_numpy(
            np.ascontiguousarray(gradients, dtype=np.float64))
        H_t, G_t = self._flux_tangent_fn(z)
        flux_vals = G_t.numpy()           # (n_qp, F_dim + N + Ng) = [P|Π|Λ]
        H = H_t.numpy()                   # (n_qp, dim, dim), symmetric

        bad = ~(np.isfinite(flux_vals).all(axis=1) & np.isfinite(H).all(axis=(1, 2)))
        if bad.any():
            q = int(np.argmax(bad))
            logger.error(
                "Non-finite NN response at %d/%d qp(s); first bad qp %d: z=%s",
                int(bad.sum()), n_qp, q, np.array2string(gradients[q], precision=4),
            )

        sF = slice(0, F_dim)
        sv = slice(F_dim, F_dim + N)
        sg = slice(F_dim + N, F_dim + N + Ng)
        ct_blocks = [H[:, sF, sF]]
        if N > 0:
            ct_blocks.extend([
                H[:, sF, sv], H[:, sF, sg],
                H[:, sv, sF], H[:, sv, sv], H[:, sv, sg],
                H[:, sg, sF], H[:, sg, sv], H[:, sg, sg],
            ])
        Ct_vals = _pack_ct(ct_blocks, n_qp)

        if getattr(self, "data_manager", None) is not None:
            self.data_manager.s1.fluxes[:] = flux_vals
            self.data_manager.s0.fluxes[:] = flux_vals

        return flux_vals, np.zeros((n_qp, 0)), Ct_vals

    def constitutive_update(self, gradients, state, dt):
        pass

    def commit(self) -> None:
        pass
