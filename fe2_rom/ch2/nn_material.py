"""Neural-network second-order (CH2) constitutive law for the mixed FE² macro
driver.

``NNCh2Material`` is a drop-in substitute for
:class:`~fe2_rom.ch2.material.Ch2RVEMaterial` / :class:`DummyCh2Material`: same
``gradients`` / ``fluxes`` / ``tangent_blocks`` contract, but the response comes
from a trained :class:`~fe2_rom.nn.model.EnergyNet` (``flavor="ch2"``) effective
energy W̄(Ū, Ĝ) instead of nested second-order RVE solves — the two-scale
problem becomes a single macroscopic one.

All quantities are automatic derivatives of one scalar
``W(F̄, Ḡ) = W̄(U(F̄), R(F̄)ᵀ·Ḡ)`` (differentiable polar decomposition inside,
see :func:`fe2_rom.nn.model.make_lab_energy`):

* fluxes  ``[P | Q] = ∂W/∂z`` with ``z = [F_vec | G_vec]`` (MFront F-packing +
  row-major Ḡ, identical to the QuadratureMap gradient layout — no reordering),
* tangents = ∂²W/∂z², sliced into the 2×2 block grid. The Hessian is exactly
  symmetric, so the cross blocks satisfy dQ̄/dF̄ = (dP̄/dḠ)ᵀ by construction, and
  the objective reduction makes Q̄ exactly symmetric in Ḡ's last two indices.

Stateless (no internal variables, no history), hence not checkpointed.
"""
from __future__ import annotations

import logging

import numpy as np

from dolfinx_materials.generic import Material

from fe2_rom.ch1.material import _f_order
from fe2_rom.ch2.material import _build_tangent_blocks, _pack_ct

logger = logging.getLogger(__name__)


class NNCh2Material(Material):
    """Second-order law backed by a trained ``flavor="ch2"`` EnergyNet.

    Parameters
    ----------
    model : str | fe2_rom.nn.model.EnergyNet
        Path to a ``.pt`` file written by ``EnergyNet.save`` (flavor ``"ch2"``)
        or a model instance. Weights are frozen; evaluation is float64.
    gdim : int, optional
        Validated against the model config (which is the source of truth);
        defaults to the model's ``gdim``.
    torch_threads : int
        Torch intra-op threads per MPI rank (keep 1 for FE²-style runs).
    """

    def __init__(self, model, gdim: int | None = None, torch_threads: int = 1):
        # fe2_rom.nn guards the dolfinx-before-torch load order.
        from fe2_rom.nn.model import EnergyNet
        import torch

        torch.set_num_threads(torch_threads)

        if isinstance(model, str):
            model = EnergyNet.load(model)
        if model.flavor != "ch2":
            raise ValueError(
                f"NNCh2Material needs a flavor='ch2' EnergyNet, "
                f"got flavor={model.flavor!r}.")
        model = model.double().eval()
        model.requires_grad_(False)
        self._model = model

        if gdim is not None and gdim != model.gdim:
            raise ValueError(f"gdim={gdim} != model.gdim={model.gdim}")
        self._gdim = model.gdim
        _, self._F_dim = _f_order(self._gdim)
        self._G_dim = self._gdim ** 3

        self._torch = torch
        self.refresh_from_model()

        self.step_failed: bool = False
        self.failure_reason: str = ""
        super().__init__()

    def refresh_from_model(self) -> None:
        """Rebuild the flux/tangent closures from the model's *current* weights.

        Needed when the weights are mutated in place (e.g. during a fit), since
        the reference-state correction baked into ``make_lab_energy`` is captured
        at build time."""
        from fe2_rom.nn.model import make_lab_energy
        from torch.func import grad, jacfwd, vmap

        W_fn = make_lab_energy(self._model)  # detached reference correction

        def grad_with_aux(z):
            G = grad(W_fn)(z)
            return G, G

        # One batched forward-over-reverse pass returns Hessian + gradient.
        self._flux_tangent_fn = vmap(jacfwd(grad_with_aux, has_aux=True))

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

        z = self._torch.from_numpy(
            np.ascontiguousarray(gradients, dtype=np.float64))
        H_t, G_t = self._flux_tangent_fn(z)
        flux_vals = G_t.numpy()           # (n_qp, F_dim + G_dim) = [P | Q]
        H = H_t.numpy()                   # (n_qp, dim, dim), symmetric

        bad = ~(np.isfinite(flux_vals).all(axis=1) & np.isfinite(H).all(axis=(1, 2)))
        if bad.any():
            q = int(np.argmax(bad))
            logger.error(
                "Non-finite NN response at %d/%d qp(s); first bad qp %d: z=%s",
                int(bad.sum()), n_qp, q,
                np.array2string(gradients[q], precision=4),
            )

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
