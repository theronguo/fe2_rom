"""Neural-network first-order (CH1) constitutive law for the FE² macro driver.

``NNRVEMaterial`` is a drop-in substitute for
:class:`~fe2_rom.ch1.material.RVEMaterial`: same ``gradients`` / ``fluxes``
contract and the same ``(fluxes, isvs, tangent)`` return signature, but the
response comes from a trained :class:`~fe2_rom.nn.model.EnergyNet`
(``flavor="ch1"``) effective energy W̄(Ū) instead of nested RVE solves — the
two-scale problem collapses to a single macroscopic one.

Both the stress and the tangent are automatic derivatives of one scalar
``W(F̄) = W̄(U(F̄))`` (differentiable polar decomposition inside, see
:func:`fe2_rom.nn.model.make_lab_energy`):

* PK1 stress  ``P̄ = ∂W/∂F̄``      (gradient w.r.t. the MFront F-vector),
* tangent     ``dP̄/dF̄ = ∂²W/∂F̄²`` (Hessian).

Frame indifference holds by construction (the net sees only the right stretch
Ū), so ∂W/∂F̄ automatically carries the rotation-related part of P̄.

Stateless (no internal variables, no history), hence not checkpointed — the
macro driver should run with ``full=False`` semantics (no RVE checkpoint); a
fresh NN run is cheap.
"""
from __future__ import annotations

import logging

import numpy as np

from dolfinx_materials.generic import Material

from fe2_rom.ch1.material import _f_order

logger = logging.getLogger(__name__)


class NNRVEMaterial(Material):
    """First-order law backed by a trained ``flavor="ch1"`` EnergyNet.

    Parameters
    ----------
    model : str | fe2_rom.nn.model.EnergyNet
        Path to a ``.pt`` file written by ``EnergyNet.save`` (flavor ``"ch1"``)
        or a model instance. Weights are frozen; evaluation is float64.
    gdim : int, optional
        Validated against the model config (which is the source of truth);
        defaults to the model's ``gdim``.
    torch_threads : int
        Torch intra-op threads per MPI rank (keep 1 for FE²-style runs).
    """

    def __init__(self, model, gdim: int | None = None, torch_threads: int = 1):
        # fe2_rom.nn guards the dolfinx-before-torch load order.
        from fe2_rom.nn.model import EnergyNet, make_lab_energy
        import torch
        from torch.func import grad, jacfwd, vmap

        torch.set_num_threads(torch_threads)

        if isinstance(model, str):
            model = EnergyNet.load(model)
        if model.flavor != "ch1":
            raise ValueError(
                f"NNRVEMaterial needs a flavor='ch1' EnergyNet, "
                f"got flavor={model.flavor!r}.")
        model = model.double().eval()
        model.requires_grad_(False)
        self._model = model

        if gdim is not None and gdim != model.gdim:
            raise ValueError(f"gdim={gdim} != model.gdim={model.gdim}")
        self._gdim = model.gdim
        _, self._F_dim = _f_order(self._gdim)

        W_fn = make_lab_energy(model)  # detached reference correction

        def grad_with_aux(z):
            G = grad(W_fn)(z)
            return G, G

        # One batched forward-over-reverse pass returns Hessian + gradient.
        self._flux_tangent_fn = vmap(jacfwd(grad_with_aux, has_aux=True))
        self._torch = torch

        self.step_failed: bool = False
        self.failure_reason: str = ""
        super().__init__()

    @property
    def gradients(self):
        return {"F": self._F_dim}

    @property
    def fluxes(self):
        return {"PK1": self._F_dim}

    def integrate(self, gradients: np.ndarray, dt: float = 0.0):
        n_qp = gradients.shape[0]
        z = self._torch.from_numpy(
            np.ascontiguousarray(gradients, dtype=np.float64))
        H_t, G_t = self._flux_tangent_fn(z)
        P_flat = G_t.numpy()          # (n_qp, F_dim)
        A_flat = H_t.numpy()          # (n_qp, F_dim, F_dim) = dP/dF, symmetric

        bad = ~(np.isfinite(P_flat).all(axis=1) & np.isfinite(A_flat).all(axis=(1, 2)))
        if bad.any():
            q = int(np.argmax(bad))
            logger.error(
                "Non-finite NN response at %d/%d qp(s); first bad qp %d: F=%s",
                int(bad.sum()), n_qp, q,
                np.array2string(gradients[q], precision=4),
            )

        self.data_manager.s1.set_item({"PK1": P_flat})
        return (
            self.data_manager.s1.fluxes,
            self.data_manager.s1.internal_state_variables,
            A_flat,
        )

    def constitutive_update(self, gradients, state, dt):
        pass

    def commit(self) -> None:
        pass
