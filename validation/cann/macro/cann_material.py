"""``CANNMaterial`` — a CH1 ``dolfinx_materials`` constitutive law backed by a
trained CANN, for the macroscopic comparison.

Same contract as :class:`fe2_rom.ch1.material.RVEMaterial` /
:class:`fe2_rom.mm.material.DummyMicromorphicMaterial`
(``gradients={"F"}`` / ``fluxes={"PK1"}`` / ``integrate`` returning ``(P, isv,
A)``), so it plugs into the (now material-pluggable) ``ch1.MacroSolver`` exactly
like the Dummy law plugs into ``MacroMicromorphicSolver`` — but the response is a
closed-form surrogate, no nested RVE.

The strain energy is the trained CANN's isochoric energy plus the same
volumetric penalty as the Mooney–Rivlin reference::

    W(F) = Ψ_CANN(C) + ½ κ (J − 1)²,   C = FᵀF,  J = det F

with ``P = ∂W/∂F`` and ``A = ∂²W/∂F²`` by torch automatic differentiation
(one batched forward-over-reverse pass per ``integrate`` call). 3D only.
"""
from __future__ import annotations

import logging

import numpy as np

from dolfinx_materials.generic import Material

from fe2_rom.ch1.material import _f_order

logger = logging.getLogger(__name__)


class CANNMaterial(Material):
    """First-order law backed by a trained CANN + volumetric penalty.

    Parameters
    ----------
    model : validation/cann/cann.py ``CANN`` instance
        Trained, ``flavor='ch1'`` CANN. Weights frozen; evaluated in float64.
    kappa : float
        Bulk penalty for ``½κ(J−1)²`` (match the Mooney–Rivlin reference).
    gdim : int
        Geometric dimension (3).
    torch_threads : int
        Torch intra-op threads (keep 1 for FE-style runs).
    """

    def __init__(self, model, kappa: float, gdim: int = 3, torch_threads: int = 1):
        import torch
        from torch.func import grad, jacfwd, vmap
        from fe2_rom.nn.model import _F_ORDER
        from cann import _det3   # explicit 3×3 det (hessian-safe, unlike linalg.det)

        torch.set_num_threads(torch_threads)
        if gdim != 3:
            raise ValueError("CANNMaterial is 3D only (gdim=3).")
        model = model.double().eval()
        model.requires_grad_(False)
        self._model = model
        self._gdim = gdim
        _, self._F_dim = _f_order(gdim)

        # F-vector (MFront ordering) -> 3x3 matrix basis
        basis = torch.zeros(self._F_dim, gdim, gdim, dtype=torch.float64)
        for k, ij in enumerate(_F_ORDER[gdim]):
            if ij is not None:
                basis[k, ij[0], ij[1]] = 1.0
        kap = float(kappa)

        def W_fn(z):                       # z: (F_dim,) MFront F-vector
            F = (z[:, None, None] * basis).sum(0)      # (3, 3)
            C = F.T @ F
            J = _det3(F)
            return model.energy(C) + 0.5 * kap * (J - 1.0) ** 2

        grad_fn = grad(W_fn)

        def grad_with_aux(z):
            G = grad_fn(z)
            return G, G

        # forward-over-reverse: returns (Hessian, gradient) batched over qp
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
        z = self._torch.from_numpy(np.ascontiguousarray(gradients, dtype=np.float64))
        H_t, G_t = self._flux_tangent_fn(z)
        P_flat = G_t.numpy()              # (n_qp, F_dim) = ∂W/∂F = PK1
        A_flat = H_t.numpy()              # (n_qp, F_dim, F_dim) = dP/dF

        bad = ~(np.isfinite(P_flat).all(axis=1) & np.isfinite(A_flat).all(axis=(1, 2)))
        if bad.any():
            q = int(np.argmax(bad))
            logger.error("Non-finite CANN response at %d/%d qp(s); first bad qp %d: F=%s",
                         int(bad.sum()), n_qp, q,
                         np.array2string(gradients[q], precision=4))

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
