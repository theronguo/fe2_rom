"""Parameter sensitivities of the NN effective energy — the load-bearing
primitive for fitting :class:`~fe2_rom.nn.model.EnergyNet` weights to macro
observations (e.g. a DNS reaction curve) through a discrete adjoint.

The macro residual depends on the weights θ only through the per-quadrature-point
flux ``P = ∂W/∂z`` that the NN material feeds the weak form. The reaction-adjoint
needs, per accepted load level,

    cᵀ ∂R/∂θ = Σ_q (∂P_q/∂θ)ᵀ a_q ,

where ``c`` is the co-state field and ``a_q`` is its quadrature-weighted gradient
mapped into flux space at quadrature point ``q``. :func:`flux_weight_vjp` returns
exactly this contraction as a flat vector over the trainable parameters — one
extra differentiation of the same graph the material already builds for
``P = grad_z W``, now also w.r.t. θ.

The reference-state correction (``W0``/``g0`` in :meth:`EnergyNet.reference_terms`)
itself depends on θ, so it is rebuilt with ``create_graph=True`` here — the
gradient is taken w.r.t. the *deployed, corrected* model, matching what the macro
solver evaluates.

``params_to_vector`` / ``set_params_from_vector`` give scipy a flat float64 view
of the trainable weights (deterministic order = ``model.named_parameters()``),
and define the column order of the returned VJP.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.func import functional_call, grad, vmap

from fe2_rom.nn.model import make_lab_energy

__all__ = [
    "trainable_param_names",
    "params_to_vector",
    "set_params_from_vector",
    "flux_weight_vjp",
]


def trainable_param_names(model: nn.Module) -> list[str]:
    """Parameter names in the canonical (vector/VJP) order.

    All :meth:`nn.Module.parameters` are fit; the calibration *buffers*
    (``x_mean`` / ``x_std`` / ``W_scale`` / ``x_ref``) are not parameters and are
    intentionally excluded. The model's ``requires_grad`` flag is irrelevant —
    the VJP differentiates these weights functionally — so the NN material may
    stay frozen for the forward pass."""
    return [n for n, _ in model.named_parameters()]


def params_to_vector(model: nn.Module) -> np.ndarray:
    """Flat float64 copy of the weights (scipy ``x0``)."""
    ps = list(model.parameters())
    return torch.nn.utils.parameters_to_vector(ps).detach().cpu().numpy()


def set_params_from_vector(model: nn.Module, vec) -> None:
    """In-place write of a flat vector back into the weights."""
    ps = list(model.parameters())
    t = torch.as_tensor(np.asarray(vec, dtype=np.float64))
    torch.nn.utils.vector_to_parameters(t, ps)


class _LabEnergyModule(nn.Module):
    """Wrap an :class:`EnergyNet` so ``forward(z)`` is the lab-frame scalar
    ``W(z)`` — lets :func:`torch.func.functional_call` substitute parameters so
    we can differentiate ``W`` w.r.t. θ functionally. The reference correction is
    rebuilt inside ``forward`` so it tracks the substituted parameters."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, z):
        # reference_terms uses requires_grad_()/autograd.grad, which functorch
        # transforms forbid — recompute the same (W0, g0) with torch.func.grad
        # so the correction stays differentiable w.r.t. the substituted θ.
        m = self.model
        x_ref = m.x_ref
        W0 = m.raw_energy_x(x_ref)
        g0 = grad(m.raw_energy_x)(x_ref)
        return make_lab_energy(m, ref=(W0, g0))(z)


def flux_weight_vjp(model, z_batch, a_batch) -> np.ndarray:
    """Return ``Σ_q (∂P_q/∂θ)ᵀ a_q`` as a flat vector over trainable weights.

    Parameters
    ----------
    model : EnergyNet
        The effective-energy network (only ``requires_grad=True`` parameters
        contribute, in :func:`trainable_param_names` order).
    z_batch : array (n_q, z_dim)
        Per-qp gradients in the MFront packing (same layout the material's
        ``integrate`` consumes).
    a_batch : array (n_q, flux_dim)
        Per-qp cotangents in flux space (the quadrature-weighted co-state
        gradient; ``flux_dim == z_dim`` since ``P`` and ``z`` are conjugate).

    Returns
    -------
    np.ndarray, shape (n_params,)
    """
    wrapper = _LabEnergyModule(model)
    params = {n: p for n, p in wrapper.named_parameters()}
    buffers = dict(wrapper.named_buffers())

    z = torch.as_tensor(np.asarray(z_batch, dtype=np.float64))
    a = torch.as_tensor(np.asarray(a_batch, dtype=np.float64))

    def W_of(p, z_single):
        return functional_call(wrapper, (p, buffers), (z_single,))

    flux_of = grad(W_of, argnums=1)  # ∂W/∂z for a single z → (z_dim,)

    def per_qp(p, z_single, a_single):
        return (a_single * flux_of(p, z_single)).sum()

    def total(p):
        return vmap(per_qp, in_dims=(None, 0, 0))(p, z, a).sum()

    g = grad(total)(params)
    flat = torch.nn.utils.parameters_to_vector([g[n] for n in params])
    return flat.detach().cpu().numpy()
