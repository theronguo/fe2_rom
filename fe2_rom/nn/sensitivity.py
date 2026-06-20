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
extra differentiation of the same scalar the material already differentiates for
``P = grad_z W``, now also w.r.t. θ.

Everything is functional in JAX: the reference-state correction
(``W0``/``g0`` in :meth:`EnergyNet.reference_terms`) is recomputed inside
:func:`make_lab_energy` from the substituted weights, so the gradient is taken
w.r.t. the *deployed, corrected* model — exactly what the macro solver evaluates.

``params_to_vector`` / ``set_params_from_vector`` give scipy a flat float64 view
of the trainable weights (deterministic leaf order = ``model.trainable_filter()``
partition), and define the column order of the returned VJP. Because an
:class:`EnergyNet` is an immutable equinox pytree, ``set_params_from_vector``
returns a *new* model rather than mutating in place.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from fe2_rom.nn.model import make_lab_energy

__all__ = [
    "trainable_param_names",
    "params_to_vector",
    "set_params_from_vector",
    "flux_weight_vjp",
]


def _trainable_partition(model):
    """``(params, static)`` split on the model's trainable/buffer filter."""
    return eqx.partition(model, model.trainable_filter())


def trainable_param_names(model) -> list[str]:
    """Key-path strings of the trainable leaves, in canonical (vector/VJP) order.

    Diagnostic only — the calibration buffers are excluded by
    :meth:`EnergyNet.trainable_filter`, mirroring torch's ``named_parameters``
    selection."""
    params, _ = _trainable_partition(model)
    paths = jax.tree_util.tree_leaves_with_path(params)
    return [jax.tree_util.keystr(p) for p, _ in paths]


def params_to_vector(model) -> np.ndarray:
    """Flat float64 copy of the trainable weights (scipy ``x0``)."""
    params, _ = _trainable_partition(model)
    leaves = jax.tree_util.tree_leaves(params)
    if not leaves:
        return np.zeros(0)
    return np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p in leaves])


def set_params_from_vector(model, vec):
    """Return a copy of ``model`` with the trainable weights set from ``vec``.

    (Equinox models are immutable, so this does **not** mutate ``model``.)
    """
    params, static = _trainable_partition(model)
    leaves, treedef = jax.tree_util.tree_flatten(params)
    vec = np.asarray(vec, dtype=np.float64)
    new_leaves, off = [], 0
    for leaf in leaves:
        n = leaf.size
        new_leaves.append(jnp.asarray(
            vec[off:off + n].reshape(leaf.shape), dtype=jnp.float64))
        off += n
    params = jax.tree_util.tree_unflatten(treedef, new_leaves)
    return eqx.combine(params, static)


def flux_weight_vjp(model, z_batch, a_batch) -> np.ndarray:
    """Return ``Σ_q (∂P_q/∂θ)ᵀ a_q`` as a flat vector over trainable weights.

    Parameters
    ----------
    model : EnergyNet
        The effective-energy network (the trainable leaves contribute, in
        :func:`params_to_vector` order).
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
    params, static = _trainable_partition(model)
    z = jnp.asarray(np.asarray(z_batch, dtype=np.float64))
    a = jnp.asarray(np.asarray(a_batch, dtype=np.float64))

    def total(params):
        m = eqx.combine(params, static)
        W_fn = make_lab_energy(m)              # ref tracks substituted weights
        flux = jax.vmap(jax.grad(W_fn))(z)     # (n_q, z_dim) = ∂W/∂z per qp
        return jnp.sum(a * flux)

    grads = jax.grad(total)(params)
    leaves = jax.tree_util.tree_leaves(grads)
    if not leaves:
        return np.zeros(0)
    return np.concatenate([np.asarray(g, dtype=np.float64).ravel() for g in leaves])
