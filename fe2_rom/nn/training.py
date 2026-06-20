"""Optax training of :class:`~fe2_rom.nn.model.EnergyNet` with a Sobolev loss,
shared across the CH1 / MM / CH2 flavors.

The loss matches the effective energy *and* its first derivatives — by
Hellmann–Feynman the homogenized stresses are exactly the partial derivatives
of W̄ at the converged micro state::

    ch1:  P̄ = ∂W̄/∂F̄
    mm :  P̄ = ∂W̄/∂F̄,   Πᵢ = ∂W̄/∂vᵢ,   Λᵢ = ∂W̄/∂gᵢ
    ch2:  P̄ = ∂W̄/∂F̄,   Q̄  = ∂W̄/∂Ḡ

so every sample contributes ``1 + (flux dimensions)`` scalar targets. Stresses
are what enter the macro residual; matching them directly gives far better
flux/tangent accuracy than an energy-only fit.

Dataset ``.npz`` layout (n samples) by flavor::

    common:  F (n, gdim, gdim)    sampled F̄
             W (n,)               effective energy density W̄
             P (n, gdim, gdim)    effective 1st PK stress P̄
    mm:      v (n, N)             mode amplitudes
             g (n, N, gdim)       mode-gradient amplitudes
             Pi (n, N)            effective Π
             Lambda (n, N, gdim)  effective Λ
    ch2:     G (n, gdim, gdim, gdim)   gradient Ḡ (symmetric in last two indices)
             Q (n, gdim, gdim, gdim)   double stress Q̄

Everything runs in float64 — the deployed material feeds consistent tangents to
SNES. Train with :func:`train_energy`, which calibrates the input/output
normalisation, runs Adam with a reduce-on-plateau schedule, and returns the
trained (immutable) model plus a loss history.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from fe2_rom.nn.model import (
    EnergyNet, make_lab_energy, cast_f64, _F_DIM, _F_ORDER, _U_COMPONENTS,
)


# ---------------------------------------------------------------------------
# Flux layout (order of the ∂W/∂z target vector, matching make_lab_energy)
# ---------------------------------------------------------------------------

def flux_spec(model: EnergyNet) -> list[tuple[str, int]]:
    """Ordered ``[(name, dim), …]`` split of the gradient target ∂W/∂z.

    Zero-dimensional fluxes (e.g. mm with ``n_modes == 0``) are dropped.
    """
    F_dim, gdim, N = _F_DIM[model.gdim], model.gdim, model.n_modes
    if model.flavor == "ch1":
        spec = [("P", F_dim)]
    elif model.flavor == "mm":
        spec = [("P", F_dim), ("Pi", N), ("Lambda", N * gdim)]
    elif model.flavor == "ch2":
        spec = [("P", F_dim), ("Q", gdim ** 3)]
    else:  # pragma: no cover — EnergyNet validates flavor
        raise ValueError(model.flavor)
    return [(name, dim) for name, dim in spec if dim > 0]


# ---------------------------------------------------------------------------
# Data packing
# ---------------------------------------------------------------------------

def _mat_to_fvec_batch(T: np.ndarray, gdim: int) -> np.ndarray:
    """(n, gdim, gdim) → (n, F_dim) in MFront ordering (zero placeholder)."""
    n = T.shape[0]
    out = np.zeros((n, _F_DIM[gdim]), dtype=np.float64)
    for k, ij in enumerate(_F_ORDER[gdim]):
        if ij is not None:
            out[:, k] = T[:, ij[0], ij[1]]
    return out


def pack_dataset(data: dict, model: EnergyNet):
    """Arrays → (z, W, dWdz) float64 arrays in the MFront packing of make_lab_energy."""
    gdim, flavor = model.gdim, model.flavor
    asf = lambda a: np.asarray(a, dtype=np.float64)
    F = asf(data["F"])
    n = F.shape[0]
    z_parts = [_mat_to_fvec_batch(F, gdim)]
    g_parts = [_mat_to_fvec_batch(asf(data["P"]), gdim)]
    if flavor == "mm":
        z_parts += [asf(data["v"]), asf(data["g"]).reshape(n, -1)]
        g_parts += [asf(data["Pi"]), asf(data["Lambda"]).reshape(n, -1)]
    elif flavor == "ch2":
        z_parts += [asf(data["G"]).reshape(n, -1)]
        g_parts += [asf(data["Q"]).reshape(n, -1)]
    z = np.concatenate(z_parts, axis=1)
    dWdz = np.concatenate(g_parts, axis=1)
    return z, asf(data["W"]), dWdz


def reduced_coords(z: np.ndarray, model: EnergyNet) -> np.ndarray:
    """z (lab packing) → x (model input) for input-standardisation calibration.

    Uses the symmetric-stretch approximation (R ≈ I, so Ū = F̄ components and
    Ĝ = Ḡ) — this only sizes the per-input mean/std, so the approximation is
    harmless; the loss itself uses the exact polar reduction in make_lab_energy.
    """
    gdim, F_dim = model.gdim, _F_DIM[model.gdim]
    order = _F_ORDER[gdim]
    cols = [z[:, order.index((i, j))] for (i, j) in _U_COMPONENTS[gdim]]
    return np.concatenate([np.stack(cols, axis=1), z[:, F_dim:]], axis=1)


# ---------------------------------------------------------------------------
# Normalisation calibration
# ---------------------------------------------------------------------------

def calibrate_normalization(model: EnergyNet, z_train: np.ndarray,
                            W_train: np.ndarray) -> EnergyNet:
    """Return a copy of ``model`` with the input standardisation (``x_mean`` /
    ``x_std``) and output scale (``W_scale``) calibrated from the training split."""
    x = reduced_coords(np.asarray(z_train), model)
    x_mean = jnp.asarray(x.mean(0), dtype=jnp.float64)
    x_std = jnp.asarray(np.clip(x.std(0), 1e-8, None), dtype=jnp.float64)
    W_scale = jnp.asarray(max(float(np.asarray(W_train).std()), 1e-12),
                          dtype=jnp.float64)
    return eqx.tree_at(lambda m: (m.x_mean, m.x_std, m.W_scale),
                       model, replace=(x_mean, x_std, W_scale))


def _target_scales(model: EnergyNet, W: np.ndarray, dWdz: np.ndarray) -> dict:
    """Per-target std (energy + each flux block) so loss weights stay O(1)."""
    std = lambda a: max(float(np.asarray(a).std()), 1e-12)
    scales = {"W": std(W)}
    off = 0
    for name, dim in flux_spec(model):
        scales[name] = std(dWdz[:, off:off + dim])
        off += dim
    return scales


# ---------------------------------------------------------------------------
# Sobolev loss + training loop
# ---------------------------------------------------------------------------

def _make_loss(model: EnergyNet, scales: dict, weights: dict):
    """Build ``loss(model, z, W, dWdz) -> (total, parts)`` for the given flavor."""
    spec = flux_spec(model)
    names = ["W"] + [name for name, _ in spec]
    w = {k: float(weights.get(k, 1.0)) for k in names}
    s = {k: float(scales.get(k, 1.0)) for k in names}

    def loss(model, z, W_t, G_t):
        W_fn = make_lab_energy(model)               # ref tracks weights
        W_p, G_p = jax.vmap(jax.value_and_grad(W_fn))(z)
        mse = lambda a, b, sc: jnp.mean(((a - b) / sc) ** 2)
        loss_W = mse(W_p, W_t, s["W"])
        total = w["W"] * loss_W
        parts = {"W": loss_W}
        off = 0
        for name, dim in spec:
            sl = slice(off, off + dim)
            off += dim
            li = mse(G_p[:, sl], G_t[:, sl], s[name])
            total = total + w[name] * li
            parts[name] = li
        return total, parts

    return loss


def train_energy(model: EnergyNet, npz_path: str, *, epochs: int = 2000,
                 batch_size: int = 256, lr: float = 1e-3,
                 weights: dict | None = None, val_fraction: float = 0.1,
                 seed: int = 0, lr_factor: float = 0.5, lr_patience: int = 50,
                 verbose: bool = True):
    """Sobolev-train ``model`` on the dataset ``npz_path``.

    Adam + reduce-on-plateau (``lr_factor`` after ``lr_patience`` stagnant val
    epochs), float64 throughout. Calibrates the model's normalisation from the
    training split first.

    Returns ``(trained_model, history)`` where ``history`` is a list of
    ``{"epoch", "train_loss", "val_loss", "lr"}`` dicts.
    """
    model = cast_f64(model)
    weights = weights or {}

    with np.load(npz_path) as data:
        z, W, dWdz = pack_dataset(dict(data), model)

    n = z.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(val_fraction * n)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    model = calibrate_normalization(model, z[train_idx], W[train_idx])
    scales = _target_scales(model, W[train_idx], dWdz[train_idx])
    loss_fn = _make_loss(model, scales, weights)

    z_tr = jnp.asarray(z[train_idx]); W_tr = jnp.asarray(W[train_idx])
    G_tr = jnp.asarray(dWdz[train_idx])
    z_va = jnp.asarray(z[val_idx]); W_va = jnp.asarray(W[val_idx])
    G_va = jnp.asarray(dWdz[val_idx])

    filter_spec = model.trainable_filter()
    params, static = eqx.partition(model, filter_spec)
    optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=jnp.asarray(lr))
    opt_state = optimizer.init(params)

    @eqx.filter_jit
    def train_step(params, opt_state, z, W_t, G_t):
        def lo(p):
            return loss_fn(eqx.combine(p, static), z, W_t, G_t)
        (total, parts), grads = jax.value_and_grad(lo, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = eqx.apply_updates(params, updates)
        return params, opt_state, total

    @eqx.filter_jit
    def val_loss(params, z, W_t, G_t):
        total, _ = loss_fn(eqx.combine(params, static), z, W_t, G_t)
        return total

    n_tr = z_tr.shape[0]
    n_batches = max(1, n_tr // batch_size)
    best_val, stale, history = np.inf, 0, []

    for epoch in range(epochs):
        order = rng.permutation(n_tr)
        for b in range(n_batches):
            idx = order[b * batch_size:(b + 1) * batch_size]
            params, opt_state, _ = train_step(
                params, opt_state, z_tr[idx], W_tr[idx], G_tr[idx])

        tr = float(val_loss(params, z_tr, W_tr, G_tr))
        va = float(val_loss(params, z_va, W_va, G_va))
        cur_lr = float(opt_state.hyperparams["learning_rate"])
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va,
                        "lr": cur_lr})

        if va < best_val - 1e-12:
            best_val, stale = va, 0
        else:
            stale += 1
            if stale >= lr_patience:
                opt_state.hyperparams["learning_rate"] = jnp.asarray(
                    cur_lr * lr_factor)
                stale = 0
        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            print(f"[{epoch:5d}] train={tr:.4e} val={va:.4e} lr={cur_lr:.2e}")

    return eqx.combine(params, static), history
