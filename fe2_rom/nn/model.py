"""EnergyNet — neural effective strain-energy density, shared across the
first-order (CH1), micromorphic (MM) and second-order (CH2) homogenization
layers via a single ``flavor`` switch:

* ``"ch1"`` : W̄(F̄)            — reduced input Ū
* ``"mm"``  : W̄(F̄, v, g)      — reduced input (Ū, v, g)
* ``"ch2"`` : W̄(F̄, Ḡ)         — reduced input (Ū, Ĝ)

Structure (all part of the autodiff graph, so stresses/tangents derived by
automatic differentiation inherit every constraint exactly):

* **Objectivity**: the core network only ever sees the right stretch Ū
  (6 unique components in 3D, 3 in 2D), never F̄. The lab-frame energy is the
  composite ``W(F̄, …) = W̃(U(F̄), …)`` built by :func:`make_lab_energy` with the
  differentiable polar decomposition from :mod:`fe2_rom.nn.polar` — frame
  indifference holds by construction, and ∂W/∂F̄ automatically contains the
  rotation-related (skew) part of P̄. For ``"ch2"`` the third-order gradient is
  additionally rotated into the reference frame, ``Ĝ_pJK = R_ip Ḡ_iJK`` (R from
  F̄ = R·U) and symmetrised in its last two indices so the double stress
  ``Q̄ = ∂W/∂Ḡ`` is objective *and* exactly symmetric in (J, K) by construction.
* **Sign symmetry** (``"mm"`` only): the enrichment modes are ±-degenerate
  buckling modes, so W̃(U, v, g) = W̃(U, −v, −g) is enforced by evaluating the
  core MLP on both sign branches and averaging.
* **Exact reference state**: an analytic correction subtracts the value and
  gradient of the (symmetrised) network at the reference (U=I, extras=0), so the
  model has exactly zero energy *and* zero stresses there.

Input packing of the *reduced* coordinates x (what the core MLP consumes)::

    ch1: x = [ U_unique ]
    mm : x = [ U_unique | v_1..v_N | g_1[0..d-1], g_2[0..d-1], … ]
    ch2: x = [ U_unique | Ĝ (g³, mode/row-major, last-two-symmetric) ]

with ``U_unique = (U00, U11, U22, U01, U02, U12)`` in 3D, ``(U00, U11, U01)`` in
2D.

The packed lab-frame coordinate z consumed by :func:`make_lab_energy` follows the
MFront / :mod:`fe2_rom.{ch1,mm,ch2}.material` conventions so gradient/Hessian
slices line up with the QuadratureMap layout without any reordering::

    ch1: z = [ F_vec (F_dim) ]
    mm : z = [ F_vec (F_dim) | v (N) | g (N*gdim, mode-major) ]
    ch2: z = [ F_vec (F_dim) | G_vec (gdim³, row-major) ]

Implementation note: :class:`EnergyNet` is an :mod:`equinox` module (a frozen
pytree), so "setting weights" produces a *new* model rather than mutating in
place — the NN materials and :class:`~fe2_rom.fit.MacroFit` swap the model
object via ``update_model`` / ``set_params_from_vector``.
"""
from __future__ import annotations

import json
from typing import Callable

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import equinox as eqx

from fe2_rom.nn.polar import right_stretch, polar

# MFront F-vector ordering — keep in sync with fe2_rom/{ch1,mm,ch2}/material.py
# (duplicated here so fe2_rom.nn stays importable without dolfinx_materials).
_F_ORDER_2D: tuple = ((0, 0), (1, 1), None, (0, 1), (1, 0))
_F_ORDER_3D: tuple = (
    (0, 0), (1, 1), (2, 2),
    (0, 1), (1, 0),
    (0, 2), (2, 0),
    (1, 2), (2, 1),
)
_F_ORDER = {2: _F_ORDER_2D, 3: _F_ORDER_3D}
_F_DIM = {2: 5, 3: 9}

# Unique (upper-triangle) components of the symmetric stretch U.
_U_COMPONENTS = {
    2: ((0, 0), (1, 1), (0, 1)),
    3: ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)),
}

_FLAVORS = ("ch1", "mm", "ch2")
_ACTIVATIONS: dict[str, Callable] = {
    "softplus": jax.nn.softplus,
    "silu": jax.nn.silu,
    "tanh": jnp.tanh,
}

# Subclasses register here (by "module.qualname") so EnergyNet.load / the NN
# materials can dispatch a saved checkpoint back to its own class.
_NET_REGISTRY: dict = {}

_is_array = eqx.is_inexact_array


def cast_f64(tree):
    """Cast every inexact-array leaf of ``tree`` to float64 (leaving the rest)."""
    return jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if _is_array(x) else x, tree)


def _extra_dim(flavor: str, gdim: int, n_modes: int) -> int:
    """Reduced (MLP-input) size of the non-stretch coordinates."""
    if flavor == "ch1":
        return 0
    if flavor == "mm":
        return n_modes + n_modes * gdim
    if flavor == "ch2":
        return gdim ** 3
    raise ValueError(f"flavor must be one of {_FLAVORS}, got {flavor!r}.")


class _MLP(eqx.Module):
    """Plain MLP core mapping standardised reduced input ``(n_in,)`` → scalar."""

    layers: tuple
    out: eqx.nn.Linear
    act: Callable = eqx.field(static=True)

    def __init__(self, n_in: int, hidden, act: Callable, *, key):
        keys = jax.random.split(key, len(hidden) + 1)
        sizes = [n_in, *hidden]
        self.layers = tuple(
            eqx.nn.Linear(sizes[i], sizes[i + 1], key=keys[i])
            for i in range(len(hidden)))
        self.out = eqx.nn.Linear(hidden[-1], 1, key=keys[-1])
        self.act = act

    def __call__(self, x):
        for lin in self.layers:
            x = self.act(lin(x))
        return self.out(x)


class EnergyNet(eqx.Module):
    """Neural effective energy on the reduced, objective coordinates x.

    Parameters
    ----------
    gdim : int
        Geometric dimension (2 or 3).
    flavor : str
        ``"ch1"`` / ``"mm"`` / ``"ch2"`` — selects the extra coordinates and the
        objectivity reduction (see module docstring).
    n_modes : int
        Number of micromorphic enrichment modes (``"mm"`` only; ignored
        otherwise).
    hidden : sequence[int]
        Hidden layer widths of the default MLP core.
    activation : str
        ``"softplus"`` / ``"silu"`` / ``"tanh"`` for the default MLP core.
    key : jax.Array, optional
        PRNG key for the default core initialisation (deterministic default).

    Custom architectures
    --------------------
    The default core is a plain MLP built by :meth:`_build_core`. To plug in your
    own network, subclass and override :meth:`_build_core` (return an
    :class:`equinox.Module` mapping standardised reduced input ``(n_in,)`` →
    ``(1,)``) — input standardisation, the ``"mm"`` sign symmetry, the exact
    reference correction and the objective reduction in :func:`make_lab_energy`
    are all inherited, since they depend only on :meth:`_core`. If your core has
    non-trainable buffer arrays, expose them through :meth:`freeze_buffers` so
    they are excluded from the fit. If your ``__init__`` takes extra arguments,
    store them as attributes *before* ``super().__init__`` and override
    :meth:`config` so the model round-trips through ``save``/``load``.

    Subclasses are auto-registered, so ``EnergyNet.load(path)`` (and the NN
    materials, which load from a path) dispatch back to the right class — provided
    the subclass module is importable/imported when loading. Otherwise pass an
    instance: ``NNCh2Material(MyEnergyNet.load(path))``.
    """

    mlp: eqx.Module
    # Input standardisation + output scale (identity until calibrated) and the
    # reference reduced coordinates x0 = (U=I, extras=0). These are *buffers* —
    # excluded from the trainable parameter vector (see :meth:`trainable_filter`).
    x_mean: jax.Array
    x_std: jax.Array
    W_scale: jax.Array
    x_ref: jax.Array

    gdim: int = eqx.field(static=True)
    flavor: str = eqx.field(static=True)
    n_modes: int = eqx.field(static=True)
    hidden: tuple = eqx.field(static=True)
    activation: str = eqx.field(static=True)
    n_u: int = eqx.field(static=True)
    extra_dim: int = eqx.field(static=True)
    n_in: int = eqx.field(static=True)
    sign_symmetry: bool = eqx.field(static=True)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _NET_REGISTRY[f"{cls.__module__}.{cls.__qualname__}"] = cls

    def __init__(self, gdim: int = 3, flavor: str = "mm", n_modes: int = 0,
                 hidden=(64, 64, 64), activation: str = "softplus", *, key=None):
        if gdim not in (2, 3):
            raise ValueError(f"gdim must be 2 or 3, got {gdim}.")
        if flavor not in _FLAVORS:
            raise ValueError(f"flavor must be one of {_FLAVORS}, got {flavor!r}.")
        self.gdim = gdim
        self.flavor = flavor
        self.n_modes = n_modes
        self.hidden = tuple(hidden)
        self.activation = activation
        self.n_u = gdim * (gdim + 1) // 2
        self.extra_dim = _extra_dim(flavor, gdim, n_modes)
        self.n_in = self.n_u + self.extra_dim
        # Sign degeneracy only holds for the micromorphic enrichment amplitudes.
        self.sign_symmetry = (flavor == "mm" and n_modes > 0)

        if key is None:
            key = jax.random.PRNGKey(0)
        self.mlp = cast_f64(self._build_core(key))

        self.x_mean = jnp.zeros(self.n_in)
        self.x_std = jnp.ones(self.n_in)
        self.W_scale = jnp.ones(())
        # Reference reduced coordinates x0 = (U=I, extras=0).
        x0 = jnp.zeros(self.n_in)
        for k, (i, j) in enumerate(_U_COMPONENTS[gdim]):
            if i == j:
                x0 = x0.at[k].set(1.0)
        self.x_ref = x0

    # -- architecture (override for a custom network) -----------------------

    def _build_core(self, key) -> eqx.Module:
        """Build the core network mapping standardised reduced input ``(n_in,)``
        → ``(1,)``.

        Override in a subclass to plug in a custom architecture; see the class
        docstring. The default is an MLP with widths ``self.hidden`` and
        activation ``self.activation``.
        """
        return _MLP(self.n_in, self.hidden, _ACTIVATIONS[self.activation], key=key)

    # -- core evaluations ---------------------------------------------------

    def _core(self, x):
        """Core network on standardised input; single x of shape (n_in,) → scalar."""
        xn = (x - self.x_mean) / self.x_std
        return self.mlp(xn).squeeze(-1) * self.W_scale

    def raw_energy_x(self, x):
        """Energy on reduced coords, sign-symmetrised when applicable."""
        if not self.sign_symmetry:
            return self._core(x)
        u, rest = x[..., :self.n_u], x[..., self.n_u:]
        x_flip = jnp.concatenate([u, -rest], axis=-1)
        return 0.5 * (self._core(x) + self._core(x_flip))

    def reference_terms(self):
        """(W0, g0) of the (symmetrised) net at x_ref.

        Everything is functional in JAX, so these track the current weights
        automatically when differentiated w.r.t. them (training / fit) and are
        plain constants otherwise (frozen-weight inference)."""
        W0, g0 = jax.value_and_grad(self.raw_energy_x)(self.x_ref)
        return W0, g0

    def energy_x(self, x, ref):
        """Corrected energy: exactly 0 energy/stress at x_ref.

        ``ref`` is the (W0, g0) pair from :meth:`reference_terms`.
        """
        W0, g0 = ref
        return self.raw_energy_x(x) - W0 - (g0 * (x - self.x_ref)).sum(-1)

    # -- trainable / buffer partition ---------------------------------------

    def trainable_filter(self):
        """Bool pytree (model-shaped): ``True`` on trainable leaves, ``False`` on
        calibration buffers — the partition the fit / training optimise over.

        Equivalent to torch's ``model.named_parameters()`` selection (the
        ``x_mean`` / ``x_std`` / ``W_scale`` / ``x_ref`` buffers are excluded).
        """
        spec = jax.tree_util.tree_map(_is_array, self)  # all arrays True
        spec = eqx.tree_at(
            lambda m: (m.x_mean, m.x_std, m.W_scale, m.x_ref),
            spec, replace=(False, False, False, False))
        if hasattr(self.mlp, "freeze_buffers"):
            spec = eqx.tree_at(lambda m: m.mlp, spec,
                               replace=self.mlp.freeze_buffers(spec.mlp))
        return spec

    # -- persistence ----------------------------------------------------------

    def config(self) -> dict:
        return {"gdim": self.gdim, "flavor": self.flavor,
                "n_modes": self.n_modes, "hidden": list(self.hidden),
                "activation": self.activation}

    def save(self, path: str) -> None:
        meta = json.dumps({
            "class": f"{type(self).__module__}.{type(self).__qualname__}",
            "config": self.config()}).encode("utf-8")
        with open(path, "wb") as f:
            f.write(len(meta).to_bytes(8, "little"))
            f.write(meta)
            eqx.tree_serialise_leaves(f, self)

    @classmethod
    def load(cls, path: str) -> "EnergyNet":
        """Load a checkpoint, dispatching to the saved subclass when possible.

        If the checkpoint records a registered subclass (its module is
        imported), that class is instantiated; otherwise ``cls`` is used —
        correct when ``load`` is called on the right class directly.
        """
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            blob = json.loads(f.read(n).decode("utf-8"))
            target = _NET_REGISTRY.get(blob.get("class"), cls)
            skeleton = target(**blob["config"])
            model = eqx.tree_deserialise_leaves(f, skeleton)
        return cast_f64(model)


def make_lab_energy(model: EnergyNet, ref=None):
    """Build the lab-frame scalar W(z) in the MFront packing of ``model.flavor``.

    All fluxes and tangent blocks are derivatives of this one scalar (single z)::

        ch1: P                = ∂W/∂z,          dP/dF = ∂²W/∂z²
        mm : [P | Π | Λ]      = ∂W/∂z,          3×3 tangent grid = ∂²W/∂z²
        ch2: [P | Q]          = ∂W/∂z,          2×2 tangent grid = ∂²W/∂z²

    ``ref`` defaults to ``model.reference_terms()``; pass it explicitly only to
    reuse a precomputed correction.
    """
    if ref is None:
        ref = model.reference_terms()
    gdim, F_dim, flavor = model.gdim, _F_DIM[model.gdim], model.flavor

    # Basis matrices E_k with (E_k)_{ij} = 1 for slot k ↔ (i, j); the 2D
    # out-of-plane placeholder slot maps to the zero matrix (W ignores it).
    import numpy as _np
    basis_np = _np.zeros((F_dim, gdim, gdim))
    for k, ij in enumerate(_F_ORDER[gdim]):
        if ij is not None:
            basis_np[k, ij[0], ij[1]] = 1.0
    basis = jnp.asarray(basis_np)

    iu = jnp.asarray([ij[0] for ij in _U_COMPONENTS[gdim]])
    ju = jnp.asarray([ij[1] for ij in _U_COMPONENTS[gdim]])

    if flavor == "ch2":
        def W_fn(z):
            F = (z[:F_dim, None, None] * basis).sum(0)
            R, U = polar(F)
            u_vec = U[iu, ju]
            G = z[F_dim:].reshape(gdim, gdim, gdim)
            # Rotate the spatial index into the reference frame (objectivity),
            # then symmetrise (J, K) so ∂W/∂Ḡ is exactly symmetric there.
            Ghat = jnp.einsum("ip,ijk->pjk", R, G)
            Ghat = 0.5 * (Ghat + jnp.swapaxes(Ghat, -1, -2))
            x = jnp.concatenate([u_vec, Ghat.reshape(-1)])
            return model.energy_x(x, ref)
    else:  # ch1 / mm — extras pass through unchanged
        def W_fn(z):
            F = (z[:F_dim, None, None] * basis).sum(0)
            U = right_stretch(F)
            u_vec = U[iu, ju]
            x = jnp.concatenate([u_vec, z[F_dim:]])
            return model.energy_x(x, ref)

    return W_fn
