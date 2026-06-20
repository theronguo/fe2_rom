"""Constitutive Artificial Neural Network (Linka et al., JCP 429 (2021) 110010,
§2.2) implemented as an :class:`~fe2_rom.nn.model.EnergyNet` subclass.

The generalized-invariant strain-energy *architecture* of the paper is supplied
through the :meth:`~fe2_rom.nn.model.EnergyNet._build_core` hook, so the CANN
reuses the EnergyNet scaffolding (objective reduced coordinates ``x = Ū`` from
the polar decomposition, input standardisation, save/load) and only specifies
its own network here in the validation folder.

The core (:class:`_InvariantEnergy`) realises Eqs. (13)–(15):

* generalized structure tensors ``L̄_r = Σ_j w_{rj} L_{rj}`` with the isotropic
  ``L_{r0}=⅓I`` and learnable simplex weights ``w_{rj}`` (sigmoid + normalise) —
  the preferred-direction structure tensors ``L_{rj}`` are **provided by the
  user** (``structure_tensors``);
* generalized invariants ``Ī_r=tr[C L̄_r]``, ``J̄_r=tr[(det C·C⁻¹) L̄_r]`` and
  ``III_C=det C`` (dropped when ``incompressible``);
* the two-step network of Fig. 1 / Fig. B2: a sub-ANN per invariant feeding a
  final strain-energy sub-ANN ``Psi`` (softplus throughout).

Since EnergyNet hands the core the reduced stretch coordinates ``x = Ū``
(``C = Ū·Ū``), the core reconstructs ``C`` and evaluates the invariants — making
the model a valid EnergyNet (``make_lab_energy`` → first Piola, objective). For
the paper's incompressible formulation we additionally expose ``energy(C)`` and
``stress_S(C) = 2 ∂Ψ/∂C`` (Eq. 2) directly from ``C`` (bypassing the EnergyNet
reference correction, which would force the — physically non-zero, hydrostatic —
incompressible reference stress to vanish; the pressure term handles
equilibrium at deployment).

JAX / equinox port: an :class:`~fe2_rom.nn.model.EnergyNet` is an immutable
pytree, so calibration helpers (:meth:`CANN.set_input_norm`) return a *new*
model rather than mutating in place.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from fe2_rom.nn.model import EnergyNet, _U_COMPONENTS

_ACTIVATIONS: dict[str, Callable] = {
    "softplus": jax.nn.softplus, "silu": jax.nn.silu, "tanh": jnp.tanh}

_is_array = eqx.is_inexact_array


def _det3(M):
    """Determinant of a 3×3 (batched ``(…,3,3)``) — explicit polynomial form.

    Used instead of ``jnp.linalg.det`` because the latter's forward-over-reverse
    second derivative (needed for the macro tangent ∂²Ψ/∂F²) can return NaN; the
    polynomial form is smooth everywhere.
    """
    return (M[..., 0, 0] * (M[..., 1, 1] * M[..., 2, 2] - M[..., 1, 2] * M[..., 2, 1])
            - M[..., 0, 1] * (M[..., 1, 0] * M[..., 2, 2] - M[..., 1, 2] * M[..., 2, 0])
            + M[..., 0, 2] * (M[..., 1, 0] * M[..., 2, 1] - M[..., 1, 1] * M[..., 2, 0]))


def _adjugate3(M):
    """Adjugate of a 3×3 ``(…,3,3)`` — ``adj(M) = det(M) · M⁻¹`` (polynomial,
    so its 2nd derivative is finite, unlike ``det(M)·jnp.linalg.inv(M)``)."""
    a00 = M[..., 1, 1] * M[..., 2, 2] - M[..., 1, 2] * M[..., 2, 1]
    a01 = M[..., 0, 2] * M[..., 2, 1] - M[..., 0, 1] * M[..., 2, 2]
    a02 = M[..., 0, 1] * M[..., 1, 2] - M[..., 0, 2] * M[..., 1, 1]
    a10 = M[..., 1, 2] * M[..., 2, 0] - M[..., 1, 0] * M[..., 2, 2]
    a11 = M[..., 0, 0] * M[..., 2, 2] - M[..., 0, 2] * M[..., 2, 0]
    a12 = M[..., 0, 2] * M[..., 1, 0] - M[..., 0, 0] * M[..., 1, 2]
    a20 = M[..., 1, 0] * M[..., 2, 1] - M[..., 1, 1] * M[..., 2, 0]
    a21 = M[..., 0, 1] * M[..., 2, 0] - M[..., 0, 0] * M[..., 2, 1]
    a22 = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]
    r0 = jnp.stack([a00, a01, a02], axis=-1)
    r1 = jnp.stack([a10, a11, a12], axis=-1)
    r2 = jnp.stack([a20, a21, a22], axis=-1)
    return jnp.stack([r0, r1, r2], axis=-2)


def _glorot_linear(in_f, out_f, key):
    """``eqx.nn.Linear`` with Glorot-uniform weights and zero bias (Table B1)."""
    lin = eqx.nn.Linear(in_f, out_f, key=key)
    W = jax.nn.initializers.glorot_uniform()(key, (out_f, in_f), jnp.float64)
    b = jnp.zeros((out_f,), jnp.float64)
    return eqx.tree_at(lambda l: (l.weight, l.bias), lin, (W, b))


def _to_hashable(structure_tensors):
    """Nested tuples (hashable → ok as an equinox static field)."""
    out = []
    for L in structure_tensors:
        if L is None:
            out.append(None)
        else:
            out.append(tuple(map(tuple, np.asarray(L, dtype=float).reshape(-1, 3))))
    return tuple(out)


def _struct_array(L, gdim):
    if L is None:
        return jnp.zeros((0, gdim, gdim))
    return jnp.asarray(np.asarray(L, dtype=float).reshape(-1, gdim, gdim))


class _FeatMLP(eqx.Module):
    """A per-invariant sub-ANN (Linear+act stack, no output layer)."""
    layers: tuple
    act: Callable = eqx.field(static=True)

    def __call__(self, x):
        for lin in self.layers:
            x = self.act(lin(x))
        return x


class _Psi(eqx.Module):
    """Final strain-energy neuron: ``Linear(.,1)`` then optional activation."""
    lin: eqx.nn.Linear
    act: Callable | None = eqx.field(static=True)

    def __call__(self, x):
        y = self.lin(x)
        return y if self.act is None else self.act(y)


class _InvariantEnergy(eqx.Module):
    """Strain-energy core (Eqs. 13–15, Fig. B2). Maps EnergyNet's reduced
    stretch coordinates ``x = Ū`` ``(n_u,)`` (single sample) to Ψ ``(1,)``."""

    subs: tuple                 # one _FeatMLP per invariant   (trainable)
    psi: _Psi                   # final energy network         (trainable)
    w_raw: tuple                # per-r simplex logits or None (trainable)
    _Ubasis: jax.Array          # buffers below
    _iso: jax.Array
    L_pref: tuple               # tuple of (J_r, gdim, gdim) arrays
    inv_mean: jax.Array
    inv_std: jax.Array
    gdim: int = eqx.field(static=True)
    incompressible: bool = eqx.field(static=True)
    R: int = eqx.field(static=True)
    n_inv: int = eqx.field(static=True)
    Jr: tuple = eqx.field(static=True)

    def __init__(self, gdim, structure_tensors, hidden, incompressible,
                 activation, psi_activation, *, key):
        self.gdim = gdim
        self.incompressible = incompressible
        self.R = len(structure_tensors)
        act = _ACTIVATIONS[activation]

        # symmetric basis to rebuild U from its unique components Ū
        ucomp = _U_COMPONENTS[gdim]
        basis = np.zeros((len(ucomp), gdim, gdim))
        for k, (i, j) in enumerate(ucomp):
            basis[k, i, j] = 1.0
            basis[k, j, i] = 1.0
        self._Ubasis = jnp.asarray(basis)

        # generalized structure tensors (Eq. 15): isotropic L_{r0}=⅓I + weights
        self._iso = jnp.eye(gdim) / gdim
        L_pref, Jr, w_raw = [], [], []
        for L in structure_tensors:
            Lpref = _struct_array(L, gdim)
            L_pref.append(Lpref)
            Jr.append(int(Lpref.shape[0]))
            w_raw.append(jnp.zeros(Lpref.shape[0] + 1) if Lpref.shape[0] > 0 else None)
        self.L_pref = tuple(L_pref)
        self.Jr = tuple(Jr)
        self.w_raw = tuple(w_raw)

        self.n_inv = 2 * self.R + (0 if incompressible else 1)

        # invariant function sub-ANNs (one per invariant) + strain energy Psi
        keys = jax.random.split(key, self.n_inv + 1)
        subs = []
        for s in range(self.n_inv):
            kk = jax.random.split(keys[s], len(hidden))
            layers, n_prev = [], 1
            for li, h in enumerate(hidden):
                layers.append(_glorot_linear(n_prev, h, kk[li]))
                n_prev = h
            subs.append(_FeatMLP(tuple(layers), act))
        self.subs = tuple(subs)
        self.psi = _Psi(_glorot_linear(self.n_inv * hidden[-1], 1, keys[-1]),
                        act if psi_activation else None)

        # invariant input standardisation (absorbed by the chain rule)
        self.inv_mean = jnp.zeros(self.n_inv)
        self.inv_std = jnp.ones(self.n_inv)

    # buffers excluded from the trainable parameter vector
    def freeze_buffers(self, spec):
        n = 4 + len(self.L_pref)
        return eqx.tree_at(
            lambda m: (m._Ubasis, m._iso, m.inv_mean, m.inv_std, *m.L_pref),
            spec, replace=(False,) * n)

    # -- Eq. (15) -----------------------------------------------------------
    def structure_tensors(self):
        Ls = []
        for r in range(self.R):
            if self.Jr[r] == 0:
                Ls.append(self._iso)
                continue
            w = jax.nn.sigmoid(self.w_raw[r])
            w = w / w.sum()
            stack = jnp.concatenate([self._iso[None], self.L_pref[r]], 0)
            Ls.append((w[:, None, None] * stack).sum(0))
        return Ls

    # -- Eq. (14) -----------------------------------------------------------
    def invariants_from_C(self, C):
        adj = _adjugate3(C)               # det(C)·C⁻¹, smooth (hessian-safe)
        cols = []
        for L in self.structure_tensors():
            cols.append(jnp.einsum("...ij,ji->...", C, L))      # Ī_r = tr[C L̄_r]
            cols.append(jnp.einsum("...ij,ji->...", adj, L))    # J̄_r = tr[adj(C) L̄_r]
        if not self.incompressible:
            cols.append(_det3(C))
        return jnp.stack(cols, axis=-1)

    def energy_from_C(self, C):
        """Single C ``(gdim, gdim)`` → Ψ ``(1,)``."""
        inv = (self.invariants_from_C(C) - self.inv_mean) / self.inv_std
        feats = [self.subs[k](inv[k:k + 1]) for k in range(self.n_inv)]
        return self.psi(jnp.concatenate(feats, axis=-1))      # (1,)

    # EnergyNet contract: standardized reduced coords x = Ū -> Ψ (1,)
    def __call__(self, x):
        U = jnp.einsum("k,kij->ij", x, self._Ubasis)
        return self.energy_from_C(U @ U)


class CANN(EnergyNet):
    """CANN as an EnergyNet (architecture via the ``_build_core`` hook).

    Parameters
    ----------
    gdim : int
        Geometric dimension (2 or 3).
    structure_tensors : list | None
        Per generalized structure tensor ``r``: the user's preferred-direction
        tensors ``L_{rj}`` as ``(J_r, 3, 3)`` (or a single ``(3, 3)``), or
        ``None`` for isotropic ``L̄_r = ⅓I``. ``None`` → one isotropic tensor
        (``R = 1`` isotropic material, no weights to learn).
    hidden, activation : architecture of every invariant sub-ANN (Fig. B2: (8, 8)).
    incompressible : drop the ``III_C`` invariant input (§4.1.1).
    psi_activation : apply ``activation`` to the final ``Ψ`` neuron.
    """

    _structure_tensors: tuple = eqx.field(static=True)
    _incompressible: bool = eqx.field(static=True)
    _psi_activation: bool = eqx.field(static=True)

    def __init__(self, gdim=3, structure_tensors=None, hidden=(8, 8),
                 incompressible=True, activation="softplus", psi_activation=True,
                 *, key=None):
        if structure_tensors is None:
            structure_tensors = [None]
        # stash CANN config so _build_core (called inside super().__init__) sees it
        self._structure_tensors = _to_hashable(structure_tensors)
        self._incompressible = incompressible
        self._psi_activation = psi_activation
        super().__init__(gdim=gdim, flavor="ch1", hidden=hidden,
                         activation=activation, key=key)

    def _build_core(self, key) -> eqx.Module:
        return _InvariantEnergy(
            self.gdim, list(self._structure_tensors), self.hidden,
            self._incompressible, self.activation, self._psi_activation, key=key)

    # -- CANN-specific (incompressible) API: work directly in C -------------
    @property
    def R(self):
        return self.mlp.R

    @property
    def n_inv(self):
        return self.mlp.n_inv

    def invariants(self, C):
        """Generalized invariants for a batch of ``C`` ``(…, gdim, gdim)``."""
        return self.mlp.invariants_from_C(jnp.asarray(C, jnp.float64))

    def energy(self, C):
        """Strain energy Ψ(C) for a batch of ``C`` (no EnergyNet reference
        correction — see module docstring)."""
        C = jnp.asarray(C, jnp.float64)
        e = jax.vmap(self.mlp.energy_from_C)(C) if C.ndim == 3 \
            else self.mlp.energy_from_C(C)
        return e.squeeze(-1)

    def stress_S(self, C):
        """Second Piola–Kirchhoff stress ``S = 2 ∂Ψ/∂C`` (Eq. 2, no pressure)."""
        C = jnp.asarray(C, jnp.float64)
        psi = lambda C1: self.mlp.energy_from_C(C1).squeeze(-1)
        G = jax.vmap(jax.grad(psi))(C) if C.ndim == 3 else jax.grad(psi)(C)
        return 2.0 * 0.5 * (G + jnp.swapaxes(G, -1, -2))

    def set_input_norm(self, C_train) -> "CANN":
        """Return a copy with the invariant standardisation calibrated from
        training ``C`` (immutable update)."""
        inv = jax.vmap(self.mlp.invariants_from_C)(jnp.asarray(C_train, jnp.float64))
        mean = inv.mean(0)
        std = jnp.clip(inv.std(0), 1e-8, None)
        return eqx.tree_at(lambda m: (m.mlp.inv_mean, m.mlp.inv_std),
                           self, (mean, std))

    def config(self):
        cts = [None if L is None else [list(row) for row in L]
               for L in self._structure_tensors]
        return {"gdim": self.gdim, "structure_tensors": cts,
                "hidden": list(self.hidden), "incompressible": self._incompressible,
                "activation": self.activation, "psi_activation": self._psi_activation}
