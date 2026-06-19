"""EnergyNet — neural effective strain-energy density, shared across the
first-order (CH1), micromorphic (MM) and second-order (CH2) homogenization
layers via a single ``flavor`` switch:

* ``"ch1"`` : W̄(F̄)            — reduced input Ū
* ``"mm"``  : W̄(F̄, v, g)      — reduced input (Ū, v, g)
* ``"ch2"`` : W̄(F̄, Ḡ)         — reduced input (Ū, Ĝ)

Structure (all part of the autograd graph, so stresses/tangents derived by
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
"""
from __future__ import annotations

import json

import torch
from torch import nn

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
_ACTIVATIONS = {"softplus": nn.Softplus, "silu": nn.SiLU, "tanh": nn.Tanh}

# Subclasses register here (by "module.qualname") so EnergyNet.load / the NN
# materials can dispatch a saved checkpoint back to its own class.
_NET_REGISTRY: dict = {}


def _extra_dim(flavor: str, gdim: int, n_modes: int) -> int:
    """Reduced (MLP-input) size of the non-stretch coordinates."""
    if flavor == "ch1":
        return 0
    if flavor == "mm":
        return n_modes + n_modes * gdim
    if flavor == "ch2":
        return gdim ** 3
    raise ValueError(f"flavor must be one of {_FLAVORS}, got {flavor!r}.")


class EnergyNet(nn.Module):
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

    Custom architectures
    --------------------
    The default core is a plain MLP built by :meth:`_build_core`. To plug in your
    own network, subclass and override :meth:`_build_core` (return a module that
    maps standardised reduced input ``(…, n_in)`` → ``(…, 1)``) — input
    standardisation, the ``"mm"`` sign symmetry, the exact reference correction
    and the objective reduction in :func:`make_lab_energy` are all inherited,
    since they depend only on :meth:`_core`. If your ``__init__`` takes extra
    arguments, store them as attributes *before* ``super().__init__`` and
    override :meth:`config` so the model round-trips through ``save``/``load``::

        class MyEnergyNet(EnergyNet):
            def __init__(self, *, width=128, depth=4, **kw):
                self._width, self._depth = width, depth
                super().__init__(**kw)
            def _build_core(self):
                return MyArch(self.n_in, self._width, self._depth)
            def config(self):
                return {**super().config(), "width": self._width,
                        "depth": self._depth}

    Subclasses are auto-registered, so ``EnergyNet.load(path)`` (and the NN
    materials, which load from a path) dispatch back to the right class — provided
    the subclass module is importable/imported when loading. Otherwise pass an
    instance: ``NNCh2Material(MyEnergyNet.load(path))``.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _NET_REGISTRY[f"{cls.__module__}.{cls.__qualname__}"] = cls

    def __init__(self, gdim: int = 3, flavor: str = "mm", n_modes: int = 0,
                 hidden=(64, 64, 64), activation: str = "softplus"):
        super().__init__()
        if gdim not in (2, 3):
            raise ValueError(f"gdim must be 2 or 3, got {gdim}.")
        if flavor not in _FLAVORS:
            raise ValueError(f"flavor must be one of {_FLAVORS}, got {flavor!r}.")
        self.gdim = gdim
        self.flavor = flavor
        self.n_modes = n_modes
        self.hidden = list(hidden)
        self.activation = activation
        self.n_u = gdim * (gdim + 1) // 2
        self.extra_dim = _extra_dim(flavor, gdim, n_modes)
        self.n_in = self.n_u + self.extra_dim
        # Sign degeneracy only holds for the micromorphic enrichment amplitudes.
        self.sign_symmetry = (flavor == "mm" and n_modes > 0)

        self.mlp = self._build_core()

        # Input standardisation + output scale (identity until calibrated).
        self.register_buffer("x_mean", torch.zeros(self.n_in))
        self.register_buffer("x_std", torch.ones(self.n_in))
        self.register_buffer("W_scale", torch.ones(()))
        # Reference reduced coordinates x0 = (U=I, extras=0).
        x0 = torch.zeros(self.n_in)
        for k, (i, j) in enumerate(_U_COMPONENTS[gdim]):
            if i == j:
                x0[k] = 1.0
        self.register_buffer("x_ref", x0)

        self.double()

    # -- architecture (override for a custom network) -----------------------

    def _build_core(self) -> nn.Module:
        """Build the core network mapping standardised reduced input
        ``(…, n_in)`` → ``(…, 1)``.

        Override in a subclass to plug in a custom architecture; see the class
        docstring. The default is an MLP with widths ``self.hidden`` and
        activation ``self.activation``.
        """
        act = _ACTIVATIONS[self.activation]
        layers, n_prev = [], self.n_in
        for h in self.hidden:
            layers += [nn.Linear(n_prev, h), act()]
            n_prev = h
        layers.append(nn.Linear(n_prev, 1))
        return nn.Sequential(*layers)

    # -- core evaluations ---------------------------------------------------

    def _core(self, x: torch.Tensor) -> torch.Tensor:
        """Core network on standardised input; x shape (..., n_in) → (...)."""
        xn = (x - self.x_mean) / self.x_std
        return self.mlp(xn).squeeze(-1) * self.W_scale

    def raw_energy_x(self, x: torch.Tensor) -> torch.Tensor:
        """Energy on reduced coords, sign-symmetrised when applicable."""
        if not self.sign_symmetry:
            return self._core(x)
        u, rest = x[..., :self.n_u], x[..., self.n_u:]
        x_flip = torch.cat([u, -rest], dim=-1)
        return 0.5 * (self._core(x) + self._core(x_flip))

    def reference_terms(self, create_graph: bool = False):
        """(W0, g0) of the (symmetrised) net at x_ref.

        ``create_graph=True`` keeps the parameter dependence (use during
        training so the optimiser sees exactly the deployed, corrected model);
        ``False`` returns detached constants for frozen-weight inference.
        """
        x0 = self.x_ref.clone().requires_grad_(True)
        W0 = self.raw_energy_x(x0)
        (g0,) = torch.autograd.grad(W0, x0, create_graph=create_graph)
        if not create_graph:
            W0, g0 = W0.detach(), g0.detach()
        return W0, g0

    def energy_x(self, x: torch.Tensor, ref) -> torch.Tensor:
        """Corrected energy: exactly 0 energy/stress at x_ref.

        ``ref`` is the (W0, g0) pair from :meth:`reference_terms`.
        """
        W0, g0 = ref
        return self.raw_energy_x(x) - W0 - (g0 * (x - self.x_ref)).sum(-1)

    # -- persistence ----------------------------------------------------------

    def config(self) -> dict:
        return {"gdim": self.gdim, "flavor": self.flavor,
                "n_modes": self.n_modes, "hidden": self.hidden,
                "activation": self.activation}

    def save(self, path: str) -> None:
        torch.save({"class": f"{type(self).__module__}.{type(self).__qualname__}",
                    "config": json.dumps(self.config()),
                    "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str) -> "EnergyNet":
        """Load a checkpoint, dispatching to the saved subclass when possible.

        If the checkpoint records a registered subclass (its module is
        imported), that class is instantiated; otherwise ``cls`` is used —
        which is correct when ``load`` is called on the right class directly,
        and backward-compatible with checkpoints saved without the class key.
        """
        blob = torch.load(path, map_location="cpu", weights_only=True)
        target = _NET_REGISTRY.get(blob.get("class"), cls)
        model = target(**json.loads(blob["config"]))
        model.load_state_dict(blob["state_dict"])
        model.double()
        return model


def make_lab_energy(model: EnergyNet, ref=None):
    """Build the lab-frame scalar W(z) in the MFront packing of ``model.flavor``.

    All fluxes and tangent blocks are derivatives of this one scalar::

        ch1: P                = ∂W/∂z,          dP/dF = ∂²W/∂z²
        mm : [P | Π | Λ]      = ∂W/∂z,          3×3 tangent grid = ∂²W/∂z²
        ch2: [P | Q]          = ∂W/∂z,          2×2 tangent grid = ∂²W/∂z²

    ``ref`` defaults to detached reference terms (frozen-weight inference);
    pass ``model.reference_terms(create_graph=True)`` during training.
    """
    if ref is None:
        ref = model.reference_terms(create_graph=False)
    gdim, F_dim, flavor = model.gdim, _F_DIM[model.gdim], model.flavor

    # Basis matrices E_k with (E_k)_{ij} = 1 for slot k ↔ (i, j); the 2D
    # out-of-plane placeholder slot maps to the zero matrix (W ignores it).
    basis = torch.zeros(F_dim, gdim, gdim, dtype=torch.float64)
    for k, ij in enumerate(_F_ORDER[gdim]):
        if ij is not None:
            basis[k, ij[0], ij[1]] = 1.0

    iu = torch.tensor([ij[0] for ij in _U_COMPONENTS[gdim]])
    ju = torch.tensor([ij[1] for ij in _U_COMPONENTS[gdim]])

    if flavor == "ch2":
        def W_fn(z: torch.Tensor) -> torch.Tensor:
            F = (z[:F_dim, None, None] * basis).sum(0)
            R, U = polar(F)
            u_vec = U[iu, ju]
            G = z[F_dim:].reshape(gdim, gdim, gdim)
            # Rotate the spatial index into the reference frame (objectivity),
            # then symmetrise (J, K) so ∂W/∂Ḡ is exactly symmetric there.
            Ghat = torch.einsum("ip,ijk->pjk", R, G)
            Ghat = 0.5 * (Ghat + Ghat.transpose(-1, -2))
            x = torch.cat([u_vec, Ghat.reshape(-1)])
            return model.energy_x(x, ref)
    else:  # ch1 / mm — extras pass through unchanged
        def W_fn(z: torch.Tensor) -> torch.Tensor:
            F = (z[:F_dim, None, None] * basis).sum(0)
            U = right_stretch(F)
            u_vec = U[iu, ju]
            x = torch.cat([u_vec, z[F_dim:]])
            return model.energy_x(x, ref)

    return W_fn
