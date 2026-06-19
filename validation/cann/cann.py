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
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from fe2_rom.nn.model import EnergyNet, _U_COMPONENTS

_ACTIVATIONS = {"softplus": nn.Softplus, "silu": nn.SiLU, "tanh": nn.Tanh}


def _det3(M):
    """Determinant of a 3×3 (batched ``(…,3,3)``) — explicit polynomial form.

    Used instead of ``torch.linalg.det`` because the latter's forward-over-reverse
    second derivative (needed for the macro tangent ∂²Ψ/∂F²) returns NaN; the
    polynomial form is smooth everywhere.
    """
    return (M[..., 0, 0] * (M[..., 1, 1] * M[..., 2, 2] - M[..., 1, 2] * M[..., 2, 1])
            - M[..., 0, 1] * (M[..., 1, 0] * M[..., 2, 2] - M[..., 1, 2] * M[..., 2, 0])
            + M[..., 0, 2] * (M[..., 1, 0] * M[..., 2, 1] - M[..., 1, 1] * M[..., 2, 0]))


def _adjugate3(M):
    """Adjugate of a 3×3 ``(…,3,3)`` — ``adj(M) = det(M) · M⁻¹`` (polynomial,
    so its 2nd derivative is finite, unlike ``det(M)·torch.linalg.inv(M)``)."""
    a00 = M[..., 1, 1] * M[..., 2, 2] - M[..., 1, 2] * M[..., 2, 1]
    a01 = M[..., 0, 2] * M[..., 2, 1] - M[..., 0, 1] * M[..., 2, 2]
    a02 = M[..., 0, 1] * M[..., 1, 2] - M[..., 0, 2] * M[..., 1, 1]
    a10 = M[..., 1, 2] * M[..., 2, 0] - M[..., 1, 0] * M[..., 2, 2]
    a11 = M[..., 0, 0] * M[..., 2, 2] - M[..., 0, 2] * M[..., 2, 0]
    a12 = M[..., 0, 2] * M[..., 1, 0] - M[..., 0, 0] * M[..., 1, 2]
    a20 = M[..., 1, 0] * M[..., 2, 1] - M[..., 1, 1] * M[..., 2, 0]
    a21 = M[..., 0, 1] * M[..., 2, 0] - M[..., 0, 0] * M[..., 2, 1]
    a22 = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]
    r0 = torch.stack([a00, a01, a02], dim=-1)
    r1 = torch.stack([a10, a11, a12], dim=-1)
    r2 = torch.stack([a20, a21, a22], dim=-1)
    return torch.stack([r0, r1, r2], dim=-2)


def _as_struct(L, gdim):
    if L is None:
        return torch.zeros(0, gdim, gdim, dtype=torch.float64)
    return torch.as_tensor(np.asarray(L), dtype=torch.float64).reshape(-1, gdim, gdim)


class _InvariantEnergy(nn.Module):
    """Strain-energy core (Eqs. 13–15, Fig. B2). Maps EnergyNet's reduced
    stretch coordinates ``x = Ū`` ``(…, n_u)`` to Ψ ``(…, 1)``."""

    def __init__(self, gdim, structure_tensors, hidden, incompressible,
                 activation, psi_activation):
        super().__init__()
        self.gdim = gdim
        self.incompressible = incompressible
        self.R = len(structure_tensors)

        # symmetric basis to rebuild U from its unique components Ū
        ucomp = _U_COMPONENTS[gdim]
        basis = torch.zeros(len(ucomp), gdim, gdim, dtype=torch.float64)
        for k, (i, j) in enumerate(ucomp):
            basis[k, i, j] = 1.0
            basis[k, j, i] = 1.0
        self.register_buffer("_Ubasis", basis)

        # generalized structure tensors (Eq. 15): isotropic L_{r0}=⅓I + weights
        self.register_buffer("_iso", torch.eye(gdim, dtype=torch.float64) / gdim)
        self._Jr = []
        for r, L in enumerate(structure_tensors):
            Lpref = _as_struct(L, gdim)
            self.register_buffer(f"L_pref_{r}", Lpref)
            self._Jr.append(Lpref.shape[0])
            if Lpref.shape[0] > 0:
                self.register_parameter(
                    f"w_raw_{r}", nn.Parameter(torch.zeros(Lpref.shape[0] + 1, dtype=torch.float64)))

        self.n_inv = 2 * self.R + (0 if incompressible else 1)

        # invariant function sub-ANNs (one per invariant) + strain energy Psi
        act = _ACTIVATIONS[activation]
        self.subs = nn.ModuleList()
        for _ in range(self.n_inv):
            layers, n_prev = [], 1
            for h in hidden:
                layers += [nn.Linear(n_prev, h), act()]
                n_prev = h
            self.subs.append(nn.Sequential(*layers))
        psi = [nn.Linear(self.n_inv * hidden[-1], 1)]
        if psi_activation:
            psi.append(act())
        self.psi = nn.Sequential(*psi)

        # invariant input standardisation (absorbed by the chain rule)
        self.register_buffer("inv_mean", torch.zeros(self.n_inv, dtype=torch.float64))
        self.register_buffer("inv_std", torch.ones(self.n_inv, dtype=torch.float64))
        self.double()

    # -- Eq. (15) -----------------------------------------------------------
    def structure_tensors(self):
        Ls = []
        for r in range(self.R):
            if self._Jr[r] == 0:
                Ls.append(self._iso)
                continue
            w = torch.sigmoid(getattr(self, f"w_raw_{r}"))
            w = w / w.sum()
            stack = torch.cat([self._iso[None], getattr(self, f"L_pref_{r}")], 0)
            Ls.append((w[:, None, None] * stack).sum(0))
        return Ls

    # -- Eq. (14) -----------------------------------------------------------
    def invariants_from_C(self, C):
        adj = _adjugate3(C)               # det(C)·C⁻¹, smooth (hessian-safe)
        cols = []
        for L in self.structure_tensors():
            cols.append(torch.einsum("...ij,ji->...", C, L))      # Ī_r = tr[C L̄_r]
            cols.append(torch.einsum("...ij,ji->...", adj, L))    # J̄_r = tr[adj(C) L̄_r]
        if not self.incompressible:
            cols.append(_det3(C))
        return torch.stack(cols, dim=-1)

    def energy_from_C(self, C):
        inv = (self.invariants_from_C(C) - self.inv_mean) / self.inv_std
        feats = [self.subs[k](inv[..., k:k + 1]) for k in range(self.n_inv)]
        return self.psi(torch.cat(feats, dim=-1))      # (..., 1)

    # EnergyNet contract: standardized reduced coords x = Ū -> Ψ (..., 1)
    def forward(self, x):
        U = torch.einsum("...k,kij->...ij", x, self._Ubasis)
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

    def __init__(self, gdim=3, structure_tensors=None, hidden=(8, 8),
                 incompressible=True, activation="softplus", psi_activation=True):
        if structure_tensors is None:
            structure_tensors = [None]
        # stash CANN config so _build_core (called inside super().__init__) sees it
        self._structure_tensors = structure_tensors
        self._incompressible = incompressible
        self._psi_activation = psi_activation
        super().__init__(gdim=gdim, flavor="ch1", hidden=hidden, activation=activation)
        self._glorot_init()   # Glorot init for weights & biases (Table B1)

    def _build_core(self) -> nn.Module:
        return _InvariantEnergy(self.gdim, self._structure_tensors, self.hidden,
                                self._incompressible, self.activation,
                                self._psi_activation)

    def _glorot_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # -- CANN-specific (incompressible) API: work directly in C -------------
    @property
    def R(self):
        return self.mlp.R

    @property
    def n_inv(self):
        return self.mlp.n_inv

    def invariants(self, C):
        return self.mlp.invariants_from_C(C)

    def energy(self, C):
        """Strain energy Ψ(C) (no EnergyNet reference correction — see module
        docstring)."""
        return self.mlp.energy_from_C(C).squeeze(-1)

    def stress_S(self, C, create_graph=None):
        """Second Piola–Kirchhoff stress ``S = 2 ∂Ψ/∂C`` (Eq. 2, no pressure)."""
        if create_graph is None:
            create_graph = self.training
        with torch.enable_grad():
            C = C.detach().clone().requires_grad_(True)
            psi = self.energy(C).sum()
            (G,) = torch.autograd.grad(psi, C, create_graph=create_graph)
        return 2.0 * 0.5 * (G + G.transpose(-1, -2))

    def set_input_norm(self, C_train):
        """Calibrate the invariant standardisation from training ``C``."""
        with torch.no_grad():
            inv = self.mlp.invariants_from_C(C_train.to(torch.float64))
            self.mlp.inv_mean.copy_(inv.mean(0))
            self.mlp.inv_std.copy_(inv.std(0).clamp_min(1e-8))

    def config(self):
        cts = [None if L is None else np.asarray(L).tolist()
               for L in self._structure_tensors]
        return {"gdim": self.gdim, "structure_tensors": cts, "hidden": self.hidden,
                "incompressible": self._incompressible, "activation": self.activation,
                "psi_activation": self._psi_activation}
