"""fe2_rom.ch2 — second-order computational homogenization (CH2).

Extends the first-order layer (:mod:`fe2_rom.ch1`) with a strain-gradient
enrichment, analogous to how :mod:`fe2_rom.mm` adds patterning modes. The
microscopic ansatz is

    u_total = (F̄ − I)·X + ½ X·Ḡ·X + w

with a new third-order macro variable ``Ḡ`` (the gradient of the deformation
gradient, ``Ḡ_iJK = ∂F̄_iJ/∂X_K``, symmetric in its last two indices). The
fluctuation ``w`` is constrained by ``⟨w⟩ = 0`` (Eq. 19) plus the two
boundary-average constraints ``∫_top w = ∫_right w = 0`` (Eqs. 17–18). The RVE
reports the effective stress ``P̄`` (Eq. 26), the double stress ``Q̄`` (Eq. 27)
and the four macro tangents ``d{P̄,Q̄}/d{F̄,Ḡ}``.

The macroscopic problem is the mixed (saddle-point) formulation of Eqs. 3–6,
coupling the displacement ``u``, an independent deformation-gradient field
``F̂`` and a Lagrange multiplier ``L̄`` enforcing ``F̂ = F̄``.

Submodules
----------
    fe2_rom.ch2.constraints  — boundary-average constraints (Eqs. 17–18).
    fe2_rom.ch2.averages     — double stress Q̄ + macro tangent factories.
    fe2_rom.ch2.microsolver  — full-order second-order RVE solver (MicroSolver).
    fe2_rom.ch2.material     — constitutive bridge (Ch2RVEMaterial, dummy law).
    fe2_rom.ch2.macrosolver  — mixed FE² macro driver (MacroSecondOrderSolver).
"""
from .constraints import ZeroBoundaryAverage
from .averages import (
    EffectiveQ,
    basis_2tensor_directions,
    x_weighted_directions,
    basis_2tensor_factory,
    x_weighted_factory,
)
from .microsolver import MicroSolver
from .material import Ch2RVEMaterial, DummyCh2Material
from .macrosolver import MacroSecondOrderSolver

__all__ = [
    "ZeroBoundaryAverage",
    "EffectiveQ",
    "basis_2tensor_directions",
    "x_weighted_directions",
    "basis_2tensor_factory",
    "x_weighted_factory",
    "MicroSolver",
    "Ch2RVEMaterial",
    "DummyCh2Material",
    "MacroSecondOrderSolver",
]
