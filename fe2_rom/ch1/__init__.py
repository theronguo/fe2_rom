"""fe2_rom.ch1 — first-order (classical) continuum homogenization.

Submodules:
    fe2_rom.ch1.microsolver    — full-order periodic RVE solver (MicroSolver).
    fe2_rom.ch1.macrosolver    — FE² macro driver (MacroSolver, RVEMaterial).
    fe2_rom.ch1.material       — constitutive bridge for dolfinx_materials.
    fe2_rom.ch1.constraints    — volume-averaging linear constraints.
    fe2_rom.ch1.exceptions     — RVEConvergenceError.
    fe2_rom.ch1.solvers        — NewtonSolverFE2.
"""
from .averages import (
    AverageQuantity,
    HomogenizationContext,
    EffectiveFbar,
    EffectiveW,
    EffectivePbar,
    EffectiveAbar,
    TangentBlock,
    resolve_average_quantities,
    STRING_KEY_MAP,
)
from .constraints import (
    LinearConstraint,
    ZeroVolumeAverage,
)
from .exceptions import RVEConvergenceError
from .solvers import NewtonSolverFE2
from .microsolver import MicroSolver
from .material import RVEMaterial
from .macrosolver import MacroSolver
from .homogenization import effective_stiffness, uniaxial_moduli

__all__ = [
    "AverageQuantity", "HomogenizationContext",
    "EffectiveFbar", "EffectiveW", "EffectivePbar", "EffectiveAbar",
    "TangentBlock", "resolve_average_quantities", "STRING_KEY_MAP",
    "LinearConstraint",
    "ZeroVolumeAverage",
    "RVEConvergenceError",
    "NewtonSolverFE2",
    "MicroSolver",
    "RVEMaterial",
    "MacroSolver",
    "effective_stiffness",
    "uniaxial_moduli",
]
