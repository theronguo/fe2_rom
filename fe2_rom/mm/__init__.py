"""fe2_rom.mm — micromorphic continuum homogenization.

Submodules:
    fe2_rom.mm.microsolver  — full-order micromorphic RVE solver (MicroSolver).
    fe2_rom.mm.macrosolver  — micromorphic FE² macro driver.
    fe2_rom.mm.material     — micromorphic constitutive bridge.
    fe2_rom.mm.constraints  — φ-dependent integral constraints.
"""
from .averages import EffectivePi, EffectiveLambda
from .constraints import ZeroVolumeAverageDot, ZeroVolumeAverageOuter
from .microsolver import MicroSolver
from .material import DummyMicromorphicMaterial, MicromorphicRVEMaterial
from .macrosolver import MacroMicromorphicSolver

__all__ = [
    "EffectivePi", "EffectiveLambda",
    "ZeroVolumeAverageDot", "ZeroVolumeAverageOuter",
    "MicroSolver",
    "DummyMicromorphicMaterial",
    "MicromorphicRVEMaterial",
    "MacroMicromorphicSolver",
]
