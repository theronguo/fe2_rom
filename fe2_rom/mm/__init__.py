"""fe2_rom.mm — micromorphic continuum homogenization.

Submodules:
    fe2_rom.mm.microsolver  — full-order micromorphic RVE solver (MicroSolver).
    fe2_rom.mm.macrosolver  — micromorphic FE² macro driver.
    fe2_rom.mm.material     — micromorphic constitutive bridge.
    fe2_rom.mm.constraints  — φ-dependent integral constraints.
    fe2_rom.mm.enrichment_modes — build φ enrichment modes (from RVE buckling
        or from analytical/lambda mode functions).
    fe2_rom.mm.training_data — generate (F̄, v, g) ROM training snapshots.
"""
from .averages import EffectivePi, EffectiveLambda
from .constraints import ZeroVolumeAverageDot, ZeroVolumeAverageOuter
from .microsolver import MicroSolver
from .material import DummyMicromorphicMaterial, MicromorphicRVEMaterial
from .macrosolver import MacroMicromorphicSolver
from .enrichment_modes import (
    extract_buckling_modes, make_symmetric_compression,
    make_analytical_modes, set_analytical_modes, EnrichmentModes,
)
from .training_data import generate_training_data, TrainingData

__all__ = [
    "EffectivePi", "EffectiveLambda",
    "ZeroVolumeAverageDot", "ZeroVolumeAverageOuter",
    "MicroSolver",
    "DummyMicromorphicMaterial",
    "MicromorphicRVEMaterial",
    "MacroMicromorphicSolver",
    "extract_buckling_modes", "make_symmetric_compression",
    "make_analytical_modes", "set_analytical_modes", "EnrichmentModes",
    "generate_training_data", "TrainingData",
]
