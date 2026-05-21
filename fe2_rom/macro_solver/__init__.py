"""Macroscopic FE² solver utilities.

Use :class:`RVEMaterial` as a drop-in dolfinx_materials Material whose
constitutive response is computed by a nested RVE solver
(``PeriodicHyperelasticHomogenizationSolver`` or ``RVESolver``).
"""
from .material import RVEMaterial
from .macro import MacroSolver
from .material_micromorphic import DummyMicromorphicMaterial, MicromorphicRVEMaterial
from .macro_micromorphic import MacroMicromorphicSolver

__all__ = [
    "RVEMaterial",
    "MacroSolver",
    "DummyMicromorphicMaterial",
    "MicromorphicRVEMaterial",
    "MacroMicromorphicSolver",
]
