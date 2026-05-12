"""fe2-rom: Reduced-order FE² for hyperelastic materials in DOLFINx.

Subpackages:
    fe2_rom.hyperelastic_solver — full-order FE solver, stability, homogenization.
    fe2_rom.rve_rom             — POD + ECM hyper-reduction and reduced RVE solver.
"""
from . import hyperelastic_solver, rve_rom

__version__ = "0.1.0"
__all__ = ["hyperelastic_solver", "rve_rom"]
