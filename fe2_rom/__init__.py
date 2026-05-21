"""fe2-rom: Reduced-order FE² for hyperelastic materials in DOLFINx.

Subpackages:
    fe2_rom.hyperelastic_solver — full-order FE solver, stability, output utilities.
    fe2_rom.rom                 — POD + ECM hyper-reduction and reduced RVE solvers.
    fe2_rom.ch1                 — first-order (classical) continuum homogenization.
    fe2_rom.mm                  — micromorphic continuum homogenization.
"""
from . import hyperelastic_solver, rom, ch1, mm

__version__ = "0.1.0"
__all__ = ["hyperelastic_solver", "rom", "ch1", "mm"]
