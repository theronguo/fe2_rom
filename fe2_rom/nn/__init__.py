"""Neural-network surrogates for the effective energy (CH1 / MM / CH2).

The Lightning training utilities live in ``fe2_rom.nn.training`` and are
imported lazily.

IMPORTANT — import order: torch's pip wheel bundles an old ``libgfortran``
that breaks dolfinx's MUMPS if torch is loaded first
(``GFORTRAN_10 not found``). dolfinx must be imported before torch; the
guard below enforces that whenever ``fe2_rom.nn`` is the entry point.
Never ``import torch`` before ``fe2_rom`` in scripts.
"""
try:  # load conda's libgfortran before torch's bundled (older) copy
    import dolfinx  # noqa: F401
except ImportError:
    pass

from fe2_rom.nn.polar import right_stretch, polar
from fe2_rom.nn.model import EnergyNet, make_lab_energy

__all__ = ["right_stretch", "polar", "EnergyNet", "make_lab_energy"]
