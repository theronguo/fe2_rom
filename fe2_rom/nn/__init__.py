"""Neural-network surrogates for the effective energy (CH1 / MM / CH2).

The effective energy is a JAX / :mod:`equinox` :class:`~fe2_rom.nn.model.EnergyNet`;
stresses and tangents are automatic derivatives of one scalar (see
:func:`~fe2_rom.nn.model.make_lab_energy`). The optax training utilities live in
:mod:`fe2_rom.nn.training` and are imported lazily.

Float64 is enabled process-wide (``jax_enable_x64``) so the deployed materials
feed consistent tangents to SNES — matching ``dolfinx_materials`` / ``jaxmat``,
which run in double precision.
"""
import jax

jax.config.update("jax_enable_x64", True)

from fe2_rom.nn.polar import right_stretch, polar
from fe2_rom.nn.model import EnergyNet, make_lab_energy

__all__ = ["right_stretch", "polar", "EnergyNet", "make_lab_energy"]
