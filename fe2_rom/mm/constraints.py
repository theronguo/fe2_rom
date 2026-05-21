"""Micromorphic integral constraints involving mode functions φ.

These supplement :class:`fe2_rom.ch1.constraints.ZeroVolumeAverage` with two
additional families that orthogonalise the fluctuation field against each
global mode φᵢ:

    ⟨w · φ⟩      = 0   (ZeroVolumeAverageDot,   1 row per mode)
    ⟨(w · φ) X⟩  = 0   (ZeroVolumeAverageOuter, gdim rows per mode)
"""

from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem

from fe2_rom.ch1.constraints import LinearConstraint


class ZeroVolumeAverageDot(LinearConstraint):
    """Constraint ⟨w · φ⟩ = 0. Contributes 1 scalar row.

    phi must be a periodic ``fem.Function`` on V.
    """

    def __init__(self, phi: fem.Function):
        self._phi = phi

    def build(self, V, dx, mesh, mpc=None):
        v = ufl.TestFunction(V)
        forms = [fem.form(ufl.inner(v, self._phi) * dx)]
        rhs = np.zeros(1)
        return forms, rhs


class ZeroVolumeAverageOuter(LinearConstraint):
    """Constraint ⟨(w · φ) X⟩ = 0 (vector). Contributes ``gdim`` scalar rows.

    phi must be a periodic ``fem.Function`` on V.
    """

    def __init__(self, phi: fem.Function):
        self._phi = phi

    def build(self, V, dx, mesh, mpc=None):
        gdim = mesh.geometry.dim
        X = ufl.SpatialCoordinate(mesh)
        v = ufl.TestFunction(V)
        forms = [fem.form(ufl.inner(v, self._phi) * X[b] * dx) for b in range(gdim)]
        rhs = np.zeros(gdim)
        return forms, rhs
