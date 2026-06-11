"""Micromorphic-specific effective quantities.

Extends the base ``AverageQuantity`` framework from ``fe2_rom.ch1.averages``
with the two forward quantities that appear in the micromorphic homogenization:

    ``EffectivePi``     — Πᵢ = ⟨P : ∇φᵢ⟩
    ``EffectiveLambda`` — Λᵢ = ⟨Pᵀ·φᵢ + X·(P : ∇φᵢ)⟩
"""

from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem

from fe2_rom.ch1.averages import AverageQuantity, _BatchAverager


class EffectivePi(AverageQuantity):
    """Micromorphic forward quantity ``Πᵢ = ⟨P : ∇φᵢ⟩`` for each mode ``i``.

    Constructed from the user-provided list of global modes ``phi``.
    Returns a NumPy array of shape ``(N,)`` with one entry per mode.
    """

    name = "Pi"

    def __init__(self, phi: list[fem.Function]):
        self._phi = list(phi)
        self._N = len(self._phi)

    def setup(self, context):
        integrands = [ufl.inner(context.P_ufl, ufl.grad(phi)) for phi in self._phi]
        self._avg = _BatchAverager(integrands, context.weight, context.dx, context.mesh)

    def compute(self, context, adjoints=None):
        return self._avg.compute(context.comm, context.vol_global)


class EffectiveLambda(AverageQuantity):
    """Micromorphic forward quantity ``Λᵢ = ⟨Pᵀ·φᵢ + X·(P : ∇φᵢ)⟩`` (vector per
    mode). Returns a NumPy array of shape ``(N, gdim)``.
    """

    name = "Lambda"

    def __init__(self, phi: list[fem.Function]):
        self._phi = list(phi)
        self._N = len(self._phi)

    def setup(self, context):
        gdim = context.mesh.geometry.dim
        self._gdim = gdim
        X = ufl.SpatialCoordinate(context.mesh)
        # Flat (i, d) C-order, matching the reshape below.
        integrands = [
            ufl.dot(self._phi[i], context.P_ufl)[d]
            + X[d] * ufl.inner(context.P_ufl, ufl.grad(self._phi[i]))
            for i in range(self._N) for d in range(gdim)
        ]
        self._avg = _BatchAverager(integrands, context.weight, context.dx, context.mesh)

    def compute(self, context, adjoints=None):
        return self._avg.compute(context.comm, context.vol_global).reshape(self._N, self._gdim)
