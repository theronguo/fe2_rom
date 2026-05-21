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

from fe2_rom.ch1.averages import AverageQuantity, _avg_scalar


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
        self._forms = [
            fem.form(ufl.inner(context.P_ufl, ufl.grad(phi)) * context.weight * context.dx)
            for phi in self._phi
        ]

    def compute(self, context, adjoints=None):
        Pi = np.zeros(self._N)
        for i in range(self._N):
            Pi[i] = _avg_scalar(self._forms[i], context.comm, context.vol_global)
        return Pi


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
        self._forms = [
            [
                fem.form(
                    (
                        ufl.dot(self._phi[i], context.P_ufl)[d]
                        + X[d] * ufl.inner(context.P_ufl, ufl.grad(self._phi[i]))
                    )
                    * context.weight * context.dx
                )
                for d in range(gdim)
            ]
            for i in range(self._N)
        ]

    def compute(self, context, adjoints=None):
        Lam = np.zeros((self._N, self._gdim))
        for i in range(self._N):
            for d in range(self._gdim):
                Lam[i, d] = _avg_scalar(self._forms[i][d], context.comm, context.vol_global)
        return Lam
