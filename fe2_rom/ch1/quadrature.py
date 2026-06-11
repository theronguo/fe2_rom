"""Parallel-efficient ``QuadratureMap`` for FE² macro solves.

The upstream ``dolfinx_materials.QuadratureMap`` evaluates the constitutive law
on ``size_local + num_ghosts`` cells, i.e. it *redundantly* re-solves the
(expensive) material on the ghost layer of every MPI partition. For FE² — where
each quadrature point is a nested RVE solve — that roughly doubles the work on
small/strongly-partitioned macro meshes.

:class:`OwnedCellQuadratureMap` instead integrates the material on **owned cells
only** and fills the ghost-cell flux / tangent / internal-state values by
``scatter_forward`` (copied from the owning rank). Those ghost-cell coefficients
are exactly what the *parallel matrix (Jacobian) assembly* needs — without the
scatter the Jacobian is wrong at partition interfaces and the macro Newton
degrades from quadratic to linear convergence (the residual is unaffected, so it
still converges, just slowly).

Subclass of the upstream class — no modification of the ``dolfinx_materials``
git clone required. Requires a ghosted macro mesh (``GhostMode.shared_facet``);
in serial there are no ghost cells and the scatter is a no-op.
"""
from __future__ import annotations

import numpy as np
from dolfinx_materials.quadrature_map import QuadratureMap


class OwnedCellQuadratureMap(QuadratureMap):
    """``QuadratureMap`` that integrates owned cells only + scatters to ghosts."""

    def __init__(self, mesh, deg, material, cells=None):
        if cells is None:
            imap = mesh.topology.index_map(mesh.topology.dim)
            cells = np.arange(imap.size_local, dtype=np.int32)  # owned cells only
        super().__init__(mesh, deg, material, cells=cells)

    def _scatter_to_ghosts(self) -> None:
        """Update ghost-cell quadrature values from their owning rank."""
        self.jacobian_flatten.x.scatter_forward()
        for f in self.fluxes.values():
            f.x.scatter_forward()
        for isv in self.internal_state_variables.values():
            isv.x.scatter_forward()

    def update(self):
        super().update()
        self._scatter_to_ghosts()

    def advance(self):
        super().advance()
        self._scatter_to_ghosts()
