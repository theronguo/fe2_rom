"""Second-order (CH2) boundary-integral constraints on the fluctuation field.

The second-order scale transition (Kouznetsova, Geers, and Brekelmans) needs,
on top of the volume constraint ``⟨w⟩ = 0`` (Eq. 19, realised by
:class:`fe2_rom.ch1.constraints.ZeroVolumeAverage`), two boundary-average
constraints that pin the *net fluctuation* on the top and right edges of the
RVE (Eqs. 17–18). Under periodic boundary conditions on ``w`` the same
conditions then hold for the bottom and left edges, so only two edges are
constrained explicitly:

    ∫_{∂Ω_top}   w ds = 0     (gdim rows, Eq. 17)
    ∫_{∂Ω_right} w ds = 0     (gdim rows, Eq. 18)

Together with ``⟨w⟩ = 0`` these remove the rigid-body modes *and* ensure the
fluctuation does not absorb any part of the imposed macroscopic deformation
gradient ``F̄`` or its gradient ``Ḡ`` — without them the homogenized higher-order
stress ``Q̄`` (Eq. 27) is wrong. They are enforced through the same projected
Newton solve (``P K P x = P b``) used by the first-order / micromorphic solvers.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import ufl
from dolfinx import fem, mesh as dmesh

from fe2_rom.ch1.constraints import LinearConstraint


class ZeroBoundaryAverage(LinearConstraint):
    """Constraint ``∫_Γ w ds = 0`` over the boundary subset ``Γ``.

    ``Γ`` is selected by ``locator`` (a geometric ``x -> bool`` array callable,
    as used by ``dolfinx.mesh.locate_entities_boundary``). Contributes ``gdim``
    scalar rows — one per displacement component. The fluctuation ``w`` is the
    trial field on ``V``; each row of the constraint matrix ``C`` is obtained by
    assembling the linear form ``∫_Γ v_d ds`` against a test function ``v``.
    """

    def __init__(self, locator: Callable, name: str = "edge"):
        self._locator = locator
        self._name = name

    def build(self, V, dx, mesh, mpc=None):
        gdim = mesh.geometry.dim
        fdim = mesh.topology.dim - 1
        facets = dmesh.locate_entities_boundary(mesh, fdim, self._locator)
        markers = np.ones(facets.shape[0], dtype=np.int32)
        facet_tags = dmesh.meshtags(mesh, fdim, facets, markers)
        ds = ufl.Measure(
            "ds", domain=mesh, subdomain_data=facet_tags, subdomain_id=1
        )
        v = ufl.TestFunction(V)
        forms = [fem.form(v[d] * ds) for d in range(gdim)]
        rhs = np.zeros(gdim)
        return forms, rhs

    def validate(self, V, mpc, tol: float = 1e-10) -> None:
        # No φ-function to validate (cf. the micromorphic constraints); the
        # locator-driven facet selection carries no periodicity assumptions.
        return
