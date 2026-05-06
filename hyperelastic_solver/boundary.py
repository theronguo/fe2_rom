import numpy as np
import ufl
from dolfinx import fem, mesh as dmesh
from mpi4py import MPI


class ReactionProbe:
    """Computes ∫(P·n)·direction ds on a boundary surface.

    Owns its own meshtags and ds measure so it is self-contained.
    P_ufl references the solver's u by closure, so it always reflects the
    current solution without any extra copy.
    """

    def __init__(self, mesh, facets, P_ufl, direction=(0.0, 0.0, 1.0),
                 bc_value=None):
        fdim = mesh.topology.dim - 1
        sorted_facets = np.sort(facets)
        tags = np.full(len(sorted_facets), 1, dtype=np.int32)
        facet_tag = dmesh.meshtags(mesh, fdim, sorted_facets, tags)
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tag)(1)
        n_ref = ufl.FacetNormal(mesh)
        e_dir = ufl.as_vector(list(direction))
        self._form = fem.form(ufl.inner(ufl.dot(P_ufl, n_ref), e_dir) * ds)
        # bc_value tracks the associated prescribed displacement constant
        self._bc_value = bc_value

    @property
    def displacement(self) -> float:
        """Current value of the associated prescribed-displacement Constant."""
        if self._bc_value is not None:
            return float(self._bc_value.value)
        return 0.0

    def assemble(self, comm) -> float:
        """Assemble and MPI-reduce the reaction force. Safe to call on all ranks."""
        rf_local = fem.assemble_scalar(self._form)
        return comm.allreduce(rf_local, op=MPI.SUM)
