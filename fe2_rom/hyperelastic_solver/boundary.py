import numpy as np
from dolfinx import fem, la
from mpi4py import MPI


class ReactionProbe:
    """Consistent (residual-based) reaction force on a constrained surface.

    Assembles the residual ``b = assemble(R_form)`` *without* applying the
    Dirichlet BCs and sums it at the probe's constrained dofs.  At a converged
    state the interior rows vanish (to Newton tolerance) and the Dirichlet
    rows carry exactly the discrete nodal forces the supports exert on the
    body — the consistent reactions, in exact equilibrium with the applied
    load (external terms in ``R_form``, e.g. body forces, are accounted for).

    The boundary flux ``∫(P·n)·e ds`` of the recovered stress is deliberately
    *not* used: the pointwise FE boundary stress satisfies no discrete balance,
    and for P1 elements with nearly-incompressible materials on irregular cut
    surfaces it can overread by tens of percent (observed +40% on the
    sphere-strut lattice DNS in validation/mm_3d).

    A component-``k`` constraint transmits force only along ``e_k``, so the
    reported value is the summed residual at those dofs weighted by
    ``direction[k]`` (the projection of ``e_k`` onto ``direction``); tangential
    components of ``direction`` cannot be measured on a single-component BC.
    """

    def __init__(self, V, parent_dofs, residual_form, subspace_index,
                 direction=(0.0, 0.0, 1.0), bc_value=None):
        n_owned = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
        parent_dofs = np.asarray(parent_dofs, dtype=np.int32)
        # Ghost rows are summed on their owning rank after scatter_reverse —
        # keep only owned dofs so nothing is double-counted in the reduction.
        self._dofs = parent_dofs[parent_dofs < n_owned]
        self._form = residual_form
        self._weight = float(direction[subspace_index])
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
        b = fem.assemble_vector(self._form)  # BCs NOT applied — rows hold reactions
        b.scatter_reverse(la.InsertMode.add)
        rf_local = float(b.array[self._dofs].sum()) if self._dofs.size else 0.0
        return self._weight * comm.allreduce(rf_local, op=MPI.SUM)
