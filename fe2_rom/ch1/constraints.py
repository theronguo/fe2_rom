"""Pluggable linear integral constraints for periodic homogenization.

Each constraint represents one or more scalar equations of the form
    c_alpha(w) = b_alpha
where c_alpha is a linear functional on V realised by a 1-form
    c_alpha_form(v) = integral_{Omega} l_alpha(x) . v(x) dx,
v is a test function in V, and w is the (trial) fluctuation field.

The solver assembles each c_alpha_form into a PETSc Vec, which becomes a row
of the constraint matrix C used in the Schur-complement Newton step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI


class LinearConstraint(ABC):
    """Abstract integral constraint C . w = b on the fluctuation field."""

    @abstractmethod
    def build(self, V, dx, mesh, mpc=None) -> tuple[list, np.ndarray]:
        """Return (forms, rhs).

        forms : list of compiled ``fem.Form`` of length n_rows, each linear in
                a ``ufl.TestFunction(V)``.  Row alpha of C is obtained by
                ``dolfinx_mpc.assemble_vector(forms[alpha], mpc)``.
        rhs   : numpy array of shape (n_rows,).
        """

    def validate(self, V, mpc, tol: float = 1e-10) -> None:
        _check_function_on_space(self._phi, V)
        _check_periodic(self._phi, mpc, tol)


def _check_function_on_space(phi: fem.Function, V) -> None:
    if phi.function_space.mesh is not V.mesh:
        raise ValueError("phi must be defined on the same mesh as V.")
    if phi.function_space.ufl_element() != V.ufl_element():
        raise ValueError(
            "phi must live on a FunctionSpace with the same UFL element as V "
            f"(got {phi.function_space.ufl_element()} vs {V.ufl_element()})."
        )


def _check_periodic(phi: fem.Function, mpc, tol: float) -> None:
    """Verify phi satisfies the periodic MPC ties to within tol."""
    if mpc is None:
        return
    phi_copy = phi.copy()
    mpc.backsubstitution(phi_copy.x.petsc_vec)
    phi_copy.x.scatter_forward()
    local_diff = float(np.max(np.abs(phi_copy.x.array - phi.x.array))) if phi.x.array.size else 0.0
    global_diff = phi.function_space.mesh.comm.allreduce(local_diff, op=MPI.MAX)
    if global_diff > tol:
        raise ValueError(
            f"phi does not respect the periodic MPC ties "
            f"(max slave-master discrepancy {global_diff:.3e} > tol {tol:.3e}). "
            "Eigenmodes from a non-periodic stability analysis cannot be used directly."
        )


class ZeroVolumeAverage(LinearConstraint):
    """Constraint <w> = 0, component-wise. Contributes ``gdim`` scalar rows."""

    def build(self, V, dx, mesh, mpc=None):
        gdim = mesh.geometry.dim
        v = ufl.TestFunction(V)
        forms = [fem.form(v[d] * dx) for d in range(gdim)]
        rhs = np.zeros(gdim)
        return forms, rhs
    
    def validate(self, V, mpc, tol: float = 1e-10) -> None:
        return


