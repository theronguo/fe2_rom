import numpy as np
import ufl
from dolfinx import fem

from .material import MaterialModel


def build_weak_forms(mesh, V, u, material: MaterialModel, body_force=None, dx=None,
                     neumann_terms=None):
    """Compile residual and tangent stiffness forms for a hyperelastic problem.

    Returns (R_form, J_form, F_var, P_ufl, J_ufl) where F_var is the variable-
    wrapped deformation gradient, P_ufl is the first PK stress, and J_ufl is det(F).
    F_var and P_ufl share the same UFL graph as R_form, so they reflect the
    current u without any extra interpolation step.

    fem.form() calls are collective — must be called on all MPI ranks.
    """
    if dx is None:
        dx = ufl.Measure("dx", domain=mesh)
    if body_force is None:
        body_force = fem.Constant(mesh, tuple([0.0] * mesh.geometry.dim))

    space_dims = mesh.geometry.dim
    F_var = ufl.variable(ufl.Identity(space_dims) + ufl.grad(u))
    P_ufl = material.first_pk_stress(F_var)
    J_ufl = ufl.det(F_var)

    v = ufl.TestFunction(V)
    R = ufl.inner(ufl.grad(v), P_ufl) * dx - ufl.inner(v, body_force) * dx
    for traction, ds_measure in (neumann_terms or []):
        R -= ufl.inner(v, traction) * ds_measure
    J_nonlinear = ufl.derivative(R, u)

    return fem.form(R), fem.form(J_nonlinear), F_var, P_ufl, J_ufl


def basis_tensor_ufl(gdim: int, i: int, j: int):
    """UFL 2-tensor with a single non-zero entry at position (i, j).

    Useful as ∂F̄/∂F̄[i,j] when building tangent RHS forms.
    """
    arr = np.zeros((gdim, gdim))
    arr[i, j] = 1.0
    return ufl.as_matrix(arr.tolist())


def build_homogenization_weak_form(mesh, V, u, Fbar, material: MaterialModel, *,
                                   u_total_extra=None, dx=None):
    """Compile residual + Jacobian and prepare a tangent-RHS builder for periodic
    hyperelastic homogenization.

    The total displacement is taken as
        u_total = (Fbar - I) X + u_total_extra(...) + u(fluctuation)
    so that
        F = grad(u_total) + I = Fbar + grad(u_total_extra) + grad(u).

    For first-order homogenization pass ``u_total_extra=None`` (or a zero vector).
    For micromorphic homogenization pass the φ-contribution
    ``sum_i (v_i + X · g_i) φ_i``.

    Returns:
      R_form, J_form: compiled residual and Jacobian.
      F_var:          ufl.variable wrapping F (for diff w.r.t. F).
      P_ufl, J_ufl, W_ufl, A_ufl: hyperelastic UFL expressions.
      u_total:        UFL expression for the total displacement.
      build_tangent_rhs_forms: callable ``(dF_dmu_list) -> list[fem.Form]`` that
                               compiles forward-sensitivity adjoint RHS forms.

    fem.form() calls are collective — must be called on all MPI ranks.
    """
    if dx is None:
        dx = ufl.Measure("dx", domain=mesh)
    gdim = mesh.geometry.dim

    # ufl.zero(gdim) has no associated mesh, which trips up ufl.grad below.
    # Only include the extra term symbolically when the caller actually supplied
    # a domain-aware expression (e.g. the micromorphic Σᵢ(vᵢ + X·gᵢ)φᵢ).
    if u_total_extra is None:
        F_var = ufl.variable(Fbar + ufl.grad(u))
    else:
        F_var = ufl.variable(Fbar + ufl.grad(u_total_extra) + ufl.grad(u))
    P_ufl = material.first_pk_stress(F_var)
    J_ufl = ufl.det(F_var)
    W_ufl = material.strain_energy(F_var)
    A_ufl = material.tangent_moduli(F_var)

    X = ufl.SpatialCoordinate(mesh)
    if u_total_extra is None:
        u_total = (Fbar - ufl.Identity(gdim)) * X + u
    else:
        u_total = (Fbar - ufl.Identity(gdim)) * X + u_total_extra + u

    v = ufl.TestFunction(V)
    R = ufl.inner(ufl.grad(v), P_ufl) * dx
    J_nonlinear = ufl.derivative(R, u)

    def build_tangent_rhs_forms(dF_dmu_list: list) -> list:
        """Compile per-component adjoint RHS forms.

        For each 2-tensor expression ``dF_dmu`` in the list, this builds
            -∫ ⟨∇v, A : dF_dmu⟩ dx
        which is the RHS for the forward sensitivity equation K p = rhs,
        where p = ∂w/∂μ_k for the k-th scalar component of macro variable μ.

        For Fbar, ``dF_dmu`` is the basis 2-tensor e_i ⊗ e_j (see
        ``basis_tensor_ufl``). For v_i and g_{i,d} of the micromorphic
        scheme, it is the appropriate UFL expression involving φ_i and X.
        """
        a, b, c, d = ufl.indices(4)
        A_4 = ufl.diff(P_ufl, F_var)
        compiled = []
        for dF_dmu in dF_dmu_list:
            dP_dmu = ufl.as_tensor(A_4[a, b, c, d] * dF_dmu[c, d], (a, b))
            compiled.append(fem.form(-ufl.inner(ufl.grad(v), dP_dmu) * dx))
        return compiled

    return (
        fem.form(R), fem.form(J_nonlinear),
        F_var, P_ufl, J_ufl, W_ufl, A_ufl, u_total,
        build_tangent_rhs_forms,
    )
