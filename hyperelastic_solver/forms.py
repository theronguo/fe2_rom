import ufl
from dolfinx import fem

from .material import MaterialModel


def build_weak_forms(mesh, V, u, material: MaterialModel, body_force=None, dx=None):
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
    J_nonlinear = ufl.derivative(R, u)

    return fem.form(R), fem.form(J_nonlinear), F_var, P_ufl, J_ufl
