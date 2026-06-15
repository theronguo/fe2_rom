"""Second-order (CH2) effective quantities and macro-tangent factories.

Adds the higher-order *double stress* ``Q̄`` (Eq. 27) to the
:class:`~fe2_rom.ch1.averages.AverageQuantity` framework, plus the UFL factories
that feed the generic :class:`~fe2_rom.ch1.averages.TangentBlock` for the four
macro tangents required by the mixed second-order macro solver:

    dP̄/dF̄   (= EffectiveAbar, "dPbar_dFbar")     dP̄/dḠ   ("dPbar_dG")
    dQ̄/dF̄   ("dQbar_dFbar")                       dQ̄/dḠ   ("dQbar_dG")

Index conventions (matching ``∇F̂`` so the macro coupling is transparent):

    F_iJ      = F̄_iJ + X_K Ḡ_iJK + ∂w_i/∂X_J         (Eq. 11)
    Ḡ_iJK     = ∂F̄_iJ/∂X_K     (symmetric in J↔K)
    Q̄_iJK     = ⟨ ½ (X_K P_iJ + X_J P_iK) ⟩          (Eq. 27)

By conjugacy the double-stress projection ``M_Q`` and the kinematic sensitivity
``∂F/∂Ḡ`` are the *same* x-weighted 2-tensor family (:func:`x_weighted_factory`):

    M_Q^{(iJK)} = ∂F/∂Ḡ_{iJK} = ½ ( X_K e_i⊗e_J + X_J e_i⊗e_K )

so ``Q̄_iJK = ⟨ M_Q^{(iJK)} : P ⟩`` and ``dQ̄/dμ = ⟨ M_Q : A : (∂F/∂μ + ∇p_μ) ⟩``
slot straight into ``TangentBlock``.
"""
from __future__ import annotations

import ufl

from fe2_rom.ch1.averages import AverageQuantity, _BatchAverager
from fe2_rom.hyperelastic_solver.forms import basis_tensor_ufl


class EffectiveQ(AverageQuantity):
    """Effective higher-order (double) stress ``Q̄_iJK = ⟨½(X_K P_iJ + X_J P_iK)⟩``
    (Eq. 27). Returns a NumPy array of shape ``(gdim, gdim, gdim)``, symmetric in
    the last two indices.

    Requires the RVE to be centred at the origin (the X-weighting is taken about
    the RVE centre), as the second-order ansatz already demands.
    """

    name = "Qbar"

    def setup(self, context):
        gdim = context.mesh.geometry.dim
        self._gdim = gdim
        X = ufl.SpatialCoordinate(context.mesh)
        P = context.P_ufl
        # Flat (i, J, K) C-order, matching the reshape below.
        integrands = [
            0.5 * (X[k] * P[i, j] + X[j] * P[i, k])
            for i in range(gdim) for j in range(gdim) for k in range(gdim)
        ]
        self._avg = _BatchAverager(integrands, context.weight, context.dx, context.mesh)

    def compute(self, context, adjoints=None):
        g = self._gdim
        return self._avg.compute(context.comm, context.vol_global).reshape(g, g, g)


# ---------------------------------------------------------------------------
# UFL factories for the TangentBlock machinery. Each returns a list of UFL
# 2-tensors in C-order over the relevant multi-index, built on demand once the
# HomogenizationContext (hence the mesh / SpatialCoordinate) is available.
# ---------------------------------------------------------------------------

def basis_2tensor_directions(mesh):
    """The ``gdim²`` single-entry basis 2-tensors ``e_i⊗e_j`` (C-order ``(i,j)``).

    Serves as both the ``P̄`` stress projection (``Q̄ = ⟨M:P⟩`` with ``M = e_i⊗e_j``
    gives ``P̄_ij``) and the ``∂F/∂F̄`` kinematic direction.
    """
    g = mesh.geometry.dim
    return [basis_tensor_ufl(g, i, j) for i in range(g) for j in range(g)]


def x_weighted_directions(mesh):
    """The x-weighted 2-tensor family shared by the double-stress projection
    ``M_Q`` and the ``∂F/∂Ḡ`` kinematic direction. For flat index ``(a, b, c)``
    (C-order):

        ½ ( X[c] · e_a⊗e_b + X[b] · e_a⊗e_c ).

    Length ``gdim³``; each entry is symmetric under ``b ↔ c`` (mirroring the
    intrinsic ``Ḡ_iJK = Ḡ_iKJ`` symmetry). The micro solver uses this both to
    build the ``Ḡ`` adjoint RHS forms and (via :func:`x_weighted_factory`) inside
    the ``Q̄`` tangent blocks, so the two stay index-aligned.
    """
    g = mesh.geometry.dim
    X = ufl.SpatialCoordinate(mesh)
    out = []
    for a in range(g):
        for b in range(g):
            for c in range(g):
                out.append(
                    0.5 * (X[c] * basis_tensor_ufl(g, a, b)
                           + X[b] * basis_tensor_ufl(g, a, c))
                )
    return out


# Context-taking wrappers matching the TangentBlock factory signature
# ``f(context) -> list[ufl.Expr]``.
def basis_2tensor_factory(context):
    return basis_2tensor_directions(context.mesh)


def x_weighted_factory(context):
    return x_weighted_directions(context.mesh)
