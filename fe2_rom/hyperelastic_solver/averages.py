"""Pluggable effective (volume-averaged) quantities for periodic homogenization.

Each subclass of ``AverageQuantity`` computes one effective quantity from the
converged RVE state. Quantities that depend on the sensitivity of the
fluctuation w with respect to macro variables (e.g. effective tangent moduli)
declare those needs via ``required_macro_adjoints`` and consume the precomputed
forward-sensitivity Functions in their ``compute`` method.

The convention for adjoint indexing: for a macro variable named ``name`` with
shape ``(*dims)``, ``adjoints[name]`` is a flat ``list`` of ``fem.Function``
objects of length ``prod(dims)``, with the multi-index ``(i, j, ...)`` mapping
to position ``np.ravel_multi_index((i, j, ...), dims)`` (C order).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI


@dataclass
class HomogenizationContext:
    """Symbolic and runtime data shared between the solver and pluggable averages."""

    mesh: Any
    V: Any
    dx: ufl.Measure
    comm: MPI.Comm
    vol_global: float
    F_var: Any
    P_ufl: Any
    A_ufl: Any
    W_ufl: Any
    u: fem.Function
    u_total: Any
    macro_vars: dict
    phi: list = field(default_factory=list)


def _avg_scalar(form, comm, vol_global) -> float:
    val = fem.assemble_scalar(form)
    val = comm.allreduce(val, op=MPI.SUM)
    return val / vol_global


class AverageQuantity(ABC):
    """A volume-averaged effective quantity emitted by the homogenization solver."""

    name: str = ""
    required_macro_adjoints: list[str] = []

    def setup(self, context: HomogenizationContext) -> None:
        """Pre-compile UFL forms. Called once at solver setup."""
        return

    @abstractmethod
    def compute(self, context: HomogenizationContext, adjoints: dict | None = None):
        """Return the quantity at the current trial state."""


class EffectiveFbar(AverageQuantity):
    """Echo the current macroscopic deformation gradient F̄."""

    name = "Fbar"

    def compute(self, context, adjoints=None):
        return context.macro_vars["Fbar"].value.copy()


class EffectiveW(AverageQuantity):
    """Effective strain energy density W̄ = ⟨W(F)⟩."""

    name = "Wbar"

    def setup(self, context):
        self._form = fem.form(context.W_ufl * context.dx)

    def compute(self, context, adjoints=None):
        return _avg_scalar(self._form, context.comm, context.vol_global)


class EffectivePbar(AverageQuantity):
    """Effective first Piola-Kirchhoff stress P̄ = ⟨P(F)⟩."""

    name = "Pbar"

    def setup(self, context):
        gdim = context.mesh.geometry.dim
        self._gdim = gdim
        self._forms = [
            [fem.form(context.P_ufl[i, j] * context.dx) for j in range(gdim)]
            for i in range(gdim)
        ]

    def compute(self, context, adjoints=None):
        gdim = self._gdim
        P_eff = np.zeros((gdim, gdim))
        for i in range(gdim):
            for j in range(gdim):
                P_eff[i, j] = _avg_scalar(self._forms[i][j], context.comm, context.vol_global)
        return P_eff


class EffectiveAbar(AverageQuantity):
    """Effective tangent moduli Ā[i,j,k,l] = d⟨P[i,j]⟩ / dF̄[k,l].

    Uses ``gdim²`` forward sensitivities ``∂w/∂F̄[k,l]`` supplied by the
    saddle-point solver. The closed form is

        Ā[i,j,k,l] = ⟨A[i,j,k,l]⟩ + ⟨A[i,j,:,:] : ∇p_{kl}⟩

    where ``p_{kl}`` solves ``K p_{kl} = -∂R/∂F̄[k,l]``.
    """

    name = "dPbar_dFbar"
    required_macro_adjoints = ["Fbar"]

    def setup(self, context):
        gdim = context.mesh.geometry.dim
        self._gdim = gdim
        # Pre-allocate adjoint storage; solver copies values into these before compute().
        self._adjoint_slots = [
            [fem.Function(context.V) for _ in range(gdim)] for _ in range(gdim)
        ]
        self._A_avg_forms = [[[[
            fem.form(context.A_ufl[i, j, k, l] * context.dx)
            for l in range(gdim)] for k in range(gdim)]
            for j in range(gdim)] for i in range(gdim)]
        self._A_fluc_forms = [[[[
            fem.form(
                ufl.inner(context.A_ufl[i, j, :, :],
                          ufl.grad(self._adjoint_slots[k][l])) * context.dx
            )
            for l in range(gdim)] for k in range(gdim)]
            for j in range(gdim)] for i in range(gdim)]

    def compute(self, context, adjoints):
        gdim = self._gdim
        Fbar_adjoints = adjoints["Fbar"]
        for k in range(gdim):
            for l in range(gdim):
                slot = self._adjoint_slots[k][l]
                src = Fbar_adjoints[k * gdim + l]
                slot.x.array[:] = src.x.array
                slot.x.scatter_forward()
        A_eff = np.zeros((gdim, gdim, gdim, gdim))
        for i in range(gdim):
            for j in range(gdim):
                for k in range(gdim):
                    for l in range(gdim):
                        avg = _avg_scalar(self._A_avg_forms[i][j][k][l],
                                          context.comm, context.vol_global)
                        fluc = _avg_scalar(self._A_fluc_forms[i][j][k][l],
                                           context.comm, context.vol_global)
                        A_eff[i, j, k, l] = avg + fluc
        return A_eff


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
            fem.form(ufl.inner(context.P_ufl, ufl.grad(phi)) * context.dx)
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
                    * context.dx
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


class TangentBlock(AverageQuantity):
    """Generic ``dQ/dμ`` block for quantities of the form ``Q = ⟨inner(M, P)⟩``.

    ``M`` is a 2-tensor independent of ``F`` and macro variables, indexed by
    ``q_shape`` (in C order). ``μ`` is a macro variable whose forward
    sensitivities ``p_{μ_k} = ∂w/∂μ_k`` were computed by the Newton solver and
    are passed as ``adjoints[macro_var_name]``. The block reads

        T[q, μ] = ⟨ M[q] : A : (∂F/∂μ_k + ∇p_{μ_k}) ⟩ / V

    and is returned as a NumPy array of shape ``q_shape + mu_shape``.

    M and ∂F/∂μ are supplied as factories ``f(context) -> list[ufl.Expr]`` so
    they can reference symbolic state (``phi``, ``X``, etc.) constructed by
    the solver.
    """

    def __init__(self, name: str, macro_var_name: str,
                 q_shape: tuple, mu_shape: tuple,
                 M_factory, dF_dmu_factory):
        self.name = name
        self.required_macro_adjoints = [macro_var_name]
        self._macro_var = macro_var_name
        self._q_shape = tuple(q_shape)
        self._mu_shape = tuple(mu_shape)
        self._M_factory = M_factory
        self._dF_dmu_factory = dF_dmu_factory

    def setup(self, context):
        M_list = self._M_factory(context)
        dF_list = self._dF_dmu_factory(context)
        n_q = int(np.prod(self._q_shape)) if self._q_shape else 1
        n_mu = int(np.prod(self._mu_shape)) if self._mu_shape else 1
        if len(M_list) != n_q:
            raise ValueError(
                f"TangentBlock {self.name!r}: M_factory returned {len(M_list)} "
                f"entries; expected {n_q} for q_shape={self._q_shape}"
            )
        if len(dF_list) != n_mu:
            raise ValueError(
                f"TangentBlock {self.name!r}: dF_dmu_factory returned {len(dF_list)} "
                f"entries; expected {n_mu} for mu_shape={self._mu_shape}"
            )

        i, j, k, l = ufl.indices(4)
        A = context.A_ufl
        self._adj_slots = [fem.Function(context.V) for _ in range(n_mu)]
        self._forms = [[None] * n_mu for _ in range(n_q)]
        for qi in range(n_q):
            M = M_list[qi]
            for mi in range(n_mu):
                dF_total = dF_list[mi] + ufl.grad(self._adj_slots[mi])
                A_dF = ufl.as_tensor(A[i, j, k, l] * dF_total[k, l], (i, j))
                self._forms[qi][mi] = fem.form(ufl.inner(M, A_dF) * context.dx)
        self._n_q = n_q
        self._n_mu = n_mu

    def compute(self, context, adjoints):
        src_list = adjoints[self._macro_var]
        for mi, slot in enumerate(self._adj_slots):
            slot.x.array[:] = src_list[mi].x.array
            slot.x.scatter_forward()
        flat = np.zeros((self._n_q, self._n_mu))
        for qi in range(self._n_q):
            for mi in range(self._n_mu):
                flat[qi, mi] = _avg_scalar(
                    self._forms[qi][mi], context.comm, context.vol_global
                )
        return flat.reshape(self._q_shape + self._mu_shape)


STRING_KEY_MAP: dict[str, type[AverageQuantity]] = {
    "F": EffectiveFbar,
    "Fbar": EffectiveFbar,
    "W": EffectiveW,
    "Wbar": EffectiveW,
    "P": EffectivePbar,
    "Pbar": EffectivePbar,
    "A": EffectiveAbar,
    "Abar": EffectiveAbar,
    "dPbar_dFbar": EffectiveAbar,
}


def resolve_average_quantities(items) -> list[AverageQuantity]:
    """Normalize a list of ``AverageQuantity`` instances or string keys into instances.

    Strings are mapped via ``STRING_KEY_MAP`` and instantiated with no
    arguments — only valid for quantities that have no required state.
    """
    out: list[AverageQuantity] = []
    for item in items:
        if isinstance(item, AverageQuantity):
            out.append(item)
        elif isinstance(item, str):
            cls = STRING_KEY_MAP.get(item)
            if cls is None:
                raise ValueError(
                    f"Unknown average quantity key {item!r}. "
                    f"Valid keys: {sorted(STRING_KEY_MAP)}"
                )
            out.append(cls())
        else:
            raise TypeError(
                "average_quantities entries must be AverageQuantity or str, "
                f"got {type(item).__name__}"
            )
    return out
