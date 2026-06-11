"""Pluggable effective (volume-averaged) quantities for first-order periodic homogenization.

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
from dolfinx import fem, la
from mpi4py import MPI

from fe2_rom.ch1.objectivity import (
    assemble_symmetric_tangent,
    symmetric_basis_tensors,
    symmetric_index_pairs,
)


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
    # Optional integration weight (used by the ECM-reduced ROM averages: ω(x)
    # such that ∫_Ω f dx ≈ ∫_submesh f ω dx). UFL-multiplicable: a Function,
    # Constant, or plain float. Defaults to 1.0 (no reweighting) for the FOM.
    weight: Any = 1.0


def _avg_scalar(form, comm, vol_global) -> float:
    val = fem.assemble_scalar(form)
    val = comm.allreduce(val, op=MPI.SUM)
    return val / vol_global


class _BatchAverager:
    """Volume-average many scalar integrands in a single vector assembly.

    Builds a ``DG-0`` vector space with one component per integrand and assembles
    ``Σ_k integrand_k · v_k · weight · dx`` *once*; summing the result over cells
    gives ``⟨integrand_k⟩ = (1/V) ∫ integrand_k · weight dx`` for every ``k``.

    This replaces ``len(integrands)`` separate ``fem.assemble_scalar`` calls —
    whose per-call Python/FFI overhead dominates the effective-tangent cost on
    the small ECM submesh — with one ``fem.assemble_vector``. The tensor field
    (e.g. the rank-4 tangent ``A``) is also evaluated once per cell instead of
    once per component. Results are identical to the per-scalar path up to
    round-off (same integrand, same quadrature, same volume).
    """

    def __init__(self, integrands, weight, dx, mesh):
        self._n = len(integrands)
        self._V0 = fem.functionspace(mesh, ("DG", 0, (self._n,)))
        v = ufl.TestFunction(self._V0)
        L = integrands[0] * v[0]
        for k in range(1, self._n):
            L = L + integrands[k] * v[k]
        self._form = fem.form(L * weight * dx)
        self._nloc = self._V0.dofmap.index_map.size_local

    def compute(self, comm, vol_global):
        b = fem.assemble_vector(self._form)
        b.scatter_reverse(la.InsertMode.add)
        local = b.array[: self._nloc * self._n].reshape(self._nloc, self._n).sum(axis=0)
        return comm.allreduce(local, op=MPI.SUM) / vol_global


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
        self._form = fem.form(context.W_ufl * context.weight * context.dx)

    def compute(self, context, adjoints=None):
        return _avg_scalar(self._form, context.comm, context.vol_global)


class EffectivePbar(AverageQuantity):
    """Effective first Piola-Kirchhoff stress P̄ = ⟨P(F)⟩."""

    name = "Pbar"

    def setup(self, context):
        gdim = context.mesh.geometry.dim
        self._gdim = gdim
        integrands = [context.P_ufl[i, j] for i in range(gdim) for j in range(gdim)]
        self._avg = _BatchAverager(integrands, context.weight, context.dx, context.mesh)

    def compute(self, context, adjoints=None):
        gdim = self._gdim
        return self._avg.compute(context.comm, context.vol_global).reshape(gdim, gdim)


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
        A = context.A_ufl
        # Flat (i,j,k,l) C-order, matching the reshape below.
        avg_integrands = [
            A[i, j, k, l]
            for i in range(gdim) for j in range(gdim)
            for k in range(gdim) for l in range(gdim)
        ]
        fluc_integrands = [
            ufl.inner(A[i, j, :, :], ufl.grad(self._adjoint_slots[k][l]))
            for i in range(gdim) for j in range(gdim)
            for k in range(gdim) for l in range(gdim)
        ]
        self._avg_b = _BatchAverager(avg_integrands, context.weight, context.dx, context.mesh)
        self._fluc_b = _BatchAverager(fluc_integrands, context.weight, context.dx, context.mesh)

    def compute(self, context, adjoints):
        gdim = self._gdim
        Fbar_adjoints = adjoints["Fbar"]
        for k in range(gdim):
            for l in range(gdim):
                slot = self._adjoint_slots[k][l]
                slot.x.array[:] = Fbar_adjoints[k * gdim + l].x.array
                slot.x.scatter_forward()
        avg = self._avg_b.compute(context.comm, context.vol_global)
        fluc = self._fluc_b.compute(context.comm, context.vol_global)
        return (avg + fluc).reshape(gdim, gdim, gdim, gdim)


class EffectiveAbarReduced(AverageQuantity):
    """U-frame reduced effective tangent ``Ã[i,j,p,q] = ∂P̃[i,j]/∂U[p,q]``.

    Used by the objectivity (``F̄ = R U``) reduction: the solver is driven with
    the *symmetric* stretch ``U`` and this quantity returns the tangent w.r.t.
    ``U``, computed from only ``n_sym`` = 6 (3D) / 3 (2D) adjoint directions (the
    symmetric basis tensors ``S^{(pq)}``) instead of ``gdim²``:

        ``dP̃_S[i,j] = ⟨A[i,j,:,:] : S⟩ + ⟨A[i,j,:,:] : ∇p_S⟩``

    and assembles the symmetric tangent (see
    :func:`fe2_rom.ch1.objectivity.assemble_symmetric_tangent`). The lab-frame
    ``dP̄/dF̄ = dR·P̃ + R·Ã·dU`` is reconstructed downstream in the solver's
    ``__call__`` from the analytic polar derivatives. Emits the key
    ``"dPbar_dFbar"`` (same as :class:`EffectiveAbar`) but carrying the *U-frame*
    tangent until that reconstruction runs.
    """

    name = "dPbar_dFbar"
    required_macro_adjoints = ["Fbar"]

    def setup(self, context):
        gdim = context.mesh.geometry.dim
        self._gdim = gdim
        self._S = symmetric_basis_tensors(gdim)
        self._n_sym = len(self._S)
        self._adjoint_slots = [fem.Function(context.V) for _ in range(self._n_sym)]
        A = context.A_ufl
        avg_integrands = [
            A[i, j, k, l]
            for i in range(gdim) for j in range(gdim)
            for k in range(gdim) for l in range(gdim)
        ]
        # Flat (s, i, j) C-order, matching the reshape below.
        fluc_integrands = [
            ufl.inner(A[i, j, :, :], ufl.grad(self._adjoint_slots[s]))
            for s in range(self._n_sym)
            for i in range(gdim) for j in range(gdim)
        ]
        self._avg_b = _BatchAverager(avg_integrands, context.weight, context.dx, context.mesh)
        self._fluc_b = _BatchAverager(fluc_integrands, context.weight, context.dx, context.mesh)

    def compute(self, context, adjoints):
        gdim = self._gdim
        src = adjoints["Fbar"]
        for s in range(self._n_sym):
            self._adjoint_slots[s].x.array[:] = src[s].x.array
            self._adjoint_slots[s].x.scatter_forward()
        avgA = self._avg_b.compute(context.comm, context.vol_global).reshape(
            gdim, gdim, gdim, gdim)
        fluc = self._fluc_b.compute(context.comm, context.vol_global).reshape(
            self._n_sym, gdim, gdim)
        dP_dir = np.empty((self._n_sym, gdim, gdim), dtype=float)
        for s in range(self._n_sym):
            direct = np.einsum("ijkl,kl->ij", avgA, self._S[s])
            dP_dir[s] = direct + fluc[s]
        return assemble_symmetric_tangent(dP_dir, gdim)


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
        # Flat (qi, mi) C-order, matching the reshape below.
        integrands = []
        for qi in range(n_q):
            M = M_list[qi]
            for mi in range(n_mu):
                dF_total = dF_list[mi] + ufl.grad(self._adj_slots[mi])
                A_dF = ufl.as_tensor(A[i, j, k, l] * dF_total[k, l], (i, j))
                integrands.append(ufl.inner(M, A_dF))
        self._avg = _BatchAverager(integrands, context.weight, context.dx, context.mesh)
        self._n_q = n_q
        self._n_mu = n_mu

    def compute(self, context, adjoints):
        src_list = adjoints[self._macro_var]
        for mi, slot in enumerate(self._adj_slots):
            slot.x.array[:] = src_list[mi].x.array
            slot.x.scatter_forward()
        flat = self._avg.compute(context.comm, context.vol_global)
        return flat.reshape(self._q_shape + self._mu_shape)


class TangentBlockReduced(TangentBlock):
    """U-frame reduced ``∂Q̃/∂U`` block (objectivity reduction) for a forward
    quantity ``Q = ⟨M:P⟩`` differentiated w.r.t. ``F̄``.

    Identical machinery to :class:`TangentBlock` with ``macro_var="Fbar"`` but
    the ``∂F/∂μ`` perturbations are the ``n_sym`` = 6 (3D) / 3 (2D) *symmetric*
    basis directions (sensitivities w.r.t. the stretch ``U``). The per-direction
    results are assembled into a tensor symmetric in the last two (stretch)
    indices, so this returns ``q_shape + (gdim, gdim)`` = ``∂Q̃/∂U``. The
    lab-frame ``dQ̄/dF̄ = ∂Q̃/∂U : dU/dF̄`` is reconstructed downstream in the
    solver's output transform.
    """

    def __init__(self, name: str, q_shape: tuple, M_factory):
        def _sym_dF_factory(context):
            gd = context.mesh.geometry.dim
            return [ufl.as_matrix(S.tolist()) for S in symmetric_basis_tensors(gd)]
        # mu_shape fixed up in setup() once gdim is known.
        super().__init__(name, "Fbar", q_shape, (1,), M_factory, _sym_dF_factory)

    def setup(self, context):
        gdim = context.mesh.geometry.dim
        self._gdim = gdim
        self._mu_shape = (len(symmetric_index_pairs(gdim)),)
        super().setup(context)

    def compute(self, context, adjoints):
        flat = super().compute(context, adjoints)         # q_shape + (n_sym,)
        arr = np.moveaxis(flat, -1, 0)                    # (n_sym, *q_shape)
        return assemble_symmetric_tangent(arr, self._gdim)  # (*q_shape, g, g)


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


def resolve_average_quantities(items, extra_factories=None) -> list[AverageQuantity]:
    """Normalize a list of ``AverageQuantity`` instances or string keys into instances.

    Strings are mapped via ``STRING_KEY_MAP`` and instantiated with no
    arguments — only valid for quantities that have no required state.
    ``extra_factories`` is an optional ``{key: zero-arg callable}`` map that
    takes precedence over ``STRING_KEY_MAP`` — solvers use it to offer keys
    for quantities that need solver-held state (e.g. the φ-bound ``"Pi"`` /
    ``"Lambda"`` / tangent blocks of the micromorphic solvers, via
    ``_string_key_factories()``).
    """
    extra_factories = extra_factories or {}
    out: list[AverageQuantity] = []
    for item in items:
        if isinstance(item, AverageQuantity):
            out.append(item)
        elif isinstance(item, str):
            factory = extra_factories.get(item) or STRING_KEY_MAP.get(item)
            if factory is None:
                raise ValueError(
                    f"Unknown average quantity key {item!r}. "
                    f"Valid keys: {sorted(set(STRING_KEY_MAP) | set(extra_factories))}"
                )
            out.append(factory())
        else:
            raise TypeError(
                "average_quantities entries must be AverageQuantity or str, "
                f"got {type(item).__name__}"
            )
    return out
