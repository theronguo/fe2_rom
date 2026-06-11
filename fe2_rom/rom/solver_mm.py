"""Micromorphic (higher-order) ROM-based periodic homogenization.

Subclass of :class:`ReducedMicroSolver` that extends the kinematic ansatz with a finite
set of user-supplied global modes ``φᵢ`` (``i = 1..N``):

    u_total = (F̄ - I)·X + Σᵢ (vᵢ + X·gᵢ) φᵢ + w(X)

with extra macro variables ``v ∈ R^N`` and ``g ∈ R^{N×gdim}``. Effective
quantities reported at each load step:

    P̄        = ⟨P(F)⟩
    Πᵢ        = ⟨P : ∇φᵢ⟩
    Λᵢ        = ⟨Pᵀ·φᵢ + X·(P : ∇φᵢ)⟩

and the full 3×3 grid of tangents ``d{P̄, Π, Λ} / d{F̄, v, g}``.

Constraints (⟨w⟩ = 0, ⟨w·φᵢ⟩ = 0, ⟨(w·φᵢ)X⟩ = 0) are *not* enforced explicitly:
the POD basis ``basis_u_sub`` already spans constraint-satisfying snapshots.

φᵢ are passed at construction as a list of length ``N`` of either
:class:`dolfinx.fem.Function` on the submesh ``V_sub`` *or* raw NumPy arrays of
matching DOF count.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import ufl
from dolfinx import fem
from petsc4py import PETSc

from fe2_rom.ch1.averages import (
    EffectiveAbar,
    EffectiveAbarReduced,
    EffectiveFbar,
    EffectivePbar,
    TangentBlock,
    TangentBlockReduced,
)
from fe2_rom.ch1.objectivity import reconstruct_dscalar_dFbar
from fe2_rom.mm.averages import EffectiveLambda, EffectivePi
from ..hyperelastic_solver.forms import basis_tensor_ufl
from .solver_ch1 import ReducedMicroSolver, _dF_Fbar_factory
from .ecm import _parent_to_sub_array

logger = logging.getLogger(__name__)


# --- UFL factory helpers ----------------------------------------------------

def _basis_vec(gdim: int, d: int):
    return ufl.as_vector([1.0 if k == d else 0.0 for k in range(gdim)])


def _M_Pbar_factory(context):
    gdim = context.mesh.geometry.dim
    return [basis_tensor_ufl(gdim, i, j) for i in range(gdim) for j in range(gdim)]


def _M_Pi_factory(phi):
    def f(context):
        return [ufl.grad(p) for p in phi]
    return f


def _M_Lambda_factory(phi):
    def f(context):
        gdim = context.mesh.geometry.dim
        X = ufl.SpatialCoordinate(context.mesh)
        out = []
        for i in range(len(phi)):
            for e in range(gdim):
                out.append(
                    ufl.outer(phi[i], _basis_vec(gdim, e)) + X[e] * ufl.grad(phi[i])
                )
        return out
    return f


def _dF_v_factory(phi):
    def f(context):
        return [ufl.grad(p) for p in phi]
    return f


def _dF_g_factory(phi):
    def f(context):
        gdim = context.mesh.geometry.dim
        X = ufl.SpatialCoordinate(context.mesh)
        out = []
        for i in range(len(phi)):
            for d in range(gdim):
                out.append(
                    ufl.outer(phi[i], _basis_vec(gdim, d)) + X[d] * ufl.grad(phi[i])
                )
        return out
    return f


# --- Solver -----------------------------------------------------------------

class ReducedMicroSolver(ReducedMicroSolver):
    """ROM solver with micromorphic enrichment over ``N`` global modes.

    Parameters
    ----------
    phi : sequence of length ``N``
        Each entry is either a :class:`dolfinx.fem.Function` on ``V_sub`` or a
        1D NumPy array of length ``V_sub.dofmap.index_map.size_local *
        block_size``. Lives on the same mesh that the ROM uses.
    *args, **kwargs
        Forwarded to :class:`ReducedMicroSolver`.

    Notes
    -----
    Call as ``solver(Fbar, v=None, g=None)``. Missing ``v``/``g`` default to
    zero arrays of the appropriate shape (``(N,)`` and ``(N, gdim)``). With
    ``N == 0`` the solver degenerates to the parent's behaviour.
    """

    def __init__(self, *args, phi: Sequence, **kwargs) -> None:
        self._phi_pending = list(phi)
        self._N_modes = len(self._phi_pending)
        # Ramp endpoints — stashed by ``__call__`` before delegating to super.
        self._target_v: np.ndarray | None = None
        self._target_g: np.ndarray | None = None
        self._prev_v: np.ndarray | None = None
        self._prev_g: np.ndarray | None = None
        super().__init__(*args, **kwargs)

    # ---- hooks ----

    def _setup_macro_vars(self) -> None:
        N = self._N_modes
        gdim = self.gdim

        # Allocate φ Functions on V_sub (used in the ROM forms) and on V_full
        # (used by the u_total visualisation). Inputs are accepted in three
        # forms: a Function on V_sub, a Function on (any) V_full-like space,
        # or a raw NumPy array of either V_full or V_sub DOF count.
        self._phi = []
        self._phi_full = []
        sub_size = fem.Function(self.V_sub).x.array.size
        full_size = fem.Function(self.V_full).x.array.size
        for i, src in enumerate(self._phi_pending):
            f_sub = fem.Function(self.V_sub, name=f"phi_{i}")
            f_full = fem.Function(self.V_full, name=f"phi_full_{i}")
            if isinstance(src, fem.Function):
                src_size = src.x.array.size
                if src_size == sub_size:
                    f_sub.x.array[:] = src.x.array
                    # No full-mesh source available — leave f_full at zero
                    # (u_total visualisation will be incomplete).
                    f_full = None
                elif src_size == full_size:
                    f_full.x.array[:] = src.x.array
                    f_full.x.scatter_forward()
                    f_sub.x.array[:] = _parent_to_sub_array(
                        f_full.x.array, self.V_full, self.V_sub, self._sub_cell_map,
                    )
                else:
                    raise ValueError(
                        f"phi[{i}] Function size {src_size} matches neither "
                        f"V_sub ({sub_size}) nor V_full ({full_size})"
                    )
            else:
                arr = np.asarray(src, dtype=PETSc.ScalarType).reshape(-1)
                if arr.size == full_size:
                    f_full.x.array[:] = arr
                    f_full.x.scatter_forward()
                    f_sub.x.array[:] = _parent_to_sub_array(
                        f_full.x.array, self.V_full, self.V_sub, self._sub_cell_map,
                    )
                elif arr.size == sub_size:
                    f_sub.x.array[:] = arr
                    f_full = None
                else:
                    raise ValueError(
                        f"phi[{i}] array size {arr.size} matches neither "
                        f"V_sub ({sub_size}) nor V_full ({full_size})"
                    )
            f_sub.x.scatter_forward()
            if f_full is not None:
                f_full.x.scatter_forward()
            self._phi.append(f_sub)
            self._phi_full.append(f_full)

        if N > 0:
            self.v_const = fem.Constant(
                self._submesh, np.zeros(N, dtype=PETSc.ScalarType),
            )
            self.g_const = fem.Constant(
                self._submesh, np.zeros((N, gdim), dtype=PETSc.ScalarType),
            )
            self.macro_vars["v"] = self.v_const
            self.macro_vars["g"] = self.g_const
            # Full-mesh mirrors of v / g, kept in sync at write time so the
            # u_total visualisation expression on V_full sees the correct
            # macro state. Constants on submesh can't be used inside a UFL
            # expression on the full mesh.
            self.v_const_full = fem.Constant(
                self._mesh_full, np.zeros(N, dtype=PETSc.ScalarType),
            )
            self.g_const_full = fem.Constant(
                self._mesh_full, np.zeros((N, gdim), dtype=PETSc.ScalarType),
            )
        else:
            self.v_const = None
            self.g_const = None
            self.v_const_full = None
            self.g_const_full = None
        self._v_conv = np.zeros(N, dtype=PETSc.ScalarType)
        self._g_conv = np.zeros((N, gdim), dtype=PETSc.ScalarType)

    def _build_F_ufl_extra(self):
        if self._N_modes == 0:
            return None
        X = ufl.SpatialCoordinate(self._submesh)
        gdim = self.gdim
        u_extra = None
        for i in range(self._N_modes):
            coeff = self.v_const[i]
            for d in range(gdim):
                coeff = coeff + X[d] * self.g_const[i, d]
            term = coeff * self._phi[i]
            u_extra = term if u_extra is None else u_extra + term
        return u_extra

    def _build_u_total_extra_full(self, mesh):
        """``Σᵢ (vᵢ + X·gᵢ) φᵢ`` expressed on the *full* mesh for u_total
        visualisation. Requires φᵢ on V_full (passed in as full-mesh
        Functions); otherwise returns ``None`` and visualisation skips the
        enrichment term.
        """
        if self._N_modes == 0:
            return None
        if any(p is None for p in self._phi_full):
            logger.warning(
                "u_total visualisation: φᵢ on the full mesh not available; "
                "the enrichment term Σᵢ (vᵢ + X·gᵢ) φᵢ will be missing from "
                "u_total. Pass φᵢ as fem.Function on V_full to enable it."
            )
            return None
        gdim = self.gdim
        X = ufl.SpatialCoordinate(mesh)
        u_extra = None
        for i in range(self._N_modes):
            coeff = self.v_const_full[i]
            for d in range(gdim):
                coeff = coeff + X[d] * self.g_const_full[i, d]
            term = coeff * self._phi_full[i]
            u_extra = term if u_extra is None else u_extra + term
        return u_extra

    def _sync_full_mesh_constants(self) -> None:
        if self._N_modes == 0 or self.v_const_full is None:
            return
        self.v_const_full.value[:] = self.v_const.value
        self.g_const_full.value[:] = self.g_const.value

    def _make_dF_dmu_factories(self) -> dict:
        # F̄ directions: 6/3 symmetric (objective reduction) or full gdim²,
        # selected by the base-class helper that honours ``objective_reduction``.
        factories = {"Fbar": self._fbar_dF_factory}
        if self._N_modes > 0:
            phi = self._phi
            factories["v"] = _dF_v_factory(phi)
            factories["g"] = _dF_g_factory(phi)
        return factories

    def _objectivize_quantities(self, qs: list) -> list:
        """Swap quantities for their U-frame reduced variants (objectivity
        reduction): ``EffectiveAbar`` → ``EffectiveAbarReduced`` and the
        ``…_dFbar`` blocks → ``TangentBlockReduced``; ``/dv``, ``/dg`` and the
        forward ``Pi``/``Lambda`` are objective invariants and left untouched."""
        out = []
        for q in qs:
            if type(q) is EffectiveAbar:
                out.append(EffectiveAbarReduced())
            elif isinstance(q, TangentBlock) and q._macro_var == "Fbar":
                out.append(TangentBlockReduced(q.name, q._q_shape, q._M_factory))
            else:
                out.append(q)
        return out

    def _objective_transform_extra(self, d: dict, R, dR, dU) -> None:
        """Lab-frame conversion of the micromorphic blocks (co-rotational
        ansatz ``φ → R φ``). ``Pi``/``Lambda`` and their ``/dv``, ``/dg``
        derivatives are rotation-invariant; only the ``P̄`` blocks rotate and the
        ``…_dFbar`` blocks contract the U-frame reduced tangent with ``dU/dF̄``."""
        if "dPbar_dv" in d:
            d["dPbar_dv"] = np.einsum("im,mJn->iJn", R, np.asarray(d["dPbar_dv"]))
        if "dPbar_dg" in d:
            d["dPbar_dg"] = np.einsum("im,mJnd->iJnd", R, np.asarray(d["dPbar_dg"]))
        if "dPi_dFbar" in d:
            d["dPi_dFbar"] = reconstruct_dscalar_dFbar(
                np.asarray(d["dPi_dFbar"]), dU)
        if "dLambda_dFbar" in d:
            d["dLambda_dFbar"] = reconstruct_dscalar_dFbar(
                np.asarray(d["dLambda_dFbar"]), dU)

    # Keys (in default registration order) of the φ-bound micromorphic
    # quantities offered by ``_string_key_factories``.
    _MM_QUANTITY_KEYS = (
        "Pi", "Lambda",
        "dPbar_dv", "dPbar_dg",
        "dPi_dFbar", "dPi_dv", "dPi_dg",
        "dLambda_dFbar", "dLambda_dv", "dLambda_dg",
    )

    def _string_key_factories(self) -> dict:
        """φ-bound micromorphic quantities, usable as string keys in
        ``average_quantities`` (e.g. ``["Wbar", "Pbar", "Pi", "Lambda"]``)."""
        gdim = self.gdim
        N = self._N_modes
        phi = self._phi
        if N == 0:
            return {}

        s_Fbar = (gdim, gdim)
        s_v = (N,)
        s_g = (N, gdim)
        s_Pbar = (gdim, gdim)
        s_Pi = (N,)
        s_Lambda = (N, gdim)

        return {
            "Pi": lambda: EffectivePi(phi),
            "Lambda": lambda: EffectiveLambda(phi),
            # dPbar / d{v, g} (dPbar/dFbar = EffectiveAbar, key "dPbar_dFbar")
            "dPbar_dv": lambda: TangentBlock(
                "dPbar_dv", "v", s_Pbar, s_v,
                _M_Pbar_factory, _dF_v_factory(phi)),
            "dPbar_dg": lambda: TangentBlock(
                "dPbar_dg", "g", s_Pbar, s_g,
                _M_Pbar_factory, _dF_g_factory(phi)),
            # dPi / d{Fbar, v, g}
            "dPi_dFbar": lambda: TangentBlock(
                "dPi_dFbar", "Fbar", s_Pi, s_Fbar,
                _M_Pi_factory(phi), _dF_Fbar_factory),
            "dPi_dv": lambda: TangentBlock(
                "dPi_dv", "v", s_Pi, s_v,
                _M_Pi_factory(phi), _dF_v_factory(phi)),
            "dPi_dg": lambda: TangentBlock(
                "dPi_dg", "g", s_Pi, s_g,
                _M_Pi_factory(phi), _dF_g_factory(phi)),
            # dLambda / d{Fbar, v, g}
            "dLambda_dFbar": lambda: TangentBlock(
                "dLambda_dFbar", "Fbar", s_Lambda, s_Fbar,
                _M_Lambda_factory(phi), _dF_Fbar_factory),
            "dLambda_dv": lambda: TangentBlock(
                "dLambda_dv", "v", s_Lambda, s_v,
                _M_Lambda_factory(phi), _dF_v_factory(phi)),
            "dLambda_dg": lambda: TangentBlock(
                "dLambda_dg", "g", s_Lambda, s_g,
                _M_Lambda_factory(phi), _dF_g_factory(phi)),
        }

    def _make_default_average_quantities(self) -> list:
        qs: list = [EffectiveFbar(), EffectivePbar(), EffectiveAbar()]
        factories = self._string_key_factories()
        qs.extend(factories[key]() for key in self._MM_QUANTITY_KEYS if key in factories)
        return qs

    def _restore_trial_state(self) -> None:
        if self._N_modes == 0:
            return
        self.v_const.value[:] = self._v_conv
        self.g_const.value[:] = self._g_conv

    def _update_macro_load_schedule(self, t: float) -> None:
        if self._N_modes == 0:
            return
        v_prev = self._prev_v if self._prev_v is not None else self._v_conv
        g_prev = self._prev_g if self._prev_g is not None else self._g_conv
        v_tgt = self._target_v if self._target_v is not None else self._v_conv
        g_tgt = self._target_g if self._target_g is not None else self._g_conv
        self.v_const.value[:] = v_prev + t * (v_tgt - v_prev)
        self.g_const.value[:] = g_prev + t * (g_tgt - g_prev)

    def _commit_extra_state(self) -> None:
        if self._N_modes == 0:
            return
        self._v_conv[:] = self.v_const.value
        self._g_conv[:] = self.g_const.value

    # ---- driver override ----

    def __call__(
        self,
        Fbar: np.ndarray,
        v: "np.ndarray | None" = None,
        g: "np.ndarray | None" = None,
        **kwargs,
    ):
        if self._N_modes > 0:
            if v is None:
                v = np.zeros(self._N_modes, dtype=PETSc.ScalarType)
            if g is None:
                g = np.zeros((self._N_modes, self.gdim), dtype=PETSc.ScalarType)
            v_arr = np.asarray(v, dtype=PETSc.ScalarType).reshape(self._N_modes)
            g_arr = np.asarray(g, dtype=PETSc.ScalarType).reshape(
                self._N_modes, self.gdim,
            )
            self._target_v = v_arr
            self._target_g = g_arr
            self._prev_v = self._v_conv.copy()
            self._prev_g = self._g_conv.copy()
        return super().__call__(Fbar, **kwargs)
