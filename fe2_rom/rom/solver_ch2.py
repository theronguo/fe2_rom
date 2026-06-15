"""Second-order (CH2) ROM-based periodic homogenization.

Subclass of :class:`fe2_rom.rom.solver_ch1.ReducedMicroSolver` that extends the
kinematic ansatz with the strain-gradient enrichment (paper Eq. 10):

    u_total = (F̄ - I)·X + ½ X·Ḡ·X + w(X)

with the extra third-order macro variable ``Ḡ`` (``Ḡ_iJK = ∂F̄_iJ/∂X_K``,
symmetric in ``J ↔ K``). Effective quantities reported at each load step:

    P̄ = ⟨P(F)⟩                       (Eq. 26)
    Q̄_iJK = ⟨ ½ (X_K P_iJ + X_J P_iK) ⟩   (Eq. 27)

plus the four macro tangents ``d{P̄, Q̄} / d{F̄, Ḡ}``.

Constraints (⟨w⟩ = 0, ∫_top w = ∫_right w = 0) are *not* enforced explicitly:
the POD basis already spans constraint-satisfying snapshots. The reduced
weighted-stress / higher-order-stress integrals are evaluated on the ECM
submesh with the magic-point weights ``ω`` (the ``context.weight`` carried by
the :class:`~fe2_rom.ch1.averages.HomogenizationContext`).

Call as ``solver(Fbar, G=None)``; a missing ``G`` defaults to zeros.
"""
from __future__ import annotations

import logging

import numpy as np
import ufl
from dolfinx import fem
from petsc4py import PETSc

from fe2_rom.ch1.averages import (
    EffectiveAbar,
    EffectiveFbar,
    EffectivePbar,
    TangentBlock,
)
from fe2_rom.ch2.averages import (
    EffectiveQ,
    basis_2tensor_factory,
    x_weighted_directions,
    x_weighted_factory,
)
from .solver_ch1 import ReducedMicroSolver as _Ch1ReducedMicroSolver

logger = logging.getLogger(__name__)


class ReducedMicroSolver(_Ch1ReducedMicroSolver):
    """ROM solver with the second-order strain-gradient enrichment."""

    _CH2_QUANTITY_KEYS = ("Qbar", "dPbar_dG", "dQbar_dFbar", "dQbar_dG")

    def __init__(self, *args, **kwargs):
        if kwargs.get("objective_reduction", False):
            raise NotImplementedError(
                "objective_reduction is not supported by the second-order (ch2) "
                "ROM; drive each RVE with the full (F̄, Ḡ).")
        self._ch2_target_G: np.ndarray | None = None
        self._ch2_prev_G: np.ndarray | None = None
        super().__init__(*args, **kwargs)

    # ---- hooks ----

    def _setup_macro_vars(self) -> None:
        g = self.gdim
        self.G_const = fem.Constant(
            self._submesh, np.zeros((g, g, g), dtype=PETSc.ScalarType))
        self._G_conv = np.zeros((g, g, g), dtype=PETSc.ScalarType)
        self.macro_vars["G"] = self.G_const

    def _build_F_ufl_extra(self):
        g = self.gdim
        X = ufl.SpatialCoordinate(self._submesh)
        G = self.G_const
        comps = []
        for i in range(g):
            expr = 0.0
            for J in range(g):
                for Kk in range(g):
                    expr = expr + G[i, J, Kk] * X[J] * X[Kk]
            comps.append(0.5 * expr)
        return ufl.as_vector(comps)

    def _make_dF_dmu_factories(self) -> dict:
        return {
            "Fbar": self._fbar_dF_factory,
            "G": lambda context: x_weighted_directions(context.mesh),
        }

    def _string_key_factories(self) -> dict:
        g = self.gdim
        s_F = (g, g)
        s_G = (g, g, g)
        s_P = (g, g)
        s_Q = (g, g, g)
        return {
            "Qbar": lambda: EffectiveQ(),
            "dPbar_dG": lambda: TangentBlock(
                "dPbar_dG", "G", s_P, s_G,
                basis_2tensor_factory, x_weighted_factory),
            "dQbar_dFbar": lambda: TangentBlock(
                "dQbar_dFbar", "Fbar", s_Q, s_F,
                x_weighted_factory, basis_2tensor_factory),
            "dQbar_dG": lambda: TangentBlock(
                "dQbar_dG", "G", s_Q, s_G,
                x_weighted_factory, x_weighted_factory),
        }

    def _make_default_average_quantities(self) -> list:
        qs = [EffectiveFbar(), EffectivePbar(), EffectiveAbar()]
        fac = self._string_key_factories()
        qs.extend(fac[key]() for key in self._CH2_QUANTITY_KEYS)
        return qs

    def _restore_trial_state(self) -> None:
        self.G_const.value[:] = self._G_conv

    def _update_macro_load_schedule(self, t: float) -> None:
        G_prev = self._ch2_prev_G if self._ch2_prev_G is not None else self._G_conv
        G_t = self._ch2_target_G if self._ch2_target_G is not None else self._G_conv
        self.G_const.value[:] = G_prev + t * (G_t - G_prev)

    def _commit_extra_state(self) -> None:
        self._G_conv[:] = self.G_const.value

    # ---- checkpoint hooks ----

    def dump_state(self) -> dict:
        return {
            "F_bar_conv": np.asarray(self.F_bar_conv, dtype=np.float64).copy(),
            "coeffs_conv": np.asarray(self.coeffs_conv, dtype=np.float64).copy(),
            "G_conv": np.asarray(self._G_conv, dtype=np.float64).copy(),
        }

    def load_state(self, d: dict) -> None:
        self.F_bar_conv[:] = np.asarray(d["F_bar_conv"], dtype=PETSc.ScalarType)
        self.coeffs_conv[:] = np.asarray(d["coeffs_conv"])
        self._G_conv[:] = np.asarray(d["G_conv"], dtype=PETSc.ScalarType)
        self.F_bar.value[:] = self.F_bar_conv
        self._restore_state(self.coeffs_conv)
        self._restore_trial_state()

    # ---- driver override ----

    def __call__(self, Fbar: np.ndarray, G: "np.ndarray | None" = None, **kwargs):
        g = self.gdim
        if G is None:
            G = np.zeros((g, g, g), dtype=PETSc.ScalarType)
        G_arr = np.asarray(G, dtype=PETSc.ScalarType).reshape(g, g, g)
        self._ch2_target_G = G_arr
        self._ch2_prev_G = self._G_conv.copy()
        return super().__call__(Fbar, **kwargs)
