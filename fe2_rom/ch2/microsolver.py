"""Second-order (CH2) periodic hyperelastic homogenization.

Subclass of :class:`fe2_rom.ch1.MicroSolver` adding the strain-gradient
enrichment of the second-order ansatz (Eq. 10):

    u_total = (F̄ − I)·X + ½ X·Ḡ·X + w

with a single new third-order macro variable ``Ḡ`` (the gradient of the
deformation gradient, ``Ḡ_iJK = ∂F̄_iJ/∂X_K``, symmetric in its last two indices
``J ↔ K``). The microscopic deformation gradient becomes (Eq. 11)

    F_iJ = F̄_iJ + X_K Ḡ_iJK + ∂w_i/∂X_J .

The RVE reports the effective stress ``P̄`` (Eq. 26), the double stress ``Q̄``
(Eq. 27) and the four macro tangents ``d{P̄,Q̄}/d{F̄,Ḡ}``.

Constraints on the fluctuation ``w`` (Eqs. 17–19), enforced through the projected
Newton solve (``P K P x = P b``):

    ⟨w⟩            = 0   (gdim rows, ZeroVolumeAverage,   Eq. 19)
    ∫_{max face d} w ds = 0   (gdim rows per axis, ZeroBoundaryAverage, Eqs. 17–18)

The boundary-average rows are imposed on the ``gdim`` "max" faces (top / right /
front); under periodic BCs the matching min faces follow automatically. Periodic
BCs on ``w`` are mandatory, so the solver runs in the full-periodicity regime
(``corner_periodic=True`` for a box, or ``lattice_vectors`` for a polygon).

Two-phase use, mirroring the micromorphic solver:

1. Build the solver.
2. Call ``self(Fbar, G)`` with the macroscopic deformation gradient and its
   gradient. ``G`` defaults to zero (⇒ first-order behaviour).
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
from fe2_rom.ch1.constraints import ZeroVolumeAverage
from fe2_rom.ch1.microsolver import MicroSolver as _Ch1MicroSolver
from fe2_rom.ch2.averages import (
    EffectiveQ,
    basis_2tensor_factory,
    x_weighted_directions,
    x_weighted_factory,
)
from fe2_rom.ch2.constraints import ZeroBoundaryAverage

logger = logging.getLogger(__name__)


class MicroSolver(_Ch1MicroSolver):
    """Full-order second-order homogenization driver.

    With ``G`` left at zero the solver reduces exactly to the first-order
    response of the parent (plus the extra — but inactive — constraints).
    """

    # φ-bound keys are absent here; these are the CH2 tangent / double-stress
    # quantities offered via ``_string_key_factories`` (default registration
    # order).
    _CH2_QUANTITY_KEYS = ("Qbar", "dPbar_dG", "dQbar_dFbar", "dQbar_dG")

    def __init__(self, mesh_path, comm, gdim, material, **kwargs):
        if kwargs.get("objective_reduction", False):
            raise NotImplementedError(
                "objective_reduction is not supported by the second-order (ch2) "
                "solver; drive each RVE with the full (F̄, Ḡ)."
            )
        # Ramp endpoints for Ḡ, stashed by __call__ and read by
        # _update_macro_load_schedule.
        self._ch2_target_G: np.ndarray | None = None
        self._ch2_prev_G: np.ndarray | None = None
        # Full periodicity (PBC on w) is required by the second-order
        # decomposition. On a box that is the corner-periodic regime; an
        # arbitrary polygon supplies lattice_vectors instead.
        if kwargs.get("lattice_vectors") is None:
            kwargs["corner_periodic"] = True
        super().__init__(mesh_path, comm, gdim, material, **kwargs)

    # ------------------------------------------------------------------
    # Subclass hooks: macro variable Ḡ and the ½ X·Ḡ·X ansatz term
    # ------------------------------------------------------------------

    def _setup_macro_vars(self):
        g = self.gdim
        self.G_const = fem.Constant(
            self._mesh, np.zeros((g, g, g), dtype=PETSc.ScalarType)
        )
        self._G_conv = np.zeros((g, g, g), dtype=PETSc.ScalarType)
        self.macro_vars = {"Fbar": self.F_bar, "G": self.G_const}

    def _build_u_total_extra(self):
        g = self.gdim
        X = ufl.SpatialCoordinate(self._mesh)
        G = self.G_const
        # u_extra_i = ½ Σ_{J,K} Ḡ_iJK X_J X_K
        comps = []
        for i in range(g):
            expr = 0.0
            for J in range(g):
                for K in range(g):
                    expr = expr + G[i, J, K] * X[J] * X[K]
            comps.append(0.5 * expr)
        return ufl.as_vector(comps)

    def _build_macro_var_rhs_forms(self, build_tangent_rhs_forms):
        return {
            "Fbar": build_tangent_rhs_forms(self._fbar_adjoint_directions()),
            "G": build_tangent_rhs_forms(x_weighted_directions(self._mesh)),
        }

    # ------------------------------------------------------------------
    # Constraints (Eqs. 17–19)
    # ------------------------------------------------------------------

    @staticmethod
    def _make_max_face_locator(axis: int, value: float, tol: float):
        def loc(x):
            return np.isclose(x[axis], value, atol=tol, rtol=0.0)
        return loc

    def _build_constraint_forms(self, constraints):
        if constraints is None:
            tol = 1e-8 * max(1.0, float(np.max(self.maxs - self.mins)))
            constraints = [ZeroVolumeAverage()]  # Eq. 19: ⟨w⟩ = 0
            for axis in range(self.gdim):  # Eqs. 17–18: ∫_{max face} w = 0
                constraints.append(
                    ZeroBoundaryAverage(
                        self._make_max_face_locator(axis, self.maxs[axis], tol),
                        name=f"max_axis{axis}",
                    )
                )
        forms = []
        for c in constraints:
            c_forms, _ = c.build(self.V, self.dx, self._mesh, self.mpc)
            forms.extend(c_forms)
        logger.debug(
            "CH2 constraint forms: %d (%d mean + %d boundary-average)",
            len(forms), self.gdim, self.gdim * self.gdim,
        )
        return forms

    # ------------------------------------------------------------------
    # Effective quantities (Q̄ and the macro tangents)
    # ------------------------------------------------------------------

    def _string_key_factories(self) -> dict:
        g = self.gdim
        s_F = (g, g)
        s_G = (g, g, g)
        s_P = (g, g)
        s_Q = (g, g, g)
        return {
            "Qbar": lambda: EffectiveQ(),
            # dP̄/dḠ — dP̄/dF̄ is EffectiveAbar ("dPbar_dFbar").
            "dPbar_dG": lambda: TangentBlock(
                "dPbar_dG", "G", s_P, s_G,
                basis_2tensor_factory, x_weighted_factory),
            # dQ̄/d{F̄, Ḡ}
            "dQbar_dFbar": lambda: TangentBlock(
                "dQbar_dFbar", "Fbar", s_Q, s_F,
                x_weighted_factory, basis_2tensor_factory),
            "dQbar_dG": lambda: TangentBlock(
                "dQbar_dG", "G", s_Q, s_G,
                x_weighted_factory, x_weighted_factory),
        }

    def _make_default_average_quantities(self):
        qs = [
            EffectiveFbar(),
            EffectivePbar(),
            EffectiveAbar(),  # name="dPbar_dFbar"
        ]
        fac = self._string_key_factories()
        qs.extend(fac[key]() for key in self._CH2_QUANTITY_KEYS)
        return qs

    # ------------------------------------------------------------------
    # Trial-state / load-ramp / commit / checkpoint hooks for Ḡ
    # ------------------------------------------------------------------

    def _restore_trial_state(self):
        self.G_const.value[:] = self._G_conv

    def _update_macro_load_schedule(self, t: float):
        G_prev = self._ch2_prev_G if self._ch2_prev_G is not None else self._G_conv
        G_t = self._ch2_target_G if self._ch2_target_G is not None else self._G_conv
        self.G_const.value[:] = G_prev + t * (G_t - G_prev)

    def _commit_extra_state(self):
        self._G_conv[:] = self.G_const.value

    def _dump_extra_state(self) -> dict:
        return {"G_conv": np.asarray(self._G_conv, dtype=np.float64).copy()}

    def _load_extra_state(self, d) -> None:
        G = np.asarray(d["G_conv"], dtype=PETSc.ScalarType)
        if G.shape != self._G_conv.shape:
            raise RuntimeError(
                f"G_conv shape mismatch: got {G.shape}, expected "
                f"{self._G_conv.shape}"
            )
        self._G_conv[:] = G
        self.G_const.value[:] = self._G_conv

    # ------------------------------------------------------------------
    # Driver override
    # ------------------------------------------------------------------

    def __call__(self, Fbar: np.ndarray, G: "np.ndarray | None" = None, **kwargs):
        g = self.gdim
        if G is None:
            G = np.zeros((g, g, g), dtype=PETSc.ScalarType)
        G_arr = np.asarray(G, dtype=PETSc.ScalarType).reshape(g, g, g)
        self._ch2_target_G = G_arr
        self._ch2_prev_G = self._G_conv.copy()
        return super().__call__(Fbar, **kwargs)
