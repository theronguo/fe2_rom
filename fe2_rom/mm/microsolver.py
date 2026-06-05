"""Higher-order (micromorphic) periodic homogenization.

Subclass of :class:`MicroSolver` that adds a
finite set of user-provided global modes ``φᵢ`` (``i = 1..N``) and the
displacement ansatz

    u_total = (F̄ - I)·X + Σᵢ (vᵢ + X·gᵢ) φᵢ + w

with extra macro variables ``v ∈ R^N`` and ``g ∈ R^{N×gdim}``. Reports the
3×3 grid of effective quantity / macro-variable tangents (``dPbar``, ``dPi``,
``dLambda``) × (``dFbar``, ``dv``, ``dg``).

Constraints on the fluctuation field ``w``
------------------------------------------
The decomposition is unique only if the three families of integral constraints
hold:

    ⟨w⟩            = 0   (gdim rows, ZeroVolumeAverage)
    ⟨w · φᵢ⟩       = 0   (N rows,    ZeroVolumeAverageDot)
    ⟨(w · φᵢ) X_b⟩ = 0   (N·gdim rows, ZeroVolumeAverageOuter)

These are built in ``_build_constraint_forms`` and enforced via the projected
Newton solver (P K P x = P b).  Because the φᵢ are zero at construction and
populated later by ``compute_linear_buckling_modes``, the projected solver
must be told to rebuild its constraint vectors — call ``rebuild_constraints()``
(or let ``compute_linear_buckling_modes`` do it automatically).
"""

from __future__ import annotations

import logging

import numpy as np
import ufl
from dolfinx import fem
from petsc4py import PETSc

from fe2_rom.ch1.averages import EffectiveAbar, EffectiveFbar, EffectivePbar, TangentBlock
from fe2_rom.mm.averages import EffectiveLambda, EffectivePi
from fe2_rom.ch1.constraints import ZeroVolumeAverage
from fe2_rom.mm.constraints import ZeroVolumeAverageDot, ZeroVolumeAverageOuter
from fe2_rom.hyperelastic_solver.forms import basis_tensor_ufl
from fe2_rom.ch1.microsolver import MicroSolver as _Ch1MicroSolver

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# UFL factory helpers — used by TangentBlock to construct the M and ∂F/∂μ
# 2-tensor lists on demand once the HomogenizationContext is available.
# ---------------------------------------------------------------------------

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
                out.append(ufl.outer(phi[i], _basis_vec(gdim, e))
                           + X[e] * ufl.grad(phi[i]))
        return out
    return f


def _dF_Fbar_factory(context):
    gdim = context.mesh.geometry.dim
    return [basis_tensor_ufl(gdim, i, j) for i in range(gdim) for j in range(gdim)]


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
                out.append(ufl.outer(phi[i], _basis_vec(gdim, d))
                           + X[d] * ufl.grad(phi[i]))
        return out
    return f


# ---------------------------------------------------------------------------
# Micromorphic solver
# ---------------------------------------------------------------------------

class MicroSolver(
    _Ch1MicroSolver
):
    """Higher-order homogenization driver with ``N`` user-provided modes.

    Construction is two-phase:

    1. Build the solver with the number of modes ``N``. The ``φᵢ`` Functions
       are allocated as zeros on ``self.V`` and exposed via ``self._phi``.
    2. Populate each ``self._phi[i]`` (e.g. ``phi.interpolate(...)`` or by
       writing into ``phi.x.array``) before calling ``self(Fbar, v, g)``.

    With ``N == 0`` the solver degenerates to the parent's behaviour: no
    extra macro variables, no extra ansatz term, no extra effective
    quantities. This is the sanity-check path in the plan.
    """

    def __init__(self, mesh_path, comm, gdim, material, *, N: int, **kwargs):
        self._N_modes = int(N)
        # Ramp endpoints stashed by __call__ and read by _update_macro_load_schedule.
        self._mm_target_v: np.ndarray | None = None
        self._mm_target_g: np.ndarray | None = None
        self._mm_prev_v: np.ndarray | None = None
        self._mm_prev_g: np.ndarray | None = None
        # Full periodicity is required for the micromorphic decomposition. On a
        # box that is the corner-periodic regime; for an arbitrary polygon the
        # caller supplies ``lattice_vectors`` (mutually exclusive with
        # ``corner_periodic``), which selects the equivalent polygon path.
        if kwargs.get("lattice_vectors") is None:
            kwargs["corner_periodic"] = True
        super().__init__(mesh_path, comm, gdim, material, **kwargs)

    # ---- subclass hooks ----

    def _build_constraint_forms(self, constraints):
        """Build constraint forms for the projected Newton solver.

        When ``constraints`` is ``None`` and ``corner_periodic=True``, builds
        the full micromorphic constraint set:

            ⟨w⟩ = 0              (gdim rows — ZeroVolumeAverage)
            ⟨w · φᵢ⟩ = 0         (N rows    — ZeroVolumeAverageDot)
            ⟨(w · φᵢ) X_b⟩ = 0   (N·gdim   — ZeroVolumeAverageOuter)

        The φ-dependent forms capture ``self._phi[i]`` by reference so that
        ``rebuild_constraints()`` (called after LBA) assembles correct vectors.
        Pass an explicit ``constraints`` list to override entirely.
        """
        all_constraints = [ZeroVolumeAverage()]
        for phi in self._phi:
            all_constraints.append(ZeroVolumeAverageDot(phi))
            all_constraints.append(ZeroVolumeAverageOuter(phi))

        forms = []
        for c in all_constraints:
            c_forms, _ = c.build(self.V, self.dx, self._mesh, self.mpc)
            forms.extend(c_forms)

        n_base = self.gdim
        n_dot = self._N_modes
        n_outer = self._N_modes * self.gdim
        logger.debug(
            "Micromorphic constraint forms: %d total "
            "(%d mean + %d dot + %d outer)",
            len(forms), n_base, n_dot, n_outer,
        )
        return forms

    def rebuild_constraints(self) -> None:
        """Reassemble constraint vectors after φ-modes change.

        Call this (or let ``compute_linear_buckling_modes`` call it) whenever
        ``self._phi`` is updated so the projected Newton solver sees the
        correct ⟨w·φᵢ⟩ = 0 and ⟨(w·φᵢ)X⟩ = 0 rows.
        """
        self._newton.rebuild_constraint_vecs()

    def _setup_phi(self):
        self._phi = [
            fem.Function(self.V, name=f"phi_{i}") for i in range(self._N_modes)
        ]

    def _setup_macro_vars(self):
        N = self._N_modes
        gdim = self.gdim
        if N > 0:
            self.v_const = fem.Constant(
                self._mesh, np.zeros(N, dtype=PETSc.ScalarType),
            )
            self.g_const = fem.Constant(
                self._mesh, np.zeros((N, gdim), dtype=PETSc.ScalarType),
            )
        else:
            self.v_const = None
            self.g_const = None
        self._v_conv = np.zeros(N, dtype=PETSc.ScalarType)
        self._g_conv = np.zeros((N, gdim), dtype=PETSc.ScalarType)

        self.macro_vars = {"Fbar": self.F_bar}
        if N > 0:
            self.macro_vars["v"] = self.v_const
            self.macro_vars["g"] = self.g_const

    def _build_u_total_extra(self):
        if self._N_modes == 0:
            return None
        X = ufl.SpatialCoordinate(self._mesh)
        gdim = self.gdim
        u_extra = None
        for i in range(self._N_modes):
            coeff = self.v_const[i]
            for d in range(gdim):
                coeff = coeff + X[d] * self.g_const[i, d]
            term = coeff * self._phi[i]
            u_extra = term if u_extra is None else u_extra + term
        return u_extra

    def _build_macro_var_rhs_forms(self, build_tangent_rhs_forms):
        gdim = self.gdim
        dF_dFbar = [basis_tensor_ufl(gdim, i, j)
                    for i in range(gdim) for j in range(gdim)]
        rhs = {"Fbar": build_tangent_rhs_forms(dF_dFbar)}
        if self._N_modes > 0:
            dF_dv = [ufl.grad(self._phi[i]) for i in range(self._N_modes)]
            rhs["v"] = build_tangent_rhs_forms(dF_dv)
            X = ufl.SpatialCoordinate(self._mesh)
            dF_dg = []
            for i in range(self._N_modes):
                for d in range(gdim):
                    dF_dg.append(
                        ufl.outer(self._phi[i], _basis_vec(gdim, d))
                        + X[d] * ufl.grad(self._phi[i])
                    )
            rhs["g"] = build_tangent_rhs_forms(dF_dg)
        return rhs

    def _make_default_average_quantities(self):
        gdim = self.gdim
        N = self._N_modes
        phi = self._phi

        qs: list = [
            EffectiveFbar(),
            EffectivePbar(),
            EffectiveAbar(),  # name="dPbar_dFbar"
        ]
        if N == 0:
            return qs

        qs.extend([EffectivePi(phi), EffectiveLambda(phi)])

        # Shapes used to size the tangent blocks.
        s_Fbar = (gdim, gdim)
        s_v = (N,)
        s_g = (N, gdim)
        s_Pbar = (gdim, gdim)
        s_Pi = (N,)
        s_Lambda = (N, gdim)

        qs.extend([
            # dPbar / d{v, g} — Fbar already covered by EffectiveAbar.
            TangentBlock(
                "dPbar_dv", "v", s_Pbar, s_v,
                _M_Pbar_factory, _dF_v_factory(phi),
            ),
            TangentBlock(
                "dPbar_dg", "g", s_Pbar, s_g,
                _M_Pbar_factory, _dF_g_factory(phi),
            ),
            # dPi / d{Fbar, v, g}
            TangentBlock(
                "dPi_dFbar", "Fbar", s_Pi, s_Fbar,
                _M_Pi_factory(phi), _dF_Fbar_factory,
            ),
            TangentBlock(
                "dPi_dv", "v", s_Pi, s_v,
                _M_Pi_factory(phi), _dF_v_factory(phi),
            ),
            TangentBlock(
                "dPi_dg", "g", s_Pi, s_g,
                _M_Pi_factory(phi), _dF_g_factory(phi),
            ),
            # dLambda / d{Fbar, v, g}
            TangentBlock(
                "dLambda_dFbar", "Fbar", s_Lambda, s_Fbar,
                _M_Lambda_factory(phi), _dF_Fbar_factory,
            ),
            TangentBlock(
                "dLambda_dv", "v", s_Lambda, s_v,
                _M_Lambda_factory(phi), _dF_v_factory(phi),
            ),
            TangentBlock(
                "dLambda_dg", "g", s_Lambda, s_g,
                _M_Lambda_factory(phi), _dF_g_factory(phi),
            ),
        ])
        return qs

    def _restore_trial_state(self):
        if self._N_modes == 0:
            return
        self.v_const.value[:] = self._v_conv
        self.g_const.value[:] = self._g_conv

    def _update_macro_load_schedule(self, t: float):
        if self._N_modes == 0:
            return
        v_prev = self._mm_prev_v if self._mm_prev_v is not None else self._v_conv
        g_prev = self._mm_prev_g if self._mm_prev_g is not None else self._g_conv
        v_t = self._mm_target_v if self._mm_target_v is not None else self._v_conv
        g_t = self._mm_target_g if self._mm_target_g is not None else self._g_conv
        self.v_const.value[:] = v_prev + t * (v_t - v_prev)
        self.g_const.value[:] = g_prev + t * (g_t - g_prev)

    def _commit_extra_state(self):
        if self._N_modes == 0:
            return
        self._v_conv[:] = self.v_const.value
        self._g_conv[:] = self.g_const.value

    # ---- checkpoint hooks ----------------------------------------------

    def _dump_extra_state(self) -> dict:
        if self._N_modes == 0:
            return {}
        return {
            "v_conv": np.asarray(self._v_conv, dtype=np.float64).copy(),
            "g_conv": np.asarray(self._g_conv, dtype=np.float64).copy(),
        }

    def _load_extra_state(self, d) -> None:
        if self._N_modes == 0:
            return
        v = np.asarray(d["v_conv"], dtype=PETSc.ScalarType)
        g = np.asarray(d["g_conv"], dtype=PETSc.ScalarType)
        if v.shape != self._v_conv.shape:
            raise RuntimeError(
                f"v_conv shape mismatch: got {v.shape}, expected "
                f"{self._v_conv.shape}"
            )
        if g.shape != self._g_conv.shape:
            raise RuntimeError(
                f"g_conv shape mismatch: got {g.shape}, expected "
                f"{self._g_conv.shape}"
            )
        self._v_conv[:] = v
        self._g_conv[:] = g
        # Seed trial values too so the next solve warm-starts.
        self.v_const.value[:] = self._v_conv
        self.g_const.value[:] = self._g_conv
        # φ may have been populated after construction (LBA or external
        # load) — refresh the constraint forms so the projected Newton
        # sees the correct φ-dependent constraints. Safe to call even
        # when no extra state is loaded.
        self.rebuild_constraints()

    # ---- linear buckling basis -----------------------------------------

    def compute_linear_buckling_modes(
        self, n_modes: int, *,
        Fbar: "np.ndarray | None" = None,
        v: "np.ndarray | None" = None,
        g: "np.ndarray | None" = None,
        tol: float = 1e-6,
        slepc_options: "dict | None" = None,
        visualize_modes: bool = False,
        modes_filename: str = "buckling_modes.bp",
        save_modes: bool = False,
        n_skip: "int | None" = None,
    ) -> np.ndarray:
        """Linear buckling analysis: load the first ``n_modes`` eigenmodes of the
        tangent ``K`` at a reference macro state into ``self._phi``.

        Thin wrapper around the base
        :meth:`fe2_rom.ch1.MicroSolver.compute_buckling_spectrum`: it sets the
        reference state, delegates the eigensolve / normalization / optional
        save+visualize, then loads the returned modes into ``self._phi`` via
        :meth:`load_buckling_modes` (which also rebuilds the φ-dependent
        constraints).

        Procedure:

        1. Reset φ to zero and solve the RVE at the reference state
           ``(Fbar, v, g)`` (default ``Fbar=I``, ``v=0``, ``g=0`` — undeformed;
           pass an ``Fbar`` close to / past the critical load for true buckling
           modes). Because φ=0 makes the φ-dependent constraint rows vanish, the
           reference solve temporarily restricts to the ``gdim`` base rows
           (``ZeroVolumeAverage``).
        2. Delegate to ``compute_buckling_spectrum`` (assemble ``K``, solve the
           ``n_modes + n_skip`` smallest eigenpairs, skip ``n_skip`` gauge modes,
           backsubstitute the periodic ties, unit-H¹ normalize).
        3. Load the modes into ``self._phi`` and rebuild constraints.

        ``n_skip`` defaults to ``self._count_zero_modes()`` (``gdim+1`` under full
        periodicity: rigid-body translations + one MPC gauge mode). After this
        call the reference solve is *not* committed (``F_bar_conv``/``_u_conv``
        unchanged), so the next ``self(F̄, v, g)`` ramps from the same baseline.

        Returns the ``n_modes`` physical eigenvalues (NaN where SLEPc did not
        converge).
        """
        if n_modes <= 0:
            raise ValueError(f"n_modes must be positive, got {n_modes}")
        if n_modes > self._N_modes:
            raise ValueError(
                f"Requested {n_modes} modes but solver was constructed with "
                f"N={self._N_modes}. Build the solver with N >= n_modes."
            )

        # 1. Reference state: φ=0 so K is the standard hyperelastic tangent.
        for phi in self._phi:
            phi.x.array[:] = 0.0
            phi.x.scatter_forward()
        if Fbar is not None:
            # φ=0 ⇒ the φ-dependent constraint rows are zero and G = C Cᵀ is
            # singular; restrict to the gdim base rows (ZeroVolumeAverage) for the
            # reference solve, then restore.
            all_forms = self._newton._constraint_forms_raw
            self._newton._constraint_forms_raw = all_forms[: self.gdim]
            self.rebuild_constraints()
            logger.info("Linear buckling: reference solve at F̄ = \n%s", Fbar)
            self(Fbar, v, g)
            self._newton._constraint_forms_raw = all_forms

        # 2. Delegate the eigensolve / normalize / save+visualize to the base.
        eigvals, modes = self.compute_buckling_spectrum(
            n_modes, Fbar=None, tol=tol, slepc_options=slepc_options, n_skip=n_skip,
            visualize_modes=visualize_modes, modes_filename=modes_filename,
            save_modes=save_modes, return_modes=True,
        )

        # 3. Load the modes into the φ basis (+ rebuild φ-dependent constraints).
        self.load_buckling_modes(modes)
        return eigvals

    def load_buckling_modes(self, modes) -> None:
        """Load mode fields into ``self._phi`` and rebuild the φ-dependent
        constraints.

        ``modes`` is a list of :class:`dolfinx.fem.Function` on ``self.V`` (as
        returned by ``compute_buckling_spectrum(..., return_modes=True)``) or of
        dof arrays (e.g. ``np.load``-ed ``phi_<i>.npy`` snapshots). Up to
        ``self._N_modes`` entries are loaded; any remaining φ slots are zeroed.
        Call this whenever ``self._phi`` changes so the projected Newton solver
        picks up the new ``⟨w·φᵢ⟩`` / ``⟨(w·φᵢ)X⟩`` constraint rows.
        """
        for i in range(self._N_modes):
            if i < len(modes):
                m = modes[i]
                arr = m.x.array if hasattr(m, "x") else np.asarray(m)
                self._phi[i].x.array[:] = arr
            else:
                self._phi[i].x.array[:] = 0.0
            self._phi[i].x.scatter_forward()
        self.rebuild_constraints()

    # ---- driver override ----

    def __call__(self, Fbar: np.ndarray,
                 v: "np.ndarray | None" = None,
                 g: "np.ndarray | None" = None,
                 **kwargs):
        if self._N_modes > 0:
            if v is None:
                v = np.zeros(self._N_modes, dtype=PETSc.ScalarType)
            if g is None:
                g = np.zeros((self._N_modes, self.gdim), dtype=PETSc.ScalarType)
            v_arr = np.asarray(v, dtype=PETSc.ScalarType).reshape(self._N_modes)
            g_arr = np.asarray(g, dtype=PETSc.ScalarType).reshape(
                self._N_modes, self.gdim,
            )
            self._mm_target_v = v_arr
            self._mm_target_g = g_arr
            self._mm_prev_v = self._v_conv.copy()
            self._mm_prev_g = self._g_conv.copy()
        return super().__call__(Fbar, **kwargs)
