"""Higher-order (micromorphic) periodic homogenization.

Subclass of :class:`PeriodicHyperelasticHomogenizationSolver` that adds a
finite set of user-provided global modes ``φᵢ`` (``i = 1..N``) and the
displacement ansatz

    u_total = (F̄ - I)·X + Σᵢ (vᵢ + X·gᵢ) φᵢ + w

with extra macro variables ``v ∈ R^N`` and ``g ∈ R^{N×gdim}``. Reports the
3×3 grid of effective quantity / macro-variable tangents (``dPbar``, ``dPi``,
``dLambda``) × (``dFbar``, ``dv``, ``dg``).

This first cut keeps the parent's corner-pinning gauge. Periodicity /
integral gauges on ``φ`` and the corresponding Lagrange-multiplier
constraints (``⟨w·φᵢ⟩ = 0``, ``⟨(w·φᵢ) X⟩ = 0``) are not added here.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import ufl
from dolfinx import fem, io
from petsc4py import PETSc
from slepc4py import SLEPc

from .averages import (
    EffectiveAbar,
    EffectiveFbar,
    EffectiveLambda,
    EffectivePbar,
    EffectivePi,
    TangentBlock,
)
from .forms import basis_tensor_ufl
from .solver import PeriodicHyperelasticHomogenizationSolver

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

class MicromorphicHyperelasticHomogenizationSolver(
    PeriodicHyperelasticHomogenizationSolver
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
        super().__init__(mesh_path, comm, gdim, material, **kwargs)

    # ---- subclass hooks ----

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
    ) -> np.ndarray:
        """Linear buckling analysis: load the first ``n_modes`` eigenmodes of
        the tangent ``K`` at a reference macro state into ``self._phi``.

        Procedure:

        1. Solve the RVE at the reference state ``(Fbar, v, g)``. Defaults to
           ``Fbar=I``, ``v=0``, ``g=0`` (undeformed); pass an ``Fbar`` close
           to (or past) the critical load to capture true buckling modes.
        2. Assemble ``K`` at that state and solve ``K φ = λ φ`` for the
           ``n_modes`` smallest-magnitude eigenpairs (SLEPc shift-invert at
           ``σ = 0``, ``TARGET_REAL``).
        3. Apply the periodic MPC backsubstitution to each eigenvector, scale
           to unit ``ℓ²`` norm, and store it in ``self._phi[i]``.

        After this call the reference solve has *not* been committed (the
        converged restart in ``F_bar_conv``/``_u_conv`` is unchanged), so the
        next ``self(F̄, v, g)`` ramps from the same baseline as before.

        Returns the ``n_modes`` eigenvalues as a NumPy array.
        """
        if n_modes <= 0:
            raise ValueError(f"n_modes must be positive, got {n_modes}")
        if n_modes > self._N_modes:
            raise ValueError(
                f"Requested {n_modes} modes but solver was constructed with "
                f"N={self._N_modes}. Build the solver with N >= n_modes."
            )

        # 1. Reference solve at the chosen macro state.
        if Fbar is None:
            Fbar = np.eye(self.gdim, dtype=PETSc.ScalarType)
        logger.info("Linear buckling: reference solve at F̄ = \n%s", Fbar)
        # Reset φ to zero so the reference state is the standard hyperelastic
        # one (no φ-contribution mixing into K).
        for phi in self._phi:
            phi.x.array[:] = 0.0
            phi.x.scatter_forward()
        self(Fbar, v, g, pert_amplitude_init=0.0)

        # 2. Assemble K and solve the eigenproblem.
        K = self._newton.assemble_stiffness()
        try:
            eps = SLEPc.EPS().create(self.comm)
            eps.setOperators(K)
            eps.setProblemType(SLEPc.EPS.ProblemType.HEP)
            st = eps.getST()
            st.setType(SLEPc.ST.Type.SINVERT)
            eps.setTarget(0.0)
            eps.setDimensions(nev=n_modes)
            eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
            eps.setTolerances(tol=tol)
            if slepc_options is not None:
                opts = PETSc.Options()
                for key, val in slepc_options.items():
                    opts[key] = val
            eps.setFromOptions()
            eps.solve()

            n_conv = eps.getConverged()
            if n_conv < n_modes:
                logger.warning(
                    "Linear buckling: requested %d modes, SLEPc converged %d",
                    n_modes, n_conv,
                )

            # 3. Load eigenvectors into self._phi.
            eigvals = np.zeros(n_modes)
            n_load = min(n_modes, n_conv)
            for i in range(n_load):
                lam = eps.getEigenvalue(i).real
                eigvals[i] = lam
                phi_vec = self._phi[i].x.petsc_vec
                eps.getEigenvector(i, phi_vec)
                if self.mpc is not None:
                    self.mpc.backsubstitution(phi_vec)
                self._phi[i].x.scatter_forward()
                nrm = phi_vec.norm()
                if nrm > 0.0:
                    phi_vec.scale(1.0 / nrm)
                    self._phi[i].x.scatter_forward()
            # Zero any remaining slots (if SLEPc didn't converge enough).
            for i in range(n_load, n_modes):
                self._phi[i].x.array[:] = 0.0
                self._phi[i].x.scatter_forward()
                eigvals[i] = np.nan

            logger.info(
                "Linear buckling eigenvalues (smallest |λ|): %s",
                np.array2string(eigvals, precision=4),
            )

            # Optional ParaView output — one timestep per mode (t = mode index).
            # The file contains a single vector field ``phi``; scrubbing the
            # time slider in ParaView walks through φ₀, φ₁, …, φ_{N−1}.
            if visualize_modes and n_load > 0:
                out_path = os.path.join(self.output_dir, modes_filename)
                os.makedirs(self.output_dir, exist_ok=True)
                phi_viz = fem.Function(self.V, name="phi")
                writer = io.VTXWriter(
                    self.comm, out_path, [phi_viz], engine="BP4",
                )
                try:
                    for i in range(n_load):
                        phi_viz.x.array[:] = self._phi[i].x.array
                        phi_viz.x.scatter_forward()
                        writer.write(float(i))
                finally:
                    writer.close()
                logger.info(
                    "Wrote %d buckling mode(s) to %s (timestep = mode index)",
                    n_load, out_path,
                )
            return eigvals
        finally:
            eps.destroy()
            PETSc.Mat.destroy(K)

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
