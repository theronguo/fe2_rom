import os
import logging
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import scipy

from ..hyperelastic_solver import NeoHookean, VTXManager, TimeStepper, setup_logging, silence_c_stdout
from ..hyperelastic_solver.exceptions import RVEConvergenceError
from ..hyperelastic_solver.averages import (
    HomogenizationContext,
    EffectiveFbar,
    EffectivePbar,
    EffectiveAbar,
    resolve_average_quantities,
)
from ..hyperelastic_solver.forms import basis_tensor_ufl
from dolfinx import io, fem, mesh as dmesh
from mpi4py import MPI
from petsc4py import PETSc
import ufl

logger = logging.getLogger(__name__)


def _dF_Fbar_factory(context):
    gdim = context.mesh.geometry.dim
    return [basis_tensor_ufl(gdim, i, j) for i in range(gdim) for j in range(gdim)]


class RVESolver:
    """ROM-based hyperelastic periodic homogenization solver using ECM.

    Modular by design. Subclasses extend three orthogonal axes:

    * **macro variables** — ``_setup_macro_vars`` registers extra
      ``fem.Constant`` inputs in ``self.macro_vars``.
    * **kinematic ansatz** — ``_build_F_ufl_extra`` returns an extra
      displacement contribution ``u_extra(X)``; its gradient is added to
      ``F = F̄ + ∇u_fluc``.
    * **effective quantities & tangents** — ``_make_default_average_quantities``
      returns ``AverageQuantity`` / ``TangentBlock`` instances from
      ``fe2_rom.hyperelastic_solver.averages``. ``_make_dF_dmu_factories``
      registers the ``∂F/∂μ`` UFL tensors used to build ROM adjoint RHS forms.

    Public API (preserved):
        solver = RVESolver(mesh_path, rom_dir, material)
        outputs = solver(Fbar)  # list[dict], one entry per accepted step
    """

    def __init__(
        self,
        mesh_path: str,
        rom_dir: str,
        material,
        *,
        gdim: int = 2,
        degree: int = 2,
        comm=MPI.COMM_WORLD,
        output_dir: str = "output",
        visualize_fields: list[str] | None = None,
        average_quantities: dict | None = None,
        newton_options: dict | None = None,
        timestepper_options: dict | None = None,
        averages_only_final: bool = False,
    ) -> None:
        newton_options = newton_options if newton_options is not None else {
            "rel_tol": 1e-8, "abs_tol": 1e-6, "max_iter": 50, "div_rel_tol": 10.0,
        }
        timestepper_options = timestepper_options if timestepper_options is not None else {
            "t_end": 1.0, "dt_init": 1.0, "dt_min": 1e-5, "dt_max": 1.0, "good_newton_steps": 5,
        }
        if visualize_fields is None:
            visualize_fields = ["u_fluc"]

        self.comm = comm
        self.gdim = gdim
        self._material = material
        self._newton_options = newton_options
        self._timestepper = TimeStepper(**timestepper_options)
        self._averages_only_final = averages_only_final
        self.output_dir = output_dir

        # --- ROM data ---
        indices          = np.load(os.path.join(rom_dir, "indices.npy"))
        self.basis_u_sub = np.load(os.path.join(rom_dir, "basis_u_sub.npy"))
        omega_sub        = np.load(os.path.join(rom_dir, "omega_sub.npy"))
        self.basis_u     = np.load(os.path.join(rom_dir, "basis_u.npy"))
        self.N = self.basis_u_sub.shape[1]
        logger.debug("ROM data loaded: N=%d modes", self.N)

        # --- Mesh & submesh ---
        with silence_c_stdout():
            mesh = io.gmsh.read_from_msh(mesh_path, comm, 0, gdim=gdim).mesh
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        tdim = mesh.topology.dim
        V_full = fem.functionspace(mesh, ("Lagrange", degree, (gdim,)))
        submesh, _, _, _ = dmesh.create_submesh(mesh, tdim, indices)
        self._mesh_full = mesh
        self._submesh = submesh
        self._dx_sub = ufl.Measure("dx", domain=submesh)

        # --- Function spaces ---
        V_sub  = fem.functionspace(submesh, ("Lagrange", degree, (gdim,)))
        Q0_sub = fem.functionspace(submesh, ("DG", 0))
        self.V_sub = V_sub
        self.V_full = V_full

        # --- Functions ---
        self._omega_func = fem.Function(Q0_sub)
        self._omega_func.x.array[:] = omega_sub

        self.u_fluc = fem.Function(V_sub)
        self.u_full = fem.Function(V_full, name="u_fluc")
        self._v     = fem.Function(V_sub)
        self._w     = fem.Function(V_sub)

        # --- Macro variables (hook) ---
        self.F_bar = fem.Constant(submesh, np.eye(gdim, dtype=PETSc.ScalarType))
        self.F_bar_conv = np.eye(gdim, dtype=PETSc.ScalarType)
        self.macro_vars: dict = {"Fbar": self.F_bar}
        self._setup_macro_vars()

        # --- Kinematics & constitutive ---
        u_extra = self._build_F_ufl_extra()
        F_inner = self.F_bar + ufl.grad(self.u_fluc)
        if u_extra is not None:
            F_inner = F_inner + ufl.grad(u_extra)
        F_ufl = ufl.variable(F_inner)
        self._F_ufl = F_ufl
        self._u_extra_ufl = u_extra
        self._P_ufl = material.first_pk_stress(F_ufl)
        self.A_ufl  = material.tangent_moduli(F_ufl)

        # --- Residual / Jacobian (ROM coefficients) ---
        ii, jj, kk, ll = ufl.indices(4)
        A_grad_v = ufl.as_tensor(self.A_ufl[ii, jj, kk, ll] * ufl.grad(self._v)[kk, ll], (ii, jj))
        self._r_form = fem.form(
            ufl.inner(self._P_ufl, ufl.grad(self._v)) * self._omega_func * self._dx_sub
        )
        self._j_form = fem.form(
            ufl.inner(A_grad_v, ufl.grad(self._w)) * self._omega_func * self._dx_sub
        )

        # ECM volume
        self._vol = fem.assemble_scalar(fem.form(1.0 * self._omega_func * self._dx_sub))
        logger.debug("Effective domain volume (ECM): %.6f", self._vol)

        # --- HomogenizationContext + average-quantity registry ---
        self._context = HomogenizationContext(
            mesh=submesh, V=V_sub, dx=self._dx_sub, comm=comm,
            vol_global=self._vol,
            F_var=F_ufl, P_ufl=self._P_ufl, A_ufl=self.A_ufl,
            W_ufl=None, u=self.u_fluc, u_total=None,
            macro_vars=self.macro_vars,
            phi=getattr(self, "_phi", []),
            weight=self._omega_func,
        )

        if average_quantities is None:
            self._quantities = self._make_default_average_quantities()
        else:
            self._quantities = resolve_average_quantities(average_quantities)
        for q in self._quantities:
            q.setup(self._context)

        # --- ROM adjoint machinery ---
        # ∂F/∂μ factories per macro var; materialised once into UFL tensor lists.
        self._dF_dmu_factories = self._make_dF_dmu_factories()
        self._dF_dmu_lists = {
            name: factory(self._context)
            for name, factory in self._dF_dmu_factories.items()
        }

        # Compile ROM adjoint RHS forms:
        #   D[p, mi] = ∫ inner(A : ∂F/∂μ_mi, grad(φ_p)) ω dx
        # parameterised at assembly time by overwriting self._v.x.array with
        # the p-th submesh POD basis vector. Convention matches FOM
        # ``TangentBlock`` (last two indices of A contract with ∂F/∂μ).
        self._D_forms: dict[str, list] = {}
        for name, dF_list in self._dF_dmu_lists.items():
            forms_mi = []
            for dF in dF_list:
                A_dF = ufl.as_tensor(self.A_ufl[ii, jj, kk, ll] * dF[kk, ll], (ii, jj))
                forms_mi.append(fem.form(
                    ufl.inner(A_dF, ufl.grad(self._v)) * self._omega_func * self._dx_sub
                ))
            self._D_forms[name] = forms_mi

        # Storage Functions for forward sensitivities p_μ_k = Σ_p α[p,k] φ_p,
        # one V_sub Function per flat macro-var component. Passed to each
        # AverageQuantity.compute() via the ``adjoints`` dict.
        self._adjoint_funcs: dict[str, list] = {
            name: [fem.Function(V_sub) for _ in self._dF_dmu_lists[name]]
            for name in self._dF_dmu_lists
        }

        # Only macro vars that some registered quantity actually needs.
        needs = set()
        for q in self._quantities:
            needs.update(q.required_macro_adjoints)
        self._adjoint_macro_vars = [n for n in self._dF_dmu_lists if n in needs]

        # warm-started ROM coefficients.
        self.coeffs = np.zeros(self.N)
        self.coeffs_conv = np.zeros(self.N)

        logger.debug("Newton solver initialized with options: %s", newton_options)
        logger.debug("Time stepper initialized with options: %s", timestepper_options)

        # --- Visualization ---
        self._u_total_expr = None
        self._F_bar_full = None

        fields = []
        for field in visualize_fields:
            if field == "u_fluc":
                fields.append(self.u_full)
            elif field == "u_total":
                self._F_bar_full = fem.Constant(mesh, np.eye(gdim, dtype=PETSc.ScalarType))
                X = ufl.SpatialCoordinate(mesh)
                u_total_ufl = (self._F_bar_full - ufl.Identity(gdim)) * X + self.u_full
                extra_full = self._build_u_total_extra_full(mesh)
                if extra_full is not None:
                    u_total_ufl = u_total_ufl + extra_full
                self.u_total = fem.Function(V_full, name="u_total")
                self._u_total_expr = fem.Expression(u_total_ufl, V_full.element.interpolation_points)
                fields.append(self.u_total)
        if fields:
            self.vtx = VTXManager(comm, os.path.join(output_dir, "solution.bp"), fields)
        else:
            self.vtx = None
        logger.debug("Visualization fields: %s", visualize_fields)
        logger.debug("Average quantities: %s", [q.name for q in self._quantities])
        logger.debug("Adjoint macro vars: %s", self._adjoint_macro_vars)
        logger.debug("Setup complete")

    # --- Hooks for subclasses --------------------------------------------------

    def _setup_macro_vars(self) -> None:
        """Register additional macro variables in ``self.macro_vars``."""
        return

    def _build_F_ufl_extra(self):
        """Return ``u_extra(X)`` (UFL expression) whose gradient is added to
        ``F = F̄ + grad(u_fluc)``. Default ``None`` (no extra term).
        """
        return None

    def _build_u_total_extra_full(self, mesh):
        """Return UFL expression on the *full* mesh to add to the visualised
        ``u_total = (F̄−I)·X + u_fluc + <this>``. Default ``None``.

        Subclasses extending the kinematic ansatz must override both this and
        ``_build_F_ufl_extra``; the two terms should be the same field
        symbolically (one expressed on the submesh, one on the full mesh).
        """
        return None

    def _sync_full_mesh_constants(self) -> None:
        """Hook called inside ``_write_fields`` before interpolating ``u_total``.
        Subclasses copy any submesh-side macro-variable values into their
        full-mesh ``fem.Constant`` mirrors here.
        """
        return

    def _make_dF_dmu_factories(self) -> dict:
        """Return ``{name: factory(context) -> list[UFL 2-tensor]}`` for every
        macro variable in ``self.macro_vars``. The factory must return a flat
        C-ordered list of length ``prod(shape(macro_var))``.
        """
        return {"Fbar": _dF_Fbar_factory}

    def _make_default_average_quantities(self) -> list:
        """Default quantity registry for the base solver."""
        return [EffectiveFbar(), EffectivePbar(), EffectiveAbar()]

    def _restore_trial_state(self) -> None:
        """Restore subclass macro-variable trial state to the last committed
        values (called at the start of every ``__call__``)."""
        return

    def _update_macro_load_schedule(self, t: float) -> None:
        """Update subclass macro variables for current ramp parameter t∈[0,1].

        ``F̄`` is already handled by the base class.
        """
        return

    def _commit_extra_state(self) -> None:
        """Promote subclass trial macro variables to committed."""
        return

    # --- State management -----------------------------------------------------

    def _restore_state(self, coeffs: np.ndarray) -> None:
        self.coeffs[:] = coeffs
        self.u_fluc.x.array[:] = sum(
            self.coeffs[ii] * self.basis_u_sub[:, ii] for ii in range(self.N)
        )

    # --- Visualization --------------------------------------------------------

    def _write_fields(self, t: float) -> None:
        if self.vtx is not None:
            self.u_full.x.array[:] = sum(
                self.coeffs[ii] * self.basis_u[:, ii] for ii in range(self.N)
            )
            if self._u_total_expr is not None:
                self._F_bar_full.value[:] = self.F_bar.value
                self._sync_full_mesh_constants()
                self.u_total.interpolate(self._u_total_expr)
            self.vtx.write(t)

    # --- Averaging / adjoints -------------------------------------------------

    def _assemble_jacobian(self) -> np.ndarray:
        N = self.N
        J = np.zeros((N, N))
        for ii in range(N):
            self._v.x.array[:] = self.basis_u_sub[:, ii]
            for jj in range(N):
                self._w.x.array[:] = self.basis_u_sub[:, jj]
                J[ii, jj] = fem.assemble_scalar(self._j_form)
        return J

    def _solve_adjoints(self) -> None:
        """Compute forward sensitivities p_μ_k = ∂u_fluc/∂μ_k and store as
        ``fem.Function``s in ``self._adjoint_funcs[name][k]``.
        """
        if not self._adjoint_macro_vars:
            return
        N = self.N
        J = self._assemble_jacobian()
        # Symmetrize via LDLT and flip negative pivots (same trick as FE²
        # adjoint path) so we can still invert when J is indefinite past
        # buckling.
        L, Dldl, _ = scipy.linalg.ldl(J)
        if (Dldl < -1e-8).any():
            logger.warning("Indefinite Jacobian in ROM adjoint solve; flipping small negative pivots")
        Dldl[Dldl < -1e-8] = -Dldl[Dldl < -1e-8]
        J_sym = L @ Dldl @ L.T

        for name in self._adjoint_macro_vars:
            forms_mi = self._D_forms[name]
            n_mu = len(forms_mi)
            D = np.zeros((N, n_mu))
            for p in range(N):
                self._v.x.array[:] = self.basis_u_sub[:, p]
                for mi in range(n_mu):
                    D[p, mi] = fem.assemble_scalar(forms_mi[mi])
            alpha = np.linalg.solve(J_sym, -D)  # (N, n_mu)
            slots = self._adjoint_funcs[name]
            for mi in range(n_mu):
                slots[mi].x.array[:] = self.basis_u_sub @ alpha[:, mi]

    def _collect_averages(self, with_tangents: bool = True) -> dict:
        if with_tangents:
            self._solve_adjoints()
        result: dict = {}
        for q in self._quantities:
            need_adj = bool(q.required_macro_adjoints)
            if need_adj and not with_tangents:
                continue
            result[q.name] = q.compute(
                self._context,
                self._adjoint_funcs if need_adj else None,
            )
        return result

    # --- Newton solver --------------------------------------------------------

    def _newton_solve(self, rel_tol: float, abs_tol: float, max_iter: int) -> int:
        """Newton solve assuming macro-var Constants are already set."""
        N = self.N
        it = 0
        res_norm = np.inf
        res_norm_0 = None
        rel_res = np.inf
        residual = np.zeros(N)
        jacobian = np.zeros((N, N))
        while it < max_iter:
            for ii in range(N):
                self._v.x.array[:] = self.basis_u_sub[:, ii]
                residual[ii] = fem.assemble_scalar(self._r_form)
                for jj in range(N):
                    self._w.x.array[:] = self.basis_u_sub[:, jj]
                    jacobian[ii, jj] = fem.assemble_scalar(self._j_form)

            res_norm = np.linalg.norm(residual)
            if res_norm_0 is None:
                res_norm_0 = res_norm if res_norm > 0.0 else 1.0
            rel_res = res_norm / res_norm_0
            logger.debug("  iter %3d: |r| = %.3e  |r|/|r0| = %.3e", it, res_norm, rel_res)
            if rel_res < rel_tol:
                logger.info("Newton converged in %d iteration(s) [rel=%.3e < rel_tol=%.3e]",
                            it, rel_res, rel_tol)
                break
            if res_norm < abs_tol:
                logger.info("Newton converged in %d iteration(s) [|r|=%.3e < abs_tol=%.3e]",
                            it, res_norm, abs_tol)
                break
            if it > 0 and rel_res > self._newton_options["div_rel_tol"]:
                raise RuntimeError(f"Newton diverging: |r|/|r0| = {rel_res:.3e}")

            try:
                L, D, _ = scipy.linalg.ldl(jacobian)
                if not np.isfinite(L).all() or not np.isfinite(D).all():
                    raise np.linalg.LinAlgError("NaN/Inf in LDL factors")
                if (D < -1e-8).any():
                    logger.warning("Indefinite Jacobian in ROM adjoint solve; flipping small negative pivots")
                D[D < -1e-8] = -D[D < -1e-8]
                dcoeffs = np.linalg.solve(L @ D @ L.T, -residual)
            except (np.linalg.LinAlgError, ValueError) as e:
                raise RuntimeError(f"LDL factorisation failed: {e}") from e

            self.coeffs += dcoeffs
            self.u_fluc.x.array[:] = sum(
                self.coeffs[ii] * self.basis_u_sub[:, ii] for ii in range(N)
            )
            it += 1

        if rel_res >= rel_tol and res_norm >= abs_tol:
            raise RuntimeError(
                f"Newton did not converge in {max_iter} iterations "
                f"(rel={rel_res:.3e}, |r|={res_norm:.3e})"
            )

        return it

    # --- Main entry point -----------------------------------------------------

    def __call__(self, Fbar: np.ndarray, plot_time_start: float = 0.0) -> list:
        """Adaptive load stepping from the last committed state to ``Fbar``.

        Returns a list with one dict per accepted step (including t=0).
        """
        rel_tol  = self._newton_options["rel_tol"]
        abs_tol  = self._newton_options["abs_tol"]
        max_iter = self._newton_options["max_iter"]

        Fbar_prev = self.F_bar_conv.copy()
        Fbar_target = np.asarray(Fbar, dtype=PETSc.ScalarType)
        self.F_bar.value[:] = Fbar_prev
        self._restore_state(self.coeffs_conv)
        self._restore_trial_state()

        def load_schedule(t: float) -> None:
            self.F_bar.value[:] = Fbar_prev + t * (Fbar_target - Fbar_prev)
            self._update_macro_load_schedule(t)

        self._write_fields(plot_time_start)
        output_quantities = (
            []
            if self._averages_only_final
            else [self._collect_averages(with_tangents=True)]
        )

        self._timestepper.reset()
        while not self._timestepper.finished:
            trial_time = self._timestepper.step_forward()
            logger.info("── Step  t=%.5f  dt=%.2e", trial_time, self._timestepper.dt)
            load_schedule(trial_time)

            coeffs_save = self.coeffs.copy()
            try:
                n_iters = self._newton_solve(rel_tol, abs_tol, max_iter)
            except RuntimeError as e:
                ok = self._timestepper.reject()
                self._restore_state(coeffs_save)
                load_schedule(self._timestepper.t_current)
                logger.debug("  reason: %s", e)
                if not ok:
                    logger.error("Minimum time step dt=%.2e reached — stopping.", self._timestepper.dt_min)
                    raise RVEConvergenceError(
                        f"RVE timestepper hit dt_min={self._timestepper.dt_min:.2e} "
                        f"at t={self._timestepper.t_current:.4f}"
                    )
                logger.warning("Newton did not converge — halving dt to %.2e", self._timestepper.dt)
                continue

            self._timestepper.accept(n_iters)
            if not self._averages_only_final:
                output_quantities.append(self._collect_averages(with_tangents=True))
            self._write_fields(self._timestepper.t_current + plot_time_start)

        if self._averages_only_final:
            output_quantities.append(self._collect_averages(with_tangents=True))

        return output_quantities

    def commit(self) -> None:
        """Promote trial state to the converged restart point."""
        self.F_bar_conv[:] = self.F_bar.value
        self.coeffs_conv[:] = self.coeffs
        self._commit_extra_state()
