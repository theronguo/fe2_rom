import os
import logging
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import scipy

from ..hyperelastic_solver import NeoHookean, VTXManager, TimeStepper, setup_logging, silence_c_stdout
from ..hyperelastic_solver.exceptions import RVEConvergenceError
from dolfinx import io, fem, mesh as dmesh
from mpi4py import MPI
from petsc4py import PETSc
import ufl

logger = logging.getLogger(__name__)


class RVESolver:
    """ROM-based hyperelastic periodic homogenization solver using ECM.

    Usage pattern:
        solver = RVESolver(mesh_path, rom_dir, material)
        output_quantities = solver(Fbar_target)

    output_quantities is a list with one entry per accepted load step;
    each entry is a list aligned with average_fields, e.g. [Pbar].
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
        average_fields: list[str] | None = None,
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
        if average_fields is None:
            average_fields = ["P"]

        self.comm = comm
        self.gdim = gdim
        self.average_fields = average_fields
        self._newton_options = newton_options
        self._timestepper = TimeStepper(**timestepper_options)
        # If True, ``__call__`` returns only the final converged step's
        # averages — intermediate ramp steps skip the (expensive) tangent /
        # PK1 reductions.  Useful for FE² inner solves where only the
        # endpoint matters.
        self._averages_only_final = averages_only_final

        # --- ROM data ---
        indices          = np.load(os.path.join(rom_dir, "indices.npy"))
        self.basis_u_sub = np.load(os.path.join(rom_dir, "basis_u_sub.npy"))
        omega_sub        = np.load(os.path.join(rom_dir, "omega_sub.npy"))
        self.basis_u     = np.load(os.path.join(rom_dir, "basis_u.npy"))
        self.N = self.basis_u_sub.shape[1]
        logger.debug("ROM data loaded: N=%d modes", self.N)

        # --- Mesh & submesh ---
        # (no cell/facet tags in ROM — mesh is only used to build the full-DOF output function)
        with silence_c_stdout():
            mesh = io.gmsh.read_from_msh(mesh_path, comm, 0, gdim=gdim).mesh
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        tdim = mesh.topology.dim
        submesh, _, _, _ = dmesh.create_submesh(mesh, tdim, indices)
        self._dx_sub = ufl.Measure("dx", domain=submesh)

        # --- Function spaces ---
        V_sub  = fem.functionspace(submesh, ("Lagrange", degree, (gdim,)))
        Q0_sub = fem.functionspace(submesh, ("DG", 0))
        V      = fem.functionspace(mesh, ("Lagrange", degree, (gdim,)))

        # --- Functions ---
        self._omega_func = fem.Function(Q0_sub)
        self._omega_func.x.array[:] = omega_sub

        self.u_fluc = fem.Function(V_sub)
        self.u_full = fem.Function(V, name="u_fluc")
        self._v     = fem.Function(V_sub)
        self._w     = fem.Function(V_sub)

        # --- Kinematics & constitutive ---
        self.F_bar = fem.Constant(submesh, np.eye(gdim, dtype=PETSc.ScalarType))
        # Last *converged* F_bar — updated only when ``__call__`` completes
        # the full ramp.  ``F_bar`` may hold a failed trial value.
        self.F_bar_conv = np.eye(gdim, dtype=PETSc.ScalarType)
        F_ufl = ufl.variable(self.F_bar + ufl.grad(self.u_fluc))
        self._P_ufl = material.first_pk_stress(F_ufl)
        self.A_ufl = material.tangent_moduli(F_ufl)

        # --- Compiled forms ---
        ii, jj, kk, ll = ufl.indices(4)
        A_grad_v = ufl.as_tensor(self.A_ufl[ii, jj, kk, ll] * ufl.grad(self._v)[kk, ll], (ii, jj))
        self._r_form = fem.form(
            ufl.inner(self._P_ufl, ufl.grad(self._v)) * self._omega_func * self._dx_sub
        )
        self._j_form = fem.form(
            ufl.inner(A_grad_v, ufl.grad(self._w)) * self._omega_func * self._dx_sub
        )

        # ECM weights approximate full-domain integrals, so this gives the full volume.
        self._vol = fem.assemble_scalar(fem.form(1.0 * self._omega_func * self._dx_sub))
        logger.debug("Effective domain volume (ECM): %.6f", self._vol)

        if "P" in average_fields:
            self._P_avg_forms = [
                [fem.form(self._P_ufl[i, j] * self._omega_func * self._dx_sub)
                 for j in range(gdim)]
                for i in range(gdim)
            ]
        else:
            self._P_avg_forms = None

        if "A" in average_fields:
            self._D_forms = [
                [fem.form(
                    ufl.inner(
                        ufl.as_tensor([[self.A_ufl[i, j, m, n] for n in range(gdim)] for m in range(gdim)]),
                        ufl.grad(self._v),
                    ) * self._omega_func * self._dx_sub
                ) for j in range(gdim)]
                for i in range(gdim)
            ]
            self._A_avg_forms = [
                [[[fem.form(self.A_ufl[i, j, k, l] * self._omega_func * self._dx_sub)
                   for l in range(gdim)]
                  for k in range(gdim)]
                 for j in range(gdim)]
                for i in range(gdim)
            ]
        else:
            self._D_forms = None
            self._A_avg_forms = None

        # warm-started ROM coefficients (persist across __call__ invocations).
        # ``coeffs`` is the *trial* state — mutated mid-Newton, may hold a
        # value from an outer iteration the macro SNES later rejects.
        # ``coeffs_conv`` mirrors ``F_bar_conv``: only updated on ``commit()``.
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
                # (F_bar - I)*X + u_fluc on the full mesh; F_bar_full stays in sync at write time
                self._F_bar_full = fem.Constant(mesh, np.eye(gdim, dtype=PETSc.ScalarType))
                X = ufl.SpatialCoordinate(mesh)
                u_total_ufl = (self._F_bar_full - ufl.Identity(gdim)) * X + self.u_full
                self.u_total = fem.Function(V, name="u_total")
                self._u_total_expr = fem.Expression(u_total_ufl, V.element.interpolation_points)
                fields.append(self.u_total)
        if fields:
            self.vtx = VTXManager(comm, os.path.join(output_dir, "solution.bp"), fields)
        else:
            self.vtx = None
        logger.debug("Visualization fields: %s", visualize_fields)
        logger.debug("Averaging fields: %s", average_fields)
        logger.debug("Setup complete")

    # --- State management ---

    def _restore_state(self, coeffs: np.ndarray) -> None:
        self.coeffs[:] = coeffs
        self.u_fluc.x.array[:] = sum(
            self.coeffs[ii] * self.basis_u_sub[:, ii] for ii in range(self.N)
        )

    # --- Visualization ---

    def _write_fields(self, t: float) -> None:
        if self.vtx is not None:
            self.u_full.x.array[:] = sum(
                self.coeffs[ii] * self.basis_u[:, ii] for ii in range(self.N)
            )
            if self._u_total_expr is not None:
                self._F_bar_full.value[:] = self.F_bar.value
                self.u_total.interpolate(self._u_total_expr)
            self.vtx.write(t)

    # --- Averaging ---

    def compute_average_first_pk_stress(self) -> np.ndarray:
        """Volume-average first Piola-Kirchhoff stress from the current converged state."""
        Pbar = np.zeros((self.gdim, self.gdim), dtype=float)
        for i in range(self.gdim):
            for j in range(self.gdim):
                Pbar[i, j] = fem.assemble_scalar(self._P_avg_forms[i][j]) / self._vol
        return Pbar

    def compute_effective_tangent_moduli(self) -> np.ndarray:
        """Effective tangent moduli from the current converged state using ROM adjoints.

        Mirrors PeriodicHyperelasticHomogenizationSolver.compute_effective_tangent_moduli.
        The ROM adjoint problem J·α[k,l] = −D[:,k,l] replaces the FEM adjoint PDE,
        where D[p,i,j] = ∫ inner(A[i,j,:,:], grad(φ_p))·ω dx_sub.
        """
        N = self.N
        gdim = self.gdim

        # Reassemble Jacobian at current state; factorize once for all (k,l) adjoint solves
        jacobian = np.zeros((N, N))
        for ii in range(N):
            self._v.x.array[:] = self.basis_u_sub[:, ii]
            for jj in range(N):
                self._w.x.array[:] = self.basis_u_sub[:, jj]
                jacobian[ii, jj] = fem.assemble_scalar(self._j_form)
        L, Dldl, _ = scipy.linalg.ldl(jacobian)
        Dldl[Dldl < 1e-8] = -Dldl[Dldl < 1e-8]
        J_sym = L @ Dldl @ L.T

        # D[p, i, j] = ∫ inner(A[i,j,:,:], grad(φ_p)) · ω dx_sub
        D = np.zeros((N, gdim, gdim))
        for i in range(gdim):
            for j in range(gdim):
                for p in range(N):
                    self._v.x.array[:] = self.basis_u_sub[:, p]
                    D[p, i, j] = fem.assemble_scalar(self._D_forms[i][j])

        # Adjoint solve: J · α[k,l] = −D[:,k,l]
        alpha = np.zeros((gdim, gdim, N))
        for k in range(gdim):
            for l in range(gdim):
                alpha[k, l, :] = np.linalg.solve(J_sym, -D[:, k, l])

        # A_eff[i,j,k,l] = (A_avg[i,j,k,l] + D[:,i,j] · α[k,l,:]) / vol
        A_eff = np.zeros((gdim, gdim, gdim, gdim), dtype=float)
        for i in range(gdim):
            for j in range(gdim):
                for k in range(gdim):
                    for l in range(gdim):
                        A_avg_local = fem.assemble_scalar(self._A_avg_forms[i][j][k][l])
                        A_fluc_local = float(D[:, i, j] @ alpha[k, l, :])
                        A_eff[i, j, k, l] = (A_avg_local + A_fluc_local) / self._vol

        return A_eff

    # --- Newton solver ---

    def _newton_solve(self, rel_tol: float, abs_tol: float, max_iter: int) -> int:
        """Newton solve assuming F_bar.value is already set. Raises RuntimeError on failure."""
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
                # use an inexact Newton step to push the system towards instability
                L, D, _ = scipy.linalg.ldl(jacobian)
                if not np.isfinite(L).all() or not np.isfinite(D).all():
                    raise np.linalg.LinAlgError("NaN/Inf in LDL factors")
                D[D < 1e-8] = -D[D < 1e-8]
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

    # --- Main entry point ---

    def _fd_check_tangent_moduli(self, Fbar: np.ndarray, A_conv: np.ndarray,
                                  eps: float = 1e-6) -> np.ndarray:
        """Central-difference FD check of A_conv.

        Perturbs each Fbar[k,l] by ±eps, re-solves Newton (warm-started from the
        current coeffs), and compares A_fd[:,:,k,l] = (Pbar_+ - Pbar_-) / (2*eps)
        against A_conv[:,:,k,l] component-wise.
        """
        gdim     = self.gdim
        rel_tol  = self._newton_options["rel_tol"]
        abs_tol  = self._newton_options["abs_tol"]
        max_iter = self._newton_options["max_iter"]

        coeffs_ref = self.coeffs.copy()
        A_fd = np.full((gdim, gdim, gdim, gdim), np.nan)
        failed = []

        for k in range(gdim):
            for l in range(gdim):
                results = []
                for sign in (+1, -1):
                    Fbar_pert = Fbar.copy()
                    Fbar_pert[k, l] += sign * eps
                    self._restore_state(coeffs_ref)
                    self.F_bar.value[:] = Fbar_pert
                    try:
                        self._newton_solve(rel_tol, abs_tol, max_iter)
                    except RuntimeError as e:
                        logger.warning("FD perturbation (%d,%d) sign=%+d failed: %s", k, l, sign, e)
                        failed.append((k, l, sign))
                        results = None
                        break
                    results.append(self.compute_average_first_pk_stress())
                if results is not None:
                    A_fd[:, :, k, l] = (results[0] - results[1]) / (2.0 * eps)

        # Restore exact converged state
        self._restore_state(coeffs_ref)
        self.F_bar.value[:] = Fbar

        if failed:
            logger.warning("FD tangent check incomplete — %d perturbation(s) failed", len(failed))
            return A_fd

        err_abs = np.abs(A_fd - A_conv)
        err_rel = err_abs / (np.abs(A_conv) + 1e-30)
        logger.info("FD tangent check (eps=%.1e): max|err|=%.3e  max|err/A|=%.3e",
                    eps, err_abs.max(), err_rel.max())
        for i in range(gdim):
            for j in range(gdim):
                for k in range(gdim):
                    for l in range(gdim):
                        logger.debug(
                            "  A[%d,%d,%d,%d]  conv=% .6e  fd=% .6e  err=% .2e",
                            i, j, k, l,
                            A_conv[i, j, k, l], A_fd[i, j, k, l],
                            err_abs[i, j, k, l],
                        )
        return A_fd

    def __call__(self, Fbar: np.ndarray, plot_time_start: float = 0.0,
                 check_tangent: bool = False) -> list:
        """Adaptive load stepping from current F_bar to target Fbar.

        Returns a list with one entry per accepted step (including t=0);
        each entry is a list aligned with average_fields, e.g. [Pbar].
        """
        rel_tol  = self._newton_options["rel_tol"]
        abs_tol  = self._newton_options["abs_tol"]
        max_iter = self._newton_options["max_iter"]

        # Restart the ramp from the last *committed* state — a previous
        # call (or earlier macro Newton iter) may have left F_bar, coeffs,
        # and u_fluc at a trial value the outer SNES eventually rejected.
        # ``_restore_state`` rebuilds ``u_fluc`` from coeffs so the compiled
        # forms see a consistent state on the first residual eval.
        Fbar_prev = self.F_bar_conv.copy()
        self.F_bar.value[:] = Fbar_prev
        self._restore_state(self.coeffs_conv)
        def load_schedule(t: float) -> None:
            self.F_bar.value[:] = Fbar_prev + t * (Fbar - Fbar_prev)

        self._write_fields(plot_time_start)
        output_quantities = (
            [] if self._averages_only_final else [self._collect_averages()]
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
                        f"at t={self._timestepper.t_current:.4f} (F_bar={self.F_bar.value!r})"
                    )
                logger.warning("Newton did not converge — halving dt to %.2e", self._timestepper.dt)
                continue

            self._timestepper.accept(n_iters)
            if not self._averages_only_final:
                output_quantities.append(self._collect_averages())
            self._write_fields(self._timestepper.t_current + plot_time_start)

        if self._averages_only_final:
            output_quantities.append(self._collect_averages())

        if check_tangent and self._timestepper.finished:
            logger.info("Running FD tangent check at final state F_bar = %s", self.F_bar.value)
            A_conv = self.compute_effective_tangent_moduli()
            self._fd_check_tangent_moduli(Fbar, A_conv)

        return output_quantities

    def commit(self) -> None:
        """Promote trial state (F_bar, coeffs) to the converged restart point.

        Call from the macro driver after a successful outer time step only.
        """
        self.F_bar_conv[:] = self.F_bar.value
        self.coeffs_conv[:] = self.coeffs

    def _collect_averages(self) -> list:
        result = []
        for field in self.average_fields:
            if field == "P":
                result.append(self.compute_average_first_pk_stress())
            elif field == "A":
                result.append(self.compute_effective_tangent_moduli())
            elif field == "F":
                result.append(self.F_bar.value.copy())
        return result


# --- Driver ---
if __name__ == "__main__":
    setup_logging(MPI.COMM_WORLD, level=logging.INFO)
    output_dir = "output_rom"

    E = 3000.0
    nu = 0.30
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    rve = RVESolver(
        mesh_path="holes.msh",
        rom_dir="ecm_variant2_data",
        material=NeoHookean(mu=mu, lmbda=lmbda),
        output_dir=output_dir,
        visualize_fields=["u_fluc", "u_total"],
        average_fields=["F", "P", "A"],
        timestepper_options={"t_end": 1.0, "dt_init": 0.01, "dt_min": 1e-5, "dt_max": 0.01, "good_newton_steps": 5},
    )

    lam = 0.8
    Fbar_target = np.array([[lam, 0.0], [0.0, 1.0]], dtype=float)
    output_quantities = rve(Fbar_target)

    import matplotlib.pyplot as plt
    Fbar_conv = []
    Pbar_conv = []
    Abar_conv = []
    for q in output_quantities:
        Fbar_conv.append(q[0])
        Pbar_conv.append(q[1])
        Abar_conv.append(q[2])
    Fbar_conv = np.array(Fbar_conv)
    Pbar_conv = np.array(Pbar_conv)
    Abar_conv = np.array(Abar_conv)

    if Fbar_conv.size and Pbar_conv.size:
        fig, ax = plt.subplots()
        ax.plot(Fbar_conv[:, 0, 0], Pbar_conv[:, 0, 0], marker="o")
        ax.set_xlabel("Fxx")
        ax.set_ylabel("Pxx")
        ax.set_title("Pxx over Fxx")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(f"{output_dir}/Pxx_over_Fxx.pdf", dpi=300)
        plt.close(fig)

    if Fbar_conv.size and Abar_conv.size:
        fig, ax = plt.subplots()
        ax.plot(Fbar_conv[:, 0, 0], Abar_conv[:, 0, 0, 0, 0], marker="o", label="Axxxx")
        ax.plot(Fbar_conv[:, 0, 0], Abar_conv[:, 0, 0, 1, 1], marker="o", label="Ayyyy")
        ax.plot(Fbar_conv[:, 0, 0], Abar_conv[:, 0, 0, 0, 1], marker="o", label="Axxxy")
        ax.set_xlabel("Fxx")
        ax.set_ylabel("Axxxx")
        ax.set_title("Axxxx over Fxx")
        ax.grid(True)
        plt.legend()
        fig.tight_layout()
        fig.savefig(f"{output_dir}/Axxxx_over_Fxx.pdf", dpi=300)
        plt.close(fig)