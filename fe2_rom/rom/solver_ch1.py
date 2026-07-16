import os
import logging
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import scipy

from ..hyperelastic_solver import NeoHookean, VTXManager, TimeStepper, setup_logging, silence_c_stdout
from fe2_rom.ch1.exceptions import RVEConvergenceError
from fe2_rom.ch1.averages import (
    HomogenizationContext,
    EffectiveFbar,
    EffectivePbar,
    EffectiveAbar,
    EffectiveAbarReduced,
    resolve_average_quantities,
)
from fe2_rom.ch1.objectivity import (
    objective_transform_pbar,
    polar_derivatives,
    symmetric_basis_tensors,
)
from ..hyperelastic_solver.forms import basis_tensor_ufl
from dolfinx import io, fem, mesh as dmesh
from dolfinx.fem.petsc import assemble_vector as _assemble_vector_petsc
from mpi4py import MPI
from petsc4py import PETSc
import ufl

logger = logging.getLogger(__name__)


def _dF_Fbar_factory(context):
    gdim = context.mesh.geometry.dim
    return [basis_tensor_ufl(gdim, i, j) for i in range(gdim) for j in range(gdim)]


class ReducedMicroSolver:
    """ROM-based hyperelastic periodic homogenization solver using ECM.

    Modular by design. Subclasses extend three orthogonal axes:

    * **macro variables** — ``_setup_macro_vars`` registers extra
      ``fem.Constant`` inputs in ``self.macro_vars``.
    * **kinematic ansatz** — ``_build_F_ufl_extra`` returns an extra
      displacement contribution ``u_extra(X)``; its gradient is added to
      ``F = F̄ + ∇u_fluc``.
    * **effective quantities & tangents** — ``_make_default_average_quantities``
      returns ``AverageQuantity`` / ``TangentBlock`` instances from
      ``fe2_rom.ch1.averages``. ``_make_dF_dmu_factories``
      registers the ``∂F/∂μ`` UFL tensors used to build ROM adjoint RHS forms.

    Public API (preserved):
        solver = ReducedMicroSolver(mesh_path, rom_dir, material)
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
        rve_volume: float | None = None,
        objective_reduction: bool = False,
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
        # Objectivity (F̄ = R U) reduction — drive the ROM with the symmetric
        # stretch U, reconstruct lab-frame P̄/dP̄/dF̄ analytically (6/3 symmetric
        # adjoints instead of gdim²). See fe2_rom.ch1.objectivity.
        self._objective = bool(objective_reduction)
        self._material = material
        self._newton_options = newton_options
        self._timestepper = TimeStepper(**timestepper_options)
        self._averages_only_final = averages_only_final
        self.output_dir = output_dir

        # --- ROM data ---
        indices          = np.load(os.path.join(rom_dir, "indices.npy"))
        self.basis_u_sub = np.load(os.path.join(rom_dir, "basis_u_sub.npy"), mmap_mode="r")
        self.basis_u     = np.load(os.path.join(rom_dir, "basis_u.npy"), mmap_mode="r")
        self.N = self.basis_u_sub.shape[1]
        # Per-qp ECM rule (quadrature_element weights at individual Gauss points)
        # vs. the classic per-cell DG-0 rule, detected by qp_meta.json.
        _qp_meta_path = os.path.join(rom_dir, "qp_meta.json")
        self._per_qp = os.path.exists(_qp_meta_path)
        if self._per_qp:
            import json
            with open(_qp_meta_path) as _f:
                self._qp_meta = json.load(_f)
            omega_data = np.load(os.path.join(rom_dir, "omega_q_sub.npy"))
        else:
            self._qp_meta = None
            omega_data = np.load(os.path.join(rom_dir, "omega_sub.npy"))
        logger.debug("ROM data loaded: N=%d modes (per_qp=%s)", self.N, self._per_qp)

        # --- Mesh & submesh ---
        with silence_c_stdout():
            mesh = io.gmsh.read_from_msh(mesh_path, comm, 0, gdim=gdim).mesh
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        tdim = mesh.topology.dim
        V_full = fem.functionspace(mesh, ("Lagrange", degree, (gdim,)))
        submesh, sub_cell_map, _, _ = dmesh.create_submesh(mesh, tdim, indices)
        self._mesh_full = mesh
        self._submesh = submesh
        # Sub->parent cell map, kept so subclasses can transfer parent-mesh DOF
        # arrays (e.g. φ) onto V_sub by *exact dof-copy* rather than the lossy
        # cross-mesh interpolation (which is inexact on curved/degree-2 meshes).
        self._sub_cell_map = sub_cell_map
        # Per-qp rules require the integration quadrature to match the
        # quadrature_element used to build the rule exactly (same scheme+degree),
        # otherwise the saved point weights are meaningless.
        if self._per_qp:
            self._dx_sub = ufl.Measure(
                "dx", domain=submesh,
                metadata={"quadrature_scheme": self._qp_meta["scheme"],
                          "quadrature_degree": self._qp_meta["qdeg"]})
        else:
            self._dx_sub = ufl.Measure("dx", domain=submesh)

        # --- Function spaces ---
        V_sub  = fem.functionspace(submesh, ("Lagrange", degree, (gdim,)))
        self.V_sub = V_sub
        self.V_full = V_full

        # --- ECM weight function: DG-0 (per-cell) or quadrature_element (per-qp) ---
        if self._per_qp:
            import basix.ufl
            cell_name = submesh.topology.cell_type.name
            q_el = basix.ufl.quadrature_element(
                cell_name, value_shape=(),
                scheme=self._qp_meta["scheme"], degree=self._qp_meta["qdeg"])
            W_space = fem.functionspace(submesh, q_el)
        else:
            W_space = fem.functionspace(submesh, ("DG", 0))

        # --- Functions ---
        self._omega_func = fem.Function(W_space)
        self._omega_func.x.array[:] = omega_data

        self.u_fluc = fem.Function(V_sub)
        self.u_full = fem.Function(V_full, name="u_fluc")

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

        # --- Residual / Jacobian (FOM forms on submesh, projected to ROM) ---
        # Assemble once per Newton iter as full vector/matrix on V_sub, then
        # contract with the POD basis Φ:  r_rom = Φᵀ r,  J_rom = Φᵀ K Φ.
        # Avoids the N²+N scalar-assembly blow-up of the previous coefficient
        # stamping approach.
        u_tr = ufl.TrialFunction(V_sub)
        v_te = ufl.TestFunction(V_sub)
        self._v_te = v_te
        ii, jj, kk, ll = ufl.indices(4)
        A_grad_tr = ufl.as_tensor(self.A_ufl[ii, jj, kk, ll] * ufl.grad(u_tr)[kk, ll], (ii, jj))
        self._r_form = fem.form(
            ufl.inner(self._P_ufl, ufl.grad(v_te)) * self._omega_func * self._dx_sub
        )
        self._j_form = fem.form(
            ufl.inner(A_grad_tr, ufl.grad(v_te)) * self._omega_func * self._dx_sub
        )

        # Macroscopic averaging volume = |Q| (periodic-cell volume).  ECM
        # weights ω are calibrated against FOM solid integrals, so
        # ∫ω dx_sub ≈ |Ω_solid| — wrong denominator for a porous RVE.  Caller
        # must supply ``rve_volume`` (e.g. (2ℓ)² for a square cell, or the
        # exact area of a hexagonal cell).  If absent we fall back to the
        # ECM-weighted solid integral, which only matches the FOM convention
        # for non-porous RVEs.
        if rve_volume is None:
            self._vol = fem.assemble_scalar(
                fem.form(1.0 * self._omega_func * self._dx_sub)
            )
        else:
            self._vol = float(rve_volume)
        logger.debug("Averaging volume |Q|: %.6f", self._vol)

        # --- HomogenizationContext + average-quantity registry ---
        self._context = HomogenizationContext(
            mesh=submesh, V=V_sub, dx=self._dx_sub, comm=comm,
            vol_global=self._vol,
            F_var=F_ufl, P_ufl=self._P_ufl, A_ufl=self.A_ufl,
            W_ufl=material.strain_energy(F_ufl), u=self.u_fluc, u_total=None,
            macro_vars=self.macro_vars,
            phi=getattr(self, "_phi", []),
            weight=self._omega_func,
        )

        if average_quantities is None:
            self._quantities = self._make_default_average_quantities()
        else:
            self._quantities = resolve_average_quantities(
                average_quantities, self._string_key_factories())
        if self._objective:
            self._quantities = self._objectivize_quantities(self._quantities)
        for q in self._quantities:
            q.setup(self._context)

        # --- ROM adjoint machinery ---
        # ∂F/∂μ factories per macro var; materialised once into UFL tensor lists.
        self._dF_dmu_factories = self._make_dF_dmu_factories()
        self._dF_dmu_lists = {
            name: factory(self._context)
            for name, factory in self._dF_dmu_factories.items()
        }

        # Compile ROM adjoint RHS forms as linear forms in TestFunction:
        #   d_mi = ∫ A : ∂F/∂μ_mi · ∇v_te ω dx     (vector on V_sub)
        #   D[:, mi] = Φᵀ d_mi
        # Convention matches FOM ``TangentBlock`` (last two indices of A
        # contract with ∂F/∂μ).
        # Assemble all ∂R/∂μ_k columns in ONE vector over a tensor-valued
        # (gdim × n_total) Lagrange test space, so the rank-4 tangent A is
        # evaluated once per cell instead of once per column. The flat column
        # order (and the per-macro-var column slices) are recorded for the
        # projection/solve in ``_solve_adjoints``.
        self._adj_flat = [
            (name, mi)
            for name, dF_list in self._dF_dmu_lists.items()
            for mi in range(len(dF_list))
        ]
        self._n_adj_total = len(self._adj_flat)
        dF_all = [dF for dF_list in self._dF_dmu_lists.values() for dF in dF_list]
        V_big = fem.functionspace(submesh, ("Lagrange", degree, (gdim, self._n_adj_total)))
        v_big = ufl.TestFunction(V_big)
        L_batched = None
        for idx, dF in enumerate(dF_all):
            A_dF = ufl.as_tensor(self.A_ufl[ii, jj, kk, ll] * dF[kk, ll], (ii, jj))
            vte_idx = ufl.as_vector([v_big[a, idx] for a in range(self.gdim)])
            term = ufl.inner(A_dF, ufl.grad(vte_idx)) * self._omega_func * self._dx_sub
            L_batched = term if L_batched is None else L_batched + term
        self._D_form_batched = fem.form(L_batched)
        self._adj_cols: dict[str, list] = {}
        for gi, (name, _mi) in enumerate(self._adj_flat):
            self._adj_cols.setdefault(name, []).append(gi)

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
        return {"Fbar": self._fbar_dF_factory}

    def _fbar_dF_factory(self, context) -> list:
        """``∂F/∂F̄`` adjoint directions. Objective reduction: the 6 (3D) / 3
        (2D) symmetric basis tensors (sensitivities w.r.t. the stretch ``U``).
        Otherwise: the full ``gdim²`` single-entry basis tensors."""
        gdim = context.mesh.geometry.dim
        if self._objective:
            return [ufl.as_matrix(S.tolist())
                    for S in symmetric_basis_tensors(gdim)]
        return [basis_tensor_ufl(gdim, i, j)
                for i in range(gdim) for j in range(gdim)]

    def _objectivize_quantities(self, qs: list) -> list:
        """Swap effective quantities for their U-frame reduced variants when the
        objectivity reduction is active. Subclasses extend this for their own
        ``/dF̄`` tangent blocks."""
        out = []
        for q in qs:
            if type(q) is EffectiveAbar:
                out.append(EffectiveAbarReduced())
            else:
                out.append(q)
        return out

    def _make_default_average_quantities(self) -> list:
        """Default quantity registry for the base solver."""
        return [EffectiveFbar(), EffectivePbar(), EffectiveAbar()]

    def _string_key_factories(self) -> dict:
        """Solver-specific ``{key: zero-arg callable}`` string-key constructors,
        extending ``STRING_KEY_MAP`` (see the micromorphic subclass for the
        φ-bound ``"Pi"`` / ``"Lambda"`` / tangent-block keys)."""
        return {}

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
        self.u_fluc.x.array[:] = self.basis_u_sub @ self.coeffs

    # --- FOM-assembly helpers (projected to ROM via Φ) ------------------------

    def _assemble_fom_residual(self) -> np.ndarray:
        b = _assemble_vector_petsc(self._r_form)
        b.assemble()
        arr = b.array.copy()
        b.destroy()
        return arr

    def _assemble_fom_jacobian(self):
        K = fem.assemble_matrix(self._j_form).to_scipy()
        return K

    def _assemble_fom_linear(self, form) -> np.ndarray:
        b = _assemble_vector_petsc(form)
        b.assemble()
        arr = b.array.copy()
        b.destroy()
        return arr

    # --- Visualization --------------------------------------------------------

    def _write_fields(self, t: float) -> None:
        if self.vtx is not None:
            self.u_full.x.array[:] = self.basis_u @ self.coeffs
            if self._u_total_expr is not None:
                self._F_bar_full.value[:] = self.F_bar.value
                self._sync_full_mesh_constants()
                self.u_total.interpolate(self._u_total_expr)
            self.vtx.write(t)

    # --- Averaging / adjoints -------------------------------------------------

    def _assemble_jacobian(self) -> np.ndarray:
        K = self._assemble_fom_jacobian()
        Phi = self.basis_u_sub
        return Phi.T @ (K @ Phi)

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

        Phi = self.basis_u_sub
        # One batched assembly of all adjoint RHS, then project: D_all = Φᵀ B.
        # b_big lays out (node, component, column) blocked; reshaping to
        # (n_sub_dofs, n_total) recovers each column in V_sub dof order.
        b_big = self._assemble_fom_linear(self._D_form_batched)
        B = b_big.reshape(Phi.shape[0], self._n_adj_total)
        D_all = Phi.T @ B  # (N, n_total)
        for name in self._adjoint_macro_vars:
            cols = self._adj_cols[name]
            alpha = np.linalg.solve(J_sym, -D_all[:, cols])  # (N, n_mu)
            slots = self._adjoint_funcs[name]
            for mi in range(len(cols)):
                slots[mi].x.array[:] = Phi @ alpha[:, mi]

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
        Phi = self.basis_u_sub
        it = 0
        res_norm = np.inf
        res_norm_0 = None
        rel_res = np.inf
        while it < max_iter:
            r_full = self._assemble_fom_residual()
            residual = Phi.T @ r_full

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

            K = self._assemble_fom_jacobian()
            jacobian = Phi.T @ (K @ Phi)

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
            self.u_fluc.x.array[:] = Phi @ self.coeffs
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

        # Warm-start from the previous __call__: the live coeffs / F̄ (and v/g in
        # the micromorphic subclass) still hold the last solve's converged state,
        # so ramp from there rather than the committed load step. Later macro
        # Newton iterations then only cover the small F̄ increment (~1 reduced
        # step). commit() (once per accepted macro step) still defines the
        # checkpoint point; the material is path-independent, so the converged
        # solution and the consistent tangent are unchanged.
        Fbar_prev = self.F_bar.value.copy()
        Fbar_target = np.asarray(Fbar, dtype=PETSc.ScalarType)

        # Objectivity reduction: drive the ROM with the symmetric stretch U and
        # remember the polar data to rotate the outputs back to the lab frame.
        obj_R = obj_dR = obj_dU = None
        if self._objective:
            obj_R, obj_U, obj_dR, obj_dU = polar_derivatives(
                np.asarray(Fbar, dtype=float))
            Fbar_target = np.asarray(obj_U, dtype=PETSc.ScalarType)

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
            logger.info("── Step  t=%.8f  dt=%.2e", trial_time, self._timestepper.dt)
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

        if self._objective and output_quantities:
            self._objective_transform_output(
                output_quantities[-1], obj_R, obj_dR, obj_dU)

        return output_quantities

    # --- Objectivity reduction: U-frame → lab-frame output conversion ---------

    def _objective_transform_output(self, d: dict, R, dR, dU) -> None:
        """Convert a U-frame output dict to the lab frame *in place*
        (``P̄ = R P̃``, ``dP̄/dF̄`` reconstructed from the U-frame reduced tangent
        and the polar derivatives). Subclasses override
        :meth:`_objective_transform_extra` for the micromorphic blocks."""
        objective_transform_pbar(d, R, dR, dU)
        self._objective_transform_extra(d, R, dR, dU)

    def _objective_transform_extra(self, d: dict, R, dR, dU) -> None:
        """Subclass hook for extra U-frame → lab-frame conversions
        (micromorphic ``Pi``/``Lambda`` blocks). Base does nothing."""
        return

    def commit(self) -> None:
        """Promote trial state to the converged restart point."""
        self.F_bar_conv[:] = self.F_bar.value
        self.coeffs_conv[:] = self.coeffs
        self._commit_extra_state()

    # --- Checkpoint / restart -------------------------------------------------

    def dump_state(self) -> dict:
        """Warm-start state for checkpoint/restart.

        The ROM is path-independent, so only the converged reduced coordinates
        and F̄ are needed: seeding them on resume lets the first post-restart
        ``__call__`` warm-start from the last accepted macro step (a tiny
        increment) instead of cold-starting from the reference state and having
        to re-walk the entire load path — which stalls near buckling.
        """
        return {
            "F_bar_conv": np.asarray(self.F_bar_conv, dtype=float).copy(),
            "coeffs_conv": np.asarray(self.coeffs_conv, dtype=float).copy(),
        }

    def load_state(self, state: dict) -> None:
        self.F_bar_conv[:] = state["F_bar_conv"]
        self.coeffs_conv[:] = state["coeffs_conv"]
        # Seed the live state too, so the next __call__'s warm-start ramp
        # begins at the restored converged point.
        self.F_bar.value[:] = self.F_bar_conv
        self._restore_state(self.coeffs_conv)
