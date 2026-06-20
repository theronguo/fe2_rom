"""``MacroFit`` — a differentiable macro solve for fitting EnergyNet weights.

Wraps a built macro solver (CH1 / MM / CH2) whose constitutive law is an
:class:`~fe2_rom.nn.model.EnergyNet`-backed NN material, and exposes

    R = fit(theta)          # reaction at each load-schedule point   (n_pts,)
    J = fit.jac(theta)      # d reaction / d weights                 (n_pts, n_params)

ready for ``scipy.optimize.least_squares(lambda th: fit(th) - R_dns, th0, jac=fit.jac)``.

How the Jacobian is computed
----------------------------
The NN materials are *stateless hyperelastic*: the macro response at a load
level depends only on that level (and the weights θ), not on the path. So each
schedule point is an independent equilibrium ``R(w; θ) = 0`` and contributes an
independent adjoint — no time-chain, no adjoint checkpointing.

For one level, the measured reaction is the assembled residual summed over the
Dirichlet reaction DOFs, ``g(θ) = sᵀ R(w(θ); θ)`` with ``s`` the 0/1 indicator of
those DOFs. Because the QoI lives on the Dirichlet boundary, the adjoint needs
the *unconstrained* tangent blocks:

    r   = (Kᵤₙ_c)ᵀ s,   zeroed on all Dirichlet DOFs
    K_bcᵀ λ = r                     (transpose solve, MUMPS LU)
    c   = s − λ                     (co-state: c_d = s_d, c_f = −λ_f)
    dg/dθ = cᵀ ∂R/∂θ = Σ_q (∂flux_q/∂θ)ᵀ a_q

where ``a_q`` is the quadrature-weighted gradient of the co-state field mapped
into flux space — obtained by differentiating ``replace(Res, {test: c})`` w.r.t.
each flux quadrature-function (guaranteeing consistency with the forward
assembly), and the flux–weight contraction is
:func:`fe2_rom.nn.sensitivity.flux_weight_vjp`.

This wrapper is model-agnostic: it relies only on the macro solver's
``qmap`` / ``Res`` / reaction-spec contract, which CH1, MM and CH2 share.
``MacroFit.for_ch2`` builds the CH2 stack; the other flavors are analogous.
"""
from __future__ import annotations

import logging

import numpy as np
import ufl
from dolfinx import fem
import dolfinx.fem.petsc  # noqa: F401  (registers fem.petsc)
from mpi4py import MPI
from petsc4py import PETSc

from fe2_rom.nn.sensitivity import (
    flux_weight_vjp,
    params_to_vector,
    set_params_from_vector,
)

logger = logging.getLogger(__name__)


class MacroFit:
    """Differentiable wrapper around a built macro solver + NN EnergyNet.

    Parameters
    ----------
    solver
        A built (pre-``setup``) macro solver: CH2 ``MacroSecondOrderSolver``,
        MM ``MacroMicromorphicSolver`` or CH1 ``MacroSolver``. Must expose
        ``V, w, qmap, Res, _test, comm`` and (after ``setup``) ``_bcs,
        _reaction_specs, _Jac_form, _Res_form, _problem``.
    model
        The :class:`EnergyNet` whose weights are fit.
    material
        The NN material wrapping ``model`` (must provide ``refresh_from_model``).
    field_attr, test_attr
        Names of the solver's solution :class:`fem.Function` and test
        :class:`ufl.Argument` — ``("w", "_test")`` for the mixed MM/CH2 drivers,
        ``("u", "_v")`` for the CH1 ``MacroSolver``.
    """

    def __init__(self, solver, model, material, *,
                 field_attr="w", test_attr="_test"):
        self.solver = solver
        self.model = model
        self.material = material
        self.comm = solver.comm
        self._field_attr = field_attr
        self._test_attr = test_attr
        self._levels: list[float] | None = None
        self._loadhistory = None
        # forward cache keyed by the exact theta vector
        self._cache_theta: np.ndarray | None = None
        self._cache_R: np.ndarray | None = None
        self._cache_w: list[np.ndarray] | None = None
        self._cache_ok: bool = True
        self._is_setup = False

    @property
    def _field(self):
        """The solver's solution Function (``w`` for MM/CH2, ``u`` for CH1)."""
        return getattr(self.solver, self._field_attr)

    @property
    def _testfn(self):
        """The solver's test Argument (``_test`` for MM/CH2, ``_v`` for CH1)."""
        return getattr(self.solver, self._test_attr)

    # -- construction -------------------------------------------------------

    @classmethod
    def for_ch2(cls, model, mesh, *, n_qp=2, degree=1, u_degree=None,
                lagrange_degree=None, lagrange_discontinuous=None,
                compat_penalty=0.0, lagrange_stab=0.0, snes_options=None):
        """Build the CH2 stack: ``NNCh2Material`` + ``MacroSecondOrderSolver``.

        The multiplier element defaults to the inf-sup-stable P2-P1-P1
        (continuous P1 multiplier, 2D and 3D); see
        :class:`~fe2_rom.ch2.macrosolver.MacroSecondOrderSolver`."""
        from fe2_rom.ch2.macrosolver import MacroSecondOrderSolver
        from fe2_rom.ch2.nn_material import NNCh2Material

        if model.flavor != "ch2":
            raise ValueError(
                f"for_ch2 needs a flavor='ch2' EnergyNet, got {model.flavor!r}.")
        material = NNCh2Material(model)
        solver = MacroSecondOrderSolver(
            mesh, n_qp=n_qp, material=material, degree=degree, u_degree=u_degree,
            lagrange_degree=lagrange_degree,
            lagrange_discontinuous=lagrange_discontinuous,
            compat_penalty=compat_penalty, lagrange_stab=lagrange_stab,
            snes_options=snes_options, check_stability=False,
        )
        return cls(solver, model, material)

    @classmethod
    def for_ch1(cls, model, mesh, *, n_qp=2, degree=1, gdim=None,
                snes_options=None):
        """Build the CH1 stack: ``NNRVEMaterial`` + ``MacroSolver`` (closed-form
        law, no inner RVE). ``add_bc`` takes an integer displacement component."""
        from fe2_rom.ch1.macrosolver import MacroSolver
        from fe2_rom.ch1.nn_material import NNRVEMaterial

        if model.flavor != "ch1":
            raise ValueError(
                f"for_ch1 needs a flavor='ch1' EnergyNet, got {model.flavor!r}.")
        material = NNRVEMaterial(model)
        solver = MacroSolver(
            mesh, n_qp=n_qp, material=material,
            gdim=model.gdim if gdim is None else gdim,
            degree=degree, snes_options=snes_options, check_stability=False,
        )
        return cls(solver, model, material, field_attr="u", test_attr="_v")

    @classmethod
    def for_mm(cls, model, mesh, *, n_qp=2, degree=1, snes_options=None):
        """Build the MM stack: ``NNMicromorphicMaterial`` + ``MacroMicromorphicSolver``.

        ``N_modes`` is taken from the model. ``add_bc`` uses the mixed-space
        signature: ``(0, comp)`` for displacement, ``(i+1,)`` for amplitude
        ``v_i`` (see the solver)."""
        from fe2_rom.mm.macrosolver import MacroMicromorphicSolver
        from fe2_rom.mm.nn_material import NNMicromorphicMaterial

        if model.flavor != "mm":
            raise ValueError(
                f"for_mm needs a flavor='mm' EnergyNet, got {model.flavor!r}.")
        material = NNMicromorphicMaterial(model)
        solver = MacroMicromorphicSolver(
            mesh, n_qp=n_qp, N_modes=model.n_modes, material=material,
            degree=degree, snes_options=snes_options, check_stability=False,
        )
        return cls(solver, model, material)

    # -- registration (passthrough to the solver) ---------------------------

    def add_bc(self, *args, **kwargs):
        self.solver.add_bc(*args, **kwargs)

    def set_load_schedule(self, levels, loadhistory):
        """``levels``: scalar load parameters (the DNS sample points). For each,
        ``loadhistory(level)`` is called to set the BC ``Constant``\\ s before the
        solve, exactly like the macro solver's ``loadhistory(t)``."""
        self._levels = [float(x) for x in levels]
        self._loadhistory = loadhistory

    def setup(self):
        self.solver.setup()
        if not self.solver._reaction_specs:
            raise ValueError("MacroFit needs at least one measure_reaction=True BC.")
        if self._levels is None:
            raise ValueError("Call set_load_schedule(...) before setup().")
        self._build_reaction_selector()
        self._build_bc_dofs()
        self._is_setup = True
        return self

    # -- weights ------------------------------------------------------------

    def get_weights(self) -> np.ndarray:
        return params_to_vector(self.model)

    @property
    def n_params(self) -> int:
        return int(self.get_weights().size)

    def _set_weights(self, theta) -> None:
        # EnergyNet is an immutable equinox pytree: rebuild it and hand the new
        # object to the material (which rebuilds its jitted closures).
        self.model = set_params_from_vector(self.model, theta)
        self.material.update_model(self.model)

    # -- forward ------------------------------------------------------------

    def __call__(self, theta) -> np.ndarray:
        """Reaction at each schedule point for weights ``theta`` (shape (n_pts,))."""
        theta = np.asarray(theta, dtype=np.float64)
        self._run_forward(theta)
        return self._cache_R.copy()

    def _run_forward(self, theta) -> None:
        if (self._cache_theta is not None
                and self._cache_R is not None
                and np.array_equal(theta, self._cache_theta)):
            return  # reuse the cached forward (scipy calls fun then jac)

        self._set_weights(theta)
        s = self.solver
        w = self._field
        # Start each trajectory from the reference state; continue level-to-level.
        w.x.array[:] = 0.0
        w.x.scatter_forward()

        R = np.zeros(len(self._levels))
        w_states: list[np.ndarray] = []
        ok = True
        for k, level in enumerate(self._levels):
            self._loadhistory(level)
            try:
                s._problem.solve()
                conv = int(s._problem.solver.getConvergedReason()) > 0
            except Exception:  # SNES blew up
                conv = False
            ok = ok and conv
            R[k] = self._reaction_value()
            w_states.append(w.x.array.copy())

        self._cache_theta = theta.copy()
        self._cache_R = R
        self._cache_w = w_states
        self._cache_ok = ok
        if not ok:
            logger.warning("Macro forward: not all load levels converged.")

    def _reaction_value(self) -> float:
        """Sum of the assembled residual over all reaction DOFs (matches the
        macro solver's ``_record_reactions``)."""
        s = self.solver
        b = fem.petsc.assemble_vector(fem.form(s.Res))
        try:
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            n_local = s.V.dofmap.index_map.size_local * s.V.dofmap.index_map_bs
            local = b.array_r
            total = 0.0
            for parent_dofs, _value, _direction in s._reaction_specs:
                owned = parent_dofs[parent_dofs < n_local]
                total += float(local[owned].sum()) if owned.size else 0.0
            return self.comm.allreduce(total, op=MPI.SUM)
        finally:
            b.destroy()

    # -- Jacobian (per-level adjoint) ---------------------------------------

    def jac(self, theta) -> np.ndarray:
        """``d reaction / d weights``, shape ``(n_pts, n_params)``."""
        theta = np.asarray(theta, dtype=np.float64)
        self._run_forward(theta)  # ensures cached w-states for this theta
        s = self.solver
        w = self._field
        J = np.zeros((len(self._levels), self.n_params))
        for k, level in enumerate(self._levels):
            # Restore the converged state of this level and refresh the qmap
            # (fluxes + tangent) at (w_k, theta).
            self._loadhistory(level)
            w.x.array[:] = self._cache_w[k]
            w.x.scatter_forward()
            s.qmap.update()
            J[k, :] = self._level_sensitivity()
        return J

    def _level_sensitivity(self) -> np.ndarray:
        """``dg/dθ`` for the currently-restored level via the reaction adjoint."""
        s = self.solver
        # --- assemble tangents: unconstrained (for the adjoint RHS) + BC-applied
        K_unc = fem.petsc.assemble_matrix(s._Jac_form, bcs=[])
        K_unc.assemble()
        K_bc = fem.petsc.assemble_matrix(s._Jac_form, bcs=s._bcs)
        K_bc.assemble()
        try:
            # r = K_uncᵀ s, then impose homogeneous adjoint Dirichlet (zero on BC dofs)
            r = K_unc.createVecRight()
            K_unc.multTranspose(self._s_vec, r)
            with r.localForm() as rl:
                rl.array_w[self._bc_dofs_local] = 0.0

            lam = K_bc.createVecRight()
            ksp = PETSc.KSP().create(self.comm)
            ksp.setOperators(K_bc)
            ksp.setType("preonly")
            pc = ksp.getPC()
            pc.setType("lu")
            pc.setFactorSolverType("mumps")
            ksp.solveTranspose(r, lam)
            ksp.destroy()

            # co-state c = s − λ  (c_d = s_d on reaction dofs, c_f = −λ_f)
            c = fem.Function(s.V)
            c.x.petsc_vec.waxpy(-1.0, lam, self._s_vec)  # c = s_vec - lam
            c.x.scatter_forward()

            z = self._gradient_values()          # (n_q, z_dim)
            a = self._cotangent_values(c)        # (n_q, flux_dim)
            return flux_weight_vjp(self.model, z, a)
        finally:
            K_unc.destroy()
            K_bc.destroy()

    # -- adjoint building blocks --------------------------------------------

    def _gradient_values(self) -> np.ndarray:
        """Per-owned-qp gradient array ``z`` in material.gradients order."""
        qmap = self.solver.qmap
        blocks = [qmap.get_gradient_vals(qmap.gradients[name], qmap.cells)
                  for name in self.material.gradients.keys()]
        return np.concatenate(blocks, axis=1)

    def _cotangent_values(self, c: fem.Function) -> np.ndarray:
        """Per-owned-qp flux-space cotangent ``a_q = w_q ζ_q``, by differentiating
        ``replace(Res, {test: c})`` w.r.t. each flux quadrature-function."""
        s = self.solver
        qmap = s.qmap
        Rc = ufl.replace(s.Res, {self._testfn: c})
        blocks = []
        for name in self.material.fluxes.keys():
            flux_qf = qmap.fluxes[name]
            dflux = ufl.TestFunction(flux_qf.function_space)
            form = fem.form(ufl.derivative(Rc, flux_qf, dflux))
            b = fem.petsc.assemble_vector(form)
            try:
                b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                              mode=PETSc.ScatterMode.REVERSE)
                dim = 1 if flux_qf.ufl_shape == () else int(np.prod(flux_qf.ufl_shape))
                vals = b.array_r.reshape(-1, dim)[qmap.dofs]
            finally:
                b.destroy()
            blocks.append(vals)
        return np.concatenate(blocks, axis=1)

    # -- precomputed selectors ----------------------------------------------

    def _build_reaction_selector(self) -> None:
        """PETSc vector with 1.0 on every reaction DOF (the QoI selector s)."""
        s = self.solver
        sel = fem.Function(s.V)
        n_local = s.V.dofmap.index_map.size_local * s.V.dofmap.index_map_bs
        arr = sel.x.array
        for parent_dofs, _value, _direction in s._reaction_specs:
            owned = parent_dofs[parent_dofs < n_local]
            arr[owned] = 1.0
        sel.x.scatter_forward()
        self._s_vec = sel.x.petsc_vec
        self._s_func = sel  # keep a reference alive

    def _build_bc_dofs(self) -> None:
        """Local indices of all Dirichlet DOFs (for homogeneous adjoint BC)."""
        dofs = []
        for bc in self.solver._bcs:
            d = bc._cpp_object.dof_indices()[0]
            dofs.append(np.asarray(d, dtype=np.int32))
        self._bc_dofs_local = (np.unique(np.concatenate(dofs))
                               if dofs else np.empty(0, dtype=np.int32))
