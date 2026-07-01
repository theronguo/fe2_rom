"""FE²-specific Newton solver variant."""

import logging

import dolfinx_mpc
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
from petsc4py import PETSc

from fe2_rom.hyperelastic_solver.solvers import NewtonSolver

logger = logging.getLogger(__name__)


class NewtonSolverFE2(NewtonSolver):
    """Newton solver variant for FE2 homogenization with adjoint / forward-
    sensitivity solves.

    Exposes a generic API ``solve_macro_sensitivities(rhs_forms_dict)`` that
    solves ``K p_k = rhs_k`` for each scalar component of each named macro
    variable, sharing one factorisation across all RHS. The legacy
    ``solve_adjoint()`` (Fbar-only, gdim² × gdim² nested list) is preserved
    as a thin wrapper for backward compatibility.
    """

    def __init__(self, comm, R_form, J_form, u, du, bcs, mpc=None, *,
                 Jij_forms=None, **kwargs):
        super().__init__(comm, R_form, J_form, u, du, bcs, mpc=mpc, **kwargs)
        # Optional legacy Fbar-only RHS forms (nested list[gdim][gdim]).
        # Used only by ``solve_adjoint``; ``solve_macro_sensitivities`` does
        # not depend on it.
        self._Jij_forms = Jij_forms

    def solve_macro_sensitivities(
        self, rhs_forms_dict: dict[str, list],
    ) -> dict[str, list[fem.Function]]:
        """Solve ``K p_k = rhs_k`` for every scalar component ``k`` of every
        named macro variable in ``rhs_forms_dict``.

        When constraints are active the projected formulation is used:
        ``P K P p_k = P rhs_k``, where ``P = I - C^T (C C^T)^{-1} C`` is the
        same orthogonal projector used by the Newton loop.

        Parameters
        ----------
        rhs_forms_dict
            ``{name -> list[fem.Form]}`` — each form is linear in the test
            function and represents the per-component RHS in the variation
            of the residual w.r.t. that macro variable.

        Returns
        -------
        ``{name -> list[fem.Function]}`` — the forward sensitivities
        ``p_k = ∂w/∂μ_k`` for each scalar component, in the same order as
        the input forms.
        """
        if self.mpc is not None:
            K = dolfinx_mpc.assemble_matrix(self._J_form, self.mpc, bcs=self._bcs)
        else:
            K = fem_petsc.assemble_matrix(self._J_form, bcs=self._bcs)
        K.assemble()

        use_projected = bool(self._constraint_vecs)
        if use_projected:
            _, apply_P = self._make_projector()
            sizes = K.getSizes()
            n_global = sizes[0][1]

            class _PKP:
                def mult(self_inner, mat, x, y):
                    Px = x.copy()
                    apply_P(Px)
                    K.mult(Px, y)
                    PETSc.Vec.destroy(Px)
                    apply_P(y)

                def multTranspose(self_inner, mat, x, y):
                    self_inner.mult(mat, x, y)

            K_proj = PETSc.Mat().create(self._comm)
            K_proj.setSizes(sizes)
            K_proj.setType(PETSc.Mat.Type.PYTHON)
            K_proj.setPythonContext(_PKP())
            K_proj.setUp()
            ksp = PETSc.KSP().create(self._comm)
            ksp.setOperators(K_proj, K)
            ksp.setType(PETSc.KSP.Type.MINRES)
            ksp.getPC().setType(PETSc.PC.Type.GAMG)
            ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=min(n_global, 50_000))
            ksp.setFromOptions()
        else:
            K_proj = None
            ksp = self._make_ksp(K)

        results: dict[str, list[fem.Function]] = {}
        for name, forms in rhs_forms_dict.items():
            sensitivities: list[fem.Function] = []
            for rhs_form in forms:
                if self.mpc is not None:
                    rhs = dolfinx_mpc.assemble_vector(rhs_form, self.mpc)
                    dolfinx_mpc.apply_lifting(rhs, [self._J_form], [self._bcs], self.mpc)
                else:
                    rhs = fem_petsc.assemble_vector(rhs_form)
                    fem_petsc.apply_lifting(rhs, [self._J_form], [self._bcs])
                rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                fem_petsc.set_bc(rhs, self._bcs)

                if use_projected:
                    apply_P(rhs)

                X = self._du.x.petsc_vec.duplicate()
                ksp.solve(rhs, X)

                if use_projected:
                    reason = ksp.getConvergedReason()
                    if reason < 0:
                        logger.warning(
                            "Projected MINRES for sensitivity '%s' did not converge (reason %d)",
                            name, reason,
                        )
                    apply_P(X)

                X.ghostUpdate(
                    addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD,
                )
                if self.mpc is not None:
                    self.mpc.backsubstitution(X)

                p = fem.Function(self._u.function_space)
                X.copy(p.x.petsc_vec)
                p.x.scatter_forward()
                sensitivities.append(p)

                PETSc.Vec.destroy(rhs)
                PETSc.Vec.destroy(X)
            results[name] = sensitivities

        ksp.destroy()
        if K_proj is not None:
            K_proj.destroy()
        PETSc.Mat.destroy(K)
        return results

    def solve_adjoint(self):
        """Legacy API: return nested ``adjoints[k][l]`` for Fbar components.

        Equivalent to ``solve_macro_sensitivities({"Fbar": flat_forms})`` with
        the flattened ``Jij_forms`` provided at construction.
        """
        if self._Jij_forms is None:
            raise RuntimeError(
                "solve_adjoint() requires Jij_forms to be supplied at construction; "
                "for new code use solve_macro_sensitivities(rhs_forms_dict)."
            )
        gdim = len(self._Jij_forms)
        flat = [self._Jij_forms[k][l] for k in range(gdim) for l in range(gdim)]
        flat_out = self.solve_macro_sensitivities({"Fbar": flat})["Fbar"]
        return [[flat_out[k * gdim + l] for l in range(gdim)] for k in range(gdim)]
