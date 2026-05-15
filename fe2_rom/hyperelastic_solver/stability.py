import logging

import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc

logger = logging.getLogger(__name__)


class StabilityAnalyzer:
    """Checks stability of the current equilibrium state via eigenvalue analysis.

    Uses SLEPc EPS with shift-invert spectral transform targeting eigenvalues near
    zero. A negative eigenvalue of the tangent stiffness indicates an unstable
    (post-buckling or bifurcation) equilibrium.

    Destroys the supplied K matrix internally after solving.
    """

    def __init__(self, comm, nev: int = 5, neg_tol: float = -1e-12,
                 tol: float = 1e-4,
                 petsc_options: dict | None = None,
                 n_skip_eigenvalues: int = 0):
        # Shift-invert at σ=0 sorts by |λ|. In this benchmark there are 3
        # positive eigenvalues (~1e-2) closer to 0 than the buckling-mode
        # negatives (~1e-1), so nev≤3 misses the negatives and falsely
        # reports stable. nev=5 is the smallest value that's robust here;
        # with MUMPS LDLᵀ below, even nev=5 finishes in a few seconds.
        #
        # petsc_options is a flat dict of SLEPc/PETSc option keys (no
        # leading '-') applied via PETSc.Options before eigensolver.setFrom
        # Options(). Use it to override defaults — e.g. switch the ST inner
        # solver to an iterative one when MUMPS no longer fits in memory:
        #     {"st_ksp_type": "minres",
        #      "st_pc_type": "gamg",
        #      "st_ksp_rtol": 1e-3}
        # Or change EPS type entirely (e.g. "eps_type": "jd"/"lobpcg").
        self._comm = comm
        self._nev = nev
        self._neg_tol = neg_tol
        self._tol = tol
        self._petsc_options = petsc_options
        # Number of smallest-magnitude eigenvalues to skip in the stability
        # check. Each integral linear constraint corresponds to one near-zero
        # mode of K (the constraint pins one direction of the unconstrained
        # null space). For periodic homogenization with the default ``⟨w⟩=0``
        # gauge, this equals ``gdim``; pass it from the caller so SLEPc returns
        # ``nev + n_skip`` eigenvalues and the check ignores the first
        # ``n_skip``, which are gauge-direction noise rather than physical
        # buckling modes.
        self._n_skip = n_skip_eigenvalues

    def check(self, K: PETSc.Mat, eigenfunction) -> tuple[bool, np.ndarray]:
        """Run eigenvalue analysis on K.

        If any eigenvalue is below neg_tol, writes the first corresponding
        eigenvector into eigenfunction and returns (False, eigenvalues).
        Destroys K.

        The extraction threshold for the eigenvector (< 1e-12) intentionally
        includes slightly positive eigenvalues to match the original solver's
        behaviour of perturbing near-zero modes.
        """
        # Solve for nev physical modes PLUS n_skip gauge-mode buffer, so the
        # check has nev real eigenvalues to inspect even after skipping the
        # near-zero gauge eigenvalues at the bottom of the spectrum.
        total_nev = self._nev + self._n_skip
        logger.info("Running stability analysis (nev=%d, skip=%d) ...",
                    self._nev, self._n_skip)

        eigensolver = SLEPc.EPS().create(self._comm)
        eigensolver.setOperators(K)
        eigensolver.setProblemType(SLEPc.EPS.ProblemType.HEP)
        st = eigensolver.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        # Shift-invert needs to factor (K - σ·I) once per call. Use MUMPS LDLᵀ:
        # symmetric, handles indefinite K via pivoting, ~3–5× faster than the
        # PETSc-builtin LU that SLEPc otherwise picks.
        # st_ksp = st.getKSP()
        # st_ksp.setType("preonly")
        # st_pc = st_ksp.getPC()
        # st_pc.setType("cholesky")
        # st_pc.setFactorSolverType("mumps")
        eigensolver.setTarget(0.0)
        eigensolver.setDimensions(nev=total_nev)
        eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
        eigensolver.setTolerances(tol=self._tol)

        if self._petsc_options is not None:
            opts = PETSc.Options()
            for key, val in self._petsc_options.items():
                opts[key] = val

        eigensolver.setFromOptions()
        try:
            eigensolver.solve()

            n_conv = eigensolver.getConverged()
            all_eigenvalues = np.array([
                eigensolver.getEigenvalue(i).real
                for i in range(min(n_conv, total_nev))
            ])

            if self._n_skip > 0:
                # Drop the ``n_skip`` smallest-|λ| eigenvalues — these are the
                # gauge-direction near-zero modes (one per integral constraint
                # in the saddle-point path). Selecting by |λ| (not by signed
                # value) is essential: a real buckling mode at λ ≈ −0.5 has
                # |λ| ≫ |gauge ≈ 1e-13|, so it stays in the physical set;
                # using argsort on signed value would silently discard the
                # buckling mode whenever one appears.
                abs_eig = np.abs(all_eigenvalues)
                physical_indices = np.argsort(abs_eig)[self._n_skip:]
            else:
                # Base case (corner-pinning, K is SPD): no skip. Preserve the
                # SLEPc TARGET_REAL ordering so that the "first eigenvalue
                # below neg_tol" is the buckling mode closest to zero — the
                # original solver's eigenvector-selection convention.
                physical_indices = np.arange(len(all_eigenvalues))
            eigenvalues = all_eigenvalues[physical_indices]

            logger.info("Smallest physical eigenvalues (skipped %d gauge modes): %s",
                        self._n_skip, np.array2string(eigenvalues, precision=4))

            is_stable = True
            negatives = np.where(eigenvalues < self._neg_tol)[0]
            if negatives.size > 0:
                global_idx = int(physical_indices[negatives[0]])
                eigensolver.getEigenvector(global_idx, eigenfunction.x.petsc_vec)
                eigenfunction.x.scatter_forward()
                is_stable = False
        finally:
            eigensolver.destroy()
            PETSc.Mat.destroy(K)

        return is_stable, eigenvalues
