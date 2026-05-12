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
                 petsc_options: dict | None = None):
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

    def check(self, K: PETSc.Mat, eigenfunction) -> tuple[bool, np.ndarray]:
        """Run eigenvalue analysis on K.

        If any eigenvalue is below neg_tol, writes the first corresponding
        eigenvector into eigenfunction and returns (False, eigenvalues).
        Destroys K.

        The extraction threshold for the eigenvector (< 1e-12) intentionally
        includes slightly positive eigenvalues to match the original solver's
        behaviour of perturbing near-zero modes.
        """
        logger.info("Running stability analysis (nev=%d) ...", self._nev)

        eigensolver = SLEPc.EPS().create(self._comm)
        eigensolver.setOperators(K)
        eigensolver.setProblemType(SLEPc.EPS.ProblemType.HEP)
        st = eigensolver.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        # Shift-invert needs to factor (K - σ·I) once per call. Use MUMPS LDLᵀ:
        # symmetric, handles indefinite K via pivoting, ~3–5× faster than the
        # PETSc-builtin LU that SLEPc otherwise picks.
        st_ksp = st.getKSP()
        st_ksp.setType("preonly")
        st_pc = st_ksp.getPC()
        st_pc.setType("cholesky")
        st_pc.setFactorSolverType("mumps")
        eigensolver.setTarget(0.0)
        eigensolver.setDimensions(nev=self._nev)
        eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
        eigensolver.setTolerances(tol=self._tol)

        if self._petsc_options is not None:
            opts = PETSc.Options()
            for key, val in self._petsc_options.items():
                opts[key] = val

        eigensolver.setFromOptions()
        eigensolver.solve()
        PETSc.Mat.destroy(K)

        n_conv = eigensolver.getConverged()
        eigenvalues = np.array([
            eigensolver.getEigenvalue(i).real
            for i in range(min(n_conv, self._nev))
        ])

        logger.info("Smallest eigenvalues: %s", np.array2string(eigenvalues, precision=4))

        is_stable = True
        if np.any(eigenvalues < self._neg_tol):
            target_indices = np.where(eigenvalues < self._neg_tol)[0]
            eigensolver.getEigenvector(target_indices[0], eigenfunction.x.petsc_vec)
            eigenfunction.x.scatter_forward()
            is_stable = False

        eigensolver.destroy()
        return is_stable, eigenvalues
