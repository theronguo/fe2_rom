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

    def __init__(self, comm, nev: int = 5, neg_tol: float = -1e-12):
        self._comm = comm
        self._nev = nev
        self._neg_tol = neg_tol

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
        eigensolver.setTarget(0.0)
        eigensolver.setDimensions(nev=self._nev)
        eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
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
