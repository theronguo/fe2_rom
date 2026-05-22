"""Strain-energy function used in van Bree et al. (2020), Example 1, Eq. (55).

    ψ(F) = c1 (I1 − 3) + c2 (I1 − 3)^2 − 2 c1 ln J + 1/2 K (J − 1)^2

Parameters (Table 1, both examples): c1 = 0.55 MPa, c2 = 0.3 MPa, K = 55 MPa.
"""

import ufl

from fe2_rom.hyperelastic_solver.material import MaterialModel


class BertoldiHyperelastic(MaterialModel):
    def __init__(self, c1: float, c2: float, K: float):
        self.c1 = c1
        self.c2 = c2
        self.K = K

    def strain_energy(self, F):
        # Plane-strain interpretation in 2D: extend F to F3 = diag(F, 1) so that
        # I1 = tr(C) and J = det(F) are evaluated as genuine 3D invariants.
        # Without this, (I1 - 3) = -1 at F = I_2x2 and the c2*(I1-3)^2 term
        # creates a spurious residual stress -4 c2 I at the reference state.
        gdim = F.ufl_shape[0]
        if gdim == 2:
            F3 = ufl.as_matrix([
                [F[0, 0], F[0, 1], 0.0],
                [F[1, 0], F[1, 1], 0.0],
                [0.0,     0.0,     1.0],
            ])
        else:
            F3 = F
        J = ufl.det(F3)
        I1 = ufl.tr(F3.T * F3)
        return (
            self.c1 * (I1 - 3)
            + self.c2 * (I1 - 3) ** 2
            - 2 * self.c1 * ufl.ln(J)
            + 0.5 * self.K * (J - 1) ** 2
        )
