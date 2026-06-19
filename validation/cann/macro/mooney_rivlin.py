"""Generalized Mooney–Rivlin reference material (UFL ``MaterialModel``) for the
macroscopic comparison — the *ground truth* the CANN was trained to mimic.

Strain energy (Linka et al. (2021) eq. 18, the generated reference) plus a
volumetric penalty so the displacement-only problem is well posed::

    Ψ = Σ_{i=1..3} c_i0 (I_C − 3)^i + c_0i (II_C − 3)^i + ½ κ (J − 1)²

The same ``κ`` is used by the CANN macro material so the comparison isolates the
CANN's fidelity to the *isochoric* part. Parameters are Table 1 [MPa]; kept in
sync with ``validation/cann/validate_mooney_rivlin.py`` (the CANN's training
data).
"""
from __future__ import annotations

import ufl

from fe2_rom.hyperelastic_solver import MaterialModel

# Mooney–Rivlin parameters, Linka et al. (2021) Table 1 [MPa]
C10, C20, C30 = 1.6e-1, -1.4e-3, 3.9e-5
C01, C02, C03 = 1.5e-2, -2.0e-6, 1.0e-10


class GeneralizedMooneyRivlin(MaterialModel):
    """Generalized Mooney–Rivlin (eq. 18) + ½κ(J−1)² volumetric penalty.

    2D ``F`` is embedded as ``diag(F, 1)`` (plane strain) so the invariants count
    all three principal stretches and ``J = det F₃ₓ₃`` — matching the CANN
    material; in 3D ``F`` is used directly.
    """

    def __init__(self, kappa: float, coeffs: dict | None = None):
        self.kappa = float(kappa)
        self.c = coeffs or dict(c10=C10, c20=C20, c30=C30,
                                c01=C01, c02=C02, c03=C03)

    def strain_energy(self, F):
        gdim = F.ufl_shape[0]
        if gdim == 2:
            F3 = ufl.as_matrix([
                [F[0, 0], F[0, 1], 0.0],
                [F[1, 0], F[1, 1], 0.0],
                [0.0,     0.0,     1.0],
            ])
        else:
            F3 = F
        C = F3.T * F3
        J = ufl.det(F3)
        Ic = ufl.tr(C)
        IIc = 0.5 * (Ic**2 - ufl.tr(C * C))
        a, b = Ic - 3.0, IIc - 3.0
        c = self.c
        W_iso = (c["c10"] * a + c["c20"] * a**2 + c["c30"] * a**3
                 + c["c01"] * b + c["c02"] * b**2 + c["c03"] * b**3)
        return W_iso + 0.5 * self.kappa * (J - 1.0) ** 2
