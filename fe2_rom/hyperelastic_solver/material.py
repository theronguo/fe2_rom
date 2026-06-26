from abc import ABC, abstractmethod
from typing import Callable, Any
import ufl


class MaterialModel(ABC):
    @abstractmethod
    def strain_energy(self, F) -> Any:
        """Return UFL scalar strain energy density W(F). F must be a ufl.variable."""
        ...

    def first_pk_stress(self, F) -> Any:
        """First Piola-Kirchhoff stress P = dW/dF. Requires F to be a ufl.variable."""
        return ufl.diff(self.strain_energy(F), F)
    
    def tangent_moduli(self, F) -> Any:
        """Tangent moduli A = dP/dF. Requires F to be a ufl.variable."""
        P = self.first_pk_stress(F)
        return ufl.diff(P, F)


class NeoHookean(MaterialModel):
    def __init__(self, mu: float, lmbda: float):
        self.mu = mu
        self.lmbda = lmbda

    def strain_energy(self, F):
        mu, lmbda = self.mu, self.lmbda
        J = ufl.det(F)
        C = F.T * F
        return (mu / 2) * (ufl.tr(C) - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J)) ** 2


class StVenantKirchhoff(MaterialModel):
    """St. Venant–Kirchhoff: S = ℂ:E with E = ½(FᵀF − I).

    For isotropic ℂ this is S = λ tr(E) I + 2μ E, whose energy potential is
        W(F) = ½ E:ℂ:E = (λ/2) (tr E)² + μ (E:E),
    so that S = dW/dE and the solver's P = dW/dF = F·S follow by autodiff.
    """

    def __init__(self, mu: float, lmbda: float):
        self.mu = mu
        self.lmbda = lmbda

    def strain_energy(self, F):
        mu, lmbda = self.mu, self.lmbda
        d = F.ufl_shape[0]
        E = 0.5 * (F.T * F - ufl.Identity(d))
        return (lmbda / 2) * ufl.tr(E) ** 2 + mu * ufl.inner(E, E)


class BertoldiHyperelastic(MaterialModel):
    """ψ = c1(I1−3) + c2(I1−3)² − 2c1 ln J + ½K(J−1)²  (van Bree 2020, Eq. 55).

    Plane-strain 2-D: F is extended to diag(F, 1) so I1 counts all three
    principal stretches and J = det F_3D.  This avoids the spurious residual
    stress that arises if I1 = tr(C_2D) − 2 at F = I.
    """

    def __init__(self, c1: float, c2: float, K: float):
        self.c1 = c1
        self.c2 = c2
        self.K = K

    def strain_energy(self, F):
        c1, c2, K = self.c1, self.c2, self.K
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
            c1 * (I1 - 3)
            + c2 * (I1 - 3) ** 2
            - 2 * c1 * ufl.ln(J)
            + 0.5 * K * (J - 1) ** 2
        )
